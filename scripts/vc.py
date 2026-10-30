#!/usr/bin/env python3
"""vc — batch video captions, fully local.

Pipeline per video:
  transcribe  (optional silence cut) -> whisper.cpp word timestamps -> editable captions/<stem>.srt
  preview     storyboard: a few frames with burned-in captions -> output/<stem>.preview.jpg
  render      captions/<stem>.srt -> ASS style -> burned into output/<stem>.mp4 (+ .srt copy)
  run         transcribe + render in one go (no review step)
  setup       download the Whisper models
  check       verify tools + models

Folder layout (next to the videos):
  <dir>/_work/      intermediates (cut sources, raw whisper json, meta)
  <dir>/captions/   editable SRT per video  <- review these
  <dir>/output/     finished videos + srt + preview
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


def find_tool(name):
    """ffmpeg-full first: plain Homebrew ffmpeg has no libass."""
    for prefix in ("/opt/homebrew/opt/ffmpeg-full/bin", "/usr/local/opt/ffmpeg-full/bin"):
        if (Path(prefix) / name).exists():
            return str(Path(prefix) / name)
    return shutil.which(name) or name


def find_model(name):
    for d in MODEL_DIRS:
        if (d / name).exists():
            return d / name
    return MODEL_DIRS[0] / name


FFMPEG = find_tool("ffmpeg")
FFPROBE = find_tool("ffprobe")
WHISPER = shutil.which("whisper-cli") or "whisper-cli"
MODEL_DIRS = [Path.home() / ".local/share/whisper-models", Path.home() / "whisper-models"]
MODEL_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin"
VAD_URL = "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin"
MODEL = Path(os.environ["VC_MODEL"]) if os.environ.get("VC_MODEL") else find_model(MODEL_URL.rsplit("/", 1)[1])
VAD_MODEL = find_model(VAD_URL.rsplit("/", 1)[1])
SKILL_DIR = Path(__file__).resolve().parent.parent
GLOSSARY = SKILL_DIR / "glossary.txt"
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".webm"}
HDR_TRC = {"arib-std-b67", "smpte2084"}


# ---------- helpers ----------

def run(cmd, **kw):
    cmd = [str(c) for c in cmd]
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        sys.exit(f"\n✗ command failed: {' '.join(cmd[:3])} ...\n{r.stderr[-2000:]}")
    return r


def find_videos(path: Path):
    if not path.exists():
        sys.exit(f"✗ not found: {path}")
    if path.is_file():
        return path.parent, [path]
    vids = sorted(p for p in path.iterdir()
                  if p.suffix.lower() in VIDEO_EXT and not p.name.startswith("."))
    if not vids:
        sys.exit(f"✗ no videos found in {path}")
    stems = [p.stem for p in vids]
    clashes = sorted({s for s in stems if stems.count(s) > 1})
    if clashes:  # captions and output are named after the stem
        sys.exit(f"✗ same name, different extension: {', '.join(clashes)}. Rename one of them.")
    return path, vids


def probe(src: Path):
    r = run([FFPROBE, "-v", "error", "-show_streams", "-show_format", "-of", "json", src])
    info = json.loads(r.stdout)
    v = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    if not v:
        sys.exit(f"✗ no video stream in {src.name}")
    w, h = int(v["width"]), int(v["height"])
    rot = 0
    for sd in v.get("side_data_list", []):
        if "rotation" in sd:
            rot = int(float(sd["rotation"]))
    rot = rot or int(v.get("tags", {}).get("rotate", 0))
    if abs(rot) % 180 == 90:  # ffmpeg autorotates, so the displayed frame is swapped
        w, h = h, w
    return {
        "w": w, "h": h,
        "dur": float(info["format"].get("duration", 0)),
        "has_audio": a is not None,
        "hdr": v.get("color_transfer") in HDR_TRC,
    }


def sdr_filters(meta):
    """Tonemap iPhone HDR (HLG/PQ) to SDR bt709 so it doesn't look washed out."""
    if not meta["hdr"]:
        return []
    return ["zscale=t=linear:npl=100", "format=gbrpf32le", "zscale=p=bt709",
            "tonemap=tonemap=hable:desat=0", "zscale=t=bt709:m=bt709:r=tv", "format=yuv420p"]


def fmt_srt_time(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def parse_srt_time(s):
    h, m, rest = s.strip().replace(".", ",").split(":")
    sec, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000


def write_srt(caps, path: Path):
    lines = []
    for i, c in enumerate(caps, 1):
        lines += [str(i), f"{fmt_srt_time(c['start'])} --> {fmt_srt_time(c['end'])}", c["text"], ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def read_srt(path: Path):
    caps = []
    for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8").strip()):
        rows = [r for r in block.strip().splitlines() if r.strip()]
        ti = next((i for i, r in enumerate(rows) if "-->" in r), None)
        if ti is None:
            continue
        a, b = rows[ti].split("-->")
        text = "\n".join(rows[ti + 1:]).strip()
        if text:
            caps.append({"start": parse_srt_time(a), "end": parse_srt_time(b), "text": text})
    return caps


# ---------- step 1: prep (silence cut) ----------

def keep_segments(src, dur, db, min_sil, pad):
    r = subprocess.run([FFMPEG, "-hide_banner", "-nostats", "-i", str(src), "-vn",
                        "-af", f"silencedetect=noise={db}dB:d={min_sil}", "-f", "null", "-"],
                       capture_output=True, text=True)
    starts = [float(x) for x in re.findall(r"silence_start: (-?[\d.]+)", r.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", r.stderr)]
    ends += [dur] * (len(starts) - len(ends))
    keep, t = [], 0.0
    for s, e in zip(starts, ends):
        seg = (t, min(dur, max(0.0, s) + pad))
        if seg[1] - seg[0] > 0.05:
            keep.append(seg)
        t = max(0.0, e - pad)
    if dur - t > 0.05:
        keep.append((t, dur))
    merged = []
    for a, b in keep:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def cut_silences(src, dst, meta, args):
    segs = keep_segments(src, meta["dur"], args.silence_db, args.min_silence, args.pad)
    removed = meta["dur"] - sum(b - a for a, b in segs)
    if len(segs) <= 1 and removed < 0.3:
        return None, 0.0
    pre = ",".join(sdr_filters(meta))
    parts, labels = [], ""
    for i, (a, b) in enumerate(segs):
        vchain = f"trim=start={a:.3f}:end={b:.3f},setpts=PTS-STARTPTS" + (f",{pre}" if pre else "")
        parts.append(f"[0:v]{vchain}[v{i}]")
        parts.append(f"[0:a]atrim=start={a:.3f}:end={b:.3f},asetpts=PTS-STARTPTS[a{i}]")
        labels += f"[v{i}][a{i}]"
    fc = ";".join(parts) + f";{labels}concat=n={len(segs)}:v=1:a=1[v][a]"
    run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", src, "-filter_complex", fc,
         "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", dst])
    return dst, removed


# ---------- step 2: transcribe ----------

def whisper_words(src: Path, prefix: Path, lang, prompt):
    wav = Path(f"{prefix}.wav")
    run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", src,
         "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav])
    cmd = [WHISPER, "-m", MODEL, "-f", wav, "-l", lang, "-ml", "1", "-sow",
           "-oj", "-of", prefix, "-np", "-t", os.cpu_count() or 4]
    if VAD_MODEL.exists():
        cmd += ["--vad", "-vm", VAD_MODEL]
    if prompt:
        cmd += ["--prompt", prompt]
    run(cmd)
    wav.unlink(missing_ok=True)
    data = json.loads(Path(f"{prefix}.json").read_text(encoding="utf-8"))
    words = []
    for seg in data.get("transcription", []):
        text = seg["text"].strip()
        if not text or re.fullmatch(r"[\[(].*[\])]", text):  # [BLANK_AUDIO], (music) ...
            continue
        w = {"text": text, "start": seg["offsets"]["from"] / 1000, "end": seg["offsets"]["to"] / 1000}
        if re.fullmatch(r"[^\w]+", text) and words:  # stray punctuation -> previous word
            words[-1]["text"] += text
            continue
        words.append(w)
    return words, data.get("result", {}).get("language", lang)


def group_captions(words, max_words, max_chars, max_gap=0.45):
    caps, cur = [], []
    for w in words:
        if cur:
            joined = " ".join(x["text"] for x in cur) + " " + w["text"]
            if (len(cur) >= max_words or len(joined) > max_chars
                    or w["start"] - cur[-1]["end"] > max_gap
                    or re.search(r"[.!?,;:]$", cur[-1]["text"])):
                caps.append(cur)
                cur = []
        cur.append(w)
    if cur:
        caps.append(cur)
    out = [{"start": c[0]["start"], "end": max(c[-1]["end"], c[0]["start"] + 0.3),
            "text": " ".join(x["text"] for x in c)} for c in caps]
    for a, b in zip(out, out[1:]):  # close tiny gaps (no flicker), never overlap
        if b["start"] - a["end"] < 0.3:
            a["end"] = b["start"]
        a["end"] = min(a["end"], b["start"])
    return out


def cmd_transcribe(args):
    base, vids = find_videos(Path(args.path).expanduser().resolve())
    work, capdir = base / "_work", base / "captions"
    work.mkdir(exist_ok=True)
    capdir.mkdir(exist_ok=True)
    prompt = ", ".join(t.strip() for t in GLOSSARY.read_text(encoding="utf-8").splitlines()
                       if t.strip() and not t.startswith("#")) if GLOSSARY.exists() else ""
    if args.prompt:
        prompt = f"{args.prompt}, {prompt}" if prompt else args.prompt
    for i, v in enumerate(vids, 1):
        stem = v.stem
        srt = capdir / f"{stem}.srt"
        if srt.exists() and not args.force:
            print(f"[{i}/{len(vids)}] {v.name}: captions exist, skip (--force to redo)")
            continue
        print(f"[{i}/{len(vids)}] {v.name}", end="", flush=True)
        meta = probe(v)
        if not meta["has_audio"]:
            print(": no audio, skip")
            continue
        src = v
        if args.cut_silence:
            cut, removed = cut_silences(v, work / f"{stem}.cut.mp4", meta, args)
            if cut:
                src = cut
                print(f"  ✂ {removed:.1f}s silence removed", end="", flush=True)
        words, lang = whisper_words(src, work / stem, args.lang, prompt)
        caps = group_captions(words, args.max_words, args.max_chars)
        write_srt(caps, srt)
        (work / f"{stem}.meta.json").write_text(json.dumps(
            {"original": str(v), "source": str(src), "lang": lang}, indent=2))
        print(f"  → {len(caps)} captions ({lang})")
    print(f"\n✓ review/edit the SRT files in: {capdir}")


# ---------- step 3: render ----------

STYLES = {
    # size/outline/margin are fractions of output height
    "reels": dict(size=0.052, outline=0.0036, shadow=0.0012, margin=0.27, border=1, pop=True),
    "box":   dict(size=0.045, outline=0.008, shadow=0.0, margin=0.27, border=3, pop=True),
    "clean": dict(size=0.042, outline=0.0025, shadow=0.001, margin=0.07, border=1, pop=False),
}


def ass_color(hex_rgb, alpha="00"):
    h = hex_rgb.lstrip("#")
    return f"&H{alpha}{h[4:6]}{h[2:4]}{h[0:2]}".upper()


def ass_time(t):
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02}:{s:02}.{cs:02}"


def build_ass(caps, W, H, args):
    st = STYLES[args.style]
    size = round(H * st["size"] * args.scale)
    margin_v = round(H * (args.margin if args.margin is not None else st["margin"]))
    align = 5 if args.position == "center" else 2
    back = "&H26000000" if st["border"] == 3 else "&H80000000"
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{args.font},{size},{ass_color(args.color)},&H000000FF,{back if st['border'] == 3 else '&H00000000'},{back},-1,0,0,0,100,100,0,0,{st['border']},{max(1, round(H * st['outline']))},{round(H * st['shadow'])},{align},{round(W * 0.08)},{round(W * 0.08)},{margin_v if align == 2 else 0},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    pop = r"{\fscx85\fscy85\t(0,90,\fscx100\fscy100)}" if st["pop"] and not args.no_pop else ""
    events = []
    for c in caps:
        text = c["text"].upper() if args.upper else c["text"]
        text = text.replace("{", "(").replace("}", ")").replace("\n", r"\N")
        events.append(f"Dialogue: 0,{ass_time(c['start'])},{ass_time(c['end'])},Default,,0,0,0,,{pop}{text}")
    return head + "\n".join(events) + "\n"


def caption_chain(src, caps, work, args):
    """Filters that burn the captions in. Writes work/render.ass, so run ffmpeg with cwd=work."""
    meta = probe(src)
    vf = sdr_filters(meta)
    W, H = meta["w"], meta["h"]
    if args.vertical:
        if W / H > 9 / 16 + 0.01:
            vf.append("crop=trunc(ih*9/16/2)*2:ih")
        vf.append("scale=1080:1920:force_original_aspect_ratio=decrease,"
                  "pad=1080:1920:(ow-iw)/2:(oh-ih)/2")
        W, H = 1080, 1920
    (work / "render.ass").write_text(build_ass(caps, W, H, args), encoding="utf-8")
    vf += ["ass=render.ass", "format=yuv420p",
           "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709"]
    return vf, W, H


def cmd_preview(args):
    base, vids = find_videos(Path(args.path).expanduser().resolve())
    work, capdir, outdir = base / "_work", base / "captions", base / "output"
    work.mkdir(exist_ok=True)
    outdir.mkdir(exist_ok=True)
    for i, v in enumerate(vids, 1):
        stem = v.stem
        srt = capdir / f"{stem}.srt"
        if not srt.exists():
            print(f"[{i}/{len(vids)}] {v.name}: no captions yet, run transcribe first")
            continue
        caps = read_srt(srt)
        if not caps:
            print(f"[{i}/{len(vids)}] {v.name}: captions are empty, skip")
            continue
        meta_p = work / f"{stem}.meta.json"
        src = Path(json.loads(meta_p.read_text())["source"]) if meta_p.exists() else v
        vf, W, H = caption_chain(src, caps, work, args)
        n = min(args.frames, len(caps))
        picks = [caps[round(j * (len(caps) - 1) / max(n - 1, 1))] for j in range(n)]
        frames = work / f"{stem}.preview"
        frames.mkdir(exist_ok=True)
        tile_w = 270 if H > W else 480
        for j, c in enumerate(picks):
            t = (c["start"] + c["end"]) / 2
            # -ss resets timestamps; shift them back so the captions line up
            chain = [f"setpts=PTS+{t:.3f}/TB"] + vf + [f"scale={tile_w}:-2"]
            run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{t:.3f}",
                 "-i", src.resolve(), "-vf", ",".join(chain), "-frames:v", "1", "-update", "1",
                 "-q:v", "3", frames.resolve() / f"{j:02}.jpg"], cwd=work)
        cols = min(n, 6 if H > W else 3)
        out = outdir / f"{stem}.preview.jpg"
        run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-framerate", "1",
             "-i", frames / "%02d.jpg", "-vf", f"tile={cols}x{math.ceil(n / cols)}",
             "-frames:v", "1", "-update", "1", "-q:v", "3", out])
        shutil.rmtree(frames)
        print(f"[{i}/{len(vids)}] {v.name} → {out}")
    (work / "render.ass").unlink(missing_ok=True)


def cmd_render(args):
    base, vids = find_videos(Path(args.path).expanduser().resolve())
    work, capdir, outdir = base / "_work", base / "captions", base / "output"
    work.mkdir(exist_ok=True)
    outdir.mkdir(exist_ok=True)
    for i, v in enumerate(vids, 1):
        stem = v.stem
        srt = capdir / f"{stem}.srt"
        out = outdir / f"{stem}.mp4"
        if not srt.exists():
            print(f"[{i}/{len(vids)}] {v.name}: no captions yet, run transcribe first")
            continue
        if out.exists() and not args.force:
            print(f"[{i}/{len(vids)}] {v.name}: output exists, skip (--force to redo)")
            continue
        meta_p = work / f"{stem}.meta.json"
        src = Path(json.loads(meta_p.read_text())["source"]) if meta_p.exists() else v
        caps = read_srt(srt)
        vf, W, H = caption_chain(src, caps, work, args)
        print(f"[{i}/{len(vids)}] {v.name} → {W}x{H}, {len(caps)} captions", flush=True)
        run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", src.resolve(),
             "-vf", ",".join(vf), "-c:v", "libx264", "-preset", "medium", "-crf", "19",
             "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
             out.resolve()], cwd=work)
        shutil.copy(srt, outdir / f"{stem}.srt")
    (work / "render.ass").unlink(missing_ok=True)
    print(f"\n✓ done: {outdir}")


def problems():
    found = []
    if not shutil.which(FFMPEG) or not shutil.which(FFPROBE):
        found.append("ffmpeg not found: brew install ffmpeg-full")
    else:
        filters = subprocess.run([FFMPEG, "-hide_banner", "-filters"],
                                 capture_output=True, text=True).stdout
        if not re.search(r"\sass\s", filters):
            found.append("this ffmpeg has no libass: brew install ffmpeg-full")
    if not shutil.which(WHISPER):
        found.append("whisper-cli not found: brew install whisper-cpp")
    if not MODEL.exists():
        found.append(f"no Whisper model at {MODEL}: run `vc.py setup`")
    return found


def cmd_check(_):
    print(f"ffmpeg: {FFMPEG}\nwhisper-cli: {WHISPER}\nmodel: {MODEL}")
    print(f"vad model (optional): {VAD_MODEL if VAD_MODEL.exists() else 'missing, run `vc.py setup`'}")
    found = problems()
    for p in found:
        print(f"✗ {p}")
    if not found:
        print("✓ ready")
    sys.exit(1 if found else 0)


def cmd_setup(_):
    MODEL_DIRS[0].mkdir(parents=True, exist_ok=True)
    for url, target in [(VAD_URL, VAD_MODEL), (MODEL_URL, MODEL)]:
        if target.exists():
            print(f"✓ {target.name} already there")
            continue
        print(f"downloading {target.name} (the main model is 1.6 GB) ...", flush=True)
        part = target.with_suffix(".part")
        urllib.request.urlretrieve(url, part)
        part.rename(target)
        print(f"✓ {target}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_transcribe(p):
        p.add_argument("--lang", default="auto", help="nl, en, fr, ... or auto (default)")
        p.add_argument("--prompt", default="", help="extra words/names to bias spelling")
        p.add_argument("--cut-silence", action="store_true", help="remove silences before captioning")
        p.add_argument("--silence-db", type=float, default=-35)
        p.add_argument("--min-silence", type=float, default=0.6, help="seconds of silence to cut")
        p.add_argument("--pad", type=float, default=0.15, help="seconds kept around speech")
        p.add_argument("--max-words", type=int, default=3)
        p.add_argument("--max-chars", type=int, default=20)

    def add_render(p):
        p.add_argument("--style", choices=STYLES, default="reels")
        p.add_argument("--vertical", action="store_true", help="output 1080x1920 (center crop)")
        p.add_argument("--font", default="Arial Black")
        p.add_argument("--color", default="FFFFFF", help="text colour hex, e.g. FFE600")
        p.add_argument("--scale", type=float, default=1.0, help="font size multiplier")
        p.add_argument("--position", choices=["lower", "center"], default="lower")
        p.add_argument("--margin", type=float, default=None, help="bottom margin as fraction of height")
        p.add_argument("--upper", action="store_true", help="UPPERCASE captions")
        p.add_argument("--no-pop", action="store_true", help="disable pop-in animation")

    for name, fn in [("transcribe", add_transcribe), ("render", add_render)]:
        p = sub.add_parser(name)
        p.add_argument("path")
        p.add_argument("--force", action="store_true")
        fn(p)
    p = sub.add_parser("preview")
    p.add_argument("path")
    p.add_argument("--frames", type=int, default=6, help="how many frames in the storyboard")
    add_render(p)
    p = sub.add_parser("run")
    p.add_argument("path")
    p.add_argument("--force", action="store_true")
    add_transcribe(p)
    add_render(p)
    sub.add_parser("setup")
    sub.add_parser("check")

    args = ap.parse_args()
    if args.cmd == "check":
        cmd_check(args)
    elif args.cmd == "setup":
        cmd_setup(args)
    else:
        found = problems()
        if found:
            sys.exit("✗ " + "\n✗ ".join(found))
        if args.cmd == "transcribe":
            cmd_transcribe(args)
        elif args.cmd == "preview":
            cmd_preview(args)
        elif args.cmd == "render":
            cmd_render(args)
        else:
            cmd_transcribe(args)
            cmd_render(args)


if __name__ == "__main__":
    main()
