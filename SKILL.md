---
name: video-captions
description: Use when the user wants videos captioned, subtitled, trimmed of silences, or turned into 9:16 Reels/TikToks, for a single video or a whole folder ("caption my videos", "subtitle this folder", "cut the silences", "make reels from these"). Transcription and rendering run locally with whisper.cpp and ffmpeg.
---

# Video captions (batch, local)

One script does everything: `python3 <this skill's folder>/scripts/vc.py`.
It needs ffmpeg with libass, whisper.cpp and a Whisper model. If anything fails, run `vc.py check`:
it says what is missing and how to install it.

## Workflow (always this order)

1. **Get the folder.** Ask only if the user gave none. A single file also works.
2. **Transcribe:**
   `vc.py transcribe "<dir>" [--cut-silence] [--lang nl|en|auto]`
   - Use `--cut-silence` by default for talking-head videos (removes pauses over 0.6s, keeps 0.15s
     padding). Not for b-roll or music videos.
   - Language is detected per video. Force it with `--lang` if a video mixes languages.
   - Names and jargon: `glossary.txt` next to this file is sent as the Whisper prompt. Add
     project-specific words with `--prompt "Acme, Kubernetes"`.
3. **Review before showing anything.** Read every `<dir>/captions/*.srt` and fix obvious
   mistranscriptions (names, brand spelling, broken punctuation). Never change timestamps unless a
   caption is clearly wrong. Tell the user in one line what you fixed.
4. **Storyboard:**
   `vc.py preview "<dir>" --vertical`
   Writes `<dir>/output/<name>.preview.jpg`: six frames from across the video with the captions
   burned in. Read the image yourself and check size, placement and cut-off text. Then show the user
   the path and **stop until they say go**. A full render is the slow step, so this is where the
   look gets approved. Use the same style flags here as in the render.
5. **Render:**
   `vc.py render "<dir>" --vertical`
   Output: `<dir>/output/<name>.mp4` plus `.srt` for platforms that take closed captions.
6. **Verify** one frame per video before reporting, using the `ffmpeg` that `vc.py check` prints:
   `ffmpeg -ss 2 -i out.mp4 -frames:v 1 -vf scale=360:-1 frame.png`, then read the image.
7. Report the output folder and anything that still looks wrong.

Shortcut when the user says no review is needed: `vc.py run "<dir>" --cut-silence --vertical`.

## Options

| Want | Flag |
|---|---|
| Style | `--style reels` (default) · `box` (black box behind text) · `clean` (small, low, for landscape) |
| 9:16 from landscape | `--vertical` (center crop to 1080×1920) |
| Colour | `--color FFE600` (yellow) |
| Bigger/smaller | `--scale 1.2` |
| Caps | `--upper` |
| Middle of screen | `--position center` |
| Higher/lower | `--margin 0.35` (fraction of height from bottom) |
| Font | `--font "Futura"` (any installed font name) |
| More storyboard frames | `--frames 9` (preview step) |
| Shorter/longer lines | `--max-words 2 --max-chars 14` (transcribe step) |
| Redo | `--force` (otherwise existing captions and outputs are skipped) |

Changing style: rerun `preview`, then `render --force`. Changing wording: edit the SRT, then the same.

## Gotchas

- Plain `brew install ffmpeg` has no libass, so captions cannot be burned in. Use `ffmpeg-full`.
- iPhone HDR (HLG/Dolby Vision) is tonemapped to SDR automatically. Otherwise it looks washed out
  on Instagram.
- `_work/` holds the silence-cut copies. Keep it until the final render: without it, `render` uses
  the uncut original and the captions run out of sync.
- Timestamps come from whisper.cpp word offsets, grouped into 3-word captions. Captions break on
  punctuation and on pauses over 0.45s.
- The model is `large-v3-turbo` (1.6 GB). Set `VC_MODEL=/path/to/model.bin` to use another one.
