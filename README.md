# video-captions

A Claude Code skill that captions your videos on your own machine, from one clip to a whole folder.

Give Claude a folder and it transcribes every video, corrects the captions, shows you a storyboard, and renders the videos once you approve. Your video stays on your machine: transcription and rendering are free and local. Claude reads the caption text and storyboard, which counts as normal Claude Code usage.

## What it does

1. **Cuts silences** from talking-head videos, if you want that.
2. **Transcribes** with Whisper, with word-level timing. Language is detected per video.
3. **Corrects the captions:** Claude reads the SRT files and fixes names and spelling.
4. **Shows a storyboard:** six frames with the captions burned in, so you can judge the look before the slow render.
5. **Renders** an MP4 with captions, plus an SRT file. Add `--vertical` for 1080×1920 reels.

Styles include `reels`, `box` and `clean`, with options for font, colour, size, position and caps.

## Install

You need Claude Code and Homebrew. Tested on an Apple Silicon Mac. Linux should work with the same tools, but I haven't tested it.

```bash
brew install ffmpeg-full whisper-cpp
git clone https://github.com/knallen-spec/video-captions.git ~/.claude/skills/video-captions
python3 ~/.claude/skills/video-captions/scripts/vc.py setup
python3 ~/.claude/skills/video-captions/scripts/vc.py check
```

`setup` downloads the 1.6 GB Whisper model. Plain `ffmpeg` won't work, because it can't burn in captions.

Add your name and product names to `glossary.txt` so Whisper spells them correctly.

## Use

Start a new Claude Code session and ask:

> Caption the videos in ~/Movies/reels, cut the silences and make them vertical.

You can also run `scripts/vc.py` directly. Run it with `--help` to see the commands.

## License

MIT
