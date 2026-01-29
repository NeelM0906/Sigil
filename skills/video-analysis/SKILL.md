---
name: video_analysis
description: Analyze videos - transcribe audio, extract frames, answer questions about video content
---

# Video Analysis Skill

Process and analyze video files: transcribe speech, describe content, answer questions.

## Capabilities

- 🎤 Audio transcription (Whisper)
- 🖼️ Frame extraction and analysis
- 📝 Content summarization
- ❓ Question answering about video

## When to Use

Automatically triggered when:
- User uploads video file (MP4, AVI, MOV, MKV)
- User asks to "analyze video" or "transcribe this"
- User sends video with question

## How It Works

1. User uploads video via WhatsApp
2. Skill detects video file
3. Extracts audio → Transcribes with Whisper
4. Extracts key frames → Analyzes content
5. Returns transcript + summary + answers questions

## Usage
```bash
# Analyze video
python ~/clawd/skills/video-analysis/analyze.py /path/to/video.mp4

# With specific question
python ~/clawd/skills/video-analysis/analyze.py /path/to/video.mp4 "What is the main topic?"
```

## Supported Formats

**Video:** MP4, AVI, MKV, MOV, WMV, FLV, WebM
**Audio:** MP3, WAV, M4A, FLAC, OGG, AAC

## Output

- Full transcript
- Video duration and metadata
- Main topics and summary
- Answers to specific questions