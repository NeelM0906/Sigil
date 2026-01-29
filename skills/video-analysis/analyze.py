#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video Analysis Skill for Sigil
Transcribes audio and analyzes video content
"""

import sys
import os
import json


def analyze_video(video_path, question=None):
    """
    Analyze video: transcribe audio and answer questions

    Args:
        video_path: Path to video file
        question: Optional question about the video

    Returns:
        Dict with transcript, analysis, metadata
    """

    # Check dependencies
    try:
        import whisper
    except ImportError:
        return {
            'error': 'openai-whisper not installed',
            'install': 'pip install openai-whisper'
        }

    try:
        from openai import OpenAI
    except ImportError:
        return {
            'error': 'openai not installed',
            'install': 'pip install openai'
        }

    # Check file exists
    if not os.path.exists(video_path):
        return {'error': f'Video file not found: {video_path}'}

    # Check API key
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        return {'error': 'OPENAI_API_KEY not set'}

    openai_client = OpenAI(api_key=api_key)

    print("=" * 70)
    print("VIDEO ANALYSIS")
    print("=" * 70)
    print(f"\nVideo: {video_path}")
    print(f"Size: {os.path.getsize(video_path) / (1024 * 1024):.1f} MB\n")

    # Step 1: Transcribe audio with Whisper
    print("Step 1/2: Transcribing audio...")
    print("(This may take a few minutes for long videos)")

    try:
        model = whisper.load_model("base")
        result = model.transcribe(video_path, fp16=False)
        transcript = result["text"]
        duration = result.get("duration", 0)
    except Exception as e:
        return {'error': f'Transcription failed: {e}'}

    print(f"✓ Transcription complete! Duration: {duration:.1f}s ({duration / 60:.1f} min)")
    print(f"  Transcript: {len(transcript)} characters, {len(transcript.split())} words\n")

    # Step 2: Analyze with GPT-4
    print("Step 2/2: Analyzing content...")

    # Build prompt
    if question:
        prompt = f"""Based on this video transcript, answer the question.

Question: {question}

Video Transcript:
{transcript}

Provide a clear, detailed answer based on the content."""
    else:
        prompt = f"""Analyze this video transcript and provide:

1. **Main Topic** (1-2 sentences)
2. **Key Points** (5-7 bullet points)
3. **Summary** (3-5 sentences)

Video Transcript:
{transcript}

Format with clear headers."""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a video content analyst. Analyze transcripts accurately."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        analysis = response.choices[0].message.content
    except Exception as e:
        return {'error': f'Analysis failed: {e}'}

    # Build result
    result_data = {
        'video_path': video_path,
        'duration_seconds': duration,
        'duration_minutes': round(duration / 60, 1),
        'transcript': transcript,
        'word_count': len(transcript.split()),
        'analysis': analysis,
        'question': question if question else None,
        'success': True
    }

    # Output
    print("\n" + "=" * 70)
    print("VIDEO INFORMATION")
    print("=" * 70)
    print(f"Duration: {duration:.1f}s ({duration / 60:.1f} minutes)")
    print(f"Transcript: {len(transcript)} characters, {len(transcript.split())} words")

    print("\n" + "=" * 70)
    print("TRANSCRIPT")
    print("=" * 70)
    if len(transcript) > 2000:
        print(transcript[:2000])
        print(f"\n[... {len(transcript) - 2000} more characters ...]")
    else:
        print(transcript)

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print(analysis)
    print("\n" + "=" * 70)

    return result_data


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze.py <video_path> [question]")
        print("\nExamples:")
        print('  python analyze.py "video.mp4"')
        print('  python analyze.py "video.mp4" "What is the main topic?"')
        sys.exit(1)

    video_path = sys.argv[1]
    question = ' '.join(sys.argv[2:]) if len(sys.argv) > 2 else None

    result = analyze_video(video_path, question)

    if result and result.get('success'):
        # Optionally output JSON
        if '--json' in sys.argv:
            print("\n\nJSON OUTPUT:")
            print(json.dumps(result, indent=2))
        sys.exit(0)
    else:
        print(f"\n❌ Error: {result.get('error', 'Unknown error')}")
        sys.exit(1)