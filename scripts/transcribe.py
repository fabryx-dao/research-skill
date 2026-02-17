#!/usr/bin/env python3
"""
Stage 1: Transcribe video using AssemblyAI with speaker diarization.
"""

import os
import json
import time
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
from urllib.parse import urlparse, parse_qs

import assemblyai as aai

from validate import save_validated

logger = logging.getLogger(__name__)


def extract_video_id(video_url: str) -> str:
    """Extract YouTube video ID from URL."""
    parsed = urlparse(video_url)
    
    if 'youtube.com' in parsed.netloc:
        query = parse_qs(parsed.query)
        return query.get('v', [''])[0]
    elif 'youtu.be' in parsed.netloc:
        return parsed.path.lstrip('/')
    else:
        raise ValueError(f"Invalid YouTube URL: {video_url}")


def download_audio(video_url: str, output_dir: Path) -> Path:
    """
    Download audio from YouTube video using yt-dlp.
    
    Args:
        video_url: YouTube video URL
        output_dir: Directory to save audio
    
    Returns:
        Path to downloaded audio file
    """
    video_id = extract_video_id(video_url)
    output_path = output_dir / f"video-{video_id}.mp3"
    
    if output_path.exists():
        logger.info(f"Audio already exists: {output_path}")
        return output_path
    
    logger.info(f"Downloading audio: {video_url}")
    
    cmd = [
        'yt-dlp',
        '-x',  # Extract audio
        '--audio-format', 'mp3',
        '--audio-quality', '0',  # Best quality
        '-o', str(output_path),
        video_url
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr}")
    
    if not output_path.exists():
        raise RuntimeError(f"Audio file not created: {output_path}")
    
    logger.info(f"Audio downloaded: {output_path}")
    return output_path


def transcribe_with_assemblyai(
    audio_path: Path,
    api_key: str,
    speaker_labels: bool = True
) -> Dict[str, Any]:
    """
    Transcribe audio using AssemblyAI with speaker diarization.
    
    Args:
        audio_path: Path to audio file
        api_key: AssemblyAI API key
        speaker_labels: Enable speaker diarization
    
    Returns:
        Transcript data with speaker labels
    """
    aai.settings.api_key = api_key
    
    config = aai.TranscriptionConfig(
        speech_models=['universal-2'],  # Use universal-2 model
        speaker_labels=speaker_labels,
        speakers_expected=2  # Interviewer + subject
    )
    
    logger.info(f"Uploading audio to AssemblyAI: {audio_path}")
    transcriber = aai.Transcriber()
    
    transcript = transcriber.transcribe(str(audio_path), config=config)
    
    if transcript.status == aai.TranscriptStatus.error:
        raise RuntimeError(f"AssemblyAI transcription failed: {transcript.error}")
    
    logger.info("Transcription complete")
    
    # Convert to our format
    result = {
        'success': True,
        'videoId': audio_path.stem.replace('video-', ''),
        'method': 'assemblyai',
        'duration': transcript.audio_duration,
        'speakers': [],
        'transcript': []
    }
    
    # Collect speaker labels
    speakers_seen = set()
    
    for utterance in transcript.utterances:
        speaker_label = utterance.speaker
        speakers_seen.add(speaker_label)
        
        result['transcript'].append({
            'text': utterance.text,
            'speaker': speaker_label,
            'start': utterance.start / 1000.0,  # Convert ms to seconds
            'end': utterance.end / 1000.0
        })
    
    # Map speakers (A = first speaker, B = second, etc.)
    speaker_mapping = {s: chr(65 + i) for i, s in enumerate(sorted(speakers_seen))}
    
    # Update speaker labels in transcript
    for entry in result['transcript']:
        entry['speaker'] = speaker_mapping[entry['speaker']]
    
    # Add speaker info
    result['speakers'] = [
        {'speaker': mapped, 'label': f'Speaker {mapped}'}
        for mapped in sorted(speaker_mapping.values())
    ]
    
    return result


def transcribe_video(
    video_url: str,
    api_key: str,
    output_dir: Optional[str] = None,
    item_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Complete transcription pipeline: download audio + transcribe.
    
    Args:
        video_url: YouTube video URL
        api_key: AssemblyAI API key
        output_dir: Output directory (defaults to /tmp/research)
        item_id: Override item ID (e.g., "0004"), otherwise auto-detect
    
    Returns:
        Dict with 'path' (transcript file) and 'video_id'
    """
    # Setup output directory
    if output_dir:
        out_path = Path(output_dir)
    else:
        out_path = Path('/tmp/research')
    
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Download audio
    audio_path = download_audio(video_url, out_path)
    
    # Transcribe
    transcript_data = transcribe_with_assemblyai(audio_path, api_key)
    
    # Determine item ID
    if not item_id:
        # Try to auto-detect from existing files
        existing_files = list(out_path.glob('video-*-transcript.json'))
        if existing_files:
            max_id = max([
                int(f.stem.split('-')[1])
                for f in existing_files
                if f.stem.split('-')[1].isdigit()
            ], default=0)
            item_id = f"{max_id + 1:04d}"
        else:
            item_id = "0001"
    
    # Save transcript
    transcript_path = out_path / f"video-{item_id}-transcript.json"
    save_validated(transcript_data, transcript_path, 'transcript')
    
    # Clean up audio file
    if audio_path.exists():
        audio_path.unlink()
        logger.info(f"Cleaned up audio: {audio_path}")
    
    return {
        'path': str(transcript_path),
        'video_id': item_id,
        'utterances': len(transcript_data['transcript']),
        'duration': transcript_data.get('duration', 0)
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Transcribe video with AssemblyAI')
    parser.add_argument('video_url', help='YouTube video URL')
    parser.add_argument('--api-key', help='AssemblyAI API key (or use ASSEMBLYAI_API_KEY env var)')
    parser.add_argument('--output-dir', help='Output directory')
    parser.add_argument('--item-id', help='Item ID (e.g., 0004)')
    
    args = parser.parse_args()
    
    api_key = args.api_key or os.getenv('ASSEMBLYAI_API_KEY')
    if not api_key:
        print("Error: ASSEMBLYAI_API_KEY not found")
        exit(1)
    
    result = transcribe_video(
        video_url=args.video_url,
        api_key=api_key,
        output_dir=args.output_dir,
        item_id=args.item_id
    )
    
    print(json.dumps(result, indent=2))
