#!/usr/bin/env python3
"""
Stage 2: Extract domain-relevant assertions from transcript using DSPy.
Uses chained FFT approach: Lowpass → Bandpass → Highpass
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import dspy
from dspy import Signature, InputField, OutputField

from validate import validate_and_load, save_validated

logger = logging.getLogger(__name__)


# Lowpass: Extract the core thesis/message
class LowpassFilter(Signature):
    """Extract the single core thesis or main message from a transcript.
    
    This is the lowest frequency - the fundamental idea that unifies everything.
    Think: "If you could only remember ONE thing from this, what would it be?"
    """
    
    transcript: str = InputField(desc="Full transcript text")
    domain: str = InputField(desc="Domain context (e.g., 'pyramid archaeology and industrial chemistry')")
    
    thesis: str = OutputField(
        desc="Single statement capturing the core thesis. "
             "Must be ONE complete sentence. "
             "This is the foundation - everything else elaborates on this."
    )


# Bandpass: Extract mid-level claims/framework
class BandpassFilter(Signature):
    """Extract mid-frequency claims - the conceptual framework and key assertions.
    
    Given the core thesis, extract the main claims that support/develop it.
    These are complete ideas but not detailed evidence yet.
    Think: The chapter headings or main points in an outline.
    """
    
    transcript: str = InputField(desc="Full transcript text")
    domain: str = InputField(desc="Domain context")
    thesis: str = InputField(desc="Core thesis from lowpass filter")
    
    claims: List[str] = OutputField(
        desc="List of mid-level claims (5-15 claims). "
             "Each should be a complete assertion. "
             "Should elaborate on the thesis but NOT repeat it. "
             "Don't include detailed evidence yet - just the conceptual framework."
    )


# Highpass: Extract supporting details/evidence
class HighpassFilter(Signature):
    """Extract high-frequency details - specific evidence, examples, and technical details.
    
    Given the thesis and main claims, extract the supporting details.
    These are the facts, measurements, observations that back up the framework.
    Think: The evidence, the specifics, the "receipts".
    """
    
    transcript: str = InputField(desc="Full transcript text")
    domain: str = InputField(desc="Domain context")
    thesis: str = InputField(desc="Core thesis from lowpass filter")
    claims: List[str] = InputField(desc="Mid-level claims from bandpass filter")
    
    details: List[str] = OutputField(
        desc="List of high-frequency details (20-100 details). "
             "Each should be specific evidence, measurements, or observations. "
             "Should support the claims but NOT repeat them. "
             "Include: technical specs, measurements, material properties, observations, etc."
    )


def fft(
    transcript_path: Path,
    domain: str,
    model: str = "claude-sonnet-4-5"
) -> Dict[str, Any]:
    """
    Chained FFT extraction: Lowpass → Bandpass → Highpass
    
    Mimics signal processing by extracting hierarchical abstractions:
    - Lowpass: Core thesis (1 statement)
    - Bandpass: Conceptual framework (5-15 claims)
    - Highpass: Supporting evidence (20-100 details)
    
    Args:
        transcript_path: Path to transcript JSON
        domain: Domain description
        model: DSPy model to use
    
    Returns:
        Dict with 'thesis', 'claims', 'details', 'all_statements'
    """
    # Load transcript
    transcript_data = validate_and_load(transcript_path, 'transcript')
    transcript_entries = transcript_data['transcript']
    
    # Convert to full text
    transcript_text = " ".join(entry['text'] for entry in transcript_entries)
    
    logger.info(f"Transcript: {len(transcript_entries)} utterances, {len(transcript_text)} chars")
    
    # Setup DSPy
    lm = dspy.LM(model)
    dspy.configure(lm=lm)
    
    # Create modules
    lowpass = dspy.ChainOfThought(LowpassFilter)
    bandpass = dspy.ChainOfThought(BandpassFilter)
    highpass = dspy.ChainOfThought(HighpassFilter)
    
    # Stage 1: Lowpass (extract thesis)
    logger.info("FFT Stage 1: Lowpass (extracting core thesis)")
    lp_result = lowpass(
        transcript=transcript_text,
        domain=domain
    )
    thesis = lp_result.thesis
    logger.info(f"Thesis extracted: {thesis[:100]}...")
    
    # Stage 2: Bandpass (extract claims given thesis)
    logger.info("FFT Stage 2: Bandpass (extracting conceptual framework)")
    bp_result = bandpass(
        transcript=transcript_text,
        domain=domain,
        thesis=thesis
    )
    claims = bp_result.claims
    logger.info(f"Claims extracted: {len(claims)}")
    
    # Stage 3: Highpass (extract details given thesis + claims)
    logger.info("FFT Stage 3: Highpass (extracting supporting evidence)")
    hp_result = highpass(
        transcript=transcript_text,
        domain=domain,
        thesis=thesis,
        claims=claims
    )
    details = hp_result.details
    logger.info(f"Details extracted: {len(details)}")
    
    # Combine all statements
    all_statements = [thesis] + claims + details
    
    logger.info(f"FFT complete: 1 thesis + {len(claims)} claims + {len(details)} details = {len(all_statements)} total")
    
    return {
        'thesis': thesis,
        'claims': claims,
        'details': details,
        'all_statements': all_statements
    }


def add_citation_ids(
    claims: List[str],
    source_id: str,
    video_id: str
) -> List[Dict[str, Any]]:
    """
    Add citation IDs to claims.
    
    Args:
        claims: List of claim strings
        source_id: Source ID (e.g., "003")
        video_id: Video ID (e.g., "0004")
    
    Returns:
        List of claim objects with citations
    """
    claims_with_citations = []
    
    for idx, claim in enumerate(claims, start=1):
        citation = f"{source_id}-{video_id}-{idx:03d}"
        
        claims_with_citations.append({
            'claim': claim,
            'citation': citation,
            'videoId': video_id
        })
    
    return claims_with_citations


def extract_claims(
    transcript_path: str,
    domain: str,
    filter_keywords: Optional[List[str]] = None,
    source_id: str = "003",
    video_id: Optional[str] = None,
    output_dir: Optional[str] = None,
    model: str = "claude-sonnet-4-5"
) -> Dict[str, Any]:
    """
    Complete extraction pipeline using chained FFT.
    
    Args:
        transcript_path: Path to transcript JSON
        domain: Domain description
        filter_keywords: Optional filter keywords (unused in FFT)
        source_id: Source ID for citations
        video_id: Video ID (auto-detected if not provided)
        output_dir: Output directory (defaults to transcript directory)
        model: DSPy model to use
    
    Returns:
        Dict with 'path', 'clean_path', 'count', 'thesis', 'claims_count', 'details_count'
    """
    transcript_path = Path(transcript_path)
    
    # Auto-detect video ID from filename
    if not video_id:
        match = re.search(r'video-(\d{4})', transcript_path.name)
        if match:
            video_id = match.group(1)
        else:
            raise ValueError(f"Could not extract video ID from: {transcript_path.name}")
    
    # Determine output directory
    if output_dir:
        out_path = Path(output_dir)
    else:
        out_path = transcript_path.parent
    
    # Run chained FFT extraction
    fft_result = fft(
        transcript_path=transcript_path,
        domain=domain,
        model=model
    )
    
    thesis = fft_result['thesis']
    claims = fft_result['claims']
    details = fft_result['details']
    all_statements = fft_result['all_statements']
    
    # Add citation IDs to all statements
    claims_with_citations = add_citation_ids(all_statements, source_id, video_id)
    
    # Save raw statements (for debugging)
    raw_path = out_path / f"video-{video_id}-claims.json"
    with open(raw_path, 'w') as f:
        json.dump({
            'thesis': thesis,
            'claims': claims,
            'details': details
        }, f, indent=2)
    
    logger.info(f"Saved raw FFT output: {raw_path}")
    
    # Save claims with citations (validated)
    clean_path = out_path / f"video-{video_id}-claims-clean.json"
    save_validated(claims_with_citations, clean_path, 'claims')
    
    return {
        'path': str(raw_path),
        'clean_path': str(clean_path),
        'count': len(claims_with_citations),
        'thesis': thesis,
        'claims_count': len(claims),
        'details_count': len(details)
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract claims from transcript using chained FFT')
    parser.add_argument('transcript_path', help='Path to transcript JSON')
    parser.add_argument('--domain', required=True, help='Domain description')
    parser.add_argument('--source-id', default='003', help='Source ID')
    parser.add_argument('--video-id', help='Video ID (auto-detected if not provided)')
    parser.add_argument('--output-dir', help='Output directory')
    parser.add_argument('--model', default='claude-sonnet-4-5', help='DSPy model to use')
    
    args = parser.parse_args()
    
    result = extract_claims(
        transcript_path=args.transcript_path,
        domain=args.domain,
        source_id=args.source_id,
        video_id=args.video_id,
        output_dir=args.output_dir,
        model=args.model
    )
    
    print(json.dumps(result, indent=2))
