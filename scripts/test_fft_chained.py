#!/usr/bin/env python3
"""
Test FFT chained signature approach.
Lowpass → Bandpass → Highpass with full context at each stage.
"""

import json
import dspy
from dspy import Signature, InputField, OutputField
from pathlib import Path
from typing import List


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
        desc="List of high-frequency details (10-30 details). "
             "Each should be specific evidence, measurements, or observations. "
             "Should support the claims but NOT repeat them. "
             "Include: technical specs, measurements, material properties, observations, etc."
    )


def load_transcript(path: Path) -> str:
    """Load transcript JSON and return full text."""
    with open(path) as f:
        data = json.load(f)
    
    transcript_entries = data.get('transcript', [])
    return " ".join(entry['text'] for entry in transcript_entries)


def test_fft_chained(transcript_path: str, domain: str):
    """Test the chained FFT approach."""
    
    # Setup DSPy
    lm = dspy.LM("claude-sonnet-4-5")
    dspy.configure(lm=lm)
    
    # Load transcript
    transcript = load_transcript(Path(transcript_path))
    print(f"Transcript length: {len(transcript)} chars\n")
    
    # Create modules
    lowpass = dspy.ChainOfThought(LowpassFilter)
    bandpass = dspy.ChainOfThought(BandpassFilter)
    highpass = dspy.ChainOfThought(HighpassFilter)
    
    # Stage 1: Lowpass (extract thesis)
    print("=" * 80)
    print("STAGE 1: LOWPASS FILTER (Core Thesis)")
    print("=" * 80)
    
    lp_result = lowpass(
        transcript=transcript,
        domain=domain
    )
    
    thesis = lp_result.thesis
    print(f"\nTHESIS:\n{thesis}\n")
    
    # Stage 2: Bandpass (extract claims given thesis)
    print("=" * 80)
    print("STAGE 2: BANDPASS FILTER (Mid-Level Claims)")
    print("=" * 80)
    print(f"Context: Given thesis above, extracting framework claims...\n")
    
    bp_result = bandpass(
        transcript=transcript,
        domain=domain,
        thesis=thesis
    )
    
    claims = bp_result.claims
    print(f"\nCLAIMS ({len(claims)}):")
    for i, claim in enumerate(claims, 1):
        print(f"{i}. {claim}")
    print()
    
    # Stage 3: Highpass (extract details given thesis + claims)
    print("=" * 80)
    print("STAGE 3: HIGHPASS FILTER (Supporting Details)")
    print("=" * 80)
    print(f"Context: Given thesis and {len(claims)} claims, extracting evidence...\n")
    
    hp_result = highpass(
        transcript=transcript,
        domain=domain,
        thesis=thesis,
        claims=claims
    )
    
    details = hp_result.details
    print(f"\nDETAILS ({len(details)}):")
    for i, detail in enumerate(details, 1):
        print(f"{i}. {detail}")
    print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Lowpass:  1 thesis")
    print(f"Bandpass: {len(claims)} claims")
    print(f"Highpass: {len(details)} details")
    print(f"Total:    {1 + len(claims) + len(details)} extracted statements")
    
    return {
        'thesis': thesis,
        'claims': claims,
        'details': details
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test FFT chained signature approach')
    parser.add_argument('transcript_path', help='Path to transcript JSON')
    parser.add_argument('--domain', required=True, help='Domain description')
    
    args = parser.parse_args()
    
    result = test_fft_chained(
        transcript_path=args.transcript_path,
        domain=args.domain
    )
