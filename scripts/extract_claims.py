#!/usr/bin/env python3
"""
Stage 2: Extract domain-relevant assertions from transcript using DSPy.
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


# DSPy signature for claim extraction
class ExtractClaims(Signature):
    """Extract domain-relevant assertions made in a transcript segment."""
    
    transcript_segment: str = InputField(desc="Segment of transcript text")
    domain: str = InputField(desc="Domain description (what to filter FOR)")
    
    claims: List[str] = OutputField(
        desc="List of domain-relevant assertions. "
             "Each claim should be a complete, standalone statement. "
             "Filter OUT: personal anecdotes, off-topic tangents, modern politics, etc. "
             "Include ONLY assertions directly relevant to the domain."
    )


def chunk_transcript(transcript: List[Dict[str, Any]], chunk_size: int = 20) -> List[str]:
    """
    Split transcript into overlapping chunks for processing.
    
    Args:
        transcript: List of transcript entries with 'text'
        chunk_size: Number of utterances per chunk
    
    Returns:
        List of text chunks
    """
    chunks = []
    
    for i in range(0, len(transcript), chunk_size // 2):  # 50% overlap
        chunk_entries = transcript[i:i + chunk_size]
        chunk_text = " ".join(entry['text'] for entry in chunk_entries)
        chunks.append(chunk_text)
    
    return chunks


def extract_claims_dspy(
    transcript_path: Path,
    domain: str,
    filter_keywords: Optional[List[str]] = None,
    model: str = "claude-sonnet-4-5"
) -> List[Dict[str, Any]]:
    """
    Extract domain-relevant claims using DSPy.
    
    Args:
        transcript_path: Path to transcript JSON
        domain: Domain description for filtering
        filter_keywords: Optional keywords to help filtering
        model: DSPy model to use
    
    Returns:
        List of extracted claims (without citations yet)
    """
    # Load transcript
    transcript_data = validate_and_load(transcript_path, 'transcript')
    transcript = transcript_data['transcript']
    
    # Setup DSPy
    lm = dspy.LM(model)
    dspy.configure(lm=lm)
    
    # Create extraction module
    extract = dspy.ChainOfThought(ExtractClaims)
    
    # Chunk transcript
    chunks = chunk_transcript(transcript)
    logger.info(f"Processing {len(chunks)} chunks from {len(transcript)} utterances")
    
    # Extract claims from each chunk
    all_claims = []
    
    for idx, chunk in enumerate(chunks):
        logger.info(f"Extracting from chunk {idx + 1}/{len(chunks)}")
        
        try:
            result = extract(
                transcript_segment=chunk,
                domain=domain
            )
            
            # DSPy returns typed list directly
            claims = result.claims
            
            if not isinstance(claims, list):
                logger.warning(f"Expected list, got {type(claims)} - skipping chunk {idx}")
                continue
            
            all_claims.extend(claims)
            
        except Exception as e:
            logger.warning(f"Failed to extract from chunk {idx}: {e}")
            continue
    
    logger.info(f"Extracted {len(all_claims)} raw claims")
    
    # Deduplicate similar claims
    unique_claims = _deduplicate_claims(all_claims)
    logger.info(f"After deduplication: {len(unique_claims)} unique claims")
    
    return unique_claims


def _deduplicate_claims(claims: List[str], similarity_threshold: float = 0.85) -> List[str]:
    """
    Deduplicate claims using simple text similarity.
    
    Args:
        claims: List of claim strings
        similarity_threshold: Similarity threshold (0-1)
    
    Returns:
        Deduplicated list
    """
    from difflib import SequenceMatcher
    
    unique = []
    
    for claim in claims:
        # Check if similar to any existing unique claim
        is_duplicate = False
        
        for existing in unique:
            similarity = SequenceMatcher(None, claim.lower(), existing.lower()).ratio()
            if similarity > similarity_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique.append(claim)
    
    return unique


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
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Complete extraction pipeline.
    
    Args:
        transcript_path: Path to transcript JSON
        domain: Domain description
        filter_keywords: Optional filter keywords
        source_id: Source ID for citations
        video_id: Video ID (auto-detected if not provided)
        output_dir: Output directory (defaults to transcript directory)
    
    Returns:
        Dict with 'path', 'clean_path', 'count'
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
    
    # Extract claims
    raw_claims = extract_claims_dspy(
        transcript_path=transcript_path,
        domain=domain,
        filter_keywords=filter_keywords
    )
    
    # Add citation IDs
    claims_with_citations = add_citation_ids(raw_claims, source_id, video_id)
    
    # Save raw claims (without citation validation yet)
    raw_path = out_path / f"video-{video_id}-claims.json"
    with open(raw_path, 'w') as f:
        json.dump([{'claim': c} for c in raw_claims], f, indent=2)
    
    logger.info(f"Saved raw claims: {raw_path}")
    
    # Save claims with citations (validated)
    clean_path = out_path / f"video-{video_id}-claims-clean.json"
    save_validated(claims_with_citations, clean_path, 'claims')
    
    return {
        'path': str(raw_path),
        'clean_path': str(clean_path),
        'count': len(claims_with_citations)
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract claims from transcript')
    parser.add_argument('transcript_path', help='Path to transcript JSON')
    parser.add_argument('--domain', required=True, help='Domain description')
    parser.add_argument('--source-id', default='003', help='Source ID')
    parser.add_argument('--video-id', help='Video ID (auto-detected if not provided)')
    parser.add_argument('--output-dir', help='Output directory')
    
    args = parser.parse_args()
    
    result = extract_claims(
        transcript_path=args.transcript_path,
        domain=args.domain,
        source_id=args.source_id,
        video_id=args.video_id,
        output_dir=args.output_dir
    )
    
    print(json.dumps(result, indent=2))
