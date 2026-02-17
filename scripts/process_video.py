#!/usr/bin/env python3
"""
Main orchestrator for research video processing pipeline.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(Path(__file__).parent.parent / '.progress' / 'process.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProgressTracker:
    """Track pipeline progress"""
    
    def __init__(self, progress_dir: Path):
        self.progress_dir = progress_dir
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        self.status_file = progress_dir / 'status.json'
        self.start_time = time.time()
        
    def update(self, stage: str, status: str, details: Optional[str] = None):
        """Update progress status"""
        data = {
            'stage': stage,
            'status': status,
            'details': details,
            'timestamp': datetime.utcnow().isoformat(),
            'elapsed_seconds': int(time.time() - self.start_time)
        }
        
        with open(self.status_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"[{stage}] {status}" + (f": {details}" if details else ""))


def research(
    video_url: str,
    source_id: str,
    source_name: str,
    domain: str,
    filter_keywords: Optional[list] = None,
    existing_graph: Optional[str] = None,
    existing_terms: Optional[str] = None,
    existing_theory: Optional[str] = None,
    output_dir: Optional[str] = None,
    validate_strict: bool = True,
    assemblyai_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main entry point: process video through full pipeline.
    
    Args:
        video_url: YouTube video URL
        source_id: Source ID (e.g., "003")
        source_name: Source name (e.g., "Geoffrey Drumm")
        domain: Domain description for filtering
        filter_keywords: Keywords to filter assertions
        existing_graph: Path to existing graph.json
        existing_terms: Path to existing terms.json
        existing_theory: Path to existing THEORY.md
        output_dir: Output directory for archive files
        validate_strict: Fail on validation errors
        assemblyai_key: AssemblyAI API key (or from env)
    
    Returns:
        Dict with paths to all created/updated files
    """
    
    # Setup
    progress = ProgressTracker(Path(__file__).parent.parent / '.progress')
    progress.update('init', 'started', f'Processing {video_url}')
    
    # Get API key
    api_key = assemblyai_key or os.getenv('ASSEMBLYAI_API_KEY')
    if not api_key:
        raise ValueError("ASSEMBLYAI_API_KEY not found in environment")
    
    # Import pipeline stages (absolute imports from scripts directory)
    script_dir = Path(__file__).parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    
    from transcribe import transcribe_video
    from extract_claims import extract_claims
    from build_graph import build_graph
    from synthesize_terms import synthesize_terms
    from synthesize_theory import synthesize_theory
    
    results = {}
    
    try:
        # Stage 1: Transcription
        progress.update('transcription', 'started', 'Downloading audio and transcribing')
        transcript_result = transcribe_video(
            video_url=video_url,
            api_key=api_key,
            output_dir=output_dir,
            source_id=source_id,
            source_name=source_name
        )
        results['transcript'] = transcript_result
        progress.update('transcription', 'completed', f"Saved {transcript_result['path']}")
        
        # Stage 2: Extraction
        progress.update('extraction', 'started', 'Extracting domain assertions')
        claims_result = extract_claims(
            transcript_path=transcript_result['path'],
            domain=domain,
            filter_keywords=filter_keywords,
            source_id=source_id,
            video_id=transcript_result['video_id']
        )
        results['claims'] = claims_result
        progress.update('extraction', 'completed', f"{claims_result['count']} claims extracted")
        
        # Stage 3: Graph building
        progress.update('graph', 'started', 'Building and validating graph')
        graph_result = build_graph(
            claims_path=claims_result['clean_path'],
            existing_graph_path=existing_graph,
            validate_strict=validate_strict
        )
        results['graph'] = graph_result
        progress.update('graph', 'completed', 
                       f"{graph_result['nodes_added']} nodes, {graph_result['edges_added']} edges added")
        
        # Stage 4: Term synthesis
        progress.update('terms', 'started', 'Synthesizing term definitions')
        terms_result = synthesize_terms(
            graph_path=graph_result['path'],
            existing_terms_path=existing_terms,
            claims_path=claims_result['clean_path']
        )
        results['terms'] = terms_result
        progress.update('terms', 'completed', f"{terms_result['terms_added']} terms added/updated")
        
        # Stage 5: Theory synthesis
        progress.update('theory', 'started', 'Synthesizing theory document')
        theory_result = synthesize_theory(
            source_name=source_name,
            domain=domain,
            claims_path=claims_result['clean_path'],
            existing_theory_path=existing_theory
        )
        results['theory'] = theory_result
        progress.update('theory', 'completed', f"Theory {'created' if theory_result['created'] else 'updated'}")
        
        # Complete
        progress.update('complete', 'success', 'All stages completed')
        return results
        
    except Exception as e:
        progress.update('error', 'failed', str(e))
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Process research video')
    parser.add_argument('video_url', help='YouTube video URL')
    parser.add_argument('--source-id', required=True, help='Source ID (e.g., 003)')
    parser.add_argument('--source-name', required=True, help='Source name')
    parser.add_argument('--domain', required=True, help='Domain description')
    parser.add_argument('--output-dir', help='Output directory for archive files')
    parser.add_argument('--existing-graph', help='Path to existing graph.json (for merging)')
    parser.add_argument('--existing-terms', help='Path to existing terms.json (for merging)')
    parser.add_argument('--existing-theory', help='Path to existing THEORY.md (for enhancement)')
    
    args = parser.parse_args()
    
    result = research(
        video_url=args.video_url,
        source_id=args.source_id,
        source_name=args.source_name,
        domain=args.domain,
        output_dir=args.output_dir,
        existing_graph=args.existing_graph,
        existing_terms=args.existing_terms,
        existing_theory=args.existing_theory
    )
    
    print(json.dumps(result, indent=2))
