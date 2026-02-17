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


def create_source_structure(
    base_dir: Path,
    source_id: str,
    source_name: str,
    source_slug: str,
    category: str = "research",
    channel: str = "",
    description: str = ""
) -> Dict[str, str]:
    """
    Create complete source directory structure for Deep Memory website.
    
    Args:
        base_dir: Base sources directory (e.g., ~/repos/deepmemory/site/sources)
        source_id: Source ID (e.g., "004")
        source_name: Human-readable name (e.g., "Jimmy Dore")
        source_slug: URL-safe slug (e.g., "jimmy-dore")
        category: Source category
        channel: YouTube/social channel handle
        description: Brief description
    
    Returns:
        Dict with paths to created files
    """
    source_dir = base_dir / source_slug
    archive_dir = source_dir / 'archive'
    
    # Create directories
    source_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(exist_ok=True)
    
    # Create SOURCE.json
    source_json = {
        'id': source_id,
        'name': source_name,
        'slug': source_slug,
        'category': category,
        'channel': channel,
        'description': description
    }
    source_json_path = source_dir / 'SOURCE.json'
    with open(source_json_path, 'w') as f:
        json.dump(source_json, f, indent=2)
    
    # Create empty graph.json
    graph_path = source_dir / 'graph.json'
    if not graph_path.exists():
        with open(graph_path, 'w') as f:
            json.dump({'nodes': [], 'edges': []}, f, indent=2)
    
    # Create empty terms.json
    terms_path = source_dir / 'terms.json'
    if not terms_path.exists():
        with open(terms_path, 'w') as f:
            json.dump([], f, indent=2)
    
    # Create empty THEORY.md
    theory_path = source_dir / 'THEORY.md'
    if not theory_path.exists():
        with open(theory_path, 'w') as f:
            f.write(f"# {source_name}'s Theory\n\n(No content yet - will be generated from videos)\n")
    
    logger.info(f"Created source structure: {source_dir}")
    
    return {
        'source_dir': str(source_dir),
        'archive_dir': str(archive_dir),
        'source_json': str(source_json_path),
        'graph': str(graph_path),
        'terms': str(terms_path),
        'theory': str(theory_path)
    }


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
    assemblyai_key: Optional[str] = None,
    create_source: bool = False,
    source_slug: Optional[str] = None,
    source_category: str = "research",
    source_channel: str = "",
    source_description: str = ""
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
        create_source: Create new source directory structure
        source_slug: URL-safe slug for new source (e.g., "jimmy-dore")
        source_category: Category for new source
        source_channel: YouTube channel for new source
        source_description: Description for new source
    
    Returns:
        Dict with paths to all created/updated files
    """
    
    # Setup
    progress = ProgressTracker(Path(__file__).parent.parent / '.progress')
    progress.update('init', 'started', f'Processing {video_url}')
    
    # Create source structure if requested
    if create_source:
        if not source_slug:
            # Auto-generate slug from source name
            source_slug = source_name.lower().replace(' ', '-').replace('_', '-')
        
        # Determine base directory - try to find deepmemory site
        if output_dir:
            base_dir = Path(output_dir).parent
        else:
            # Try to find deepmemory/site/sources
            home = Path.home()
            candidate = home / 'repos' / 'deepmemory' / 'site' / 'sources'
            if candidate.exists():
                base_dir = candidate
            else:
                raise ValueError("Cannot determine base sources directory. Provide output_dir or ensure ~/repos/deepmemory/site/sources exists")
        
        progress.update('source_creation', 'started', f'Creating source structure for {source_name}')
        
        source_paths = create_source_structure(
            base_dir=base_dir if not output_dir else Path(output_dir).parent,
            source_id=source_id,
            source_name=source_name,
            source_slug=source_slug,
            category=source_category,
            channel=source_channel,
            description=source_description
        )
        
        # Update paths to use newly created structure
        if not output_dir:
            output_dir = source_paths['archive_dir']
        if not existing_graph:
            existing_graph = source_paths['graph']
        if not existing_terms:
            existing_terms = source_paths['terms']
        if not existing_theory:
            existing_theory = source_paths['theory']
        
        progress.update('source_creation', 'completed', f'Source structure created at {source_paths["source_dir"]}')
    
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
    parser.add_argument('--create-source', action='store_true', help='Create new source directory structure')
    parser.add_argument('--source-slug', help='URL-safe slug for new source (auto-generated if not provided)')
    parser.add_argument('--source-category', default='research', help='Category for new source')
    parser.add_argument('--source-channel', default='', help='YouTube channel for new source')
    parser.add_argument('--source-description', default='', help='Description for new source')
    
    args = parser.parse_args()
    
    result = research(
        video_url=args.video_url,
        source_id=args.source_id,
        source_name=args.source_name,
        domain=args.domain,
        output_dir=args.output_dir,
        existing_graph=args.existing_graph,
        existing_terms=args.existing_terms,
        existing_theory=args.existing_theory,
        create_source=args.create_source,
        source_slug=args.source_slug,
        source_category=args.source_category,
        source_channel=args.source_channel,
        source_description=args.source_description
    )
    
    print(json.dumps(result, indent=2))
