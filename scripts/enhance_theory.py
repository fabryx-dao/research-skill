#!/usr/bin/env python3
"""
Stage 5: Enhance theory document from full context.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import dspy
from dspy import Signature, InputField, OutputField

from validate import validate_and_load

logger = logging.getLogger(__name__)


# DSPy signature for theory enhancement
class EnhanceTheory(Signature):
    """Synthesize or enhance theory document from research knowledge."""
    
    existing_theory: str = InputField(desc="Existing theory markdown, or 'none' if starting fresh")
    new_transcript_excerpt: str = InputField(desc="Key excerpts from new transcript")
    graph_summary: str = InputField(desc="Summary of knowledge graph structure")
    terms_summary: str = InputField(desc="Summary of defined terms")
    research_question: str = InputField(desc="What question does this researcher address?")
    
    enhanced_theory: str = OutputField(
        desc="Enhanced theory document in markdown format. "
             "If existing theory provided: ADD new sections or EXPAND existing ones "
             "(preserve existing narrative, accumulate knowledge). "
             "If no existing theory: CREATE coherent synthesis answering the research question. "
             "Structure: # Core Thesis, ## Key Concepts, ## Evidence, ## Implications. "
             "Cite evidence using [citation-id]. Write as synthesis, not summary."
    )


def extract_transcript_excerpts(transcript_data: Dict[str, Any], max_chars: int = 5000) -> str:
    """
    Extract key excerpts from transcript.
    
    For now, simple sampling. Could be enhanced with semantic extraction.
    
    Args:
        transcript_data: Transcript JSON
        max_chars: Maximum characters to extract
    
    Returns:
        Excerpt string
    """
    transcript = transcript_data.get('transcript', [])
    
    if not transcript:
        return "No transcript available"
    
    # Sample evenly across transcript
    total_utterances = len(transcript)
    sample_count = min(20, total_utterances)
    step = max(1, total_utterances // sample_count)
    
    excerpts = []
    char_count = 0
    
    for i in range(0, total_utterances, step):
        if char_count >= max_chars:
            break
        
        text = transcript[i].get('text', '')
        excerpts.append(text)
        char_count += len(text)
    
    return "\n\n".join(excerpts)


def summarize_graph(graph: Dict[str, Any]) -> str:
    """
    Create human-readable summary of graph structure.
    
    Args:
        graph: Knowledge graph
    
    Returns:
        Summary text
    """
    nodes = graph.get('nodes', [])
    edges = graph.get('edges', [])
    
    # Count by type
    type_counts = {}
    for node in nodes:
        node_type = node.get('type', 'unknown')
        type_counts[node_type] = type_counts.get(node_type, 0) + 1
    
    # Sample key relationships
    sample_edges = edges[:10]
    relationships = []
    
    for edge in sample_edges:
        relationships.append(f"- {edge['from']} → {edge['relation']} → {edge['to']}")
    
    summary = f"Graph: {len(nodes)} nodes, {len(edges)} edges\n\n"
    summary += "Node types:\n" + "\n".join([f"- {t}: {c}" for t, c in type_counts.items()])
    summary += "\n\nKey relationships:\n" + "\n".join(relationships)
    
    return summary


def summarize_terms(terms: List[Dict[str, Any]]) -> str:
    """
    Create summary of terms.
    
    Args:
        terms: List of term definitions
    
    Returns:
        Summary text
    """
    if not terms:
        return "No terms defined yet"
    
    summary = f"Terms defined: {len(terms)}\n\n"
    
    # List terms with brief excerpt
    for term in terms[:15]:  # First 15 terms
        term_name = term.get('term', 'Unknown')
        definition = term.get('definition', '')
        excerpt = definition[:100] + "..." if len(definition) > 100 else definition
        summary += f"- **{term_name}**: {excerpt}\n"
    
    return summary


def enhance_theory_dspy(
    existing_theory: Optional[str],
    transcript_path: Path,
    graph: Dict[str, Any],
    terms: List[Dict[str, Any]],
    research_question: str,
    model: str = "claude-sonnet-4-5"
) -> str:
    """
    Enhance theory document using DSPy.
    
    Args:
        existing_theory: Existing theory markdown
        transcript_path: Path to transcript
        graph: Knowledge graph
        terms: Term definitions
        research_question: Research question for context
        model: DSPy model
    
    Returns:
        Enhanced theory markdown
    """
    # Setup DSPy
    lm = dspy.LM(model)
    dspy.configure(lm=lm)
    
    # Load transcript
    transcript_data = validate_and_load(transcript_path, 'transcript')
    
    # Extract key content
    transcript_excerpt = extract_transcript_excerpts(transcript_data)
    graph_summary = summarize_graph(graph)
    terms_summary = summarize_terms(terms)
    
    # Create enhancement module
    enhance = dspy.ChainOfThought(EnhanceTheory)
    
    # Enhance theory
    result = enhance(
        existing_theory=existing_theory or "none",
        new_transcript_excerpt=transcript_excerpt,
        graph_summary=graph_summary,
        terms_summary=terms_summary,
        research_question=research_question
    )
    
    return result.enhanced_theory


def enhance_theory(
    transcript_path: str,
    graph_path: str,
    terms_path: str,
    existing_theory_path: Optional[str] = None,
    research_question: str = "What does this researcher believe about human origins and capabilities?"
) -> Dict[str, Any]:
    """
    Complete theory enhancement pipeline.
    
    Args:
        transcript_path: Path to transcript JSON
        graph_path: Path to graph.json
        terms_path: Path to terms.json
        existing_theory_path: Path to existing THEORY.md
        research_question: Research question for context
    
    Returns:
        Dict with 'path', 'sections_added'
    """
    transcript_path = Path(transcript_path)
    graph_path = Path(graph_path)
    terms_path = Path(terms_path)
    existing_theory_path = Path(existing_theory_path) if existing_theory_path else None
    
    # Load existing theory
    existing_theory = None
    if existing_theory_path and existing_theory_path.exists():
        with open(existing_theory_path) as f:
            existing_theory = f.read()
        logger.info(f"Loaded existing theory: {len(existing_theory)} chars")
    
    # Load graph and terms
    graph = validate_and_load(graph_path, 'graph')
    terms = validate_and_load(terms_path, 'terms')
    
    # Enhance theory
    enhanced_theory = enhance_theory_dspy(
        existing_theory=existing_theory,
        transcript_path=transcript_path,
        graph=graph,
        terms=terms,
        research_question=research_question
    )
    
    # Count sections (naive: count ## headers)
    section_count = enhanced_theory.count('\n##')
    
    # Save theory
    if existing_theory_path:
        output_path = existing_theory_path
    else:
        output_path = graph_path.parent / 'THEORY.md'
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(enhanced_theory)
    
    logger.info(f"Theory saved: {output_path}")
    
    return {
        'path': str(output_path),
        'sections_added': section_count,
        'char_count': len(enhanced_theory)
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhance theory document')
    parser.add_argument('transcript_path', help='Path to transcript JSON')
    parser.add_argument('graph_path', help='Path to graph.json')
    parser.add_argument('terms_path', help='Path to terms.json')
    parser.add_argument('--existing-theory', help='Path to existing THEORY.md')
    parser.add_argument('--research-question', 
                       default='What does this researcher believe about human origins and capabilities?',
                       help='Research question for context')
    
    args = parser.parse_args()
    
    result = enhance_theory(
        transcript_path=args.transcript_path,
        graph_path=args.graph_path,
        terms_path=args.terms_path,
        existing_theory_path=args.existing_theory,
        research_question=args.research_question
    )
    
    print(json.dumps(result, indent=2))
