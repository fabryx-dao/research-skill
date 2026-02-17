#!/usr/bin/env python3
"""
Stage 4: Synthesize term definitions from graph knowledge.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import dspy
from dspy import Signature, InputField, OutputField

from validate import validate_and_load, save_validated

logger = logging.getLogger(__name__)


# DSPy signature for term synthesis
class SynthesizeTerm(Signature):
    """Synthesize term definition EXCLUSIVELY from source statements - zero external knowledge."""
    
    term: str = InputField(desc="Term to define")
    related_claims: str = InputField(desc="All claims/evidence related to this term from source material")
    existing_definition: str = InputField(desc="Existing definition if term already exists, or 'none'")
    
    definition: str = OutputField(
        desc="State ONLY what the source explicitly says about this term. "
             "Zero external knowledge. Zero inference. Zero apologetic meta-commentary. "
             "Compile the author's statements. Cite evidence inline using [citation-id]. "
             "If existing definition provided, ADD new source statements (don't rewrite from scratch). "
             "No minimum length - one sentence is fine if that's all the source says."
    )


def identify_key_concepts(graph: Dict[str, Any], min_connections: int = 3) -> List[str]:
    """
    Identify key concepts that need term definitions.
    
    Prioritize nodes with:
    - Multiple connections (high degree)
    - Type: concept, structure, process, substance
    
    Args:
        graph: Knowledge graph
        min_connections: Minimum edge connections to consider
    
    Returns:
        List of node IDs for key concepts
    """
    nodes = graph.get('nodes', [])
    edges = graph.get('edges', [])
    
    # Count connections for each node
    connection_count = {}
    for edge in edges:
        connection_count[edge['from']] = connection_count.get(edge['from'], 0) + 1
        connection_count[edge['to']] = connection_count.get(edge['to'], 0) + 1
    
    # Filter nodes by type and connections
    key_concepts = []
    
    for node in nodes:
        if node['type'] in ['concept', 'structure', 'process', 'substance']:
            connections = connection_count.get(node['id'], 0)
            if connections >= min_connections:
                key_concepts.append(node['id'])
    
    # Sort by connection count (descending)
    key_concepts.sort(key=lambda nid: connection_count.get(nid, 0), reverse=True)
    
    logger.info(f"Identified {len(key_concepts)} key concepts needing definitions")
    
    return key_concepts


def collect_term_evidence(node_id: str, graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Collect all evidence related to a term from graph edges.
    
    Args:
        node_id: Node ID to collect evidence for
        graph: Knowledge graph
    
    Returns:
        List of evidence dicts with content and citation
    """
    edges = graph.get('edges', [])
    evidence = []
    
    for edge in edges:
        if edge['from'] == node_id or edge['to'] == node_id:
            # Include relation in evidence context
            for ev in edge['evidence']:
                evidence.append({
                    'content': f"{edge['relation']}: {ev['content']}",
                    'citation': ev['citation'],
                    'videoId': ev.get('videoId')
                })
    
    return evidence


def load_existing_terms(terms_path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    """
    Load existing terms into map: term_id -> term_data.
    
    Args:
        terms_path: Path to existing terms.json
    
    Returns:
        Dict mapping node IDs to term data
    """
    if not terms_path or not terms_path.exists():
        logger.info("No existing terms")
        return {}
    
    # Load without validation (may have legacy format)
    with open(terms_path) as f:
        raw_data = json.load(f)
    
    # Handle legacy format: {"terms": [...]} vs direct array [...]
    if isinstance(raw_data, dict) and 'terms' in raw_data:
        logger.warning(f"Legacy terms format detected - converting to array format")
        terms_list = raw_data['terms']
    elif isinstance(raw_data, list):
        terms_list = raw_data
    else:
        logger.error(f"Invalid terms format: expected array or {{terms: array}}, got {type(raw_data)}")
        return {}
    
    # Map by term ID (assuming term field maps to node ID in kebab-case)
    terms_map = {}
    for term in terms_list:
        # Try to match term name to node ID
        term_key = term.get('term', '').lower().replace(' ', '-')
        if term_key:
            terms_map[term_key] = term
    
    logger.info(f"Loaded {len(terms_map)} existing terms")
    return terms_map


def synthesize_term_dspy(
    term_label: str,
    evidence: List[Dict[str, Any]],
    existing_definition: Optional[str] = None,
    model: str = "claude-sonnet-4-5"
) -> str:
    """
    Synthesize term definition using DSPy.
    
    Args:
        term_label: Term name/label
        evidence: Evidence for this term
        existing_definition: Existing definition to enhance
        model: DSPy model
    
    Returns:
        Wikipedia-style definition
    """
    # Setup DSPy
    lm = dspy.LM(model)
    dspy.configure(lm=lm)
    
    # Create synthesis module
    synthesize = dspy.ChainOfThought(SynthesizeTerm)
    
    # Format evidence
    evidence_text = "\n".join([
        f"- {ev['content']} [{ev['citation']}]"
        for ev in evidence
    ])
    
    # Synthesize
    result = synthesize(
        term=term_label,
        related_claims=evidence_text,
        existing_definition=existing_definition or "none"
    )
    
    return result.definition


def synthesize_terms(
    graph_path: str,
    existing_terms_path: Optional[str] = None,
    claims_path: Optional[str] = None,
    max_terms: int = 50
) -> Dict[str, Any]:
    """
    Complete term synthesis pipeline.
    
    Args:
        graph_path: Path to graph.json
        existing_terms_path: Path to existing terms.json
        claims_path: Path to claims (for metadata)
        max_terms: Maximum terms to generate/update
    
    Returns:
        Dict with 'path', 'terms_added', 'terms_updated'
    """
    graph_path = Path(graph_path)
    existing_terms_path = Path(existing_terms_path) if existing_terms_path else None
    
    # Load graph
    graph = validate_and_load(graph_path, 'graph')
    
    # Load existing terms
    existing_terms = load_existing_terms(existing_terms_path)
    
    # Identify key concepts
    key_concepts = identify_key_concepts(graph)[:max_terms]
    
    logger.info(f"Synthesizing definitions for {len(key_concepts)} terms")
    
    # Build node lookup
    node_map = {node['id']: node for node in graph.get('nodes', [])}
    
    # Synthesize/update terms
    new_terms = []
    updated_count = 0
    added_count = 0
    
    for node_id in key_concepts:
        node = node_map.get(node_id)
        if not node:
            logger.warning(f"Node not found: {node_id}")
            continue
        
        # Collect evidence
        evidence = collect_term_evidence(node_id, graph)
        
        if not evidence:
            logger.warning(f"No evidence for term: {node['label']}")
            continue
        
        # Check if term exists
        existing_def = existing_terms.get(node_id, {}).get('definition')
        
        # Synthesize definition
        definition = synthesize_term_dspy(
            term_label=node['label'],
            evidence=evidence,
            existing_definition=existing_def
        )
        
        # Skip if DSPy failed to generate definition
        if not definition or not definition.strip():
            logger.warning(f"Skipping term '{node['label']}' - empty definition")
            continue
        
        # Determine term ID
        if node_id in existing_terms:
            term_id = existing_terms[node_id]['termId']
            updated_count += 1
        else:
            # Assign new term ID
            existing_ids = [int(t['termId']) for t in new_terms if 'termId' in t]
            existing_ids.extend([int(t['termId']) for t in existing_terms.values()])
            next_id = max(existing_ids, default=0) + 1
            term_id = f"{next_id:03d}"
            added_count += 1
        
        # Extract citations from evidence
        citations = list(set(ev['citation'] for ev in evidence))
        citations.sort()
        
        # Build term entry (matching existing format)
        term_entry = {
            'termId': term_id,
            'term': node['label'],
            'category': node['type'],
            'definition': definition,
            'citations': citations,
            'sourceId': node.get('sourceId', '003')
        }
        
        new_terms.append(term_entry)
    
    # Merge with existing terms
    all_terms = list(existing_terms.values())
    
    # Update existing or add new
    for new_term in new_terms:
        # Find and replace existing, or append
        found = False
        for idx, existing_term in enumerate(all_terms):
            if existing_term['termId'] == new_term['termId']:
                all_terms[idx] = new_term
                found = True
                break
        
        if not found:
            all_terms.append(new_term)
    
    # Sort by term ID
    all_terms.sort(key=lambda t: t['termId'])
    
    # Save
    if existing_terms_path:
        output_path = existing_terms_path
    else:
        output_path = graph_path.parent / 'terms.json'
    
    save_validated(all_terms, output_path, 'terms')
    
    logger.info(f"Terms: {added_count} added, {updated_count} updated")
    
    return {
        'path': str(output_path),
        'terms_added': added_count,
        'terms_updated': updated_count,
        'total_terms': len(all_terms)
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Synthesize term definitions')
    parser.add_argument('graph_path', help='Path to graph.json')
    parser.add_argument('--existing-terms', help='Path to existing terms.json')
    parser.add_argument('--max-terms', type=int, default=50, help='Max terms to process')
    
    args = parser.parse_args()
    
    result = synthesize_terms(
        graph_path=args.graph_path,
        existing_terms_path=args.existing_terms,
        max_terms=args.max_terms
    )
    
    print(json.dumps(result, indent=2))
