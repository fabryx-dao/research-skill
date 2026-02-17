#!/usr/bin/env python3
"""
Stage 3: Build knowledge graph from claims with validation.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import dspy
from dspy import Signature, InputField, OutputField

from validate import validate_and_load, save_validated, validate_graph_integrity

logger = logging.getLogger(__name__)


# DSPy signature for converting claims to triplets
class ClaimsToTriplets(Signature):
    """Convert domain claims to knowledge graph triplets."""
    
    claims: str = InputField(desc="List of domain assertions with citations")
    existing_nodes: str = InputField(desc="Existing node IDs, labels, and types to reuse when possible")
    allowed_node_types: str = InputField(desc="REQUIRED node types - ONLY use these")
    
    triplets: list[dict] = OutputField(
        desc="List of knowledge graph triplets. Each dict contains: "
             "from (dict with id, label, type), "
             "to (dict with id, label, type), "
             "relation (string), "
             "evidence (list of dicts with content and citation). "
             "Node IDs must be kebab-case. Node types MUST be from allowed_node_types. "
             "Reuse existing node IDs when referencing the same concepts."
    )


def sanitize_existing_graph(graph: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean up existing graph - remove nodes and edges with invalid types.
    
    Args:
        graph: Raw existing graph
    
    Returns:
        Sanitized graph with only valid node types
    """
    original_node_count = len(graph.get('nodes', []))
    original_edge_count = len(graph.get('edges', []))
    
    # Filter nodes to only valid types
    valid_nodes = []
    invalid_node_ids = set()
    
    for node in graph.get('nodes', []):
        node_type = node.get('type', '')
        if node_type in ALLOWED_NODE_TYPES:
            valid_nodes.append(node)
        else:
            invalid_node_ids.add(node['id'])
            logger.warning(f"Removing node '{node['id']}' ({node['label']}) - invalid type '{node_type}'")
    
    # Filter edges to only reference valid nodes
    valid_edges = []
    for edge in graph.get('edges', []):
        if edge['from'] in invalid_node_ids or edge['to'] in invalid_node_ids:
            continue  # Skip edges referencing removed nodes
        valid_edges.append(edge)
    
    removed_nodes = original_node_count - len(valid_nodes)
    removed_edges = original_edge_count - len(valid_edges)
    
    if removed_nodes > 0 or removed_edges > 0:
        logger.warning(f"Sanitized existing graph: removed {removed_nodes} nodes and {removed_edges} edges with invalid types")
    
    return {
        'nodes': valid_nodes,
        'edges': valid_edges
    }


def load_existing_graph(graph_path: Optional[Path]) -> Dict[str, Any]:
    """
    Load existing graph or return empty structure.
    Sanitizes existing graph to remove invalid node types.
    
    Args:
        graph_path: Path to existing graph.json
    
    Returns:
        Graph data with nodes and edges (sanitized)
    """
    if graph_path and graph_path.exists():
        logger.info(f"Loading existing graph: {graph_path}")
        # Load without strict validation (may have old invalid types)
        with open(graph_path) as f:
            raw_graph = json.load(f)
        # Sanitize before using
        return sanitize_existing_graph(raw_graph)
    else:
        logger.info("No existing graph, starting fresh")
        return {'nodes': [], 'edges': []}


def filter_invalid_triplets(triplets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter out triplets with invalid node types. Don't try to fix - just drop them.
    
    Args:
        triplets: Raw triplets from DSPy
    
    Returns:
        Filtered triplets with only valid node types
    """
    valid_triplets = []
    rejected_count = 0
    
    for triplet in triplets:
        from_type = triplet.get('from', {}).get('type', '')
        to_type = triplet.get('to', {}).get('type', '')
        
        # Check both node types
        if from_type not in ALLOWED_NODE_TYPES:
            logger.warning(f"Rejecting triplet: 'from' node has invalid type '{from_type}' (label: {triplet['from'].get('label', 'unknown')})")
            rejected_count += 1
            continue
        
        if to_type not in ALLOWED_NODE_TYPES:
            logger.warning(f"Rejecting triplet: 'to' node has invalid type '{to_type}' (label: {triplet['to'].get('label', 'unknown')})")
            rejected_count += 1
            continue
        
        # Both types valid - keep triplet
        valid_triplets.append(triplet)
    
    if rejected_count > 0:
        logger.warning(f"Rejected {rejected_count} triplets with invalid node types (kept {len(valid_triplets)})")
    
    return valid_triplets


def extract_nodes_from_triplets(
    triplets: List[Dict[str, Any]],
    source_id: str,
    existing_nodes: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extract unique nodes from triplets and merge with existing.
    
    Args:
        triplets: List of triplet dicts with from/to/relation/evidence
                  where from/to are objects with id/label/type
        source_id: Source ID for new nodes
        existing_nodes: Existing nodes from graph
    
    Returns:
        (all_nodes, new_nodes_only)
    """
    # Build existing node map
    existing_map = {node['id']: node for node in existing_nodes}
    
    # Collect all nodes mentioned in triplets (already filtered for valid types)
    nodes_seen = {}
    
    for triplet in triplets:
        for key in ['from', 'to']:
            node_data = triplet[key]
            node_id = node_data['id']
            
            # Track this node
            if node_id not in nodes_seen:
                nodes_seen[node_id] = {
                    'id': node_id,
                    'label': node_data['label'],
                    'type': node_data['type']
                }
    
    # Create new nodes for IDs not in existing graph
    new_nodes = []
    
    for node_id, node_data in nodes_seen.items():
        if node_id not in existing_map:
            new_node = {
                'id': node_id,
                'label': node_data['label'],
                'type': node_data['type'],
                'sourceId': source_id
            }
            
            new_nodes.append(new_node)
            existing_map[node_id] = new_node
    
    all_nodes = list(existing_map.values())
    
    logger.info(f"Nodes: {len(existing_nodes)} existing + {len(new_nodes)} new = {len(all_nodes)} total")
    
    return (all_nodes, new_nodes)


def merge_edges(
    new_triplets: List[Dict[str, Any]],
    existing_edges: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Merge new triplets with existing edges.
    
    If edge already exists (same from/to/relation), append evidence.
    Otherwise, add as new edge.
    
    Args:
        new_triplets: New triplets to add (from/to are objects with id/label/type)
        existing_edges: Existing edges (from/to are string IDs)
    
    Returns:
        (merged_edges, count_new_edges)
    """
    # Build edge map: (from, to, relation) -> edge
    edge_map = {}
    
    for edge in existing_edges:
        key = (edge['from'], edge['to'], edge['relation'])
        edge_map[key] = edge
    
    new_edge_count = 0
    
    for triplet in new_triplets:
        # Extract node IDs from node objects
        from_id = triplet['from']['id']
        to_id = triplet['to']['id']
        
        key = (from_id, to_id, triplet['relation'])
        
        if key in edge_map:
            # Merge evidence
            edge_map[key]['evidence'].extend(triplet['evidence'])
        else:
            # New edge
            edge_map[key] = {
                'from': from_id,
                'to': to_id,
                'relation': triplet['relation'],
                'evidence': triplet['evidence']
            }
            new_edge_count += 1
    
    merged_edges = list(edge_map.values())
    
    logger.info(f"Edges: {len(existing_edges)} existing + {new_edge_count} new = {len(merged_edges)} total")
    
    return (merged_edges, new_edge_count)


# Allowed node types (must match graph-schema.json)
ALLOWED_NODE_TYPES = [
    "term",
    "concept", 
    "structure",
    "process",
    "substance",
    "person",
    "text",
    "event",
    "location",
    "artifact"
]


def build_graph_dspy(
    claims_with_citations: List[Dict[str, Any]],
    existing_graph: Dict[str, Any],
    source_id: str,
    model: str = "claude-sonnet-4-5"
) -> Dict[str, Any]:
    """
    Build graph triplets from claims using DSPy.
    
    Args:
        claims_with_citations: Claims with citation IDs
        existing_graph: Existing graph to merge with
        source_id: Source ID for new nodes
        model: DSPy model
    
    Returns:
        Updated graph with new nodes and edges
    """
    # Setup DSPy
    lm = dspy.LM(model)
    dspy.configure(lm=lm)
    
    # Create triplet extraction module
    extract_triplets = dspy.ChainOfThought(ClaimsToTriplets)
    
    # Prepare existing nodes context (with types)
    existing_nodes_text = ", ".join([
        f"{node['id']} ({node['label']}, type: {node['type']})"
        for node in existing_graph.get('nodes', [])
    ])
    
    if not existing_nodes_text:
        existing_nodes_text = "none yet"
    
    # Format allowed node types
    allowed_types_text = ", ".join(ALLOWED_NODE_TYPES)
    
    # Format claims for extraction
    claims_text = "\n".join([
        f"- {claim['claim']} [{claim['citation']}]"
        for claim in claims_with_citations
    ])
    
    logger.info(f"Extracting triplets from {len(claims_with_citations)} claims")
    logger.info(f"Allowed node types: {allowed_types_text}")
    
    # Extract triplets
    result = extract_triplets(
        claims=claims_text,
        existing_nodes=existing_nodes_text,
        allowed_node_types=allowed_types_text
    )
    
    # Get triplets (DSPy returns structured data directly)
    triplets = result.triplets
    
    if not isinstance(triplets, list):
        logger.error(f"Expected list of triplets, got {type(triplets)}")
        raise TypeError(f"DSPy returned {type(triplets)} instead of list")
    
    logger.info(f"Extracted {len(triplets)} triplets from DSPy")
    
    # Add videoId to evidence objects by matching citations to original claims
    citation_to_video_id = {claim['citation']: claim.get('videoId') for claim in claims_with_citations}
    for triplet in triplets:
        for evidence in triplet.get('evidence', []):
            citation = evidence.get('citation')
            if citation and citation in citation_to_video_id:
                evidence['videoId'] = citation_to_video_id[citation]
    
    # Filter out triplets with invalid node types
    triplets = filter_invalid_triplets(triplets)
    logger.info(f"After filtering: {len(triplets)} valid triplets")
    
    # Extract and merge nodes
    all_nodes, new_nodes = extract_nodes_from_triplets(
        triplets=triplets,
        source_id=source_id,
        existing_nodes=existing_graph.get('nodes', [])
    )
    
    # Merge edges
    merged_edges, new_edge_count = merge_edges(
        new_triplets=triplets,
        existing_edges=existing_graph.get('edges', [])
    )
    
    # Build final graph
    final_graph = {
        'nodes': all_nodes,
        'edges': merged_edges
    }
    
    # Validate integrity (all types already filtered to be valid)
    validate_graph_integrity(final_graph, strict=True)
    
    return final_graph


def build_graph(
    claims_path: str,
    existing_graph_path: Optional[str] = None,
    validate_strict: bool = True,
    source_id: str = "003"
) -> Dict[str, Any]:
    """
    Complete graph building pipeline.
    
    Args:
        claims_path: Path to claims JSON with citations
        existing_graph_path: Path to existing graph.json
        validate_strict: Fail on validation errors
        source_id: Source ID for new nodes
    
    Returns:
        Dict with 'path', 'nodes_added', 'edges_added'
    """
    claims_path = Path(claims_path)
    
    # Load claims
    claims = validate_and_load(claims_path, 'claims', strict=validate_strict)
    
    # Load existing graph
    existing_graph_path = Path(existing_graph_path) if existing_graph_path else None
    existing_graph = load_existing_graph(existing_graph_path)
    
    original_node_count = len(existing_graph.get('nodes', []))
    original_edge_count = len(existing_graph.get('edges', []))
    
    # Build graph
    updated_graph = build_graph_dspy(
        claims_with_citations=claims,
        existing_graph=existing_graph,
        source_id=source_id
    )
    
    # Save updated graph
    if existing_graph_path:
        output_path = existing_graph_path
    else:
        output_path = claims_path.parent / 'graph.json'
    
    save_validated(updated_graph, output_path, 'graph', strict=validate_strict)
    
    nodes_added = len(updated_graph['nodes']) - original_node_count
    edges_added = len(updated_graph['edges']) - original_edge_count
    
    return {
        'path': str(output_path),
        'nodes_added': nodes_added,
        'edges_added': edges_added,
        'total_nodes': len(updated_graph['nodes']),
        'total_edges': len(updated_graph['edges'])
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Build knowledge graph from claims')
    parser.add_argument('claims_path', help='Path to claims JSON')
    parser.add_argument('--existing-graph', help='Path to existing graph.json')
    parser.add_argument('--source-id', default='003', help='Source ID')
    
    args = parser.parse_args()
    
    result = build_graph(
        claims_path=args.claims_path,
        existing_graph_path=args.existing_graph,
        source_id=args.source_id
    )
    
    print(json.dumps(result, indent=2))
