#!/usr/bin/env python3
"""
Stage 5: Synthesize coherent theory narrative from graph + terms.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import dspy
from dspy import Signature, InputField, OutputField
import tiktoken

from validate import validate_and_load

logger = logging.getLogger(__name__)

# Theory size constraints for Claude Opus 4.6
# Context: 200K tokens, Max output: 128K tokens
# Input budget: 200K - 128K = 72K
# Overhead (graph + terms + claims + prompts): ~30K
# Available for theory: 72K - 30K = 42K
THEORY_TARGET_TOKENS = 40000  # Target size (conservative)
THEORY_HARD_LIMIT_TOKENS = 60000  # Hard limit (safety margin)

def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """Count tokens using tiktoken."""
    enc = tiktoken.get_encoding(model)
    return len(enc.encode(text))


# DSPy signature for theory synthesis
class SynthesizeTheory(Signature):
    """Synthesize coherent narrative of author's worldview from knowledge graph + terms."""
    
    source_name: str = InputField(desc="Name of the source/author")
    domain: str = InputField(desc="Research domain (e.g., theology/pharmacology)")
    graph_summary: str = InputField(desc="Summary of key nodes/edges from knowledge graph")
    terms_summary: str = InputField(desc="Summary of key term definitions")
    existing_theory: str = InputField(desc="Existing THEORY.md section headers if updating, or 'none'")
    new_video_claims: str = InputField(desc="Key claims from new video if updating, or 'none'")
    
    theory: str = OutputField(
        desc="Coherent narrative synthesizing the author's worldview from current graph+terms data. "
             "Write in clear prose that connects concepts into a unified theory. "
             "Organize into thematic sections with markdown headers (## Section). "
             "State ONLY what the source material establishes - zero external knowledge. "
             "NO citations - this is clean prose. Evidence lives in graph/terms layers. "
             "If existing sections provided: synthesize fresh from current data, maintaining similar structure/themes. "
             "Focus on the most important concepts from the graph, not exhaustive coverage."
    )


def summarize_graph(graph: Dict[str, Any], max_nodes: int = 30) -> str:
    """
    Create summary of key graph nodes and relationships.
    
    Args:
        graph: Knowledge graph
        max_nodes: Maximum nodes to include
    
    Returns:
        Text summary of graph structure
    """
    nodes = graph.get('nodes', [])
    edges = graph.get('edges', [])
    
    # Count connections per node
    connection_count = {}
    for edge in edges:
        connection_count[edge['from']] = connection_count.get(edge['from'], 0) + 1
        connection_count[edge['to']] = connection_count.get(edge['to'], 0) + 1
    
    # Sort nodes by connections (most connected first)
    sorted_nodes = sorted(nodes, key=lambda n: connection_count.get(n['id'], 0), reverse=True)
    
    # Take top N
    top_nodes = sorted_nodes[:max_nodes]
    
    # Build summary
    lines = ["KEY CONCEPTS:"]
    for node in top_nodes:
        connections = connection_count.get(node['id'], 0)
        lines.append(f"- {node['label']} ({node['type']}, {connections} connections)")
    
    lines.append("\nKEY RELATIONSHIPS:")
    # Show edges involving top nodes
    top_node_ids = {n['id'] for n in top_nodes}
    relevant_edges = [e for e in edges if e['from'] in top_node_ids or e['to'] in top_node_ids]
    
    # Sample edges
    for edge in relevant_edges[:20]:
        from_node = next((n for n in nodes if n['id'] == edge['from']), None)
        to_node = next((n for n in nodes if n['id'] == edge['to']), None)
        if from_node and to_node:
            lines.append(f"- {from_node['label']} → {edge['relation']} → {to_node['label']}")
    
    return "\n".join(lines)


def summarize_terms(terms: List[Dict[str, Any]]) -> str:
    """
    Create summary of term definitions.
    
    Args:
        terms: List of term dicts
    
    Returns:
        Text summary of terms
    """
    lines = ["TERM DEFINITIONS:"]
    
    for term in terms:
        lines.append(f"\n**{term['term']}** ({term['category']})")
        lines.append(term['definition'])
    
    return "\n".join(lines)


def load_existing_theory(theory_path: Optional[Path]) -> str:
    """
    Load existing THEORY.md section headers (not full content).
    
    For large theories, passing full content creates massive prompts that timeout.
    Instead, pass only section structure so LLM knows what exists.
    
    Also validates that existing theory doesn't exceed size limits.
    
    Args:
        theory_path: Path to THEORY.md
    
    Returns:
        Section headers or "none"
    
    Raises:
        ValueError: If existing theory exceeds hard token limit
    """
    if not theory_path or not theory_path.exists():
        return "none"
    
    with open(theory_path) as f:
        content = f.read()
    
    if not content.strip():
        return "none"
    
    # Check existing theory size
    token_count = count_tokens(content)
    logger.info(f"Existing theory: {token_count:,} tokens")
    
    if token_count > THEORY_HARD_LIMIT_TOKENS:
        raise ValueError(
            f"Existing theory {token_count:,} tokens exceeds hard limit "
            f"{THEORY_HARD_LIMIT_TOKENS:,} tokens. Theory must be condensed before "
            f"adding new content. Target: {THEORY_TARGET_TOKENS:,} tokens."
        )
    
    # Extract section headers only (## lines)
    headers = []
    for line in content.split('\n'):
        if line.startswith('##'):
            headers.append(line)
    
    if not headers:
        return "none"
    
    return "EXISTING SECTIONS:\n" + "\n".join(headers[:50])  # Max 50 headers


def synthesize_theory_dspy(
    source_name: str,
    domain: str,
    graph_summary: str,
    terms_summary: str,
    existing_theory: str = "none",
    new_video_claims: Optional[str] = None,
    model: str = "claude-opus-4-6"
) -> str:
    """
    Synthesize theory using DSPy with Claude Opus 4.6.
    
    Enforces token limits:
    - Target: 40K tokens (conservative)
    - Hard limit: 60K tokens (fails if exceeded)
    
    Args:
        source_name: Name of source/author
        domain: Research domain
        graph_summary: Summary of knowledge graph
        terms_summary: Summary of terms
        existing_theory: Existing THEORY.md section headers
        new_video_claims: Key claims from new video
        model: DSPy model (default: claude-opus-4-6)
    
    Returns:
        THEORY.md markdown content
    
    Raises:
        ValueError: If generated theory exceeds hard token limit
    """
    # Setup DSPy
    lm = dspy.LM(model)
    dspy.configure(lm=lm)
    
    # Create synthesis module
    synthesize = dspy.ChainOfThought(SynthesizeTheory)
    
    # Add size guidance to the synthesis
    size_guidance = (
        f"\n\nIMPORTANT: Keep theory under {THEORY_TARGET_TOKENS:,} tokens "
        f"(hard limit {THEORY_HARD_LIMIT_TOKENS:,} tokens). "
        "Focus on most important concepts, not exhaustive coverage."
    )
    
    # Synthesize
    result = synthesize(
        source_name=source_name,
        domain=domain,
        graph_summary=graph_summary + size_guidance,
        terms_summary=terms_summary,
        existing_theory=existing_theory,
        new_video_claims=new_video_claims or "none"
    )
    
    theory_text = result.theory
    
    # Validate token count
    token_count = count_tokens(theory_text)
    
    if token_count > THEORY_HARD_LIMIT_TOKENS:
        raise ValueError(
            f"Generated theory exceeds hard limit: {token_count:,} tokens > "
            f"{THEORY_HARD_LIMIT_TOKENS:,} tokens. Theory must fit in Claude Opus 4.6 "
            f"output budget (128K tokens with headroom)."
        )
    
    if token_count > THEORY_TARGET_TOKENS:
        logger.warning(
            f"Theory size {token_count:,} tokens exceeds target {THEORY_TARGET_TOKENS:,} tokens "
            f"but within hard limit {THEORY_HARD_LIMIT_TOKENS:,} tokens"
        )
    else:
        logger.info(f"Theory size: {token_count:,} tokens (target: {THEORY_TARGET_TOKENS:,})")
    
    return theory_text


def synthesize_theory(
    graph_path: str,
    terms_path: str,
    source_name: str,
    domain: str,
    existing_theory_path: Optional[str] = None,
    claims_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Complete theory synthesis pipeline.
    
    Args:
        graph_path: Path to graph.json
        terms_path: Path to terms.json
        source_name: Name of source/author
        domain: Research domain
        existing_theory_path: Path to existing THEORY.md
        claims_path: Path to latest video claims (for updates)
    
    Returns:
        Dict with 'path', 'created', 'updated'
    """
    graph_path = Path(graph_path)
    terms_path = Path(terms_path)
    existing_theory_path = Path(existing_theory_path) if existing_theory_path else None
    
    # Load data
    graph = validate_and_load(graph_path, 'graph')
    terms = validate_and_load(terms_path, 'terms')
    
    # Load existing theory if present
    existing_theory = load_existing_theory(existing_theory_path)
    
    # Summarize graph
    graph_summary = summarize_graph(graph)
    
    # Summarize terms
    terms_summary = summarize_terms(terms)
    
    # Load new video claims if updating
    new_video_claims = None
    if claims_path:
        claims_path = Path(claims_path)
        if claims_path.exists():
            with open(claims_path) as f:
                claims = json.load(f)
            # Extract key claims
            new_video_claims = "\n".join([
                f"- {claim['claim']} [{claim.get('citation', claim.get('citationId', 'no-citation'))}]"
                for claim in claims[:20]  # Top 20 claims
            ])
    
    logger.info(f"Synthesizing theory for {source_name} ({domain})")
    
    # Synthesize theory
    theory_content = synthesize_theory_dspy(
        source_name=source_name,
        domain=domain,
        graph_summary=graph_summary,
        terms_summary=terms_summary,
        existing_theory=existing_theory,
        new_video_claims=new_video_claims
    )
    
    # Determine output path
    if existing_theory_path:
        output_path = existing_theory_path
        created = False
    else:
        output_path = graph_path.parent / 'THEORY.md'
        created = True
    
    # Write theory
    with open(output_path, 'w') as f:
        f.write(theory_content)
    
    logger.info(f"Theory {'created' if created else 'updated'}: {output_path}")
    
    return {
        'path': str(output_path),
        'created': created,
        'updated': not created
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Synthesize theory narrative')
    parser.add_argument('graph_path', help='Path to graph.json')
    parser.add_argument('terms_path', help='Path to terms.json')
    parser.add_argument('--source-name', required=True, help='Name of source/author')
    parser.add_argument('--domain', required=True, help='Research domain')
    parser.add_argument('--existing-theory', help='Path to existing THEORY.md')
    parser.add_argument('--claims', help='Path to latest video claims')
    
    args = parser.parse_args()
    
    result = synthesize_theory(
        graph_path=args.graph_path,
        terms_path=args.terms_path,
        source_name=args.source_name,
        domain=args.domain,
        existing_theory_path=args.existing_theory,
        claims_path=args.claims
    )
    
    print(json.dumps(result, indent=2))
