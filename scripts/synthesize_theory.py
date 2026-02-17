#!/usr/bin/env python3
"""
Stage 5: Synthesize coherent theory narrative from graph + terms.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

import anthropic
import tiktoken

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


# System prompt for theory synthesis
THEORY_SYNTHESIS_SYSTEM = """You synthesize coherent narratives of an author's worldview from source material.

Your task:
1. Read the existing theory (if provided) and new video claims
2. EXPAND the existing theory with new insights from the claims
3. Add new sections if new themes emerge
4. Preserve existing narrative flow
5. Write in clear prose with markdown headers (## Section)
6. State ONLY what the source material establishes - zero external knowledge
7. NO citations - evidence lives in structured data layers
8. Keep under 40K tokens - focus on most important concepts

Output format: Markdown document with thematic sections."""


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
    Load existing THEORY.md full content.
    
    Validates that existing theory doesn't exceed size limits.
    
    Args:
        theory_path: Path to THEORY.md
    
    Returns:
        Full theory content or "none"
    
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
    
    return content


def synthesize_theory_anthropic(
    source_name: str,
    domain: str,
    existing_theory: str = "none",
    new_video_claims: Optional[str] = None
) -> str:
    """
    Synthesize theory using Anthropic SDK with Claude Opus 4.6.
    
    Enforces token limits:
    - Target: 40K tokens (conservative)
    - Hard limit: 60K tokens (fails if exceeded)
    
    Args:
        source_name: Name of source/author
        domain: Research domain
        existing_theory: Existing THEORY.md full content
        new_video_claims: Domain-relevant claims from new video
    
    Returns:
        THEORY.md markdown content
    
    Raises:
        ValueError: If generated theory exceeds hard token limit
    """
    print(f"\n[1/7] Getting ANTHROPIC_API_KEY...")
    # Get API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment")
    print(f"[1/7] ✓ API key found (length: {len(api_key)})")
    
    print(f"[2/7] Creating Anthropic client...")
    client = anthropic.Anthropic(api_key=api_key)
    print(f"[2/7] ✓ Client created")
    
    print(f"[3/7] Counting input tokens...")
    existing_tokens = count_tokens(existing_theory)
    claims_tokens = count_tokens(new_video_claims or 'none')
    print(f"[3/7] ✓ Existing theory: {existing_tokens:,} tokens, Claims: {claims_tokens:,} tokens")
    logger.info(f"Input size: existing_theory={existing_tokens:,} tokens, claims={claims_tokens:,} tokens")
    
    print(f"[4/7] Building prompt...")
    # Build user prompt
    user_prompt = f"""Source: {source_name}
Domain: {domain}

Existing Theory:
{existing_theory if existing_theory != "none" else "(none - creating new theory)"}

New Video Claims:
{new_video_claims or "(none)"}

Task: {"Expand the existing theory" if existing_theory != "none" else "Create a new theory"} by synthesizing these new claims into a coherent narrative."""
    
    prompt_tokens = count_tokens(user_prompt)
    print(f"[4/7] ✓ Prompt built ({prompt_tokens:,} tokens)")
    
    print(f"[5/7] Calling Claude Opus 4.6 with streaming (max_tokens=65536)...")
    logger.info("Calling Claude Opus 4.6 with streaming...")
    
    # Call API with streaming for large max_tokens
    theory_chunks = []
    chunk_count = 0
    print(f"[5/7] Waiting for first response chunk...")
    with client.messages.stream(
        model='claude-opus-4-6',
        max_tokens=65536,  # Allow up to 65K tokens output
        system=THEORY_SYNTHESIS_SYSTEM,
        messages=[{
            'role': 'user',
            'content': user_prompt
        }]
    ) as stream:
        print(f"[5/7] ✓ Stream started, receiving chunks...")
        for text in stream.text_stream:
            theory_chunks.append(text)
            chunk_count += 1
            if chunk_count % 10 == 0:
                print(f"[5/7] ...received {chunk_count} chunks ({len(''.join(theory_chunks))} chars so far)")
    
    print(f"[5/7] ✓ Streaming complete ({chunk_count} chunks total)")
    
    print(f"[6/7] Joining chunks and getting usage stats...")
    theory_text = ''.join(theory_chunks)
    
    # Get final message for usage stats
    final_message = stream.get_final_message()
    print(f"[6/7] ✓ Generated {len(theory_text)} chars")
    print(f"[6/7] ✓ Usage: {final_message.usage.input_tokens:,} in, {final_message.usage.output_tokens:,} out")
    logger.info(f"Synthesis complete. Usage: {final_message.usage.input_tokens:,} in, {final_message.usage.output_tokens:,} out")
    
    print(f"[7/7] Validating token count...")
    # Validate token count
    token_count = count_tokens(theory_text)
    print(f"[7/7] ✓ Token count: {token_count:,} tokens")
    
    if token_count > THEORY_HARD_LIMIT_TOKENS:
        print(f"[7/7] ✗ ERROR: Exceeds hard limit ({THEORY_HARD_LIMIT_TOKENS:,} tokens)")
        raise ValueError(
            f"Generated theory exceeds hard limit: {token_count:,} tokens > "
            f"{THEORY_HARD_LIMIT_TOKENS:,} tokens. Theory must fit in Claude Opus 4.6 "
            f"output budget (128K tokens with headroom)."
        )
    
    if token_count > THEORY_TARGET_TOKENS:
        print(f"[7/7] ⚠ WARNING: Exceeds target ({THEORY_TARGET_TOKENS:,} tokens) but within hard limit")
        logger.warning(
            f"Theory size {token_count:,} tokens exceeds target {THEORY_TARGET_TOKENS:,} tokens "
            f"but within hard limit {THEORY_HARD_LIMIT_TOKENS:,} tokens"
        )
    else:
        print(f"[7/7] ✓ Within target ({THEORY_TARGET_TOKENS:,} tokens)")
        logger.info(f"Theory size: {token_count:,} tokens (target: {THEORY_TARGET_TOKENS:,})")
    
    print(f"[7/7] ✓ Validation complete\n")
    return theory_text


def synthesize_theory(
    source_name: str,
    domain: str,
    claims_path: str,
    existing_theory_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Complete theory synthesis pipeline.
    
    Theory synthesis is simple: existing_theory + new_claims → updated_theory
    
    Args:
        source_name: Name of source/author
        domain: Research domain
        claims_path: Path to video claims
        existing_theory_path: Path to existing THEORY.md
    
    Returns:
        Dict with 'path', 'created', 'updated'
    """
    claims_path = Path(claims_path)
    existing_theory_path = Path(existing_theory_path) if existing_theory_path else None
    
    # Load existing theory if present
    existing_theory = load_existing_theory(existing_theory_path)
    
    # Load new video claims
    if not claims_path.exists():
        raise FileNotFoundError(f"Claims file not found: {claims_path}")
    
    with open(claims_path) as f:
        claims = json.load(f)
    
    # Format claims
    new_video_claims = "\n".join([
        f"- {claim['claim']}"
        for claim in claims
    ])
    
    logger.info(f"Synthesizing theory for {source_name} ({domain}) from {len(claims)} claims")
    
    # Synthesize theory
    theory_content = synthesize_theory_anthropic(
        source_name=source_name,
        domain=domain,
        existing_theory=existing_theory,
        new_video_claims=new_video_claims
    )
    
    # Determine output path
    if existing_theory_path:
        output_path = existing_theory_path
        created = False
    else:
        output_path = claims_path.parent / 'THEORY.md'
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
    parser.add_argument('claims_path', help='Path to video claims')
    parser.add_argument('--source-name', required=True, help='Name of source/author')
    parser.add_argument('--domain', required=True, help='Research domain')
    parser.add_argument('--existing-theory', help='Path to existing THEORY.md')
    
    args = parser.parse_args()
    
    result = synthesize_theory(
        source_name=args.source_name,
        domain=args.domain,
        claims_path=args.claims_path,
        existing_theory_path=args.existing_theory
    )
    
    print(json.dumps(result, indent=2))
