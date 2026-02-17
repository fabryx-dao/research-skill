#!/usr/bin/env python3
"""
Validation utilities for JSON schemas and graph integrity.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from jsonschema import validate as json_validate, ValidationError

logger = logging.getLogger(__name__)

ASSETS_DIR = Path(__file__).parent.parent / 'assets'


def load_schema(schema_name: str) -> Dict[str, Any]:
    """Load JSON schema from assets directory."""
    schema_path = ASSETS_DIR / f"{schema_name}-schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found: {schema_path}")
    
    with open(schema_path) as f:
        return json.load(f)


def validate_json(data: Any, schema_name: str, strict: bool = True) -> Tuple[bool, Optional[str]]:
    """
    Validate data against JSON schema.
    
    Args:
        data: Data to validate
        schema_name: Schema name (without -schema.json suffix)
        strict: If False, log warnings instead of raising errors
    
    Returns:
        (is_valid, error_message)
    """
    try:
        schema = load_schema(schema_name)
        json_validate(instance=data, schema=schema)
        return (True, None)
    except ValidationError as e:
        error_msg = f"Validation failed for {schema_name}: {e.message}"
        if strict:
            logger.error(error_msg)
            raise
        else:
            logger.warning(error_msg)
            return (False, error_msg)
    except Exception as e:
        error_msg = f"Validation error for {schema_name}: {e}"
        logger.error(error_msg)
        if strict:
            raise
        return (False, error_msg)


def validate_graph_integrity(graph: Dict[str, Any], strict: bool = True) -> Tuple[bool, List[str]]:
    """
    Validate graph integrity beyond schema:
    - All edge.from and edge.to must reference existing nodes
    - No duplicate node IDs
    - All citations must match pattern
    
    Args:
        graph: Graph data with nodes and edges
        strict: If False, return warnings instead of raising
    
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    
    # Check for duplicate node IDs
    node_ids = [n['id'] for n in graph.get('nodes', [])]
    duplicates = [nid for nid in node_ids if node_ids.count(nid) > 1]
    if duplicates:
        errors.append(f"Duplicate node IDs: {set(duplicates)}")
    
    # Build node ID set
    node_id_set = set(node_ids)
    
    # Check all edge references
    for idx, edge in enumerate(graph.get('edges', [])):
        if edge['from'] not in node_id_set:
            errors.append(f"Edge {idx}: 'from' node '{edge['from']}' does not exist")
        
        if edge['to'] not in node_id_set:
            errors.append(f"Edge {idx}: 'to' node '{edge['to']}' does not exist")
        
        # Check citation format in evidence
        for eidx, ev in enumerate(edge.get('evidence', [])):
            citation = ev.get('citation', '')
            if not citation:
                errors.append(f"Edge {idx}, evidence {eidx}: missing citation")
            elif not _is_valid_citation(citation):
                errors.append(f"Edge {idx}, evidence {eidx}: invalid citation format '{citation}'")
    
    if errors:
        error_msg = "\n".join(errors)
        logger.error(f"Graph integrity check failed:\n{error_msg}")
        if strict:
            raise ValueError(f"Graph integrity errors: {error_msg}")
        return (False, errors)
    
    return (True, [])


def _is_valid_citation(citation: str) -> bool:
    """Check if citation matches SOURCE_ID-ITEM_ID-CLAIM_ID pattern."""
    import re
    return bool(re.match(r'^\d{3}-\d{4}-\d{3}$', citation))


def validate_and_load(path: Path, schema_name: str, strict: bool = True) -> Dict[str, Any]:
    """
    Load and validate JSON file.
    
    Args:
        path: Path to JSON file
        schema_name: Schema to validate against
        strict: Fail on validation errors
    
    Returns:
        Loaded and validated data
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    with open(path) as f:
        data = json.load(f)
    
    validate_json(data, schema_name, strict=strict)
    
    # Extra integrity check for graphs
    if schema_name == 'graph':
        validate_graph_integrity(data, strict=strict)
    
    return data


def save_validated(data: Any, path: Path, schema_name: str, strict: bool = True):
    """
    Validate and save JSON file.
    
    Args:
        data: Data to save
        path: Destination path
        schema_name: Schema to validate against
        strict: Fail on validation errors
    """
    # Validate before writing
    validate_json(data, schema_name, strict=strict)
    
    # Extra integrity check for graphs
    if schema_name == 'graph':
        validate_graph_integrity(data, strict=strict)
    
    # Write to file
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Validated and saved: {path}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: validate.py <schema-name> <file-path>")
        sys.exit(1)
    
    schema_name = sys.argv[1]
    file_path = Path(sys.argv[2])
    
    try:
        data = validate_and_load(file_path, schema_name)
        print(f"✓ Valid {schema_name}: {file_path}")
    except Exception as e:
        print(f"✗ Invalid: {e}")
        sys.exit(1)
