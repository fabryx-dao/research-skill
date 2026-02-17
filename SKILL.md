---
name: research
description: Process research videos into validated knowledge graphs. Transcribes via AssemblyAI with speaker diarization, extracts domain assertions, builds graph triplets, synthesizes terms, generates theory. Validates all outputs with JSON schemas. Use when archiving interviews, lectures, or research content into structured knowledge base. Main entry: research(video_url).
compatibility: Requires python3, dspy-ai, assemblyai, yt-dlp, jq. Needs ASSEMBLYAI_API_KEY env var.
metadata:
  author: rawksh
  version: "1.0.0"
  pipeline: "3-layer compression (raw->graph->theory->skills)"
---

# Research Skill

Process research videos through a validated extraction pipeline that produces high-quality knowledge graphs with full traceability.

## When to Use

Use this skill when:
- Processing long-form interviews (Danny Jones podcasts, Lex Fridman, etc.)
- Archiving lecture series or conference talks
- Building knowledge bases from researcher YouTube channels
- Extracting structured knowledge from video content
- You need speaker diarization (separate interviewer from subject)

**Don't use** for: Short clips, news segments, or content without deep knowledge transfer.

## Quick Start

Main meta-method (runs entire pipeline):

```python
from scripts.process_video import research

# Process video with all defaults
research(
    video_url="https://www.youtube.com/watch?v=...",
    source_id="003",
    source_name="Geoffrey Drumm",
    domain="pyramids as chemical facilities"
)
```

This will:
1. Download audio
2. Transcribe with AssemblyAI (speaker diarization)
3. Extract domain-relevant assertions
4. Build validated graph triplets
5. Synthesize terms
6. Generate/enhance theory
7. Save all files with correct naming

## Progress Tracking

Check processing status:

```bash
# View current stage
cat ~/.openclaw/workspace/skills/research/.progress/status.json

# View detailed log
tail -f ~/.openclaw/workspace/skills/research/.progress/process.log
```

## Pipeline Stages

### Stage 1: Transcription
- Downloads audio via yt-dlp
- Uploads to AssemblyAI
- Speaker diarization (separates interviewer/subject)
- Saves: `video-XXXX-transcript.json`

### Stage 2: Extraction
- Reads full transcript
- Filters domain-relevant assertions (using DSPy)
- Validates against existing concepts
- Saves: `video-XXXX-claims.json`, `video-XXXX-claims-clean.json`

### Stage 3: Graph Building
- Converts claims to triplets (subject-relation-object)
- Validates all node references exist
- Deduplicates nodes
- Merges with existing graph
- Saves: `graph.json` (updated)

### Stage 4: Term Synthesis
- Identifies new concepts needing definitions
- Generates Wikipedia-style entries
- Aggregates evidence across videos
- Saves: `terms.json` (updated)

### Stage 5: Theory Enhancement
- Synthesizes coherent narrative
- Adds new sections or expands existing
- Preserves existing structure (accumulative)
- Saves: `THEORY.md` (updated)

## Configuration

Set required environment variables:

```bash
export ASSEMBLYAI_API_KEY="your-key-here"
```

Optional config (defaults shown):

```python
research(
    video_url="...",
    source_id="003",
    source_name="Geoffrey Drumm",
    domain="pyramids as chemical facilities",
    filter_keywords=["chemical", "pyramid", "ammonia"],  # Domain filters
    existing_graph="site/sources/geoffrey-drumm/graph.json",
    existing_terms="site/sources/geoffrey-drumm/terms.json",
    existing_theory="site/sources/geoffrey-drumm/THEORY.md",
    output_dir="site/sources/geoffrey-drumm/archive/",
    validate_strict=True  # Fail on validation errors (vs. warn)
)
```

## Advanced Usage

Run individual stages:

```python
from scripts.extract_claims import extract_claims
from scripts.build_graph import build_graph
from scripts.synthesize_terms import synthesize_terms

# Just extract claims from existing transcript
claims = extract_claims(
    transcript_path="video-0004-transcript.json",
    domain="pyramids as chemical facilities",
    existing_concepts=["ammonia", "pyramid", "chemical"]
)

# Just build graph from existing claims
graph = build_graph(
    claims_path="video-0004-claims-clean.json",
    existing_graph_path="graph.json"
)
```

## Validation

All outputs are validated before writing:

- **JSON syntax**: Parsed and re-serialized
- **Schema compliance**: Checked against `assets/*.json` schemas
- **Graph integrity**: All edge references must point to existing nodes
- **Citation format**: Must match `SOURCE_ID-ITEM_ID-CLAIM_ID`
- **File naming**: Must follow `video-XXXX-*.json` convention

Failed validations trigger detailed error messages and optional retry.

## Troubleshooting

See [references/TROUBLESHOOTING.md](references/TROUBLESHOOTING.md) for common issues.

Quick checks:

```bash
# Validate existing graph
python scripts/validate_graph.py site/sources/geoffrey-drumm/graph.json

# Check AssemblyAI status
python scripts/check_transcription.py <transcript-id>

# Reprocess failed stage
python scripts/process_video.py --resume-from=stage3 --video-id=0004
```

## Output Structure

After successful run:

```
site/sources/{source-slug}/
├── archive/
│   ├── video-XXXX-metadata.json      # Video info + timestamps
│   ├── video-XXXX-transcript.json    # Full transcript (speaker-diarized)
│   ├── video-XXXX-claims.json        # Extracted assertions
│   └── video-XXXX-claims-clean.json  # With citation IDs
├── graph.json                        # Updated knowledge graph
├── terms.json                        # Updated term definitions
└── THEORY.md                         # Updated theory synthesis
```

## Performance

Typical processing time for 3-hour interview:
- Transcription: 10-15 min (AssemblyAI)
- Extraction: 5-10 min (depends on assertion count)
- Graph building: 2-5 min
- Term synthesis: 3-7 min
- Theory enhancement: 5-10 min

**Total: 25-50 minutes**

Progress updates written every 30 seconds.
