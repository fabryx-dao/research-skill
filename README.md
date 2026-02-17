# Research Skill

Transform long-form video interviews into validated knowledge graphs with full traceability.

## Quick Start

```bash
# Set API key
export ASSEMBLYAI_API_KEY="your-key-here"

# Process a video
cd ~/.openclaw/workspace/skills/research
./research https://www.youtube.com/watch?v=VIDEO_ID \
  --source-id 003 \
  --source-name "Researcher Name" \
  --domain "field of study" \
  --output-dir ~/repos/deepmemory/site/sources/researcher-slug/archive/
```

## What It Does

**3-Layer Compression Pipeline:**
1. **Transcription** (Layer 0): Audio → Speaker-diarized transcript via AssemblyAI
2. **Extraction** (Layer 1): Transcript → Domain-relevant assertions → Knowledge graph
3. **Synthesis** (Layer 2): Graph → Terms (wiki-style) → Theory (coherent narrative)

**Output:** Validated JSON files ready for knowledge base integration

## Installation

```bash
cd ~/.openclaw/workspace/skills/research
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

**Dependencies:**
- Python 3.11+
- ~400MB disk space for packages
- AssemblyAI API key (for transcription)
- Anthropic/OpenAI API key (for DSPy extraction/synthesis)

## Usage Examples

### Process First Video (New Source)

```bash
./research https://www.youtube.com/watch?v=ktNdOsXFOTg \
  --source-id 003 \
  --source-name "Geoffrey Drumm" \
  --domain "pyramids as chemical facilities" \
  --output-dir ~/repos/deepmemory/site/sources/geoffrey-drumm/archive/
```

**Output files:**
- `archive/video-0001-transcript.json` - Full transcript with speaker labels
- `archive/video-0001-claims.json` - Raw assertions
- `archive/video-0001-claims-clean.json` - With citation IDs
- `graph.json` - Knowledge graph (nodes + edges)
- `terms.json` - Term definitions
- `THEORY.md` - Synthesized theory document

### Process Subsequent Video (Merge Mode)

```bash
./research https://www.youtube.com/watch?v=NEXT_VIDEO_ID \
  --source-id 003 \
  --source-name "Geoffrey Drumm" \
  --domain "pyramids as chemical facilities" \
  --output-dir ~/repos/deepmemory/site/sources/geoffrey-drumm/archive/ \
  --existing-graph ~/repos/deepmemory/site/sources/geoffrey-drumm/graph.json \
  --existing-terms ~/repos/deepmemory/site/sources/geoffrey-drumm/terms.json \
  --existing-theory ~/repos/deepmemory/site/sources/geoffrey-drumm/THEORY.md
```

**Behavior:** Merges new data with existing (accumulative, not replacement)

### Python API

```python
from scripts.process_video import research

result = research(
    video_url="https://www.youtube.com/watch?v=...",
    source_id="003",
    source_name="Geoffrey Drumm",
    domain="pyramids as chemical facilities",
    existing_graph="path/to/graph.json",      # Optional
    existing_terms="path/to/terms.json",      # Optional
    existing_theory="path/to/THEORY.md",      # Optional
    output_dir="path/to/output/",
    validate_strict=True,                     # Fail on validation errors
    assemblyai_key="..."                      # Or from env
)

print(f"Transcript: {result['transcript']['path']}")
print(f"Graph: {result['graph']['path']}")
print(f"Nodes added: {result['graph']['nodes_added']}")
```

## Configuration

### Environment Variables

```bash
# Required
export ASSEMBLYAI_API_KEY="your-key-here"

# DSPy model (via LiteLLM)
export ANTHROPIC_API_KEY="..."  # For Anthropic models (default)
export OPENAI_API_KEY="..."     # For OpenAI models
```

### Model Selection

DSPy uses LiteLLM for model access. Default: `claude-sonnet-4`

To change model, edit `scripts/*.py` and update `model="..."` in DSPy calls.

## Architecture

```
research/
├── scripts/
│   ├── process_video.py      # Main orchestrator
│   ├── transcribe.py          # Stage 1: AssemblyAI transcription
│   ├── extract_claims.py      # Stage 2: DSPy assertion extraction
│   ├── build_graph.py         # Stage 3: Graph building + validation
│   ├── synthesize_terms.py    # Stage 4: Term definition synthesis
│   ├── enhance_theory.py      # Stage 5: Theory document enhancement
│   └── validate.py            # JSON schema validation + graph integrity
├── assets/
│   ├── transcript-schema.json # Transcript validation
│   ├── claims-schema.json     # Claims validation
│   ├── graph-schema.json      # Graph validation
│   ├── terms-schema.json      # Terms validation
│   └── metadata-schema.json   # Metadata validation
├── references/
│   └── TROUBLESHOOTING.md     # Common issues + solutions
├── venv/                      # Python virtual environment
├── research                   # Wrapper script (uses venv python)
├── requirements.txt           # Python dependencies
├── SKILL.md                   # Detailed documentation
└── README.md                  # This file
```

## Progress Tracking

```bash
# View current stage
cat .progress/status.json

# Monitor logs
tail -f .progress/process.log
```

## Validation

All outputs validated before writing:
- JSON schema compliance
- Graph integrity (all edge references exist)
- Citation format (SOURCE_ID-ITEM_ID-CLAIM_ID)
- File naming (video-XXXX-*.json)

Failed validations trigger detailed error messages.

## Performance

**Typical 3-hour interview:**
- Transcription: 10-15 min (AssemblyAI)
- Extraction: 5-10 min (depends on claim count)
- Graph: 2-5 min
- Terms: 3-7 min
- Theory: 5-10 min

**Total: 25-50 minutes**

## Troubleshooting

See [TROUBLESHOOTING.md](references/TROUBLESHOOTING.md)

Common issues:
- No space left on device → Clear caches
- API key not found → Set environment variable
- Import errors → Use venv python (`./research` not `python3`)
- Validation fails → Check citation format + node references

## License

Part of OpenClaw workspace. See main LICENSE.
