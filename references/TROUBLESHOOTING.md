# Troubleshooting

## Installation Issues

### No space left on device

**Symptom:** `pip install` fails with "OSError: [Errno 28] No space left on device"

**Solution:** Free up disk space. DSPy + dependencies need ~400MB.
```bash
# Check available space
df -h /

# Clear common caches
rm -rf ~/.cache/pip
rm -rf ~/.cache/go-build
sudo apt clean
sudo journalctl --vacuum-size=200M
```

### Virtual environment corrupted

**Symptom:** `./venv/bin/pip: No such file or directory`

**Solution:** Recreate the venv:
```bash
cd ~/.openclaw/workspace/skills/research
rm -rf venv
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

## Runtime Issues

### AssemblyAI API key not found

**Symptom:** `ValueError: ASSEMBLYAI_API_KEY not found in environment`

**Solution:** Set the API key:
```bash
export ASSEMBLYAI_API_KEY="your-key-here"
```

Or pass it directly:
```python
research(video_url="...", assemblyai_key="your-key-here")
```

### DSPy model configuration

**Symptom:** DSPy fails to connect to model

**Solution:** DSPy uses LiteLLM under the hood. Check model name format:
- Anthropic: `anthropic/claude-sonnet-4` (default)
- OpenAI: `openai/gpt-4`

Set environment variables for API keys:
```bash
export ANTHROPIC_API_KEY="..."
export OPENAI_API_KEY="..."
```

### Graph integrity validation fails

**Symptom:** `ValueError: Graph integrity errors: Edge X: 'from' node 'Y' does not exist`

**Solution:** This means DSPy generated a triplet referencing a non-existent node. This is a DSPy extraction error. Options:
1. Run validation with `strict=False` to log warnings instead of failing
2. Manually fix the graph.json by adding missing nodes or removing bad edges
3. Re-run extraction with better prompts

### Import errors after installation

**Symptom:** `ModuleNotFoundError: No module named 'dspy'`

**Solution:** Make sure you're using the venv python:
```bash
# Wrong (uses system python)
python3 scripts/process_video.py

# Right (uses venv python)
./venv/bin/python3 scripts/process_video.py

# Or use wrapper
./research --help
```

## Common Errors

### Video download fails

**Symptom:** `yt-dlp failed: ERROR: ...`

**Possible causes:**
- Video is private/deleted
- Network connectivity issue
- yt-dlp needs updating: `./venv/bin/pip install --upgrade yt-dlp`

### Transcript file wrong format

**Symptom:** Schema validation fails - expected JSON array, got string

**Solution:** Ensure transcript saved as JSON, not text:
```json
{
  "success": true,
  "videoId": "0004",
  "transcript": [{"text": "..."}]
}
```

NOT: `video-0004-transcript.txt` with plain text

### Citations don't follow format

**Symptom:** `invalid citation format 'some-wrong-format'`

**Solution:** All citations must be `SOURCE_ID-ITEM_ID-CLAIM_ID`:
- SOURCE_ID: 3 digits (e.g., 003)
- ITEM_ID: 4 digits (e.g., 0004)
- CLAIM_ID: 3 digits (e.g., 015)

Example: `003-0004-015`

## Performance Issues

### Transcription is slow

**Normal:** 10-15 minutes for 3-hour video (AssemblyAI processes at ~12x realtime)

**If slower:** Check AssemblyAI dashboard for service status

### Extraction taking too long

**Typical:** 5-10 minutes for 200-300 claims

**If stuck:** Check if DSPy is actually calling the API:
```bash
# Monitor API calls (if using Anthropic)
watch -n 2 'tail -f ~/.openclaw/agents/*/main.jsonl | grep -i anthropic'
```

## Getting Help

1. Check progress logs: `tail -f ~/.openclaw/workspace/skills/research/.progress/process.log`
2. Check status: `cat ~/.openclaw/workspace/skills/research/.progress/status.json`
3. Validate manually: `./venv/bin/python3 scripts/validate.py graph /path/to/graph.json`
