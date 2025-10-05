# MLX-SDXL-LoRA Caption Tool - AI Development Guide

## Project Overview

This is a **dual-architecture** VLM-powered image captioning tool for SDXL LoRA training with two distinct implementations:

- **`app.py`**: LM Studio integration (OpenAI-compatible API) with Gradio UI
- **`caption_tool.py`**: MLX-VLM native integration for macOS Apple Silicon

Both tools generate captions for LoRA training datasets with different VLM backends but similar user workflows.

## Key Architecture Patterns

### VLM Integration Strategy
The project implements **two VLM backends** - understand which one you're working with:

**LM Studio Backend (`app.py`)**:
- Uses OpenAI client pointing to `http://127.0.0.1:1234/v1`
- Encodes images to base64 for API transmission
- Requires external LM Studio server running

**MLX Backend (`caption_tool.py`)**:
- Direct MLX-VLM integration via `mlx_vlm` package
- Lazy model loading pattern: `load_model()` called on first use
- Uses `CaptionGenerator` class for VLM operations

### Data Persistence Pattern
**Critical**: Both implementations save captions as `.txt` files co-located with images:
```
image.jpg -> image.txt (UTF-8 encoded)
```

The `workspace_state` dict in `app.py` tracks: `images` (list of tuples), `current_page`, `workspace_dir`, `images_per_page`.

### UI Architecture (Gradio)
Both tools use **paginated gallery pattern** for large datasets:
- Gallery renders 12 images per page by default
- JavaScript onclick handlers for image selection
- Status/error messaging throughout UI
- Bulk operations for batch processing

## Development Workflows

### Testing Strategy
```bash
# Run full test suite
pytest tests/ -v --cov=. --cov-report=term

# Code quality (required for CI)
ruff check .
ruff format .
```

**Test Architecture**: Uses `temp_workspace` fixture creating test images, mocks OpenAI calls, tests workspace state management.

### Environment Setup
**Conda workflow** (recommended):
```bash
conda env create -f environment.yml
conda activate mlx-sdxl-lora
```

**Alternative pip workflow**:
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install gradio openai pillow pytest pytest-cov ruff
```

### VS Code Integration
Project uses **Ruff** for formatting/linting (not flake8/black):
- Format on save enabled
- Line length: 88 characters
- Pytest integration configured
- Python interpreter: `.venv/bin/python`

## Project-Specific Conventions

### Error Handling Pattern
Functions return **tuple responses** for UI updates:
```python
def operation() -> Tuple[str, str]:
    return status_message, updated_ui_html
```

Status messages use emoji prefixes: `✅` for success, `❌` for errors.

### Global State Management
**Critical**: `workspace_state` dict in `app.py` is global - always reset in tests using `reset_workspace_state` fixture.

### Image Format Support
Only these formats: `{".jpg", ".jpeg", ".png", ".webp", ".bmp"}` - check `SUPPORTED_FORMATS` constant.

### Caption Generation Prompts
Default prompt focuses on **LoRA training context**:
```python
SYSTEM_PROMPT = """Generate detailed, comma-separated captions describing the image content.
Focus on: subject, style, composition, colors, lighting, mood, and artistic elements."""
```

## Integration Points

### LM Studio Dependency
`app.py` requires LM Studio running locally - **no fallback**. Connection failures throw exceptions immediately.

### MLX Performance
`caption_tool.py` optimized for Apple Silicon - model loading is expensive, so implement lazy loading pattern.

### Gradio File Handling
Uses `type="filepath"` for image components and `allowed_paths=["/"]` for file access in `launch()`.

## Common Implementation Patterns

### Batch Processing
Both implementations support:
- Range-based batch processing (1-based indexing in UI, 0-based internally)
- Progress tracking with `gr.Progress()`
- Auto-save option for generated captions

### Bulk Operations
Template-based bulk updates with modes: `append`, `prepend`, `replace` - always validate template is non-empty.

### Gallery Navigation
Pagination uses `navigate_page(direction: int)` with bounds checking to prevent invalid page states.

## Development Anti-Patterns

- **Don't mix the two VLM backends** - they're incompatible
- **Don't assume model is loaded** - always check `model_loaded` state
- **Don't modify workspace_state without bounds checking** - gallery will break
- **Don't use relative paths** - workspace scanning uses absolute paths only
- **Don't forget UTF-8 encoding** for caption files - critical for international content

## Quick Reference

**Start app**: `python app.py` (LM Studio) or `python caption_tool.py` (MLX)  
**Run tests**: `pytest tests/ -v`  
**Format code**: `ruff format .`  
**Key modules**: `encode_image_to_base64()`, `scan_workspace_directory()`, `CaptionManager`  
**Test fixtures**: `temp_workspace`, `reset_workspace_state`