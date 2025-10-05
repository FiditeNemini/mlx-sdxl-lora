# 🎨 VLM-Powered Image Captioning Tool for LoRA Training

A desktop application that automates caption generation for SDXL LoRA models using a local Vision-Language Model (VLM) via LM Studio.

[![CI](https://github.com/FiditeNemini/mlx-sdxl-lora/actions/workflows/ci.yml/badge.svg)](https://github.com/FiditeNemini/mlx-sdxl-lora/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Features

- **VLM Integration**: Connects to LM Studio's local Vision-Language Model for intelligent caption generation
- **Batch Processing**: Generate captions for multiple images in one go
- **Interactive Gallery**: Paginated UI for browsing large image datasets
- **In-Place Editing**: Edit and refine captions directly in the interface
- **Bulk Updates**: Apply templates to all captions (append/prepend/replace)
- **Data Persistence**: Automatically saves captions as UTF-8 .txt files alongside images
- **Human-in-the-Loop**: Review and refine AI-generated captions before finalizing

## 📋 Prerequisites

- **Python**: 3.10 or higher
- **LM Studio**: Running locally at `http://127.0.0.1:1234` with a VLM loaded
- **Conda** (recommended) or pip for dependency management

## 🚀 Quick Start

### 1. Clone the Repository

# 🖼️ VLM-Powered Image Captioning Tool for LoRA Training

A desktop application to automate caption generation for SDXL LoRA models using a local Vision-Language Model (VLM). Built with Gradio and MLX for macOS.

## Features

- 🤖 **Local VLM Integration**: Uses MLX-optimized Vision-Language Models for fast, private caption generation
- 📦 **Batch Processing**: Process multiple images at once with progress tracking
- ✏️ **Interactive Editing**: Human-in-the-loop refinement with a built-in caption editor
- 💾 **Data Persistence**: Automatic saving of captions as .txt files alongside images
- 🎨 **User-Friendly UI**: Clean Gradio interface with multiple tabs for different workflows
- 🔄 **Caption Management**: Load existing captions, edit them, and export metadata

## Requirements

- macOS (Apple Silicon recommended for optimal MLX performance)
- Python 3.9+
- MLX-compatible hardware

## Installation

1. Clone the repository:
```bash
git clone https://github.com/FiditeNemini/mlx-sdxl-lora.git
cd mlx-sdxl-lora
```

### 2. Set Up Environment

#### Using Conda (Recommended)

```bash
# Create environment from environment.yml
conda env create -f environment.yml

# Activate environment
conda activate mlx-sdxl-lora
```

#### Using pip

```bash
# Create virtual environment
python -m venv .venv

# Activate environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install dependencies
pip install gradio openai pillow pytest pytest-cov ruff
```

### 3. Start LM Studio

1. Launch LM Studio
2. Load a Vision-Language Model (VLM)
3. Start the local server (ensure it's running at `http://127.0.0.1:1234`)

### 4. Run the Application

```bash
python app.py
```

The application will launch in your default browser at `http://127.0.0.1:7860`

## 📖 Usage Guide

### Loading Your Workspace

1. Enter the path to your image directory in the "Workspace Directory" field
2. Click "📂 Load Workspace"
3. The gallery will display all supported images (.jpg, .jpeg, .png, .webp, .bmp)

### Single Image Captioning

1. **Browse Gallery**: Navigate through pages using Previous/Next buttons
2. **Select Image**: Enter the image index (0-based) or click on an image
3. **Generate Caption**: Click "🤖 Generate Caption" to use the VLM
4. **Edit Caption**: Refine the caption in the text area
5. **Save**: Click "💾 Save Caption" to save as a .txt file

### Batch Processing

1. Navigate to the "Batch Operations" section
2. Enter start and end indices (1-based)
3. Optionally customize the system prompt
4. Enable "Auto-save" to automatically save generated captions
5. Click "🤖 Batch Generate Captions"

### Bulk Caption Updates

Apply templates to all captions:

- **Append**: Add text to the end of existing captions
- **Prepend**: Add text to the beginning of existing captions  
- **Replace**: Replace all captions with the template

Example templates:
- `masterpiece, high quality, detailed`
- `photorealistic, 4k, professional photography`
- `anime style, vibrant colors`

### Custom Prompts

Customize the VLM's captioning behavior:

1. Expand the "⚙️ Custom Prompt" section
2. Enter your custom system prompt
3. Generate captions with your custom instructions

Default prompt focuses on: subject, style, composition, colors, lighting, mood, and artistic elements.

## 🏗️ Project Structure

```
mlx-sdxl-lora/
├── app.py                      # Main application
├── tests/                      # Unit tests
│   └── test_app.py
├── environment.yml             # Conda environment specification
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions CI/CD
├── .vscode/
│   └── settings.json          # VS Code configuration
├── README.md                   # This file
├── LICENSE                     # MIT License
└── .gitignore                 # Git ignore rules
```

## 🧪 Development

### Code Quality

This project follows PEP 8 guidelines and uses Ruff for linting and formatting.

```bash
# Check code style
ruff check .

# Format code
ruff format .
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=. --cov-report=term
```

### VS Code Setup

The repository includes VS Code settings for:
- Python interpreter configuration
- Ruff integration for formatting and linting
- Pytest integration
- Recommended extensions

## 🔧 Configuration

### LM Studio URL

To change the LM Studio endpoint, edit the `LM_STUDIO_URL` constant in `app.py`:

```python
LM_STUDIO_URL = "http://127.0.0.1:1234/v1"
```

### System Prompt

Customize the default captioning prompt by modifying `SYSTEM_PROMPT` in `app.py`:

```python
SYSTEM_PROMPT = """Your custom prompt here..."""
```

### Gallery Settings

Adjust the number of images per page by modifying `workspace_state`:

```python
workspace_state = {
    "images_per_page": 12,  # Change this value
    # ...
}
```

## 🐛 Troubleshooting

### "VLM caption generation failed"

- Ensure LM Studio is running at `http://127.0.0.1:1234`
- Verify a VLM is loaded in LM Studio
- Check LM Studio's server logs for errors

### "Invalid directory"

- Ensure the workspace path exists and contains images
- Use absolute paths for best results
- Check file permissions

### "Error reading image file"

- Verify image format is supported (.jpg, .jpeg, .png, .webp, .bmp)
- Check if the image file is corrupted
- Ensure read permissions on the directory

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with appropriate tests
4. Ensure code passes linting and tests
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Gradio](https://gradio.app/) for the UI
- Powered by [LM Studio](https://lmstudio.ai/) for local VLM inference
- Uses [OpenAI Python library](https://github.com/openai/openai-python) for API compatibility

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Troubleshooting](#-troubleshooting) section
2. Search [existing issues](https://github.com/FiditeNemini/mlx-sdxl-lora/issues)
3. Open a [new issue](https://github.com/FiditeNemini/mlx-sdxl-lora/issues/new) with details

---

**Happy Captioning! 🎉**
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Starting the Application

Run the caption tool:
```bash
python caption_tool.py
```

The application will open in your default web browser at `http://127.0.0.1:7860`.

### Workflow Options

#### 1. Single Image Captioning
- Upload one image
- Optionally customize the prompt
- Generate caption
- Edit the caption if needed
- Save to disk

#### 2. Batch Processing
- Upload multiple images at once
- Set a custom prompt (optional)
- Process all images automatically
- Review results
- Save all captions

#### 3. Caption Editor
- Load any image with its caption
- Edit and refine captions
- Save changes

#### 4. Settings & Export
- Pre-load the VLM model
- Export all captions to a specific directory
- Generate metadata JSON file

## Caption Storage

Captions are saved as `.txt` files with the same name as the corresponding image:
```
my_image.jpg  → my_image.txt
photo.png     → photo.txt
```

Additionally, a `captions_metadata.json` file is created with all caption data.

## Model Information

By default, the tool uses `mlx-community/nanoLLaVA-1.5-4bit`, a lightweight VLM optimized for MLX. The model is loaded on first use to minimize startup time.

## Human-in-the-Loop Refinement

The tool implements a human-in-the-loop approach:
1. VLM generates initial captions
2. User reviews captions in the editor
3. User refines captions as needed
4. Final captions are saved for training

This ensures high-quality, accurate captions for LoRA training.

## Tips for Best Results

- Use descriptive custom prompts for specific caption styles
- Review and edit captions in the Editor tab before training
- Process images in batches for efficiency
- Pre-load the model in Settings if processing many images

## License

MIT License - See [LICENSE](LICENSE) file for details

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Acknowledgments

- Built with [Gradio](https://gradio.app/)
- Powered by [MLX](https://github.com/ml-explore/mlx) and [MLX-VLM](https://github.com/Blaizzy/mlx-vlm)
- Designed for SDXL LoRA training workflows
