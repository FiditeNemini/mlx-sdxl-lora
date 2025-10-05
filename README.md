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