"""VLM-Powered Image Captioning Tool for LoRA Training.

This application integrates with a local Vision-Language Model (VLM) via LM Studio
to generate descriptive captions for images used in SDXL LoRA training.

Features:
- VLM integration with LM Studio
- Single/batch image processing
- Paginated gallery UI for large datasets
- In-place caption editing
- Bulk update with template support
- Data persistence with UTF-8 .txt files
"""

import base64
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
from openai import OpenAI
from PIL import Image

# Configuration
LM_STUDIO_URL = "http://127.0.0.1:1234/v1"
SYSTEM_PROMPT = """You are an expert image captioning assistant for AI art model training.
Generate detailed, comma-separated captions describing the image content.
Focus on: subject, style, composition, colors, lighting, mood, and artistic elements.
Be concise but descriptive. Output only the caption text, nothing else."""

# Supported image formats
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Global state
workspace_state: Dict = {
    "images": [],  # List of tuples: (image_path, caption_path)
    "current_page": 0,
    "images_per_page": 12,
    "workspace_dir": None,
}


def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64 string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string of the image
        
    Raises:
        FileNotFoundError: If the image file does not exist
        IOError: If the image cannot be read
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found: {image_path}")
    except Exception as e:
        raise IOError(f"Error reading image file {image_path}: {str(e)}")


def generate_caption_vlm(image_path: str, custom_prompt: Optional[str] = None) -> str:
    """Generate caption for an image using VLM via LM Studio.
    
    Args:
        image_path: Path to the image file
        custom_prompt: Optional custom system prompt to override default
        
    Returns:
        Generated caption string
        
    Raises:
        Exception: If VLM communication fails
    """
    try:
        # Initialize OpenAI client with LM Studio endpoint
        client = OpenAI(base_url=LM_STUDIO_URL, api_key="not-needed")
        
        # Encode image
        base64_image = encode_image_to_base64(image_path)
        
        # Prepare messages
        messages = [
            {
                "role": "system",
                "content": custom_prompt if custom_prompt else SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                    {"type": "text", "text": "Describe this image:"},
                ],
            },
        ]
        
        # Call VLM
        response = client.chat.completions.create(
            model="local-model", messages=messages, max_tokens=500, temperature=0.7
        )
        
        # Extract caption
        caption = response.choices[0].message.content.strip()
        return caption
        
    except Exception as e:
        raise Exception(f"VLM caption generation failed: {str(e)}")


def load_caption(image_path: str) -> str:
    """Load existing caption from .txt file or return empty string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Caption text or empty string if no caption file exists
    """
    caption_path = Path(image_path).with_suffix(".txt")
    if caption_path.exists():
        try:
            with open(caption_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            return ""
    return ""


def save_caption(image_path: str, caption: str) -> None:
    """Save caption to .txt file co-located with the image.
    
    Args:
        image_path: Path to the image file
        caption: Caption text to save
        
    Raises:
        IOError: If the caption file cannot be written
    """
    caption_path = Path(image_path).with_suffix(".txt")
    try:
        with open(caption_path, "w", encoding="utf-8") as f:
            f.write(caption)
    except Exception as e:
        raise IOError(f"Error saving caption to {caption_path}: {str(e)}")


def scan_workspace_directory(directory_path: str) -> List[Tuple[str, str]]:
    """Scan directory for image files and their caption files.
    
    Args:
        directory_path: Path to the workspace directory
        
    Returns:
        List of tuples (image_path, caption_path)
        
    Raises:
        ValueError: If directory does not exist
    """
    dir_path = Path(directory_path)
    if not dir_path.exists() or not dir_path.is_dir():
        raise ValueError(f"Invalid directory: {directory_path}")
    
    images = []
    for file_path in sorted(dir_path.rglob("*")):
        if file_path.suffix.lower() in SUPPORTED_FORMATS:
            caption_path = file_path.with_suffix(".txt")
            images.append((str(file_path), str(caption_path)))
    
    return images


def load_workspace(directory_path: str) -> Tuple[str, str]:
    """Load a workspace directory.
    
    Args:
        directory_path: Path to the workspace directory
        
    Returns:
        Tuple of (status_message, gallery_html)
    """
    try:
        images = scan_workspace_directory(directory_path)
        workspace_state["images"] = images
        workspace_state["workspace_dir"] = directory_path
        workspace_state["current_page"] = 0
        
        status = f"✅ Loaded {len(images)} images from workspace"
        gallery = render_gallery()
        return status, gallery
        
    except Exception as e:
        return f"❌ Error loading workspace: {str(e)}", ""


def render_gallery() -> str:
    """Render the current page of images as an HTML gallery.
    
    Returns:
        HTML string of the gallery
    """
    images = workspace_state["images"]
    page = workspace_state["current_page"]
    per_page = workspace_state["images_per_page"]
    
    if not images:
        return "<div style='text-align: center; padding: 40px;'>No images loaded. Please load a workspace first.</div>"
    
    start_idx = page * per_page
    end_idx = min(start_idx + per_page, len(images))
    page_images = images[start_idx:end_idx]
    
    total_pages = (len(images) + per_page - 1) // per_page
    
    # Build gallery HTML
    html = f"""
    <div style='margin-bottom: 20px; text-align: center;'>
        <strong>Page {page + 1} of {total_pages}</strong> 
        (Showing {start_idx + 1}-{end_idx} of {len(images)} images)
    </div>
    <div style='display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px;'>
    """
    
    for idx, (img_path, cap_path) in enumerate(page_images):
        global_idx = start_idx + idx
        caption = load_caption(img_path)
        has_caption = bool(caption)
        caption_indicator = "✅" if has_caption else "❌"
        
        # Truncate caption for preview
        caption_preview = caption[:50] + "..." if len(caption) > 50 else caption
        
        html += f"""
        <div style='border: 2px solid {"#4CAF50" if has_caption else "#ccc"}; 
                    border-radius: 8px; padding: 10px; background: white;'>
            <img src='file/{img_path}' style='width: 100%; height: 150px; object-fit: cover; 
                 border-radius: 4px; cursor: pointer;' 
                 onclick='document.getElementById("image_selector").value={global_idx}; 
                          document.getElementById("load_image_btn").click();'/>
            <div style='margin-top: 8px; font-size: 12px;'>
                <div style='font-weight: bold;'>{caption_indicator} Image {global_idx + 1}</div>
                <div style='color: #666; margin-top: 4px;'>{caption_preview}</div>
            </div>
        </div>
        """
    
    html += "</div>"
    return html


def navigate_page(direction: int) -> str:
    """Navigate to previous or next page.
    
    Args:
        direction: -1 for previous, 1 for next
        
    Returns:
        Updated gallery HTML
    """
    images = workspace_state["images"]
    per_page = workspace_state["images_per_page"]
    total_pages = (len(images) + per_page - 1) // per_page
    
    new_page = workspace_state["current_page"] + direction
    new_page = max(0, min(new_page, total_pages - 1))
    workspace_state["current_page"] = new_page
    
    return render_gallery()


def load_image_for_editing(image_index: int) -> Tuple[object, str, str]:
    """Load an image and its caption for editing.
    
    Args:
        image_index: Index of the image in the workspace
        
    Returns:
        Tuple of (image_object, caption_text, status_message)
    """
    try:
        images = workspace_state["images"]
        if not images or image_index < 0 or image_index >= len(images):
            return None, "", "❌ Invalid image index"
        
        img_path, _ = images[image_index]
        caption = load_caption(img_path)
        
        # Load image for display
        image = Image.open(img_path)
        
        status = f"✅ Loaded image {image_index + 1} of {len(images)}"
        return image, caption, status
        
    except Exception as e:
        return None, "", f"❌ Error loading image: {str(e)}"


def save_caption_for_image(image_index: int, caption: str) -> Tuple[str, str]:
    """Save caption for a specific image.
    
    Args:
        image_index: Index of the image in the workspace
        caption: Caption text to save
        
    Returns:
        Tuple of (status_message, updated_gallery)
    """
    try:
        images = workspace_state["images"]
        if not images or image_index < 0 or image_index >= len(images):
            return "❌ Invalid image index", render_gallery()
        
        img_path, _ = images[image_index]
        save_caption(img_path, caption)
        
        status = f"✅ Caption saved for image {image_index + 1}"
        gallery = render_gallery()
        return status, gallery
        
    except Exception as e:
        return f"❌ Error saving caption: {str(e)}", render_gallery()


def generate_caption_for_image(image_index: int, custom_prompt: str = "") -> Tuple[str, str]:
    """Generate caption for a specific image using VLM.
    
    Args:
        image_index: Index of the image in the workspace
        custom_prompt: Optional custom prompt
        
    Returns:
        Tuple of (generated_caption, status_message)
    """
    try:
        images = workspace_state["images"]
        if not images or image_index < 0 or image_index >= len(images):
            return "", "❌ Invalid image index"
        
        img_path, _ = images[image_index]
        
        # Generate caption
        prompt = custom_prompt if custom_prompt.strip() else None
        caption = generate_caption_vlm(img_path, prompt)
        
        status = f"✅ Caption generated for image {image_index + 1}"
        return caption, status
        
    except Exception as e:
        return "", f"❌ Error generating caption: {str(e)}"


def batch_generate_captions(start_idx: int, end_idx: int, 
                           custom_prompt: str = "",
                           auto_save: bool = True) -> str:
    """Generate captions for a batch of images.
    
    Args:
        start_idx: Starting image index (1-based)
        end_idx: Ending image index (1-based)
        custom_prompt: Optional custom prompt
        auto_save: Whether to auto-save generated captions
        
    Returns:
        Status message
    """
    try:
        images = workspace_state["images"]
        if not images:
            return "❌ No images loaded"
        
        # Convert to 0-based indexing
        start_idx = max(0, start_idx - 1)
        end_idx = min(len(images), end_idx)
        
        if start_idx >= end_idx:
            return "❌ Invalid range"
        
        prompt = custom_prompt if custom_prompt.strip() else None
        success_count = 0
        error_count = 0
        
        for idx in range(start_idx, end_idx):
            img_path, _ = images[idx]
            try:
                caption = generate_caption_vlm(img_path, prompt)
                if auto_save:
                    save_caption(img_path, caption)
                success_count += 1
            except Exception:
                error_count += 1
        
        return f"✅ Generated {success_count} captions, {error_count} errors"
        
    except Exception as e:
        return f"❌ Batch generation failed: {str(e)}"


def bulk_update_captions(template: str, mode: str = "append") -> Tuple[str, str]:
    """Apply bulk update to all loaded captions.
    
    Args:
        template: Template text to apply
        mode: "append", "prepend", or "replace"
        
    Returns:
        Tuple of (status_message, updated_gallery)
    """
    try:
        images = workspace_state["images"]
        if not images:
            return "❌ No images loaded", render_gallery()
        
        if not template.strip():
            return "❌ Template cannot be empty", render_gallery()
        
        updated_count = 0
        for img_path, _ in images:
            try:
                current_caption = load_caption(img_path)
                
                if mode == "append":
                    new_caption = f"{current_caption}, {template}" if current_caption else template
                elif mode == "prepend":
                    new_caption = f"{template}, {current_caption}" if current_caption else template
                else:  # replace
                    new_caption = template
                
                save_caption(img_path, new_caption)
                updated_count += 1
            except Exception:
                continue
        
        status = f"✅ Updated {updated_count} captions"
        gallery = render_gallery()
        return status, gallery
        
    except Exception as e:
        return f"❌ Bulk update failed: {str(e)}", render_gallery()


def create_ui() -> gr.Blocks:
    """Create and configure the Gradio UI.
    
    Returns:
        Configured Gradio Blocks interface
    """
    with gr.Blocks(title="VLM Image Captioning Tool", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🎨 VLM-Powered Image Captioning Tool
            
            Generate high-quality captions for your LoRA training images using a local Vision-Language Model.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                # Workspace Management
                gr.Markdown("### 📁 Workspace")
                workspace_input = gr.Textbox(
                    label="Workspace Directory",
                    placeholder="/path/to/your/images",
                    scale=3,
                )
                load_workspace_btn = gr.Button("📂 Load Workspace", variant="primary")
                workspace_status = gr.Textbox(label="Status", interactive=False)
                
                # Gallery
                gr.Markdown("### 🖼️ Image Gallery")
                gallery_html = gr.HTML(label="Gallery")
                
                with gr.Row():
                    prev_page_btn = gr.Button("⬅️ Previous Page")
                    next_page_btn = gr.Button("Next Page ➡️")
            
            with gr.Column(scale=1):
                # Image Editor
                gr.Markdown("### ✏️ Caption Editor")
                
                image_selector = gr.Number(
                    label="Image Index (0-based)",
                    value=0,
                    precision=0,
                    elem_id="image_selector",
                )
                load_image_btn = gr.Button("Load Image", elem_id="load_image_btn")
                
                current_image = gr.Image(label="Current Image", type="pil")
                caption_editor = gr.TextArea(
                    label="Caption",
                    placeholder="Edit or generate caption here...",
                    lines=5,
                )
                
                with gr.Row():
                    save_caption_btn = gr.Button("💾 Save Caption", variant="primary")
                    generate_caption_btn = gr.Button("🤖 Generate Caption")
                
                editor_status = gr.Textbox(label="Editor Status", interactive=False)
                
                # Custom Prompt
                with gr.Accordion("⚙️ Custom Prompt", open=False):
                    custom_prompt = gr.TextArea(
                        label="Custom System Prompt",
                        placeholder="Leave empty to use default prompt",
                        lines=3,
                    )
        
        with gr.Row():
            with gr.Column():
                # Batch Operations
                gr.Markdown("### 🔄 Batch Operations")
                
                with gr.Row():
                    batch_start = gr.Number(label="Start Index (1-based)", value=1, precision=0)
                    batch_end = gr.Number(label="End Index (1-based)", value=10, precision=0)
                
                batch_auto_save = gr.Checkbox(label="Auto-save generated captions", value=True)
                batch_generate_btn = gr.Button("🤖 Batch Generate Captions")
                batch_status = gr.Textbox(label="Batch Status", interactive=False)
            
            with gr.Column():
                # Bulk Update
                gr.Markdown("### 📝 Bulk Caption Update")
                
                bulk_template = gr.TextArea(
                    label="Template",
                    placeholder="e.g., 'masterpiece, high quality'",
                    lines=2,
                )
                bulk_mode = gr.Radio(
                    choices=["append", "prepend", "replace"],
                    value="append",
                    label="Update Mode",
                )
                bulk_update_btn = gr.Button("📝 Apply Bulk Update")
        
        # Event Handlers
        load_workspace_btn.click(
            fn=load_workspace,
            inputs=[workspace_input],
            outputs=[workspace_status, gallery_html],
        )
        
        prev_page_btn.click(
            fn=lambda: navigate_page(-1),
            inputs=[],
            outputs=[gallery_html],
        )
        
        next_page_btn.click(
            fn=lambda: navigate_page(1),
            inputs=[],
            outputs=[gallery_html],
        )
        
        load_image_btn.click(
            fn=load_image_for_editing,
            inputs=[image_selector],
            outputs=[current_image, caption_editor, editor_status],
        )
        
        save_caption_btn.click(
            fn=save_caption_for_image,
            inputs=[image_selector, caption_editor],
            outputs=[editor_status, gallery_html],
        )
        
        generate_caption_btn.click(
            fn=generate_caption_for_image,
            inputs=[image_selector, custom_prompt],
            outputs=[caption_editor, editor_status],
        )
        
        batch_generate_btn.click(
            fn=batch_generate_captions,
            inputs=[batch_start, batch_end, custom_prompt, batch_auto_save],
            outputs=[batch_status],
        )
        
        bulk_update_btn.click(
            fn=bulk_update_captions,
            inputs=[bulk_template, bulk_mode],
            outputs=[workspace_status, gallery_html],
        )
        
        gr.Markdown(
            """
            ---
            ### 📚 Quick Guide
            
            1. **Load Workspace**: Enter the path to your image directory and click "Load Workspace"
            2. **Browse Gallery**: Navigate through pages to see all your images
            3. **Edit Captions**: Click on an image or enter its index to load it for editing
            4. **Generate Captions**: Use the VLM to automatically generate captions
            5. **Batch Processing**: Generate captions for multiple images at once
            6. **Bulk Updates**: Apply templates to all captions (append/prepend/replace)
            
            💡 **Tip**: Make sure LM Studio is running at http://127.0.0.1:1234 with a VLM loaded!
            """
        )
    
    return demo


def main():
    """Main entry point for the application."""
    demo = create_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        allowed_paths=["/"],  # Allow access to file paths
    )


if __name__ == "__main__":
    main()
