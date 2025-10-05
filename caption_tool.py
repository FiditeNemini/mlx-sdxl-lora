#!/usr/bin/env python3
"""
VLM-Powered Image Captioning Tool for LoRA Training

A Gradio-based desktop application that automates caption generation for SDXL LoRA models.
Uses a local Vision-Language Model (VLM) to generate descriptive captions from uploaded images.
Features include batch processing, interactive editing, and data persistence.
"""

import os
import gradio as gr
from pathlib import Path
from typing import List, Tuple
import json
from PIL import Image

from workspace_manager import WorkspaceManager, SecurityError
from template_engine import TemplateEngine


# Global workspace and template managers
workspace_manager = WorkspaceManager()
template_engine = TemplateEngine()


class CaptionGenerator:
    """Handles VLM-based image captioning with MLX."""

    def __init__(self):
        self.model = None
        self.processor = None
        self.model_loaded = False

    def load_model(self, model_name: str = "mlx-community/nanoLLaVA-1.5-4bit") -> str:
        """Load the VLM model lazily when first needed."""
        if self.model_loaded:
            return "Model already loaded"

        try:
            from mlx_vlm import load, generate
            from mlx_vlm.prompt_utils import apply_chat_template
            from mlx_vlm.utils import load_config

            self.model, self.processor = load(model_name)
            self.generate = generate
            self.apply_chat_template = apply_chat_template
            self.load_config = load_config
            self.model_loaded = True
            return f"Model {model_name} loaded successfully"
        except Exception as e:
            return f"Error loading model: {str(e)}"

    def generate_caption(
        self,
        image_path: str,
        prompt: str = "Describe this image in detail for training a text-to-image model.",
    ) -> str:
        """Generate a caption for a single image."""
        if not self.model_loaded:
            load_status = self.load_model()
            if "Error" in load_status:
                return load_status

        try:
            # Load and prepare the image
            image = Image.open(image_path)

            # Prepare the prompt
            config = self.load_config(self.model)
            formatted_prompt = self.apply_chat_template(
                self.processor, config, prompt, num_images=1
            )

            # Generate caption
            output = self.generate(
                self.model,
                self.processor,
                image,
                formatted_prompt,
                max_tokens=500,
                temperature=0.7,
                verbose=False,
            )

            return output.strip()
        except Exception as e:
            return f"Error generating caption: {str(e)}"


class CaptionManager:
    """Manages captions and their persistence."""

    def __init__(self):
        self.captions = {}  # {image_path: caption}
        self.image_list = []

    def add_caption(self, image_path: str, caption: str):
        """Add or update a caption for an image."""
        self.captions[image_path] = caption
        if image_path not in self.image_list:
            self.image_list.append(image_path)

    def get_caption(self, image_path: str) -> str:
        """Get caption for an image."""
        return self.captions.get(image_path, "")

    def save_captions(self, output_dir: str = None) -> str:
        """Save captions to text files next to images."""
        if not self.captions:
            return "No captions to save"

        saved_count = 0
        for image_path, caption in self.captions.items():
            try:
                # Save caption as .txt file with same name as image
                txt_path = Path(image_path).with_suffix(".txt")
                if output_dir:
                    txt_path = Path(output_dir) / txt_path.name
                    os.makedirs(output_dir, exist_ok=True)

                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(caption)
                saved_count += 1
            except Exception as e:
                print(f"Error saving caption for {image_path}: {e}")

        return f"Saved {saved_count} captions"

    def load_existing_captions(self, image_paths: List[str]):
        """Load existing captions from .txt files if they exist."""
        for image_path in image_paths:
            txt_path = Path(image_path).with_suffix(".txt")
            if txt_path.exists():
                try:
                    with open(txt_path, "r", encoding="utf-8") as f:
                        caption = f.read().strip()
                        self.add_caption(image_path, caption)
                except Exception as e:
                    print(f"Error loading caption for {image_path}: {e}")

    def export_metadata(self, output_path: str = "captions_metadata.json") -> str:
        """Export all captions to a JSON file."""
        try:
            metadata = {"captions": self.captions, "image_count": len(self.image_list)}
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            return f"Exported metadata to {output_path}"
        except Exception as e:
            return f"Error exporting metadata: {str(e)}"


# Global instances
caption_generator = CaptionGenerator()
caption_manager = CaptionManager()


def process_single_image(
    image_path: str, custom_prompt: str = None, progress=None
) -> Tuple[str, str]:
    """Process a single image and generate caption with optional progress tracking."""
    if not image_path:
        return "", "Please upload an image first"

    try:
        # Secure the file if workspace manager is available
        if workspace_manager.current_workspace:
            try:
                secure_files = workspace_manager.copy_files_to_workspace([image_path])
                if secure_files:
                    image_path = secure_files[0][0]  # Use secure path
            except SecurityError as e:
                return f"Security Error: {str(e)}", "Security validation failed"

        if progress:
            progress(0.3, desc="Generating caption...")

        prompt = (
            custom_prompt
            if custom_prompt
            else "Describe this image in detail for training a text-to-image model."
        )
        caption = caption_generator.generate_caption(image_path, prompt)

        if progress:
            progress(0.8, desc="Saving caption...")

        if not caption.startswith("Error"):
            caption_manager.add_caption(image_path, caption)
            if progress:
                progress(1.0, desc="Caption completed!")
            return caption, f"Caption generated for {Path(image_path).name}"

        return caption, "Error generating caption"

    except SecurityError as e:
        return f"Security Error: {str(e)}", "Security validation failed"
    except Exception as e:
        return f"Error: {str(e)}", "Processing failed"


def process_single_image_with_progress(
    image_path: str, custom_prompt: str = None, progress=gr.Progress()
) -> Tuple[str, str]:
    """Wrapper for single image processing with progress indicator."""
    return process_single_image(image_path, custom_prompt, progress)


def process_batch(
    image_files: List[str],
    directory_files: List[str],
    custom_prompt: str = None,
    use_template: bool = False,
    template_text: str = "",
    progress=gr.Progress(),
) -> str:
    """Process multiple images in batch with security and template support."""
    # Combine files from both sources
    all_files = []
    if image_files:
        all_files.extend(image_files)
    if directory_files:
        all_files.extend(directory_files)

    if not all_files:
        return "No images uploaded"

    try:
        # Create secure workspace if not exists
        if (
            not hasattr(workspace_manager, "current_workspace")
            or not workspace_manager.current_workspace
        ):
            workspace_manager.create_session_workspace()

        # Securely copy files to workspace
        progress(0.1, desc="Securing files in workspace...")
        try:
            secure_files = workspace_manager.copy_files_to_workspace(all_files)
            image_files_secure = [img_path for img_path, _ in secure_files]
        except SecurityError as e:
            return f"🚨 Security Error: {str(e)}"

        progress(0.2, desc="Loading existing captions...")
        # Load existing captions first
        caption_manager.load_existing_captions(image_files_secure)

        results = []
        total_files = len(image_files_secure)

        for idx, image_path in enumerate(image_files_secure):
            # Calculate progress (20% to 90% for processing)
            current_progress = 0.2 + (0.7 * (idx + 1) / total_files)
            progress(
                current_progress,
                desc=f"Processing {idx + 1}/{total_files}: {Path(image_path).name}",
            )

            # Skip if caption already exists
            if caption_manager.get_caption(image_path):
                results.append(f"✓ {Path(image_path).name}: Using existing caption")
                continue

            # Generate caption
            caption, status = process_single_image(image_path, custom_prompt)

            # Apply template if requested
            if (
                use_template
                and template_text.strip()
                and not caption.startswith("Error")
            ):
                try:
                    templated_caption = template_engine.process_template(
                        template_text, image_path, idx
                    )
                    # Combine VLM caption with template
                    caption = (
                        f"{templated_caption}, {caption}"
                        if templated_caption != template_text
                        else caption
                    )
                except Exception as e:
                    results.append(
                        f"⚠️ {Path(image_path).name}: Template error: {str(e)}"
                    )
                    continue

            if not caption.startswith("Error"):
                results.append(f"✓ {Path(image_path).name}: Caption generated")
            else:
                results.append(f"✗ {Path(image_path).name}: {caption}")

        progress(1.0, desc="Batch processing complete!")
        return "\n".join(results)

    except SecurityError as e:
        return f"🚨 Security Error: {str(e)}"
    except Exception as e:
        return f"❌ Processing Error: {str(e)}"


def update_caption(image_path: str, edited_caption: str) -> str:
    """Update a caption manually edited by the user."""
    if not image_path:
        return "No image selected"

    caption_manager.add_caption(image_path, edited_caption)
    return f"Caption updated for {Path(image_path).name}"


def preview_template(template_text: str) -> str:
    """Preview template with example data."""
    if not template_text.strip():
        return "Enter a template to see preview..."

    try:
        preview = template_engine.process_template(
            template_text, "/path/to/example_image.jpg", 0
        )
        return f"Preview: {preview}"
    except Exception as e:
        return f"Template error: {str(e)}"


def toggle_template_visibility(use_template: bool) -> Tuple[dict, dict]:
    """Toggle visibility of template-related components."""
    return (
        {"visible": use_template, "__type__": "update"},
        {"visible": use_template, "__type__": "update"},
    )


def save_all_captions(output_directory: str = None) -> str:
    """Save all captions to disk."""
    result = caption_manager.save_captions(output_directory)
    metadata_result = caption_manager.export_metadata()
    return f"{result}\n{metadata_result}"


def get_caption_for_display(image_path: str) -> str:
    """Get caption for display in the UI."""
    if not image_path:
        return ""
    return caption_manager.get_caption(image_path)


def create_ui():
    """Create the Gradio UI."""

    with gr.Blocks(
        title="VLM Image Captioning for LoRA Training", theme=gr.themes.Soft()
    ) as app:
        gr.Markdown(
            """
            # 🖼️ VLM-Powered Image Captioning Tool for LoRA Training

            Automate caption generation for SDXL LoRA models using a local Vision-Language Model.
            Upload images, generate captions, edit them interactively, and save for training.
            """
        )

        with gr.Tabs():
            # Single Image Tab
            with gr.Tab("Single Image"):
                with gr.Row():
                    with gr.Column(scale=1):
                        single_image = gr.Image(
                            type="filepath", label="Upload Image", height=400
                        )
                        single_prompt = gr.Textbox(
                            label="Custom Prompt (Optional)",
                            placeholder="Describe this image in detail for training a text-to-image model.",
                            lines=2,
                        )
                        single_generate_btn = gr.Button(
                            "Generate Caption", variant="primary"
                        )

                    with gr.Column(scale=1):
                        single_caption = gr.Textbox(
                            label="Generated Caption",
                            lines=10,
                            placeholder="Caption will appear here...",
                        )
                        single_status = gr.Textbox(label="Status", lines=1)
                        single_update_btn = gr.Button(
                            "Update Caption", variant="secondary"
                        )
                        single_save_btn = gr.Button("Save Caption", variant="secondary")

                single_generate_btn.click(
                    fn=process_single_image_with_progress,
                    inputs=[single_image, single_prompt],
                    outputs=[single_caption, single_status],
                )

                single_update_btn.click(
                    fn=update_caption,
                    inputs=[single_image, single_caption],
                    outputs=[single_status],
                )

                single_save_btn.click(
                    fn=lambda img, cap: (
                        caption_manager.save_captions() if img else "No image to save"
                    ),
                    inputs=[single_image, single_caption],
                    outputs=[single_status],
                )

            # Batch Processing Tab
            with gr.Tab("Batch Processing"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 📂 Upload Images")

                        # Multiple files upload
                        batch_images = gr.File(
                            file_count="multiple",
                            label="Upload Multiple Images",
                            file_types=["image"],
                        )

                        # Directory upload support
                        directory_upload = gr.File(
                            file_count="directory",
                            label="Or Upload Entire Directory",
                            file_types=["image"],
                        )

                        batch_prompt = gr.Textbox(
                            label="Custom Prompt (Optional)",
                            placeholder="Describe this image in detail for training a text-to-image model.",
                            lines=2,
                        )

                        gr.Markdown("### 📝 Template Options")
                        use_template = gr.Checkbox(
                            label="Use Template Placeholders", value=False
                        )
                        template_text = gr.Textbox(
                            label="Template (supports {filename}, {index})",
                            placeholder="e.g., 'Image {index}: {filename}'",
                            visible=False,
                        )
                        template_preview = gr.Textbox(
                            label="Template Preview", interactive=False, visible=False
                        )

                        batch_process_btn = gr.Button(
                            "Process Batch", variant="primary"
                        )

                    with gr.Column():
                        batch_results = gr.Textbox(
                            label="Processing Results",
                            lines=15,
                            placeholder="Results will appear here...",
                        )
                        batch_save_btn = gr.Button(
                            "Save All Captions", variant="primary"
                        )
                        batch_save_status = gr.Textbox(label="Save Status", lines=2)

                batch_process_btn.click(
                    fn=process_batch,
                    inputs=[
                        batch_images,
                        directory_upload,
                        batch_prompt,
                        use_template,
                        template_text,
                    ],
                    outputs=[batch_results],
                )

                # Template functionality event handlers
                use_template.change(
                    fn=toggle_template_visibility,
                    inputs=[use_template],
                    outputs=[template_text, template_preview],
                )

                template_text.change(
                    fn=preview_template,
                    inputs=[template_text],
                    outputs=[template_preview],
                )

                batch_save_btn.click(
                    fn=lambda: save_all_captions(),
                    inputs=[],
                    outputs=[batch_save_status],
                )

            # Editor Tab
            with gr.Tab("Caption Editor"):
                gr.Markdown("Review and edit generated captions")

                with gr.Row():
                    editor_image = gr.Image(
                        type="filepath", label="Select Image", height=400
                    )

                with gr.Row():
                    editor_caption = gr.Textbox(
                        label="Caption",
                        lines=8,
                        placeholder="Caption will load when image is selected...",
                    )

                with gr.Row():
                    editor_update_btn = gr.Button(
                        "Save Edited Caption", variant="primary"
                    )
                    editor_status = gr.Textbox(label="Status", lines=1)

                editor_image.change(
                    fn=get_caption_for_display,
                    inputs=[editor_image],
                    outputs=[editor_caption],
                )

                editor_update_btn.click(
                    fn=update_caption,
                    inputs=[editor_image, editor_caption],
                    outputs=[editor_status],
                )

            # Settings Tab
            with gr.Tab("Settings & Export"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Model Settings")
                        model_status = gr.Textbox(
                            label="Model Status",
                            value="Model will load on first use",
                            interactive=False,
                        )
                        load_model_btn = gr.Button("Pre-load Model")

                        gr.Markdown("### Export Options")
                        output_dir = gr.Textbox(
                            label="Output Directory (Optional)",
                            placeholder="Leave empty to save next to images",
                        )
                        export_btn = gr.Button("Export All Captions", variant="primary")
                        export_status = gr.Textbox(label="Export Status", lines=3)

                load_model_btn.click(
                    fn=lambda: caption_generator.load_model(),
                    inputs=[],
                    outputs=[model_status],
                )

                export_btn.click(
                    fn=save_all_captions, inputs=[output_dir], outputs=[export_status]
                )

        gr.Markdown(
            """
            ---
            ### Usage Tips:
            - **Single Image**: Upload one image, generate caption, edit if needed, and save
            - **Batch Processing**: Upload multiple images for automated caption generation
            - **Caption Editor**: Review and refine captions before saving
            - **Settings & Export**: Pre-load the model and export all captions at once

            Captions are saved as .txt files with the same name as the image.
            """
        )

    return app


def main():
    """Main entry point for the application."""
    # Initialize secure workspace
    global workspace_manager
    workspace_manager.create_session_workspace()
    allowed_paths = workspace_manager.get_allowed_paths()

    app = create_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        allowed_paths=allowed_paths,  # SECURITY FIX: Only allow session workspace
    )


if __name__ == "__main__":
    main()
