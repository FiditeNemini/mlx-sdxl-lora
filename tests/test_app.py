"""Unit tests for the VLM Image Captioning Tool."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

# Import functions from app
from app import (
    encode_image_to_base64,
    generate_caption_vlm,
    load_caption,
    save_caption,
    scan_workspace_directory,
    load_workspace,
    render_gallery,
    navigate_page,
    load_image_for_editing,
    save_caption_for_image,
    bulk_update_captions,
    workspace_state,
)


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace with test images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test images
        for i in range(5):
            img = Image.new("RGB", (100, 100), color=(i * 50, 100, 150))
            img_path = Path(tmpdir) / f"test_image_{i}.jpg"
            img.save(img_path)

            # Create corresponding caption
            if i < 3:  # Only first 3 have captions
                caption_path = img_path.with_suffix(".txt")
                caption_path.write_text(f"Test caption {i}", encoding="utf-8")

        yield tmpdir


@pytest.fixture
def reset_workspace_state():
    """Reset workspace state before each test."""
    workspace_state["images"] = []
    workspace_state["current_page"] = 0
    workspace_state["workspace_dir"] = None
    yield
    workspace_state["images"] = []
    workspace_state["current_page"] = 0
    workspace_state["workspace_dir"] = None


class TestImageEncoding:
    """Tests for image encoding functionality."""

    def test_encode_image_to_base64_success(self, temp_workspace):
        """Test successful image encoding to base64."""
        img_path = Path(temp_workspace) / "test_image_0.jpg"
        result = encode_image_to_base64(str(img_path))

        assert isinstance(result, str)
        assert len(result) > 0

    def test_encode_image_to_base64_file_not_found(self):
        """Test encoding with non-existent file."""
        with pytest.raises(FileNotFoundError):
            encode_image_to_base64("/nonexistent/path/image.jpg")


class TestCaptionOperations:
    """Tests for caption loading and saving."""

    def test_load_caption_existing(self, temp_workspace):
        """Test loading an existing caption."""
        img_path = Path(temp_workspace) / "test_image_0.jpg"
        caption = load_caption(str(img_path))

        assert caption == "Test caption 0"

    def test_load_caption_nonexistent(self, temp_workspace):
        """Test loading caption when file doesn't exist."""
        img_path = Path(temp_workspace) / "test_image_4.jpg"
        caption = load_caption(str(img_path))

        assert caption == ""

    def test_save_caption_success(self, temp_workspace):
        """Test saving a caption."""
        img_path = Path(temp_workspace) / "test_image_4.jpg"
        test_caption = "New test caption"

        save_caption(str(img_path), test_caption)

        # Verify caption was saved
        caption_path = Path(img_path).with_suffix(".txt")
        assert caption_path.exists()
        assert caption_path.read_text(encoding="utf-8") == test_caption

    def test_save_caption_overwrite(self, temp_workspace):
        """Test overwriting an existing caption."""
        img_path = Path(temp_workspace) / "test_image_0.jpg"
        new_caption = "Updated caption"

        save_caption(str(img_path), new_caption)

        # Verify caption was updated
        loaded_caption = load_caption(str(img_path))
        assert loaded_caption == new_caption


class TestWorkspaceScan:
    """Tests for workspace scanning functionality."""

    def test_scan_workspace_directory_success(self, temp_workspace):
        """Test successful workspace scanning."""
        images = scan_workspace_directory(temp_workspace)

        assert len(images) == 5
        for img_path, cap_path in images:
            assert Path(img_path).exists()
            assert Path(img_path).suffix == ".jpg"
            assert Path(cap_path).suffix == ".txt"

    def test_scan_workspace_directory_invalid(self):
        """Test scanning invalid directory."""
        with pytest.raises(ValueError):
            scan_workspace_directory("/nonexistent/directory")

    def test_scan_workspace_directory_nested(self, temp_workspace):
        """Test scanning directory with nested images."""
        # Create nested directory with images
        nested_dir = Path(temp_workspace) / "nested"
        nested_dir.mkdir()

        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        img.save(nested_dir / "nested_image.png")

        images = scan_workspace_directory(temp_workspace)

        # Should find both top-level and nested images
        assert len(images) == 6


class TestWorkspaceLoading:
    """Tests for workspace loading functionality."""

    def test_load_workspace_success(self, temp_workspace, reset_workspace_state):
        """Test successful workspace loading."""
        status, gallery = load_workspace(temp_workspace)

        assert "✅" in status
        assert "5 images" in status
        assert len(workspace_state["images"]) == 5
        assert workspace_state["workspace_dir"] == temp_workspace

    def test_load_workspace_invalid(self, reset_workspace_state):
        """Test loading invalid workspace."""
        status, gallery = load_workspace("/nonexistent/directory")

        assert "❌" in status
        assert "Error" in status


class TestGalleryRendering:
    """Tests for gallery rendering functionality."""

    def test_render_gallery_no_images(self, reset_workspace_state):
        """Test rendering gallery with no images."""
        html = render_gallery()

        assert "No images loaded" in html

    def test_render_gallery_with_images(self, temp_workspace, reset_workspace_state):
        """Test rendering gallery with images."""
        load_workspace(temp_workspace)
        html = render_gallery()

        assert "Page 1 of 1" in html
        assert "Showing 1-5 of 5 images" in html

    def test_navigate_page(self, temp_workspace, reset_workspace_state):
        """Test page navigation."""
        # Create more images to test pagination
        for i in range(15):
            img = Image.new("RGB", (100, 100), color=(i * 15, 100, 150))
            img_path = Path(temp_workspace) / f"extra_image_{i}.jpg"
            img.save(img_path)

        load_workspace(temp_workspace)
        workspace_state["images_per_page"] = 10

        # Navigate to next page
        navigate_page(1)
        assert workspace_state["current_page"] == 1

        # Navigate back
        navigate_page(-1)
        assert workspace_state["current_page"] == 0

        # Can't go before first page
        navigate_page(-1)
        assert workspace_state["current_page"] == 0


class TestImageEditing:
    """Tests for image editing functionality."""

    def test_load_image_for_editing_success(
        self, temp_workspace, reset_workspace_state
    ):
        """Test loading image for editing."""
        load_workspace(temp_workspace)

        image, caption, status = load_image_for_editing(0)

        assert image is not None
        assert caption == "Test caption 0"
        assert "✅" in status

    def test_load_image_for_editing_invalid_index(
        self, temp_workspace, reset_workspace_state
    ):
        """Test loading image with invalid index."""
        load_workspace(temp_workspace)

        image, caption, status = load_image_for_editing(100)

        assert image is None
        assert caption == ""
        assert "❌" in status

    def test_save_caption_for_image_success(
        self, temp_workspace, reset_workspace_state
    ):
        """Test saving caption for image."""
        load_workspace(temp_workspace)

        new_caption = "Updated caption via editor"
        status, gallery = save_caption_for_image(0, new_caption)

        assert "✅" in status

        # Verify caption was saved
        img_path = workspace_state["images"][0][0]
        saved_caption = load_caption(img_path)
        assert saved_caption == new_caption

    def test_save_caption_for_image_invalid_index(
        self, temp_workspace, reset_workspace_state
    ):
        """Test saving caption with invalid index."""
        load_workspace(temp_workspace)

        status, gallery = save_caption_for_image(100, "Invalid")

        assert "❌" in status


class TestBulkOperations:
    """Tests for bulk caption operations."""

    def test_bulk_update_captions_append(self, temp_workspace, reset_workspace_state):
        """Test bulk append operation."""
        load_workspace(temp_workspace)

        template = "high quality"
        status, gallery = bulk_update_captions(template, "append")

        assert "✅" in status
        assert "5 captions" in status

        # Verify first caption was appended
        img_path = workspace_state["images"][0][0]
        caption = load_caption(img_path)
        assert caption == "Test caption 0, high quality"

    def test_bulk_update_captions_prepend(self, temp_workspace, reset_workspace_state):
        """Test bulk prepend operation."""
        load_workspace(temp_workspace)

        template = "masterpiece"
        status, gallery = bulk_update_captions(template, "prepend")

        assert "✅" in status

        # Verify first caption was prepended
        img_path = workspace_state["images"][0][0]
        caption = load_caption(img_path)
        assert caption == "masterpiece, Test caption 0"

    def test_bulk_update_captions_replace(self, temp_workspace, reset_workspace_state):
        """Test bulk replace operation."""
        load_workspace(temp_workspace)

        template = "new caption for all"
        status, gallery = bulk_update_captions(template, "replace")

        assert "✅" in status

        # Verify all captions were replaced
        for img_path, _ in workspace_state["images"]:
            caption = load_caption(img_path)
            assert caption == template

    def test_bulk_update_captions_empty_template(
        self, temp_workspace, reset_workspace_state
    ):
        """Test bulk update with empty template."""
        load_workspace(temp_workspace)

        status, gallery = bulk_update_captions("", "append")

        assert "❌" in status
        assert "empty" in status.lower()


class TestVLMIntegration:
    """Tests for VLM caption generation (mocked)."""

    @patch("app.OpenAI")
    def test_generate_caption_vlm_success(self, mock_openai, temp_workspace):
        """Test successful VLM caption generation."""
        # Mock OpenAI response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "A beautiful test image"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        img_path = Path(temp_workspace) / "test_image_0.jpg"
        caption = generate_caption_vlm(str(img_path))

        assert caption == "A beautiful test image"
        mock_client.chat.completions.create.assert_called_once()

    @patch("app.OpenAI")
    def test_generate_caption_vlm_custom_prompt(self, mock_openai, temp_workspace):
        """Test VLM caption generation with custom prompt."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Custom prompted caption"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        img_path = Path(temp_workspace) / "test_image_0.jpg"
        custom_prompt = "Focus on colors only"
        caption = generate_caption_vlm(str(img_path), custom_prompt)

        assert caption == "Custom prompted caption"

        # Verify custom prompt was used
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        assert messages[0]["content"] == custom_prompt

    @patch("app.OpenAI")
    def test_generate_caption_vlm_failure(self, mock_openai, temp_workspace):
        """Test VLM caption generation failure."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("Connection failed")
        mock_openai.return_value = mock_client

        img_path = Path(temp_workspace) / "test_image_0.jpg"

        with pytest.raises(Exception) as exc_info:
            generate_caption_vlm(str(img_path))

        assert "VLM caption generation failed" in str(exc_info.value)


class TestErrorHandling:
    """Tests for error handling."""

    def test_load_caption_with_encoding_error(self, temp_workspace):
        """Test loading caption with encoding issues."""
        img_path = Path(temp_workspace) / "test_image_0.jpg"
        caption_path = img_path.with_suffix(".txt")

        # Write invalid UTF-8 (should handle gracefully)
        with open(caption_path, "wb") as f:
            f.write(b"\xff\xfe Invalid UTF-8")

        # Should return empty string on error
        caption = load_caption(str(img_path))
        assert caption == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
