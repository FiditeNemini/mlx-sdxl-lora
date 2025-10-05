"""Tests for template processing engine."""

import pytest
from template_engine import TemplateEngine


class TestTemplateEngine:
    """Test template processing functionality."""

    def test_basic_placeholder_substitution(self):
        """Test basic placeholder substitution."""
        engine = TemplateEngine()

        template = "File: {filename}, Index: {index}"
        result = engine.process_template(template, "/path/to/image.jpg", 5)

        assert result == "File: image.jpg, Index: 5"

    def test_all_placeholders(self):
        """Test all available placeholders."""
        engine = TemplateEngine()

        template = "{basename}.{extension} in {directory} - #{index} ({filename})"
        result = engine.process_template(template, "/path/to/subdir/test_image.jpg", 10)

        expected = "test_image.jpg in subdir - #10 (test_image.jpg)"
        assert result == expected

    def test_template_validation(self):
        """Test template validation."""
        engine = TemplateEngine()

        # Valid template
        valid_template = "Image {filename} at index {index}"
        result = engine.validate_template(valid_template)

        assert result['valid_placeholders'] == ['filename', 'index']
        assert result['invalid_placeholders'] == []

        # Invalid template
        invalid_template = "Image {filename} with {invalid_placeholder}"
        result = engine.validate_template(invalid_template)

        assert result['valid_placeholders'] == ['filename']
        assert result['invalid_placeholders'] == ['invalid_placeholder']

    def test_empty_template(self):
        """Test handling of empty templates."""
        engine = TemplateEngine()

        assert engine.process_template("", "/path/image.jpg", 0) == ""
        assert engine.process_template(None, "/path/image.jpg", 0) == ""

        validation = engine.validate_template("")
        assert validation['valid_placeholders'] == []
        assert validation['invalid_placeholders'] == []

    def test_template_preview(self):
        """Test template preview functionality."""
        engine = TemplateEngine()

        template = "Image {index}: {filename}"
        sample_paths = [
            "/path/image1.jpg",
            "/path/image2.png",
            "/path/image3.webp"
        ]

        previews = engine.preview_template(template, sample_paths)

        assert len(previews) == 3
        assert "Sample 1: Image 0: image1.jpg" in previews[0]
        assert "Sample 2: Image 1: image2.png" in previews[1]
        assert "Sample 3: Image 2: image3.webp" in previews[2]

    def test_special_characters_in_path(self):
        """Test handling of special characters in file paths."""
        engine = TemplateEngine()

        template = "File: {filename}, Directory: {directory}"

        # Test path with spaces and special characters
        special_path = "/path with spaces/special-chars_123/test file (copy).jpg"
        result = engine.process_template(template, special_path, 0)

        assert "test file (copy).jpg" in result
        assert "special-chars_123" in result

    def test_get_available_placeholders(self):
        """Test getting available placeholders."""
        engine = TemplateEngine()

        placeholders = engine.get_available_placeholders()

        # Check that all expected placeholders are present
        expected_keys = ['filename', 'basename', 'index', 'extension', 'directory']
        for key in expected_keys:
            assert key in placeholders
            assert isinstance(placeholders[key], str)
            assert len(placeholders[key]) > 0  # Should have description

    def test_no_placeholders_in_template(self):
        """Test template with no placeholders."""
        engine = TemplateEngine()

        template = "This is a static template with no placeholders"
        result = engine.process_template(template, "/path/image.jpg", 5)

        assert result == template

    def test_duplicate_placeholders(self):
        """Test template with duplicate placeholders."""
        engine = TemplateEngine()

        template = "{filename} and {filename} again"
        result = engine.process_template(template, "/path/test.jpg", 0)

        assert result == "test.jpg and test.jpg again"

    def test_edge_case_filenames(self):
        """Test edge cases for filename processing."""
        engine = TemplateEngine()

        # File with no extension
        result = engine.process_template("{basename}.{extension}", "/path/filename", 0)
        assert result == "filename."

        # File with multiple dots
        result = engine.process_template("{basename}.{extension}", "/path/file.name.jpg", 0)
        assert result == "file.name.jpg"

        # Hidden file
        result = engine.process_template("{filename}", "/path/.hidden", 0)
        assert result == ".hidden"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
