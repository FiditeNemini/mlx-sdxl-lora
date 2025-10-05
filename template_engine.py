"""Template processing engine for dynamic caption templates.

Supports placeholder substitution in bulk caption operations.
"""

import re
from pathlib import Path
from typing import Dict, List


class TemplateEngine:
    """Processes caption templates with dynamic placeholder substitution."""

    def __init__(self):
        self.placeholders = {
            "filename": self._get_filename,
            "basename": self._get_basename,
            "index": self._get_index,
            "extension": self._get_extension,
            "directory": self._get_directory,
        }

    def process_template(self, template: str, image_path: str, index: int = 0) -> str:
        """Process a template string with placeholder substitution.

        Args:
            template: Template string with placeholders like {filename}
            image_path: Path to the image file
            index: Index of the image in the dataset

        Returns:
            Processed template string with placeholders replaced

        Example:
            >>> engine = TemplateEngine()
            >>> engine.process_template("{filename} - image {index}", "/path/image.jpg", 5)
            "image.jpg - image 5"
        """
        if not template or not template.strip():
            return ""

        # Find all placeholders in the template
        placeholders = re.findall(r"\{(\w+)\}", template)
        processed_template = template

        # Create context for placeholder resolution
        context = {"image_path": image_path, "index": index}

        # Replace each placeholder
        for placeholder in placeholders:
            if placeholder in self.placeholders:
                value = self.placeholders[placeholder](context)
                processed_template = processed_template.replace(
                    f"{{{placeholder}}}", str(value)
                )

        return processed_template.strip()

    def validate_template(self, template: str) -> Dict[str, List[str]]:
        """Validate a template and return any issues.

        Args:
            template: Template string to validate

        Returns:
            Dictionary with 'valid_placeholders' and 'invalid_placeholders' lists
        """
        if not template:
            return {"valid_placeholders": [], "invalid_placeholders": []}

        # Find all placeholders
        placeholders = re.findall(r"\{(\w+)\}", template)

        valid_placeholders = []
        invalid_placeholders = []

        for placeholder in placeholders:
            if placeholder in self.placeholders:
                valid_placeholders.append(placeholder)
            else:
                invalid_placeholders.append(placeholder)

        return {
            "valid_placeholders": valid_placeholders,
            "invalid_placeholders": invalid_placeholders,
        }

    def get_available_placeholders(self) -> Dict[str, str]:
        """Get list of available placeholders with descriptions.

        Returns:
            Dictionary mapping placeholder names to descriptions
        """
        return {
            "filename": 'Full filename with extension (e.g., "image.jpg")',
            "basename": 'Filename without extension (e.g., "image")',
            "index": "Index of the image in the dataset (0-based)",
            "extension": 'File extension without dot (e.g., "jpg")',
            "directory": "Name of the parent directory",
        }

    def preview_template(self, template: str, sample_paths: List[str]) -> List[str]:
        """Preview how a template would be processed for sample paths.

        Args:
            template: Template string to preview
            sample_paths: List of sample image paths

        Returns:
            List of processed template results
        """
        previews = []
        for index, path in enumerate(sample_paths[:5]):  # Limit to 5 samples
            try:
                result = self.process_template(template, path, index)
                previews.append(f"Sample {index + 1}: {result}")
            except Exception as e:
                previews.append(f"Sample {index + 1}: Error - {str(e)}")

        return previews

    # Placeholder resolver functions
    def _get_filename(self, context: Dict) -> str:
        """Get full filename with extension."""
        return Path(context["image_path"]).name

    def _get_basename(self, context: Dict) -> str:
        """Get filename without extension."""
        return Path(context["image_path"]).stem

    def _get_index(self, context: Dict) -> int:
        """Get image index."""
        return context["index"]

    def _get_extension(self, context: Dict) -> str:
        """Get file extension without dot."""
        return Path(context["image_path"]).suffix.lstrip(".")

    def _get_directory(self, context: Dict) -> str:
        """Get parent directory name."""
        return Path(context["image_path"]).parent.name


# Global template engine instance
template_engine = TemplateEngine()
