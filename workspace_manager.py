"""Secure workspace management for image captioning sessions.

This module implements session-specific workspace isolation to prevent
security vulnerabilities and ensure clean separation between user sessions.
"""

import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class WorkspaceManager:
    """Manages secure, isolated workspaces for captioning sessions."""

    def __init__(self):
        self.active_workspaces: Dict[str, str] = {}
        self.session_id: Optional[str] = None
        self.current_workspace: Optional[str] = None

    def create_session_workspace(self) -> str:
        """Create a new isolated workspace for the current session.

        Returns:
            Path to the created workspace directory
        """
        # Generate unique session ID
        self.session_id = str(uuid.uuid4())

        # Create temporary workspace directory
        workspace_root = tempfile.mkdtemp(
            prefix=f"captioning_session_{self.session_id}_"
        )
        self.current_workspace = workspace_root
        self.active_workspaces[self.session_id] = workspace_root

        return workspace_root

    def copy_files_to_workspace(self, source_files: List[str]) -> List[Tuple[str, str]]:
        """Safely copy files to the session workspace.

        Args:
            source_files: List of source file paths

        Returns:
            List of tuples (workspace_path, caption_path)

        Raises:
            ValueError: If no active workspace or invalid source files
            SecurityError: If path traversal detected
        """
        if not self.current_workspace:
            raise ValueError(
                "No active workspace. Call create_session_workspace() first."
            )

        workspace_files = []

        for source_file in source_files:
            source_path = Path(source_file)

            # Check for path traversal in the filename itself BEFORE checking existence
            if ".." in str(source_path) or "/../" in str(source_path):
                raise SecurityError(
                    f"Path traversal detected in filename: {source_file}"
                )

            # Security: Prevent path traversal (do this before file existence check)
            if not self._is_safe_path(source_path):
                raise SecurityError(f"Path traversal detected: {source_file}")

            # Validate source file exists and is readable
            if not source_path.exists() or not source_path.is_file():
                continue

            # Create safe filename in workspace
            safe_filename = self._sanitize_filename(source_path.name)
            workspace_path = Path(self.current_workspace) / safe_filename

            # Copy file to workspace
            try:
                shutil.copy2(source_path, workspace_path)
                caption_path = workspace_path.with_suffix(".txt")
                workspace_files.append((str(workspace_path), str(caption_path)))
            except Exception as e:
                print(f"Warning: Could not copy {source_file}: {e}")
                continue

        return workspace_files

    def copy_directory_to_workspace(self, source_dir: str) -> List[Tuple[str, str]]:
        """Safely copy a directory to the session workspace.

        Args:
            source_dir: Path to source directory

        Returns:
            List of tuples (workspace_path, caption_path)
        """
        if not self.current_workspace:
            raise ValueError(
                "No active workspace. Call create_session_workspace() first."
            )

        source_path = Path(source_dir)
        if not source_path.exists() or not source_path.is_dir():
            raise ValueError(f"Invalid source directory: {source_dir}")

        # Security: Prevent path traversal
        if not self._is_safe_path(source_path):
            raise SecurityError(f"Path traversal detected: {source_dir}")

        # Supported image formats
        supported_formats = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        workspace_files = []

        # Recursively find image files
        for file_path in source_path.rglob("*"):
            if file_path.suffix.lower() in supported_formats:
                # Create relative path structure in workspace
                rel_path = file_path.relative_to(source_path)
                workspace_file = Path(self.current_workspace) / rel_path

                # Create directories if needed
                workspace_file.parent.mkdir(parents=True, exist_ok=True)

                try:
                    shutil.copy2(file_path, workspace_file)
                    caption_path = workspace_file.with_suffix(".txt")
                    workspace_files.append((str(workspace_file), str(caption_path)))
                except Exception as e:
                    print(f"Warning: Could not copy {file_path}: {e}")
                    continue

        return workspace_files

    def get_allowed_paths(self) -> List[str]:
        """Get list of paths that Gradio should allow access to.

        Returns:
            List of allowed paths (only current workspace)
        """
        if not self.current_workspace:
            return []
        return [self.current_workspace]

    def cleanup_workspace(self, session_id: Optional[str] = None) -> None:
        """Clean up workspace for a specific session or current session.

        Args:
            session_id: Session ID to clean up (None for current session)
        """
        target_session = session_id or self.session_id

        if target_session and target_session in self.active_workspaces:
            workspace_path = self.active_workspaces[target_session]
            try:
                shutil.rmtree(workspace_path, ignore_errors=True)
                del self.active_workspaces[target_session]

                if target_session == self.session_id:
                    self.session_id = None
                    self.current_workspace = None
            except Exception as e:
                print(f"Warning: Could not cleanup workspace {workspace_path}: {e}")

    def cleanup_all_workspaces(self) -> None:
        """Clean up all active workspaces."""
        for session_id in list(self.active_workspaces.keys()):
            self.cleanup_workspace(session_id)

    def _is_safe_path(self, path: Path) -> bool:
        """Check if a path is safe (no path traversal).

        Args:
            path: Path to validate

        Returns:
            True if path is safe, False otherwise
        """
        try:
            # Check for obvious path traversal patterns
            path_str = str(path)
            if (
                ".." in path_str
                or path_str.startswith("/proc")
                or path_str.startswith("/sys")
            ):
                return False

            # Check if the path contains relative components
            for part in path.parts:
                if part == "..":
                    return False

            # Resolve path and ensure it doesn't escape expected boundaries
            resolved = path.resolve()

            # Additional checks for common dangerous paths
            dangerous_patterns = ["/etc/", "/root/", "/home/", "/var/log/", "/tmp/"]
            resolved_str = str(resolved)

            for pattern in dangerous_patterns:
                if pattern in resolved_str and not resolved_str.startswith(
                    "/tmp/captioning_session_"
                ):
                    return False

            return True
        except Exception:
            return False

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe use in workspace.

        Args:
            filename: Original filename

        Returns:
            Sanitized filename
        """
        import re

        # First, reject if contains path traversal patterns
        if ".." in filename or "/" in filename or "\\" in filename:
            # Replace dangerous patterns more aggressively
            safe_filename = re.sub(r"\.\.+", "_DOTS_", filename)
            safe_filename = re.sub(r"[/\\]", "_PATH_", safe_filename)
        else:
            safe_filename = filename

        # Remove or replace dangerous characters
        safe_filename = re.sub(r"[^\w\-_\.]", "_", safe_filename)

        # Remove consecutive underscores
        safe_filename = re.sub(r"_+", "_", safe_filename)

        # Ensure filename is not empty and has reasonable length
        if not safe_filename or len(safe_filename) < 1:
            safe_filename = f"file_{uuid.uuid4().hex[:8]}"
        elif len(safe_filename) > 255:
            # Truncate very long filenames
            name, ext = os.path.splitext(safe_filename)
            safe_filename = name[:250] + ext

        return safe_filename


class SecurityError(Exception):
    """Exception raised for security violations."""

    pass


# Global workspace manager instance
workspace_manager = WorkspaceManager()
