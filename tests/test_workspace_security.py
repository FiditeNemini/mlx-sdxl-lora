"""Security validation tests for workspace management."""

import os
import tempfile
from pathlib import Path

import pytest

from workspace_manager import WorkspaceManager, SecurityError


class TestWorkspaceSecurity:
    """Test security aspects of workspace management."""

    def test_workspace_isolation(self):
        """Test that workspaces are properly isolated."""
        manager = WorkspaceManager()

        # Create workspace
        workspace1 = manager.create_session_workspace()
        assert workspace1
        assert Path(workspace1).exists()

        # Create another workspace (simulating different session)
        manager2 = WorkspaceManager()
        workspace2 = manager2.create_session_workspace()

        # Workspaces should be different
        assert workspace1 != workspace2

        # Cleanup
        manager.cleanup_workspace()
        manager2.cleanup_workspace()

    def test_path_traversal_prevention(self):
        """Test prevention of path traversal attacks."""
        manager = WorkspaceManager()
        manager.create_session_workspace()

        try:
            # Create a malicious file path
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                malicious_path = f.name

            # This should work (normal file)
            result = manager.copy_files_to_workspace([malicious_path])
            assert len(result) == 1

            # These should fail (path traversal attempts)
            with pytest.raises(SecurityError):
                manager.copy_files_to_workspace(["../../../etc/passwd"])

            with pytest.raises(SecurityError):
                manager.copy_files_to_workspace(["/proc/version"])

        finally:
            # Cleanup
            if os.path.exists(malicious_path):
                os.unlink(malicious_path)
            manager.cleanup_workspace()

    def test_allowed_paths_restriction(self):
        """Test that allowed paths are restricted to workspace only."""
        manager = WorkspaceManager()

        # No workspace - no allowed paths
        assert manager.get_allowed_paths() == []

        # Create workspace
        workspace = manager.create_session_workspace()
        allowed_paths = manager.get_allowed_paths()

        # Should only allow workspace path
        assert len(allowed_paths) == 1
        assert allowed_paths[0] == workspace
        assert workspace in allowed_paths[0]

        # Cleanup
        manager.cleanup_workspace()

        # After cleanup - no allowed paths
        assert manager.get_allowed_paths() == []

    def test_filename_sanitization(self):
        """Test that filenames are properly sanitized."""
        manager = WorkspaceManager()

        # Test dangerous filename sanitization
        dangerous_names = [
            "../../etc/passwd",
            "test<script>alert('xss')</script>.jpg",
            "file with spaces & symbols!@#$.jpg",
            "",  # Empty filename
        ]

        for dangerous_name in dangerous_names:
            sanitized = manager._sanitize_filename(dangerous_name)

            # Should not contain dangerous characters
            assert ".." not in sanitized
            assert "<" not in sanitized
            assert ">" not in sanitized
            assert len(sanitized) > 0  # Should not be empty

    def test_workspace_cleanup(self):
        """Test proper cleanup of workspace resources."""
        manager = WorkspaceManager()

        # Create workspace
        workspace = manager.create_session_workspace()
        workspace_path = Path(workspace)

        # Verify workspace exists
        assert workspace_path.exists()

        # Cleanup
        manager.cleanup_workspace()

        # Verify workspace is removed
        assert not workspace_path.exists()
        assert manager.current_workspace is None
        assert manager.session_id is None

    def test_multiple_workspace_cleanup(self):
        """Test cleanup of multiple workspaces."""
        managers = []
        workspaces = []

        # Create multiple workspaces
        for _ in range(3):
            manager = WorkspaceManager()
            workspace = manager.create_session_workspace()
            managers.append(manager)
            workspaces.append(Path(workspace))

        # All should exist
        for workspace_path in workspaces:
            assert workspace_path.exists()

        # Cleanup all
        for manager in managers:
            manager.cleanup_all_workspaces()

        # All should be removed
        for workspace_path in workspaces:
            assert not workspace_path.exists()

    def test_safe_directory_copying(self):
        """Test safe copying of directory structures."""
        manager = WorkspaceManager()
        workspace = manager.create_session_workspace()

        # Create test directory structure
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create nested structure with images
            (temp_path / "subdir").mkdir()
            (temp_path / "image1.jpg").touch()
            (temp_path / "subdir" / "image2.png").touch()
            (temp_path / "not_image.txt").touch()  # Should be ignored

            # Copy directory
            result = manager.copy_directory_to_workspace(str(temp_path))

            # Should copy only image files
            assert len(result) == 2

            # Verify files exist in workspace
            workspace_path = Path(workspace)
            assert (workspace_path / "image1.jpg").exists()
            assert (workspace_path / "subdir" / "image2.png").exists()
            assert not (workspace_path / "not_image.txt").exists()

        # Cleanup
        manager.cleanup_workspace()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
