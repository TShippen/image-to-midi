"""File management utilities for the image-to-MIDI pipeline.

This module provides a centralized file manager that handles temporary file
creation, tracking, and cleanup for the Gradio interface. It ensures that
files are properly managed and cleaned up to prevent accumulation.
"""

import os
import tempfile
from pathlib import Path


class GradioFileManager:
    """Manages temporary files for a Gradio session.
    
    Implements a single-active-file pattern where only one MIDI and one WAV
    file exist at any time. Files are overwritten on parameter changes to
    prevent accumulation. Works with Gradio's built-in cache management.
    
    Attributes:
        session_dir: Path to the session's temporary directory.
        current_files: Dictionary tracking current active files by type.
    """
    
    def __init__(self, session_id: str | None = None):
        """Initialize the file manager with an optional session ID.
        
        Args:
            session_id: Optional unique session identifier. If None, uses
                       Gradio's temp directory or system temp.
        """
        # Use Gradio's temp directory if available, otherwise system temp
        base_dir = os.environ.get('GRADIO_TEMP_DIR', tempfile.gettempdir())
        
        if session_id:
            self.session_dir = Path(base_dir) / f"image-to-midi-{session_id}"
        else:
            # Create a unique directory for this session
            self.session_dir = Path(tempfile.mkdtemp(prefix="image-to-midi-", dir=base_dir))
        
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.current_files: dict[str, Path] = {}
    
    def get_temp_path(self, file_type: str, extension: str = "") -> str:
        """Get the path for a temporary file of the specified type.
        
        Implements the single-active-file pattern: returns the same path
        for each file type, causing overwrites instead of accumulation.
        
        Args:
            file_type: Type of file (e.g., "midi", "wav", "plot").
            extension: File extension including dot (e.g., ".mid", ".wav").
        
        Returns:
            Absolute path to the temporary file as a string.
        """
        filename = f"current_{file_type}{extension}"
        file_path = self.session_dir / filename
        
        # Track this file for cleanup
        self.current_files[file_type] = file_path
        
        return str(file_path)
    
    def write_file(self, file_type: str, content: bytes, extension: str = "") -> str:
        """Write content to a temporary file, overwriting if it exists.
        
        Args:
            file_type: Type of file (e.g., "midi", "wav").
            content: Binary content to write to the file.
            extension: File extension including dot.
        
        Returns:
            Path to the written file as a string.
        """
        file_path = self.get_temp_path(file_type, extension)
        
        # Write atomically to prevent partial reads
        temp_path = file_path + ".tmp"
        with open(temp_path, 'wb') as f:
            f.write(content)
        
        # Atomic rename (overwrites existing file if present)
        os.replace(temp_path, file_path)
        
        return file_path
    
    def cleanup_file(self, file_type: str) -> None:
        """Remove a specific file type if it exists.
        
        Args:
            file_type: Type of file to remove.
        """
        if file_type in self.current_files:
            file_path = self.current_files[file_type]
            if file_path.exists():
                try:
                    file_path.unlink()
                except OSError:
                    pass  # File might be in use
            del self.current_files[file_type]
    
    def cleanup_all(self) -> None:
        """Remove all tracked files and the session directory.
        
        Called when the session ends or on explicit cleanup request.
        Safe to call multiple times.
        """
        # Remove all tracked files
        for file_path in list(self.current_files.values()):
            if file_path.exists():
                try:
                    file_path.unlink()
                except OSError:
                    pass  # File might be in use
        
        self.current_files.clear()
        
        # Try to remove the session directory
        if self.session_dir.exists():
            try:
                # Remove directory if empty
                self.session_dir.rmdir()
            except OSError:
                pass  # Directory might not be empty or in use
    
    def __del__(self):
        """Cleanup when the file manager is garbage collected."""
        self.cleanup_all()