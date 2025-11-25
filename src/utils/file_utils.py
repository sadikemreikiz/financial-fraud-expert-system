from pathlib import Path
from typing import List

# Supported image file extensions
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".pdf"}

def list_image_files(root: Path) -> List[Path]:
    """
    Recursively list all image files under a given directory.
    """
    root = Path(root)
    files = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(root.rglob(f"*{ext}"))
    return sorted(files)
