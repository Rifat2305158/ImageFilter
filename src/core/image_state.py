"""
Image State Management Module.

Manages application image state for original, current (processed), and history trace.
Guarantees independent copies to prevent shared memory reference mutations between
original and current images.

Strictly decoupled from Tkinter and convolution mathematics.
"""

from typing import Optional, List
import numpy as np


from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np


@dataclass
class HistoryEntry:
    """
    Record of a processing step stored in the history stack.

    Attributes:
        image (np.ndarray): Independent copy of image array prior to filter step.
        description (str): Human-readable operation description.
        metadata (Dict[str, Any]): Detailed filter parameters and representation details.
    """
    image: np.ndarray
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


def create_operation_metadata(
    description: str = "",
    image: Optional[np.ndarray] = None,
    filter_type: Optional[str] = None,
    kernel_size: Optional[Any] = None,
    parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Construct standardized metadata dictionary for an image processing operation.
    """
    meta: Dict[str, Any] = {
        "operation_name": description,
        "filter_type": filter_type,
        "kernel_size": kernel_size,
        "parameters": parameters or {},
        "image_mode": None,
    }

    if image is not None:
        meta["image_mode"] = "RGB" if (image.ndim == 3 and image.shape[2] == 3) else "Grayscale"

    if description:
        desc_lower = description.lower()
        if filter_type is None:
            if "blur" in desc_lower:
                meta["filter_type"] = "blur"
            elif "sharpen" in desc_lower:
                meta["filter_type"] = "sharpen"
            elif any(k in desc_lower for k in ["edge", "sobel", "prewitt", "roberts", "laplacian"]):
                meta["filter_type"] = "edge_detection"
            elif "custom" in desc_lower or "kernel" in desc_lower:
                meta["filter_type"] = "custom_kernel"
            else:
                meta["filter_type"] = "filter"

        if kernel_size is None:
            import re
            match = re.search(r'\((\d+)x(\d+)\)', description)
            if match:
                meta["kernel_size"] = (int(match.group(1)), int(match.group(2)))
            else:
                match_single = re.search(r'(\d+)x\d+', description)
                if match_single:
                    s = int(match_single.group(1))
                    meta["kernel_size"] = (s, s)

    return meta


class ImageState:
    """
    Encapsulates state for the original loaded image, the current working/processed image,
    and history tracking stack for Undo/Reset operations.
    """

    def __init__(self, image: Optional[np.ndarray] = None):
        self._original_image: Optional[np.ndarray] = None
        self._current_image: Optional[np.ndarray] = None
        self._history_stack: List[HistoryEntry] = []

        if image is not None:
            self.set_image(image)

    @property
    def original_image(self) -> Optional[np.ndarray]:
        """Return a copy of the original image to prevent external array mutation."""
        if self._original_image is None:
            return None
        return self._original_image.copy()

    @property
    def current_image(self) -> Optional[np.ndarray]:
        """Return a copy of the current image to prevent external array mutation."""
        if self._current_image is None:
            return None
        return self._current_image.copy()

    @property
    def previous_image(self) -> Optional[np.ndarray]:
        """
        Return a copy of the image state immediately before the latest filter operation.

        With chained filters this is the input image of the most recent step, which
        is not necessarily the original image. Returns None when no filter has been
        applied yet (empty history).
        """
        if not self._history_stack:
            return None
        return self._history_stack[-1].image.copy()

    @property
    def history(self) -> List[str]:
        """Return a list of human-readable operation descriptions in history."""
        return [entry.description for entry in self._history_stack]

    @property
    def history_entries(self) -> List[HistoryEntry]:
        """Return a shallow copy list of HistoryEntry objects."""
        return list(self._history_stack)

    @property
    def can_undo(self) -> bool:
        """Check if there is at least one step in history available for undo."""
        return len(self._history_stack) > 0

    @property
    def is_loaded(self) -> bool:
        """Check if an image session is currently active."""
        return self._original_image is not None

    def set_image(self, image: np.ndarray) -> None:
        """
        Initialize session with a new loaded image array.

        Sets original_image and current_image to independent float64 copies
        and clears previous processing history stack.

        Args:
            image (np.ndarray): 2D (H, W) or 3D (H, W, 3) NumPy array.

        Raises:
            TypeError: If image is not a NumPy array.
            ValueError: If array dimension is invalid or empty.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a NumPy array.")

        if image.size == 0:
            raise ValueError("Input image array cannot be empty.")

        if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] != 3):
            raise ValueError(
                f"Unsupported image shape {image.shape}. "
                "Expected 2D (H, W) or 3D (H, W, 3)."
            )

        arr = image.astype(np.float64)
        self._original_image = arr.copy()
        self._current_image = arr.copy()
        self._history_stack.clear()

    def update_current(
        self,
        new_image: np.ndarray,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update the current working image with a new processed result.
        Saves the previous current_image state onto the history stack before updating.

        Args:
            new_image (np.ndarray): 2D (H, W) or 3D (H, W, 3) NumPy array.
            description (str): Description of applied filter operation.
            metadata (Optional[dict]): Detailed metadata regarding the filter operation.

        Raises:
            TypeError: If new_image is not a NumPy array.
            ValueError: If new_image array dimension is invalid or empty.
        """
        if not isinstance(new_image, np.ndarray):
            raise TypeError("New image must be a NumPy array.")

        if new_image.size == 0:
            raise ValueError("New image array cannot be empty.")

        if new_image.ndim not in (2, 3) or (new_image.ndim == 3 and new_image.shape[2] != 3):
            raise ValueError(
                f"Unsupported image shape {new_image.shape}. "
                "Expected 2D (H, W) or 3D (H, W, 3)."
            )

        if self._current_image is not None:
            prev_snapshot = self._current_image.copy()
            meta = create_operation_metadata(
                description=description,
                image=prev_snapshot
            )
            if metadata:
                meta.update(metadata)

            entry = HistoryEntry(
                image=prev_snapshot,
                description=description,
                metadata=meta
            )
            self._history_stack.append(entry)

        self._current_image = new_image.astype(np.float64).copy()

    def undo(self) -> Optional[np.ndarray]:
        """
        Restore the previous current-image state from history and remove the latest operation.

        Returns:
            Optional[np.ndarray]: A copy of the restored current_image, or None if history is empty.
        """
        if not self._history_stack:
            return None

        entry = self._history_stack.pop()
        self._current_image = entry.image.copy()
        return self.current_image

    def reset(self) -> None:
        """
        Reset current_image back to an independent copy of original_image
        and clear processing history.
        """
        if self._original_image is not None:
            self._current_image = self._original_image.copy()
            self._history_stack.clear()

    def clear(self) -> None:
        """Clear all stored images and history."""
        self._original_image = None
        self._current_image = None
        self._history_stack.clear()

