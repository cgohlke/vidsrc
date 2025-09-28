# vidsrc/__init__.pyi

"""Video Frameserver for Numpy."""

from collections.abc import Sequence
from typing import Any, overload

from numpy.typing import NDArray

__version__: str

class VideoSource(Sequence[NDArray[Any]]):
    """Access frames of video file as numpy arrays.

    Parameters:
        filename:
            Name of file to open.
        framerate:
            Video frame rate in fps.
        grayscale:
            Return frames as normalized gray scale.

    """

    filename: str
    """Name of open file."""

    duration: float
    """Video duration in s."""

    framerate: float
    """Video frame rate in fps."""

    shape: tuple[int, ...]
    """Shape of video data.

    Number of frames, frame height, frame width, color channels.
    """

    def __init__(
        self,
        filename: str,
        /,
        *,
        framerate: float = 0.0,
        grayscale: bool = False,
    ) -> None: ...
    @overload
    def __getitem__(self, key: int, /) -> NDArray[Any]: ...
    @overload
    def __getitem__(
        self, key: slice[Any, Any, Any], /
    ) -> Sequence[NDArray[Any]]: ...
    def __len__(self) -> int: ...
