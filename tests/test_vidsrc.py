# test_vidsrc.py

import numpy
import pytest
import vidsrc


def test_module() -> None:
    """Test vidsrc module."""

    assert 'version' in vidsrc.__doc__
    assert vidsrc.__version__


def test_videosource() -> None:
    """Test VideoSource class."""
    video = vidsrc.VideoSource('test.avi')
    assert video.filename == 'test.avi'
    assert len(video) == 48
    assert pytest.approx(video.framerate, abs=1e-2) == 29.97
    assert pytest.approx(video.duration, abs=1e-2) == 1.6016
    assert video.shape == (48, 64, 64, 3)
    assert video[0].dtype == numpy.uint8
    assert video[0].shape == (64, 64, 3)
    mean = numpy.asarray([frame.mean() for frame in video])
    numpy.testing.assert_allclose(
        mean[[1, -1]], [6.871582, 18.197266], atol=1e-3
    )


def test_videosource_options() -> None:
    """Test VideoSource class with options."""
    video = vidsrc.VideoSource('test.avi', framerate=30.0, grayscale=True)
    assert video.filename == 'test.avi'
    assert len(video) == 48
    assert pytest.approx(video.framerate, abs=1e-2) == 30.0
    assert pytest.approx(video.duration, abs=1e-2) == 1.6016
    assert video.shape == (48, 64, 64)
    assert video[0].dtype == numpy.float64
    assert video[0].shape == (64, 64)
    mean = numpy.asarray([frame.mean() for frame in video])
    numpy.testing.assert_allclose(
        mean[[1, -1]], [0.026947, 0.071362], atol=1e-3
    )
