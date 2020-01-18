Video Frameserver for Numpy
===========================

Vidsrc is a Python library to read frames from video files as numpy arrays
via the DirectShow IMediaDet interface.

:Author:
  `Christoph Gohlke <https://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics. University of California, Irvine

:License: BSD 3-Clause

:Version: 2020.1.1

Requirements
------------
* `CPython >= 3.6 <https://www.python.org>`_
* `Numpy 1.14 <https://www.numpy.org>`_
* Microsoft Visual Studio  (build)
* DirectX 9.0c SDK  (build)
* DirectShow BaseClasses include files  (build)
* DirectShow STRMBASE.lib  (build)

Revisions
---------
2020.1.1
    Remove support for Python 2.7 and 3.5.
    Update copyright.

Notes
-----
The DirectShow IMediaDet interface is deprecated and may be removed from
future releases of Windows
(https://docs.microsoft.com/en-us/windows/desktop/directshow/imediadet).

Example
-------
>>> from vidsrc import VideoSource
>>> video = VideoSource('test.avi', grayscale=False)
>>> len(video)  # number of frames in video
48
>>> video.duration  # length in s
1.6016
>>> video.framerate  # frames per second
29.970089850329373
>>> video.shape  # frames, height, width, color channels
(48, 64, 64, 3)
>>> frame = video[0]  # access first frame
>>> frame = video[-1]  # access last frame
>>> for frame in video:
...     pass  # do_something_with(frame)
