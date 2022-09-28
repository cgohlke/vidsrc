Video Frameserver for Numpy
===========================

Vidsrc is a Python library to read frames from video files as numpy arrays
via the DirectShow IMediaDet interface.

:Author: `Christoph Gohlke <https://www.cgohlke.com>`_
:License: BSD 3-Clause
:Version: 2022.9.28

Requirements
------------

This release has been tested with the following requirements and 
dependencies (other versions may work):

- `CPython 3.8.10, 3.9.13, 3.10.7, 3.11.0rc2 <https://www.python.org>`_
- `Numpy 1.22.4 <https://pypi.org/project/numpy/>`_
- Microsoft Visual Studio 2019 (build)
- DirectX 9.0c SDK (build)
- DirectShow BaseClasses include files (build)
- DirectShow STRMBASE.lib (build)

Revisions
---------

2022.9.28

- Update metadata.

2021.6.6

- Remove support for Python 3.6 (NEP 29).
- Fix compile error on PyPy3.

2020.1.1

- Remove support for Python 2.7 and 3.5.

Notes
-----

The DirectShow IMediaDet interface is deprecated and may be removed from
future releases of Windows
(https://docs.microsoft.com/en-us/windows/desktop/directshow/imediadet).

To fix compile
``error C2146: syntax error: missing ';' before identifier 'PVOID64'``,
change ``typedef void * POINTER_64 PVOID64;``
to ``typedef void * __ptr64 PVOID64;``
in ``winnt.h``.

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