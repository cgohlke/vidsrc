# test_vidsrc.py

from __future__ import division, print_function

import sys

import numpy

import vidsrc

__doc__ = vidsrc.__doc__

print('module:', vidsrc)
print('module version:', vidsrc.__version__)
print('module doc:', vidsrc.__doc__)
print()

video = vidsrc.VideoSource("test.avi")

print('filename:', video.filename)
print('frames:', len(video))
print('framerate:', video.framerate)
print('duration:', video.duration)
print('shape:', video.shape)
print('frame averages:')
for i, frame in enumerate(video):
    print('%.2i %.2f' % (i, frame.mean()))

if __name__ == '__main__':
    import doctest
    doctest.testmod()
