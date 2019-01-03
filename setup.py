# -*- coding: utf-8 -*-
# vidsrc/setup.py

"""Vidsrc package setuptools script."""

import sys
import re

from setuptools import setup, Extension

import numpy

with open('vidsrc/vidsrc.cpp') as fh:
    code = fh.read()

version = re.search(r'#define _VERSION_ "(.*?)"', code).groups()[0]
readme = re.search(r'\*/[\r\n?|\n]{2}/\* (.*)\*/[\r\n?|\n]{2}#', code,
                   re.MULTILINE | re.DOTALL).groups()[0]
license = re.search(r'(Copyright.*)\*/[\r\n?|\n]{2}/\*', code,
                    re.MULTILINE | re.DOTALL).groups()[0]

description = readme.splitlines()[0][:-1]
readme = '\n'.join([description, '=' * len(description)]
                   + readme.splitlines()[1:])
license = license.replace('# ', '').replace('#', '')

if 'sdist' in sys.argv:
    with open('LICENSE', 'w') as fh:
        fh.write(license)
    with open('README.rst', 'w') as fh:
        fh.write(readme)
    numpy_required = '1.11.3'
else:
    numpy_required = numpy.__version__

DIRECTX_DIR = 'X:/DirectX/include'
STRMBASE_DIR = 'X:/DirectX/Samples/C++/DirectShow/BaseClasses'

setup(
    name='vidsrc',
    version=version,
    description=description,
    long_description=readme,
    author='Christoph Gohlke',
    author_email='cgohlke@uci.edu',
    url='https://www.lfd.uci.edu/~gohlke/',
    packages=['vidsrc'],
    ext_modules=[Extension(
        'vidsrc.vidsrc', ['vidsrc/vidsrc.cpp'],
        include_dirs=[numpy.get_include(), DIRECTX_DIR, STRMBASE_DIR],
        library_dirs=[STRMBASE_DIR + '/libraries'],
        libraries=['STRMBASE', 'Ole32', 'OleAut32', 'strmiids'],)],
    install_requires=['numpy>=%s' % numpy_required],
    license='BSD',
    zip_safe=False,
    platforms=['Windows'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: BSD License',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Multimedia :: Video',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: C++',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        ],
)
