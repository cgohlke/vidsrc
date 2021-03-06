# vidsrc/setup.py

"""Vidsrc package setuptools script."""

import sys
import re

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext


with open('vidsrc/vidsrc.cpp') as fh:
    code = fh.read()

version = re.search(r'#define _VERSION_ "(.*?)"', code).groups()[0]

readme = re.search(
    r'\*/(?:\r\n|\r|\n){2}/\* (.*)\*/(?:\r\n|\r|\n){2}#',
    code,
    re.MULTILINE | re.DOTALL,
).groups()[0]

description = readme.splitlines()[0][:-1]

readme = '\n'.join(
    [description, '=' * len(description)] + readme.splitlines()[1:]
)

license = re.search(
    r'(Copyright.*)\*/(?:\r\n|\r|\n){2}/\*', code, re.MULTILINE | re.DOTALL
).groups()[0]

license = license.replace('# ', '').replace('#', '')

if 'sdist' in sys.argv:
    with open('LICENSE', 'w') as fh:
        fh.write('BSD 3-Clause License\n\n')
        fh.write(license)
    with open('README.rst', 'w') as fh:
        fh.write(readme)


class build_ext(_build_ext):
    """Delay import numpy until build."""

    def finalize_options(self):
        _build_ext.finalize_options(self)
        __builtins__.__NUMPY_SETUP__ = False
        import numpy

        self.include_dirs.append(numpy.get_include())


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
    project_urls={
        'Bug Tracker': 'https://github.com/cgohlke/vidsrc/issues',
        'Source Code': 'https://github.com/cgohlke/vidsrc',
        # 'Documentation': 'https://',
    },
    python_requires='>=3.7',
    install_requires=['numpy>=1.15.1'],
    setup_requires=['setuptools>=18.0', 'numpy>=1.15.1'],
    cmdclass={'build_ext': build_ext},
    packages=['vidsrc'],
    ext_modules=[
        Extension(
            'vidsrc.vidsrc',
            ['vidsrc/vidsrc.cpp'],
            include_dirs=[DIRECTX_DIR, STRMBASE_DIR],
            library_dirs=[STRMBASE_DIR + '/libraries'],
            libraries=['STRMBASE', 'Ole32', 'OleAut32', 'strmiids'],
        )
    ],
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
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
