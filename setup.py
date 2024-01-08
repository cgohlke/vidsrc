# vidsrc/setup.py

"""Vidsrc package Setuptools script."""

import re
import sys

import numpy
from setuptools import Extension, setup


def search(pattern, code, flags=0):
    # return first match for pattern in code
    match = re.search(pattern, code, flags)
    if match is None:
        raise ValueError(f'{pattern!r} not found')
    return match.groups()[0]


with open('vidsrc/vidsrc.cpp', encoding='utf-8') as fh:
    code = fh.read()

version = search(r'#define _VERSION_ "(.*?)"', code).replace('.x.x', '.dev0')

readme = search(
    r'(?:\r\n|\r|\n)\"(.*)\"(?:\r\n|\r|\n){2}\#define _VERSION_',
    code,
    re.MULTILINE | re.DOTALL,
).replace('\\n\\', '')

description = readme.splitlines()[0][:-1]

readme = '\n'.join(
    [description, '=' * len(description)] + readme.splitlines()[1:]
)

license = search(
    r'(Copyright.*)\*/(?:\r\n|\r|\n){2}\#define _DOC_',
    code,
    re.MULTILINE | re.DOTALL,
)

license = license.replace('# ', '').replace('#', '')

if 'sdist' in sys.argv:
    with open('LICENSE', 'w', encoding='utf-8') as fh:
        fh.write('BSD 3-Clause License\n\n')
        fh.write(license)
    with open('README.rst', 'w', encoding='utf-8') as fh:
        fh.write(readme)


DIRECTX_DIR = 'D:/Programs/DirectX9/include'
STRMBASE_DIR = 'D:/Programs/DirectX9/Samples/C++/DirectShow/BaseClasses'

setup(
    name='vidsrc',
    version=version,
    license='BSD',
    description=description,
    long_description=readme,
    long_description_content_type='text/x-rst',
    author='Christoph Gohlke',
    author_email='cgohlke@cgohlke.com',
    url='https://www.cgohlke.com',
    project_urls={
        'Bug Tracker': 'https://github.com/cgohlke/vidsrc/issues',
        'Source Code': 'https://github.com/cgohlke/vidsrc',
        # 'Documentation': 'https://',
    },
    python_requires='>=3.9',
    install_requires=['numpy'],
    packages=['vidsrc'],
    ext_modules=[
        Extension(
            'vidsrc.vidsrc',
            ['vidsrc/vidsrc.cpp'],
            include_dirs=[DIRECTX_DIR, STRMBASE_DIR, numpy.get_include()],
            library_dirs=[STRMBASE_DIR + '/libraries'],
            libraries=['STRMBASE', 'Ole32', 'OleAut32', 'strmiids'],
        )
    ],
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
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
)
