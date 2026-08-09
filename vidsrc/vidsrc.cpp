/* vidsrc.cpp */

/*
Copyright (c) 2006-2026, Christoph Gohlke
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#define _DOC_ \
"Video Frameserver for Numpy.\n\
\n\
Vidsrc is a Python library to read frames from video files as numpy arrays\n\
via the DirectShow IMediaDet interface.\n\
\n\
:Author: `Christoph Gohlke <https://www.cgohlke.com>`_\n\
:License: BSD-3-Clause\n\
:Version: 2026.8.8\n\
\n\
Quickstart\n\
----------\n\
\n\
Install the vidsrc package and all dependencies from the\n\
`Python Package Index <https://pypi.org/project/vidsrc/>`_::\n\
\n\
    python -m pip install -U vidsrc\n\
\n\
See `Examples`_ for using the programming interface.\n\
\n\
Source code and support are available on\n\
`GitHub <https://github.com/cgohlke/vidsrc>`_.\n\
\n\
Requirements\n\
------------\n\
\n\
This revision was tested with the following requirements and \n\
dependencies (other versions may work):\n\
\n\
- `CPython <https://www.python.org>`_ 3.12.10, 3.13.15, 3.14.7, 3.15.0rc\n\
- `Numpy <https://pypi.org/project/numpy/>`_ 2.5.2\n\
- Microsoft Visual Studio 2026 (build)\n\
- DirectX 9.0c SDK (build)\n\
- DirectShow BaseClasses include files (build)\n\
- DirectShow STRMBASE.lib (build)\n\
\n\
Revisions\n\
---------\n\
\n\
2026.8.8\n\
\n\
- Make C++ extension ABI3 and free-threading compatible.\n\
- Drop support for Python 3.11 and numpy 2.0 (SPEC0).\n\
- Support Python 3.15.\n\
\n\
2026.1.18\n\
\n\
- Use multi-phase initialization.\n\
\n\
2025.8.1\n\
\n\
- Drop support for Python 3.10, support Python 3.14.\n\
\n\
2025.1.6\n\
\n\
- Add type hints.\n\
- Drop support for Python 3.9, support Python 3.13 and NumPy 2.\n\
\n\
2024.1.6\n\
\n\
- Support Python 3.12.\n\
- Drop support for Python 3.8 and NumPy 1.22 (NEP 29).\n\
\n\
2022.9.28\n\
\n\
- Update metadata.\n\
\n\
2021.6.6\n\
\n\
- Drop support for Python 3.6 (NEP 29).\n\
- Fix compile error on PyPy3.\n\
\n\
2020.1.1\n\
\n\
- Drop support for Python 2.7 and 3.5.\n\
\n\
Notes\n\
-----\n\
\n\
The DirectShow IMediaDet interface is deprecated and may be removed from\n\
future releases of Windows\n\
(https://docs.microsoft.com/en-us/windows/desktop/directshow/imediadet).\n\
\n\
To fix compile\n\
``error C2146: syntax error: missing ';' before identifier 'PVOID64'``,\n\
change ``typedef void * POINTER_64 PVOID64;``\n\
to ``typedef void * __ptr64 PVOID64;``\n\
in ``winnt.h``.\n\
\n\
Examples\n\
--------\n\
\n\
>>> from vidsrc import VideoSource\n\
>>> video = VideoSource('test.avi', grayscale=False)\n\
>>> len(video)  # number of frames in video\n\
48\n\
>>> video.duration  # length in s\n\
1.6016\n\
>>> video.framerate  # frames per second\n\
29.970089850329373\n\
>>> video.shape  # frames, height, width, color channels\n\
(48, 64, 64, 3)\n\
>>> frame = video[0]  # access first frame\n\
>>> frame = video[-1]  # access last frame\n\
>>> for frame in video:\n\
...     pass  # do_something_with(frame)\n\
"

#define _VERSION_ "2026.8.8"

#define PY_SSIZE_T_CLEAN
#define WIN32_LEAN_AND_MEAN
#define _WIN32_DCOM
#define NPY_NO_DEPRECATED_API NPY_2_1_API_VERSION

#include "Python.h"

#include <windows.h>
#include <atlbase.h>
#include <dshow.h>
#include <tchar.h>
#include <Qedit.h>
#include <mtype.h>
#include <new>

#include "math.h"
#include "float.h"
#include "numpy/arrayobject.h"

/* thread-local storage for COM initialization */
static Py_tss_t *com_tss = NULL;

/* Vidsrc object */

typedef struct {
    PyObject_HEAD PyObject *shape; /* shape of video data */
    PyObject *filename;            /* name of video file */
    Py_ssize_t *dims;              /* numpy array dimensions */
    long stride;                   /* numpy array stride along width */
    long width;                    /* frame width */
    long height;                   /* frame height */
    long framesize;                /* frame size */
    double framerate;              /* frames per second */
    double duration;               /* duration in seconds */
    Py_ssize_t frames;             /* number of frames */
    int grayscale;                 /* convert to gray scale */
    IMediaDet *imediadet;          /* COM object */
} Vidsrc;

static void
vidsrc_dealloc(
    Vidsrc *self)
{
    if (self->dims)
        PyMem_Free(self->dims);

    if (self->imediadet) {
        self->imediadet->Release();
        self->imediadet = NULL;
    }

    Py_XDECREF(self->shape);
    Py_XDECREF(self->filename);
    PyTypeObject *tp = Py_TYPE((PyObject *)self);
    freefunc tp_free = (freefunc)PyType_GetSlot(tp, Py_tp_free);
    tp_free(self);
    Py_DECREF(tp);
}

static PyObject *
vidsrc_new(
    PyTypeObject *type,
    PyObject *args,
    PyObject *kwds)
{
    Vidsrc *self = (Vidsrc *)PyType_GenericAlloc(type, 0);

    if (self != NULL) {
        self->filename = PyUnicode_FromString("");
        if (self->filename == NULL) {
            Py_DECREF(self);
            return NULL;
        }
        self->dims = NULL;
        self->imediadet = NULL;
        self->shape = NULL;
    }
    return (PyObject *)self;
}

static int
vidsrc_init(
    Vidsrc *self,
    PyObject *args,
    PyObject *kwds)
{
    /* clean up previous state in case __init__ is called again */
    if (self->imediadet) {
        self->imediadet->Release();
        self->imediadet = NULL;
    }
    if (self->dims) {
        PyMem_Free(self->dims);
        self->dims = NULL;
    }
    Py_CLEAR(self->shape);

    self->framerate = 0.0;
    self->duration = 0.0;
    self->frames = 0;
    self->width = 0;
    self->height = 0;
    self->stride = 0;
    self->grayscale = 0;
    self->framesize = 0;

    /* parse Python arguments */
    PyObject *tmp;
    PyObject *path = NULL;
    static const char *kwlist[] = {
        "filename", "framerate", "grayscale", NULL };

    if (
        !PyArg_ParseTupleAndKeywords(
        args,
        kwds,
        "O|di",
        const_cast<char **>(kwlist),
        &path,
        &self->framerate,
        &self->grayscale
    ))
    {
        return -1;
    }

    if (path) {
        tmp = self->filename;
        self->filename = PyUnicode_FromObject(path);
        Py_XDECREF(tmp);
        if (self->filename == NULL) {
            return -1;
        }
    }

    /* initialize COM for this thread */
    if (PyThread_tss_get(com_tss) == NULL) {
        HRESULT hr = CoInitializeEx(NULL, COINIT_APARTMENTTHREADED);
        if (FAILED(hr)) {
            PyErr_Format(
                PyExc_RuntimeError,
                "failed to initialize COM: 0x%lx",
                hr
            );
            return -1;
        }
        if (PyThread_tss_set(com_tss, (void *)1) != 0) {
            CoUninitialize();
            PyErr_NoMemory();
            return -1;
        }
    }

    HRESULT hr = CoCreateInstance(
        CLSID_MediaDet,
        NULL,
        CLSCTX_INPROC_SERVER,
        IID_IMediaDet,
        (void **)&self->imediadet
    );
    if (FAILED(hr)) {
        PyErr_Format(PyExc_OSError, "failed CoCreateInstance IMediaDet");
        return -1;
    }

    /* open video file */
    wchar_t *wstr = PyUnicode_AsWideCharString(self->filename, NULL);
    if (wstr == NULL) {
        return -1;
    }
    CComBSTR bstr(wstr);
    Py_BEGIN_ALLOW_THREADS
        hr = self->imediadet->put_Filename(bstr);
    Py_END_ALLOW_THREADS if (FAILED(hr)) {
        PyErr_Format(PyExc_OSError, "failed IMediaDet::put_Filename");
        PyMem_Free(wstr);
        return -1;
    }
    PyMem_Free(wstr);

    /* find video stream */
    long mstreams;
    bool found = false;
    hr = self->imediadet->get_OutputStreams(&mstreams);
    if (FAILED(hr)) {
        PyErr_Format(PyExc_ValueError, "invalid media type (no streams)");
        return -1;
    }

    GUID stype;
    for (long i = 0; i < mstreams; i++) {
        hr = self->imediadet->put_CurrentStream(i);
        if (SUCCEEDED(hr)) {
            hr = self->imediadet->get_StreamType(&stype);
        }
        if (FAILED(hr)) {
            break;
        }
        if (stype == MEDIATYPE_Video) {
            found = true;
            break;
        }
    }
    if (!found) {
        PyErr_Format(PyExc_ValueError, "invalid media type (no video stream)");
        return -1;
    }

    /* get video properties */
    AM_MEDIA_TYPE mtype;
    hr = self->imediadet->get_StreamMediaType(&mtype);
    if (SUCCEEDED(hr)) {
        if (
            (mtype.formattype == FORMAT_VideoInfo) &&
            (mtype.cbFormat >= sizeof(VIDEOINFOHEADER)))
        {
            VIDEOINFOHEADER *vh = (VIDEOINFOHEADER *)(mtype.pbFormat);
            self->width = vh->bmiHeader.biWidth;
            self->height = vh->bmiHeader.biHeight;
            if (self->height < 0) {
                self->height *= -1;
            }
        }
        else {
            hr = VFW_E_INVALIDMEDIATYPE;
        }
        FreeMediaType(mtype);
    }
    if (FAILED(hr)) {
        PyErr_Format(PyExc_ValueError, "invalid media type");
        return -1;
    }

    hr = self->imediadet->get_StreamLength(&self->duration);
    if (FAILED(hr)) {
        PyErr_Format(PyExc_OSError, "failed to get stream length");
        return -1;
    }

    if ((self->framerate <= 0.0) || (1.0 / self->framerate > self->duration)) {
        hr = self->imediadet->get_FrameRate(&self->framerate);
        if (FAILED(hr)) {
            self->framerate = 0.0;
        }
    }
    if (self->framerate <= 0.0) {
        self->framerate = 29.97;
    }

    self->frames = (Py_ssize_t)(self->duration * self->framerate);
    if (self->frames == 0) {
        self->frames = 1;
    }

    /* create shape tuple */
    self->shape = PyTuple_New(self->grayscale ? 3 : 4);
    if (self->shape == NULL) {
        PyErr_Format(PyExc_MemoryError, "failed to create shape tuple");
        return -1;
    }
    PyObject *item;
    item = PyLong_FromSsize_t(self->frames);
    if (item == NULL)
        return -1;
    PyTuple_SetItem(self->shape, 0, item);
    item = PyLong_FromLong(self->height);
    if (item == NULL)
        return -1;
    PyTuple_SetItem(self->shape, 1, item);
    item = PyLong_FromLong(self->width);
    if (item == NULL)
        return -1;
    PyTuple_SetItem(self->shape, 2, item);
    if (!self->grayscale) {
        item = PyLong_FromLong(3);
        if (item == NULL)
            return -1;
        PyTuple_SetItem(self->shape, 3, item);
    }

    /* read first frame to determine frame data size */
    Py_BEGIN_ALLOW_THREADS
        hr = self->imediadet->GetBitmapBits(
        0,
        &self->framesize,
        NULL,
        self->width,
        self->height
        );
    Py_END_ALLOW_THREADS

    if (FAILED(hr)) {
        PyErr_Format(PyExc_OSError, "failed to read first frame");
        return -1;
    }

    /* Windows bitmap scan lines are aligned */
    self->stride = 4;
    self->stride = ((((self->width * 3) + self->stride - 1) / self->stride) *
        self->stride);

    /* dimensions of numpy array */
    self->dims = PyMem_New(Py_ssize_t, 4);
    if (self->dims == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    self->dims[0] = self->height;
    self->dims[1] = self->width;
    self->dims[2] = 3;
    self->dims[3] = (self->width == self->stride) ? self->stride
                                                  : self->stride + 1;
    /* sanity check */
    if (self->stride * self->height !=
        self->framesize - sizeof(BITMAPINFOHEADER))
    {
        PyMem_Free(self->dims);
        self->dims = NULL;
        PyErr_Format(
            PyExc_ValueError,
            "frame size (%ld) does not match array stride (%ld)",
            self->framesize - (long)sizeof(BITMAPINFOHEADER),
            self->stride
        );
        return -1;
    }

    return 0;
}

static Py_ssize_t
vidsrc_length(
    Vidsrc *self)
{
    return self->frames;
}

static PyObject *
vidsrc_iter(
    Vidsrc *self)
{
    return PySeqIter_New((PyObject *)self);
}

static PyObject *
vidsrc_getframe(
    Vidsrc *self,
    Py_ssize_t frame)
{
    HRESULT hr;

    if (frame < 0) {
        frame = self->frames + frame;
    }
    if ((frame < 0) || (frame >= self->frames)) {
        PyErr_Format(PyExc_IndexError, "frame out of bounds: %zd", frame);
        return NULL;
    }

    unsigned char *buffer = new (std::nothrow) unsigned char[self->framesize];
    if (buffer == NULL) {
        PyErr_Format(PyExc_MemoryError, "out of memory");
        return NULL;
    }

    /* release the GIL and get frame */
    Py_BEGIN_ALLOW_THREADS
        hr = self->imediadet->GetBitmapBits(
        (double)frame / self->framerate,
        NULL,
        (char *)buffer,
        self->width,
        self->height
        );
    Py_END_ALLOW_THREADS

    if (SUCCEEDED(hr)) {
        /* create numpy array and copy memory from buffer */
        /* IMediaDet returns BGR-24 images with no extra color information */
        PyArrayObject *ret;
        unsigned char *psrc;
        unsigned char *pbuf = buffer + sizeof(BITMAPINFOHEADER);
        long i, j;

        if (self->grayscale) {
            /* convert BGR color channels (bytes) to gray scale (double) */
            ret = (PyArrayObject *)PyArray_New(
                &PyArray_Type,
                2,
                self->dims,
                NPY_DOUBLE,
                NULL,
                NULL,
                0,
                NULL,
                NULL
            );
            if (ret == NULL) {
                delete[] buffer;
                return NULL;
            }
            double *pdst = (double *)PyArray_DATA(ret);
            j = self->height;
            /* release the GIL and process the frame */
            Py_BEGIN_ALLOW_THREADS while (j--) {
                psrc = pbuf + j * self->stride;
                i = self->width;
                while (i--) {
                    *pdst++ = (0.114 * double(*(psrc)) +
                        0.587 * double(*(psrc + 1)) +
                        0.299 * double(*(psrc + 2))) /
                        255.0;
                    psrc += 3;
                }
            }
            Py_END_ALLOW_THREADS
        }
        else {
            /* convert BGR to RGB color channels (bytes) */
            ret = (PyArrayObject *)PyArray_New(
                &PyArray_Type,
                3,
                self->dims,
                NPY_UINT8,
                NULL,
                NULL,
                0,
                NULL,
                NULL
            );
            if (ret == NULL) {
                delete[] buffer;
                return NULL;
            }
            unsigned char *pdst = (unsigned char *)PyArray_DATA(ret);
            j = self->height;
            /* release the GIL and copy the frame */
            Py_BEGIN_ALLOW_THREADS while (j--) {
                psrc = pbuf + j * self->stride + 2;
                i = self->width;
                while (i--) {
                    *pdst++ = *psrc--; /* Red */
                    *pdst++ = *psrc--; /* Green */
                    *pdst++ = *psrc--; /* Blue */
                    psrc += 6;
                }
            }
            Py_END_ALLOW_THREADS
        }
        delete[] buffer;
        return PyArray_Return(ret);
    }
    else {
        delete[] buffer;
        PyErr_Format(PyExc_OSError, "failed to read frame %zd", frame);
        return NULL;
    }
}

static PyMethodDef *vidsrc_methods = NULL;

static PyMemberDef vidsrc_members[] = {
    { "filename", Py_T_OBJECT_EX, offsetof(Vidsrc, filename), Py_READONLY,
      "File name" },
    { "framerate", Py_T_DOUBLE, offsetof(Vidsrc, framerate), Py_READONLY,
      "Video frame rate in fps" },
    { "duration", Py_T_DOUBLE, offsetof(Vidsrc, duration), Py_READONLY,
      "Video duration in s" },
    { "shape", Py_T_OBJECT_EX, offsetof(Vidsrc, shape), Py_READONLY,
      "Shape of video data "
      "(number of frames, frame height, frame width, color channels)" },
    { NULL } /* Sentinel */
};

static PyType_Slot vidsrc_slots[] = {
    { Py_tp_dealloc, (void *)vidsrc_dealloc },
    { Py_tp_iter, (void *)vidsrc_iter },
    { Py_tp_methods, vidsrc_methods },
    { Py_tp_members, vidsrc_members },
    { Py_tp_init, (void *)vidsrc_init },
    { Py_tp_new, (void *)vidsrc_new },
    { Py_tp_doc, (void *)
      "Access the frames of a video file as numpy arrays.\n\n"
      "Instances must be used in a single thread only.\n\n" },
    { Py_sq_length, (void *)vidsrc_length },
    { Py_sq_item, (void *)vidsrc_getframe },
    { 0, NULL } };

static PyType_Spec vidsrc_spec = {
    "vidsrc.VideoSource",
    sizeof(Vidsrc),
    0,
    Py_TPFLAGS_DEFAULT,
    vidsrc_slots,
};

/* Vidsrc Module */

static PyMethodDef module_methods[] = {
    { NULL } /* Sentinel */
};

static int module_traverse(
    PyObject *m,
    visitproc visit,
    void *arg)
{
    (void)m;
    (void)visit;
    (void)arg;
    return 0;
}

static int module_clear(
    PyObject *m)
{
    if (com_tss != NULL && PyThread_tss_is_created(com_tss)) {
        if (PyThread_tss_get(com_tss) != NULL) {
            CoUninitialize();
        }
        PyThread_tss_delete(com_tss);
        PyThread_tss_free(com_tss);
        com_tss = NULL;
    }

    return 0;
}

static int module_exec(
    PyObject *module)
{
    if (_import_array() < 0) {
        return -1;
    }

    /* create thread-local storage slot for COM initialization */
    com_tss = PyThread_tss_alloc();
    if (com_tss == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    if (PyThread_tss_create(com_tss) != 0) {
        PyThread_tss_free(com_tss);
        com_tss = NULL;
        PyErr_NoMemory();
        return -1;
    }

    PyObject *type = PyType_FromSpec(&vidsrc_spec);
    if (type == NULL) {
        return -1;
    }

    if (PyModule_AddObjectRef(module, "VideoSource", type) < 0) {
        Py_DECREF(type);
        return -1;
    }
    Py_DECREF(type);

    if (PyModule_AddStringConstant(module, "__version__", _VERSION_) < 0) {
        return -1;
    }
    return 0;
}

static struct PyModuleDef_Slot module_slots[] = {
    { Py_mod_exec, module_exec },
    { Py_mod_multiple_interpreters, Py_MOD_PER_INTERPRETER_GIL_SUPPORTED },
#ifdef Py_MOD_GIL_NOT_USED
    { Py_mod_gil, Py_MOD_GIL_NOT_USED },
#endif
    { 0, NULL }
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "vidsrc",
    _DOC_,
    0,
    module_methods,
    module_slots,
    module_traverse,
    module_clear,
    NULL
};

PyMODINIT_FUNC
PyInit_vidsrc(void)
{
    return PyModuleDef_Init(&moduledef);
}
