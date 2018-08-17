/* vidsrc.cpp */

/*
Copyright (c) 2006-2018, Christoph Gohlke
Copyright (c) 2006-2018, The Regents of the University of California
Produced at the Laboratory for Fluorescence Dynamics
All rights reserved.

Redistribution and use in src and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of src code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
* Neither the name of the copyright holders nor the names of any
  contributors may be used to endorse or promote products derived
  from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

/* Video Frameserver for Numpy.

Read frames from video file as numpy array via DirectShow IMediaDet interface.

:Author:
  `Christoph Gohlke <https://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics. University of California, Irvine

:Version: 2018.8.15

Requirements
------------
* `CPython 2.7 or 3.5+ <http://www.python.org>`_
* `Numpy 1.14 <https://www.numpy.org>`_
* Microsoft Visual Studio  (build)
* DirectX 9.0c SDK  (build)
* DirectShow BaseClasses include files  (build)
* DirectShow STRMBASE.lib  (build)

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

*/

#define _VERSION_ "2018.8.15"

#define WIN32_LEAN_AND_MEAN
#define _WIN32_DCOM
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "Python.h"

#include <windows.h>
#include <atlbase.h>
#include <dshow.h>
#include <tchar.h>
#include <Qedit.h>
#include <mtype.h>

#include "structmember.h"
#include "math.h"
#include "float.h"
#include "numpy/arrayobject.h"

/* Vidsrc Object */

typedef struct {
    PyObject_VAR_HEAD
    PyObject *shape;       /* shape of video data */
    PyObject *filename;    /* name of video file */
    Py_ssize_t *dims;      /* numpy array dimensions */
    long stride;           /* numpy array stride along width */
    long width;            /* frame width */
    long height;           /* frame height */
    long framesize;        /* frame size */
    double framerate;      /* frames per second */
    double duration;       /* duration in seconds */
    Py_ssize_t frames;     /* number of frames */
    int grayscale;         /* convert to gray scale */
    IMediaDet *imediadet;  /* COM object */
} Vidsrc;

static void
vidsrc_dealloc(Vidsrc* self)
{
    if (self->dims)
        delete self->dims;

    if (self->imediadet) {
        self->imediadet->Release();
        self->imediadet = NULL;
    }

    Py_XDECREF(self->shape);
    Py_XDECREF(self->filename);
    Py_TYPE(self)->tp_free((PyObject*)self);

    CoUninitialize();
}

static PyObject *
vidsrc_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    Vidsrc *self = (Vidsrc *)type->tp_alloc(type, 0);

    if (self != NULL) {
        self->filename = PyUnicode_FromString("");
        if (self->filename == NULL) {
            Py_DECREF(self);
            return NULL;
        }
    }
    return (PyObject *)self;
}

static int
vidsrc_init(Vidsrc *self, PyObject *args, PyObject *kwds)
{
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
    static char *kwlist[] = {"path", "framerate", "grayscale", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|di", kwlist,
        &path, &self->framerate, &self->grayscale)) {
        return -1;
    }

    if (path) {
        tmp = self->filename;
        self->filename = PyUnicode_FromObject(path);
        Py_XDECREF(tmp);
    }

    /* init COM and connect to IMediaDet */
    CoInitializeEx(NULL, COINIT_APARTMENTTHREADED);

    HRESULT hr = CoCreateInstance(CLSID_MediaDet, NULL, CLSCTX_INPROC_SERVER,
        IID_IMediaDet, (void**)&self->imediadet);
    if (FAILED(hr)) {
        PyErr_Format(PyExc_WindowsError, "failed CoCreateInstance IMediaDet");
        return -1;
    }

    /* open video file */
    CComBSTR bstr(PyUnicode_AsUnicode(self->filename));
    Py_BEGIN_ALLOW_THREADS
        hr = self->imediadet->put_Filename(bstr);
    Py_END_ALLOW_THREADS
        if (FAILED(hr)) {
            PyErr_Format(PyExc_IOError, "failed IMediaDet::put_Filename");
            return -1;
        }

    /* find video stream */
    long mstreams;
    bool found = false;
    hr = self->imediadet->get_OutputStreams(&mstreams);
    if (FAILED(hr)) {
        PyErr_Format(PyExc_TypeError, "invalid media type (no streams)");
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
        PyErr_Format(PyExc_TypeError, "invalid media type (no video stream)");
        return -1;
    }

    /* get video properties */
    AM_MEDIA_TYPE mtype;
    hr = self->imediadet->get_StreamMediaType(&mtype);
    if (SUCCEEDED(hr)) {
        if ((mtype.formattype == FORMAT_VideoInfo) &&
            (mtype.cbFormat >= sizeof(VIDEOINFOHEADER))) {
                VIDEOINFOHEADER *vh = (VIDEOINFOHEADER*)(mtype.pbFormat);
                self->width = vh->bmiHeader.biWidth;
                self->height = vh->bmiHeader.biHeight;
                if (self->height < 0) {
                    self->height *= -1;
                }
        } else {
            hr = VFW_E_INVALIDMEDIATYPE;
        }
        FreeMediaType(mtype);
    }
    if (FAILED(hr)) {
        PyErr_Format(PyExc_TypeError, "invalid media type");
        return -1;
    }

    self->imediadet->get_StreamLength(&self->duration);

    if ((self->framerate <= 0.0) || (1.0/self->framerate > self->duration)) {
        self->imediadet->get_FrameRate(&self->framerate);
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
    PyTuple_SET_ITEM(self->shape, 0, PyLong_FromSsize_t(self->frames));
    PyTuple_SET_ITEM(self->shape, 1, PyLong_FromLong(self->height));
    PyTuple_SET_ITEM(self->shape, 2, PyLong_FromLong(self->width));
    if (!self->grayscale) {
        PyTuple_SET_ITEM(self->shape, 3, PyLong_FromLong(3));
    }

    /* read first frame to determine frame data size */
    Py_BEGIN_ALLOW_THREADS
        hr = self->imediadet->GetBitmapBits(0, &self->framesize, NULL,
                                            self->width, self->height);
    Py_END_ALLOW_THREADS

    if (FAILED(hr)) {
        PyErr_Format(PyExc_IOError, "failed to read first frame");
        return -1;
    }

    /* Windows bitmap scan lines are aligned */
    self->stride = 4;
    self->stride = ((((self->width*3) + self->stride-1) / self->stride) *
                                                                self->stride);

    /* dimensions of numpy array */
    self->dims = new Py_ssize_t[4];
    self->dims[0] = self->height;
    self->dims[1] = self->width;
    self->dims[2] = 3;
    self->dims[3] = (self->width == self->stride) ? self->stride
                                                  : self->stride + 1;
    /* sanity check */
    if (self->stride*self->height != self->framesize-sizeof(BITMAPINFOHEADER))
    {
        PyErr_Format(PyExc_ValueError,
            "frame size (%i) does not match array stride (%i))",
            self->framesize-sizeof(BITMAPINFOHEADER), self->stride);
        return -1;
    }

    return 0;
}

static Py_ssize_t
vidsrc_length(Vidsrc *self)
{
    return self->frames;
}

static PyObject *
vidsrc_iter(Vidsrc *self)
{
    return PySeqIter_New((PyObject *)self);
}

static PyObject *
vidsrc_getframe(Vidsrc* self, Py_ssize_t frame)
{
    HRESULT hr;

    if (frame < 0) {
        frame = self->frames + frame;
    } else if ((frame < 0) || (frame >= self->frames)) {
        PyErr_Format(PyExc_IndexError, "frame out of bounds: %i", frame);
        return NULL;
    }

    unsigned char* buffer = new unsigned char[self->framesize];
    if (buffer == NULL) {
        PyErr_Format(PyExc_MemoryError, "out of memory");
        return NULL;
    }

    /* release the GIL and get frame */
    Py_BEGIN_ALLOW_THREADS
    hr = self->imediadet->GetBitmapBits((double)frame/self->framerate,
                            NULL, (char *)buffer, self->width, self->height);
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
            ret = (PyArrayObject*)PyArray_New(&PyArray_Type, 2,
                self->dims, NPY_DOUBLE, NULL, NULL, 0, NULL, NULL);
            if (ret == NULL) {
                PyErr_Format(PyExc_ValueError, "failed to create Numpy array");
                return NULL;
            }
            double *pdst = (double *)PyArray_DATA(ret);
            j = self->height;
            /* release the GIL and process the frame */
            Py_BEGIN_ALLOW_THREADS
            while (j--) {
                psrc = pbuf + j*self->stride;
                i = self->width;
                while (i--) {
                    *pdst++ = (0.11 * double(*(psrc)) +
                               0.59 * double(*(psrc+1)) +
                               0.30 * double(*(psrc+2))) / 255.0;
                    psrc += 3;
                }
            }
            Py_END_ALLOW_THREADS
        } else {
            /* convert BGR to RGB color channels (bytes) */
            ret = (PyArrayObject*)PyArray_New(&PyArray_Type, 3,
                self->dims, NPY_UINT8, NULL, NULL, 0, NULL, NULL);
            if (ret == NULL) {
                PyErr_Format(PyExc_ValueError, "failed to create Numpy array");
                return NULL;
            }
            unsigned char *pdst = (unsigned char *)PyArray_DATA(ret);
            j = self->height;
            /* release the GIL and copy the frame */
            Py_BEGIN_ALLOW_THREADS
            while (j--) {
                psrc = pbuf + j*self->stride + 2;
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
        if (buffer) delete buffer;
        return PyArray_Return(ret);
    } else {
        if (buffer) delete buffer;
        PyErr_Format(PyExc_IOError, "failed to read frame %i", frame);
        return NULL;
    }
    return NULL;
}

static PyMethodDef* vidsrc_methods = NULL;

static PyMemberDef vidsrc_members[] = {
    {"filename", T_OBJECT_EX, offsetof(Vidsrc, filename), 0,
        "File name"},
    {"framerate", T_DOUBLE, offsetof(Vidsrc, framerate), 0,
        "Video frame rate in fps"},
    {"duration", T_DOUBLE, offsetof(Vidsrc, duration), 0,
        "Video duration in s"},
    {"shape", T_OBJECT_EX, offsetof(Vidsrc, shape), 0,
        "Shape of video data "
        "(number of frames, frame height, frame width, color channels)"},
    {NULL}  /* Sentinel */
};

static PySequenceMethods vidsrc_as_sequence = {
    (lenfunc)vidsrc_length,     /* sq_length */
    (binaryfunc)NULL,           /* sq_concat is handled by nb_add */
    (ssizeargfunc)NULL,
    (ssizeargfunc)vidsrc_getframe,
    (ssizessizeargfunc)NULL,    /* vidsrc_slice */
    (ssizeobjargproc)NULL,      /* sq_ass_item */
    (ssizessizeobjargproc)NULL, /* sq_ass_slice */
    (objobjproc)NULL,           /* sq_contains */
    (binaryfunc)NULL,           /* sg_inplace_concat */
    (ssizeargfunc)NULL,
};

static PyTypeObject VidsrcType = {
#if PY_MAJOR_VERSION >= 3
    PyVarObject_HEAD_INIT(0,0)
#else
    PyObject_HEAD_INIT(0)
    0,                         /* ob_size */
#endif
    "vidsrc.VideoSource",      /* tp_name */
    sizeof(Vidsrc),            /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)vidsrc_dealloc,/* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
#if PY_MAJOR_VERSION >= 3
    NULL,                      /* tp_reserved */
#else
    0,                         /* tp_compare */
#endif
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    &vidsrc_as_sequence,       /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    /* tp_flags */
    "Access the frames of a video file as numpy arrays.\n\n"
    "Instances must be used in a single thread only.\n\n",
    /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    (getiterfunc)vidsrc_iter,  /* tp_iter */
    0,                         /* tp_iternext */
    vidsrc_methods,            /* tp_methods */
    vidsrc_members,            /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)vidsrc_init,     /* tp_init */
    0,                         /* tp_alloc */
    vidsrc_new,                /* tp_new */
};

/* Vidsrc Module */

static PyMethodDef module_methods[] = {
    {NULL}  /* Sentinel */
};

const char module_doc[] =
"Video Frameserver for Numpy.\n\n"
"Read frames from video file as numpy array via DirectShow IMediaDet "
"interface.\n\n"
":Authors: `Christoph Gohlke <https://www.lfd.uci.edu/~gohlke/>`__\n\n"
":Organization:\n"
"  Laboratory for Fluorescence Dynamics. University of California, Irvine\n\n"
":Version: %s\n\n"
"Example:\n\n"
">>> from vidsrc import VideoSource\n"
">>> video = VideoSource('test.avi', grayscale=False)\n"
">>> len(video)  # number of frames in video\n"
"48\n"
">>> video.duration  # length in s\n"
"1.6016\n"
">>> video.framerate  # frames per second\n"
"29.970089850329373\n"
">>> video.shape  # frames, height, width, color channels\n"
"(48, 64, 64, 3)\n"
">>> frame = video[0]  # access first frame\n"
">>> frame = video[-1]  # access last frame\n"
">>> for frame in video:\n"
"...     pass  # do_something_with(frame)\n";

#if PY_MAJOR_VERSION >= 3

struct module_state {
    PyObject *error;
};

#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))

static int module_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int module_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "vidsrc",
        NULL,
        sizeof(struct module_state),
        module_methods,
        NULL,
        module_traverse,
        module_clear,
        NULL
};

#define INITERROR return NULL

PyMODINIT_FUNC
PyInit_vidsrc(void)

#else

#define INITERROR return

PyMODINIT_FUNC
initvidsrc(void)

#endif
{
    char *doc = (char *)PyMem_Malloc(sizeof(module_doc) + sizeof(_VERSION_));
    PyOS_snprintf(doc, sizeof(module_doc) + sizeof(_VERSION_),
                  module_doc, _VERSION_);

#if PY_MAJOR_VERSION >= 3
    moduledef.m_doc = doc;
    PyObject *module = PyModule_Create(&moduledef);
#else
    PyObject *module = Py_InitModule3("vidsrc", module_methods, doc);
#endif

    PyMem_Free(doc);
    if (module == NULL)
        INITERROR;

    if (_import_array() < 0) {
        Py_DECREF(module);
        INITERROR;
    }

    VidsrcType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&VidsrcType) < 0) {
        Py_DECREF(module);
        INITERROR;
    }

    Py_INCREF(&VidsrcType);
    PyModule_AddObject(module, "VideoSource", (PyObject *)&VidsrcType);

#if PY_MAJOR_VERSION < 3
    PyObject* s = PyString_FromString(_VERSION_);
#else
    PyObject* s = PyUnicode_FromString(_VERSION_);
#endif
    PyObject* dict = PyModule_GetDict(module);
    PyDict_SetItemString(dict, "__version__", s);
    Py_DECREF(s);

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}
