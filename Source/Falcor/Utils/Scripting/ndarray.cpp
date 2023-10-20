/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/

// This is a backport of nanobind's ndarray class to pybind11.
// See https://github.com/wjakob/nanobind/blob/master/docs/ndarray.rst

#include "ndarray.h"
#include <atomic>

namespace pybind11
{
namespace detail
{

/// Python object representing a `nb_ndarray` (which wraps a DLPack tensor)
struct nb_ndarray
{
    PyObject_HEAD;
    ndarray_handle* th;
};

template<typename T>
struct scoped_pymalloc
{
    scoped_pymalloc(size_t size = 1)
    {
        ptr = (T*)PyMem_Malloc(size * sizeof(T));
        if (!ptr)
            pybind11_fail("scoped_pymalloc(): could not allocate " + std::to_string(size) + " bytes of memory!");
    }
    ~scoped_pymalloc() { PyMem_Free(ptr); }
    T* release()
    {
        T* temp = ptr;
        ptr = nullptr;
        return temp;
    }
    T* get() const { return ptr; }
    T& operator[](size_t i) { return ptr[i]; }
    T* operator->() { return ptr; }

private:
    T* ptr{nullptr};
};

void nb_ndarray_dealloc(PyObject* self);
int nb_ndarray_getbuffer(PyObject* exporter, Py_buffer* view, int);
void nb_ndarray_releasebuffer(PyObject*, Py_buffer* view);

static PyType_Slot nb_ndarray_slots[] = {
    {Py_tp_dealloc, (void*)nb_ndarray_dealloc},
#if PY_VERSION_HEX >= 0x03090000
    {Py_bf_getbuffer, (void*)nb_ndarray_getbuffer},
    {Py_bf_releasebuffer, (void*)nb_ndarray_releasebuffer},
#endif
    {0, nullptr}};

static PyType_Spec nb_ndarray_spec = {
    /* .name = */ "pybind11.nb_ndarray",
    /* .basicsize = */ (int)sizeof(nb_ndarray),
    /* .itemsize = */ 0,
    /* .flags = */ Py_TPFLAGS_DEFAULT,
    /* .slots = */ nb_ndarray_slots};

struct nb_internals
{
    PyTypeObject* nb_ndarray;
    static nb_internals& get()
    {
        // BACKPORT
        // static nb_internals internals{(PyTypeObject*)PyType_FromSpec(&nb_ndarray_spec)};
        static nb_internals internals = []()
        {
            nb_internals internals_{(PyTypeObject*)PyType_FromSpec(&nb_ndarray_spec)};
#if PY_VERSION_HEX < 0x03090000
            // Python < 3.9 does not support buffer protocol in stable API.
            internals_.nb_ndarray->tp_as_buffer->bf_getbuffer = nb_ndarray_getbuffer;
            internals_.nb_ndarray->tp_as_buffer->bf_releasebuffer = nb_ndarray_releasebuffer;
#endif
            return internals_;
        }();
        return internals;
    }
};

// ========================================================================

struct managed_dltensor
{
    dlpack::dltensor dltensor;
    void* manager_ctx;
    void (*deleter)(managed_dltensor*);
};

struct ndarray_handle
{
    managed_dltensor* ndarray;
    std::atomic<size_t> refcount;
    PyObject* owner;
    bool free_shape;
    bool free_strides;
    bool call_deleter;
};

void nb_ndarray_dealloc(PyObject* self)
{
    ndarray_dec_ref(((nb_ndarray*)self)->th);

    freefunc tp_free;
#if defined(Py_LIMITED_API)
    tp_free = (freefunc)PyType_GetSlot(Py_TYPE(self), Py_tp_free);
#else
    tp_free = Py_TYPE(self)->tp_free;
#endif

    tp_free(self);
}

int nb_ndarray_getbuffer(PyObject* exporter, Py_buffer* view, int)
{
    nb_ndarray* self = (nb_ndarray*)exporter;

    dlpack::dltensor& t = self->th->ndarray->dltensor;

    if (t.device.device_type != device::cpu::value)
    {
        PyErr_SetString(
            PyExc_BufferError,
            "Only CPU-allocated ndarrays can be "
            "accessed via the buffer protocol!"
        );
        return -1;
    }

    const char* format = nullptr;
    switch ((dlpack::dtype_code)t.dtype.code)
    {
    case dlpack::dtype_code::Int:
        switch (t.dtype.bits)
        {
        case 8:
            format = "b";
            break;
        case 16:
            format = "h";
            break;
        case 32:
            format = "i";
            break;
        case 64:
            format = "q";
            break;
        }
        break;

    case dlpack::dtype_code::UInt:
        switch (t.dtype.bits)
        {
        case 8:
            format = "B";
            break;
        case 16:
            format = "H";
            break;
        case 32:
            format = "I";
            break;
        case 64:
            format = "Q";
            break;
        }
        break;

    case dlpack::dtype_code::Float:
        switch (t.dtype.bits)
        {
        case 16:
            format = "e";
            break;
        case 32:
            format = "f";
            break;
        case 64:
            format = "d";
            break;
        }
        break;

    default:
        break;
    }

    if (!format || t.dtype.lanes != 1)
    {
        PyErr_SetString(PyExc_BufferError, "Don't know how to convert DLPack dtype into buffer protocol format!");
        return -1;
    }

    view->format = (char*)format;
    view->itemsize = t.dtype.bits / 8;
    view->buf = (void*)((uintptr_t)t.data + t.byte_offset);
    view->obj = exporter;
    Py_INCREF(exporter);

    Py_ssize_t len = view->itemsize;
    scoped_pymalloc<Py_ssize_t> strides(t.ndim), shape(t.ndim);
    for (int32_t i = 0; i < t.ndim; ++i)
    {
        len *= (Py_ssize_t)t.shape[i];
        strides[i] = (Py_ssize_t)t.strides[i] * view->itemsize;
        shape[i] = (Py_ssize_t)t.shape[i];
    }

    view->ndim = t.ndim;
    view->len = len;
    view->readonly = false;
    view->suboffsets = nullptr;
    view->internal = nullptr;
    view->strides = strides.release();
    view->shape = shape.release();

    return 0;
}

void nb_ndarray_releasebuffer(PyObject*, Py_buffer* view)
{
    PyMem_Free(view->shape);
    PyMem_Free(view->strides);
}

static PyObject* dlpack_from_buffer_protocol(PyObject* o)
{
    scoped_pymalloc<Py_buffer> view;
    scoped_pymalloc<managed_dltensor> mt;

    if (PyObject_GetBuffer(o, view.get(), PyBUF_RECORDS))
    {
        PyErr_Clear();
        return nullptr;
    }

    char format = 'B';
    const char* format_str = view->format;
    if (format_str)
        format = *format_str;

    bool skip_first = format == '@' || format == '=';

    int32_t num = 1;
    if (*(uint8_t*)&num == 1)
    {
        if (format == '<')
            skip_first = true;
    }
    else
    {
        if (format == '!' || format == '>')
            skip_first = true;
    }

    if (skip_first && format_str)
        format = *++format_str;

    dlpack::dtype dt{};
    bool fail = format_str && format_str[1] != '\0';

    if (!fail)
    {
        switch (format)
        {
        case 'c':
        case 'b':
        case 'h':
        case 'i':
        case 'l':
        case 'q':
        case 'n':
            dt.code = (uint8_t)dlpack::dtype_code::Int;
            break;

        case 'B':
        case 'H':
        case 'I':
        case 'L':
        case 'Q':
        case 'N':
            dt.code = (uint8_t)dlpack::dtype_code::UInt;
            break;

        case 'e':
        case 'f':
        case 'd':
            dt.code = (uint8_t)dlpack::dtype_code::Float;
            break;

        default:
            fail = true;
        }

        dt.lanes = 1;
        dt.bits = (uint8_t)(view->itemsize * 8);
    }

    if (fail)
    {
        PyBuffer_Release(view.get());
        return nullptr;
    }

    mt->deleter = [](managed_dltensor* mt2)
    {
        gil_scoped_acquire guard;
        Py_buffer* buf = (Py_buffer*)mt2->manager_ctx;
        PyBuffer_Release(buf);
        PyMem_Free(mt2->dltensor.shape);
        PyMem_Free(mt2->dltensor.strides);
        PyMem_Free(mt2);
    };

    /* DLPack mandates 256-byte alignment of the 'DLTensor::data' field, but
       PyTorch unfortunately ignores the 'byte_offset' value.. :-( */
#if 0
    uintptr_t value_int = (uintptr_t) view->buf,
              value_rounded = (value_int / 256) * 256;
#else
    uintptr_t value_int = (uintptr_t)view->buf, value_rounded = value_int;
#endif

    mt->dltensor.data = (void*)value_rounded;
    mt->dltensor.device = {device::cpu::value, 0};
    mt->dltensor.ndim = view->ndim;
    mt->dltensor.dtype = dt;
    mt->dltensor.byte_offset = value_int - value_rounded;

    scoped_pymalloc<int64_t> strides(view->ndim);
    scoped_pymalloc<int64_t> shape(view->ndim);
    for (size_t i = 0; i < (size_t)view->ndim; ++i)
    {
        strides[i] = (int64_t)(view->strides[i] / view->itemsize);
        shape[i] = (int64_t)view->shape[i];
    }

    mt->manager_ctx = view.release();
    mt->dltensor.shape = shape.release();
    mt->dltensor.strides = strides.release();

    return PyCapsule_New(
        mt.release(),
        "dltensor",
        [](PyObject* o)
        {
            error_scope scope; // temporarily save any existing errors
            managed_dltensor* mt = (managed_dltensor*)PyCapsule_GetPointer(o, "dltensor");
            if (mt)
            {
                if (mt->deleter)
                    mt->deleter(mt);
            }
            else
            {
                PyErr_Clear();
            }
        }
    );
}

ndarray_handle* ndarray_import(PyObject* o, const ndarray_req* req, bool convert) noexcept
{
    object capsule;

    // If this is not a capsule, try calling o.__dlpack__()
    if (!PyCapsule_CheckExact(o))
    {
        // BACKPORT
        // capsule = steal(PyObject_CallMethod(o, "__dlpack__", nullptr));
        capsule = reinterpret_steal<object>(PyObject_CallMethod(o, "__dlpack__", nullptr));

        // BACKPORT
        // if (!capsule.is_valid())
        if (!capsule)
        {
            PyErr_Clear();
            PyTypeObject* tp = Py_TYPE(o);

            try
            {
                // BACKPORT
                // const char* module_name = borrow<str>(handle(tp).attr("__module__")).c_str();
                std::string module_name = reinterpret_borrow<str>(handle(tp->tp_dict).attr("__module__"));

                object package;
                if (strncmp(module_name.c_str(), "tensorflow.", 11) == 0)
                    package = module_::import("tensorflow.experimental.dlpack");
                else if (strcmp(module_name.c_str(), "torch") == 0)
                    package = module_::import("torch.utils.dlpack");
                else if (strncmp(module_name.c_str(), "jaxlib", 6) == 0)
                    package = module_::import("jax.dlpack");

                // BACKPORT
                // if (package.is_valid())
                if (package)
                    capsule = package.attr("to_dlpack")(handle(o));
            }
            catch (...)
            {
                // BACKPORT
                // capsule.reset();
                capsule.release();
            }
        }

        // Try creating a ndarray via the buffer protocol
        // BACKPORT
        // if (!capsule.is_valid())
        //     capsule = steal(dlpack_from_buffer_protocol(o));
        if (!capsule)
            capsule = reinterpret_steal<object>(dlpack_from_buffer_protocol(o));

        // BACKPORT
        // if (!capsule.is_valid())
        if (!capsule)
            return nullptr;
    }
    else
    {
        // BACKPORT
        // capsule = borrow(o);
        capsule = reinterpret_borrow<object>(o);
    }

    // Extract the pointer underlying the capsule
    void* ptr = PyCapsule_GetPointer(capsule.ptr(), "dltensor");
    if (!ptr)
    {
        PyErr_Clear();
        return nullptr;
    }

    // Check if the ndarray satisfies the requirements
    dlpack::dltensor& t = ((managed_dltensor*)ptr)->dltensor;

    bool pass_dtype = true, pass_device = true, pass_shape = true, pass_order = true;

    if (req->req_dtype)
        pass_dtype = t.dtype == req->dtype;

    if (req->req_device)
        pass_device = t.device.device_type == req->req_device;

    if (req->req_shape)
    {
        pass_shape &= req->ndim == (uint32_t)t.ndim;

        if (pass_shape)
        {
            for (uint32_t i = 0; i < req->ndim; ++i)
            {
                if (req->shape[i] != (size_t)t.shape[i] && req->shape[i] != pybind11::any)
                {
                    pass_shape = false;
                    break;
                }
            }
        }
    }

    scoped_pymalloc<int64_t> strides(t.ndim);
    if ((req->req_order || !t.strides) && t.ndim > 0)
    {
        size_t accum = 1;

        if (req->req_order == 'C' || !t.strides)
        {
            for (uint32_t i = (uint32_t)(t.ndim - 1);;)
            {
                strides[i] = accum;
                accum *= t.shape[i];
                if (i == 0)
                    break;
                --i;
            }
        }
        else if (req->req_order == 'F')
        {
            for (uint32_t i = 0; i < (uint32_t)t.ndim; ++i)
            {
                strides[i] = accum;
                accum *= t.shape[i];
            }
        }
        else
        {
            pass_order = false;
        }

        if (req->req_order)
        {
            if (!t.strides)
            {
                // c-style strides assumed
                pass_order = req->req_order == 'C';
            }
            else
            {
                for (uint32_t i = 0; i < (uint32_t)t.ndim; ++i)
                {
                    if (!((strides[i] == t.strides[i]) || (t.shape[i] == 1 && t.strides[i] == 0)))
                    {
                        pass_order = false;
                        break;
                    }
                }
            }
        }
    }

    // Support implicit conversion of 'dtype' and order
    if (pass_device && pass_shape && (!pass_dtype || !pass_order) && convert && capsule.ptr() != o)
    {
        PyTypeObject* tp = Py_TYPE(o);
        // BACKPORT
        // str module_name_o = borrow<str>(handle(tp).attr("__module__"));
        // const char* module_name = module_name_o.c_str();
        std::string module_name_str = reinterpret_borrow<str>(handle(tp->tp_dict).attr("__module__"));
        const char* module_name = module_name_str.c_str();

        char order = 'K';
        if (req->req_order != '\0')
            order = req->req_order;

        if (req->dtype.lanes != 1)
            return nullptr;

        const char* prefix = nullptr;
        char dtype[9];
        switch (req->dtype.code)
        {
        case (uint8_t)dlpack::dtype_code::Int:
            prefix = "int";
            break;
        case (uint8_t)dlpack::dtype_code::UInt:
            prefix = "uint";
            break;
        case (uint8_t)dlpack::dtype_code::Float:
            prefix = "float";
            break;
        default:
            return nullptr;
        }
        snprintf(dtype, sizeof(dtype), "%s%u", prefix, req->dtype.bits);

        object converted;
        try
        {
            if (strcmp(module_name, "numpy") == 0)
            {
                converted = handle(o).attr("astype")(dtype, order);
            }
            else if (strcmp(module_name, "torch") == 0)
            {
                converted = handle(o).attr("to")(arg("dtype") = module_::import("torch").attr(dtype), arg("copy") = true);
            }
            else if (strncmp(module_name, "tensorflow.", 11) == 0)
            {
                converted = module_::import("tensorflow").attr("cast")(handle(o), dtype);
            }
            else if (strncmp(module_name, "jaxlib", 6) == 0)
            {
                converted = handle(o).attr("astype")(dtype);
            }
        }
        catch (...)
        {
            // BACKPORT
            // converted.reset();
            converted.release();
        }

        // Potentially try again recursively
        // BACKPORT
        // if (!converted.is_valid())
        if (!converted)
            return nullptr;
        else
            return ndarray_import(converted.ptr(), req, false);
    }

    if (!pass_dtype || !pass_device || !pass_shape || !pass_order)
        return nullptr;

    // Create a reference-counted wrapper
    scoped_pymalloc<ndarray_handle> result;
    result->ndarray = (managed_dltensor*)ptr;
    result->refcount = 0;
    result->owner = nullptr;
    result->free_shape = false;
    result->call_deleter = true;

    // Ensure that the strides member is always initialized
    if (t.strides)
    {
        result->free_strides = false;
    }
    else
    {
        result->free_strides = true;
        t.strides = strides.release();
    }

    // Mark the dltensor capsule as "consumed"
    if (PyCapsule_SetName(capsule.ptr(), "used_dltensor") || PyCapsule_SetDestructor(capsule.ptr(), nullptr))
        pybind11_fail(
            "pybind11::detail::ndarray_import(): could not mark dltensor "
            "capsule as consumed!"
        );

    return result.release();
}

dlpack::dltensor* ndarray_inc_ref(ndarray_handle* th) noexcept
{
    if (!th)
        return nullptr;
    ++th->refcount;
    return &th->ndarray->dltensor;
}

void ndarray_dec_ref(ndarray_handle* th) noexcept
{
    if (!th)
        return;
    size_t rc_value = th->refcount--;

    if (rc_value == 0)
    {
        pybind11_fail("ndarray_dec_ref(): reference count became negative!");
    }
    else if (rc_value == 1)
    {
        Py_XDECREF(th->owner);
        managed_dltensor* mt = th->ndarray;
        if (th->free_shape)
        {
            PyMem_Free(mt->dltensor.shape);
            mt->dltensor.shape = nullptr;
        }
        if (th->free_strides)
        {
            PyMem_Free(mt->dltensor.strides);
            mt->dltensor.strides = nullptr;
        }
        if (th->call_deleter)
        {
            if (mt->deleter)
                mt->deleter(mt);
        }
        else
        {
            PyMem_Free(mt);
        }
        PyMem_Free(th);
    }
}

ndarray_handle* ndarray_create(
    void* value,
    size_t ndim,
    const size_t* shape_in,
    PyObject* owner,
    const int64_t* strides_in,
    dlpack::dtype* dtype,
    int32_t device_type,
    int32_t device_id
)
{
    /* DLPack mandates 256-byte alignment of the 'DLTensor::data' field, but
       PyTorch unfortunately ignores the 'byte_offset' value.. :-( */
#if 0
    uintptr_t value_int = (uintptr_t) value,
              value_rounded = (value_int / 256) * 256;
#else
    uintptr_t value_int = (uintptr_t)value, value_rounded = value_int;
#endif

    scoped_pymalloc<managed_dltensor> ndarray;
    scoped_pymalloc<ndarray_handle> result;
    scoped_pymalloc<int64_t> shape(ndim), strides(ndim);

    auto deleter = [](managed_dltensor* mt)
    {
        gil_scoped_acquire guard;
        ndarray_handle* th = (ndarray_handle*)mt->manager_ctx;
        ndarray_dec_ref(th);
    };

    for (size_t i = 0; i < ndim; ++i)
        shape[i] = (int64_t)shape_in[i];

    if (ndim > 0)
    {
        int64_t prod = 1;
        for (size_t i = ndim - 1;;)
        {
            if (strides_in)
            {
                strides[i] = strides_in[i];
            }
            else
            {
                strides[i] = prod;
                prod *= (int64_t)shape_in[i];
            }
            if (i == 0)
                break;
            --i;
        }
    }

    ndarray->dltensor.data = (void*)value_rounded;
    ndarray->dltensor.device.device_type = device_type;
    ndarray->dltensor.device.device_id = device_id;
    ndarray->dltensor.ndim = (int32_t)ndim;
    ndarray->dltensor.dtype = *dtype;
    ndarray->dltensor.byte_offset = value_int - value_rounded;
    ndarray->dltensor.shape = shape.release();
    ndarray->dltensor.strides = strides.release();
    ndarray->manager_ctx = result.get();
    ndarray->deleter = deleter;
    result->ndarray = (managed_dltensor*)ndarray.release();
    result->refcount = 0;
    result->owner = owner;
    result->free_shape = true;
    result->free_strides = true;
    result->call_deleter = false;
    Py_XINCREF(owner);
    return result.release();
}

static void ndarray_capsule_destructor(PyObject* o)
{
    error_scope scope; // temporarily save any existing errors
    managed_dltensor* mt = (managed_dltensor*)PyCapsule_GetPointer(o, "dltensor");

    if (mt)
        ndarray_dec_ref((ndarray_handle*)mt->manager_ctx);
    else
        PyErr_Clear();
}

PyObject* ndarray_wrap(ndarray_handle* th, int framework, return_value_policy policy) noexcept
{
    if (!th)
        return none().release().ptr();

    bool copy = policy == return_value_policy::copy || policy == return_value_policy::move;

    if ((ndarray_framework)framework == ndarray_framework::numpy)
    {
        try
        {
            // BACKPORT
            // object o = steal(PyType_GenericAlloc(internals_get().nb_ndarray, 0));
            object o = reinterpret_steal<object>(PyType_GenericAlloc(nb_internals::get().nb_ndarray, 0));
            // if (!o.is_valid())
            if (!o)
                return nullptr;
            ((nb_ndarray*)o.ptr())->th = th;
            ndarray_inc_ref(th);

            return module_::import("numpy").attr("array")(o, arg("copy") = copy).release().ptr();
        }
        catch (const std::exception& e)
        {
            PyErr_Format(
                PyExc_RuntimeError,
                "pybind11::detail::ndarray_wrap(): could not "
                "convert ndarray to NumPy array: %s",
                e.what()
            );
            return nullptr;
        }
    }

    object package;
    try
    {
        switch ((ndarray_framework)framework)
        {
        case ndarray_framework::none:
            break;

        case ndarray_framework::pytorch:
            package = module_::import("torch.utils.dlpack");
            break;

        case ndarray_framework::tensorflow:
            package = module_::import("tensorflow.experimental.dlpack");
            break;

        case ndarray_framework::jax:
            package = module_::import("jax.dlpack");
            break;

        default:
            pybind11_fail(
                "pybind11::detail::ndarray_wrap(): unknown framework "
                "specified!"
            );
        }
    }
    catch (const std::exception& e)
    {
        PyErr_Format(
            PyExc_RuntimeError,
            "pybind11::detail::ndarray_wrap(): could not import ndarray "
            "framework: %s",
            e.what()
        );
        return nullptr;
    }

    // BACKPORT
    // object o = steal(PyCapsule_New(th->ndarray, "dltensor", ndarray_capsule_destructor));
    object o = reinterpret_steal<object>(PyCapsule_New(th->ndarray, "dltensor", ndarray_capsule_destructor));

    ndarray_inc_ref(th);

    // BACKPORT
    // if (package.is_valid())
    if (package)
    {
        try
        {
            o = package.attr("from_dlpack")(o);
        }
        catch (const std::exception& e)
        {
            PyErr_Format(
                PyExc_RuntimeError,
                "pybind11::detail::ndarray_wrap(): could not "
                "import ndarray: %s",
                e.what()
            );
            return nullptr;
        }
    }

    if (copy)
    {
        try
        {
            o = o.attr("copy")();
        }
        catch (std::exception& e)
        {
            PyErr_Format(PyExc_RuntimeError, "pybind11::detail::ndarray_wrap(): copy failed: %s", e.what());
            return nullptr;
        }
    }

    return o.release().ptr();
}

} // namespace detail
} // namespace pybind11

#include "ScriptBindings.h"

namespace Falcor
{
FALCOR_SCRIPT_BINDING(ndarray)
{
    m.def(
        "inspect_ndarray",
        [](pybind11::ndarray<> ndarray)
        {
            printf("ndarray data pointer : %p\n", ndarray.data());
            printf("ndarray dimension : %zu\n", ndarray.ndim());
            for (size_t i = 0; i < ndarray.ndim(); ++i)
            {
                printf("ndarray dimension [%zu] : %zu\n", i, ndarray.shape(i));
                printf("ndarray stride    [%zu] : %zd\n", i, ndarray.stride(i));
            }
            printf(
                "Device ID = %u (cpu=%i, cuda=%i)\n",
                ndarray.device_id(),
                int(ndarray.device_type() == pybind11::device::cpu::value),
                int(ndarray.device_type() == pybind11::device::cuda::value)
            );
            printf(
                "ndarray dtype: int16=%i, uint32=%i, float32=%i\n",
                ndarray.dtype() == pybind11::dtype<int16_t>(),
                ndarray.dtype() == pybind11::dtype<uint32_t>(),
                ndarray.dtype() == pybind11::dtype<float>()
            );
        }
    );
}

} // namespace Falcor
