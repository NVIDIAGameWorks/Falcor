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

/*
    nanobind/ndarray.h: functionality to exchange n-dimensional arrays with
    other array programming frameworks (NumPy, PyTorch, etc.)

    Copyright (c) 2022 Wenzel Jakob

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.

    The API below is based on the DLPack project
    (https://github.com/dmlc/dlpack/blob/main/include/dlpack/dlpack.h)
*/
#pragma once

#include "Core/Macros.h"
#include <pybind11/pybind11.h>

namespace pybind11
{

// Forward declarations for types in dlpack.h (1)
namespace dlpack
{
struct dltensor;
struct dtype;
} // namespace dlpack

namespace detail
{

// Forward declarations for types in dlpack.h (2)
struct ndarray_handle;
struct ndarray_req;

// Try to import a reference-counted ndarray object via DLPack
FALCOR_API ndarray_handle* ndarray_import(PyObject* o, const ndarray_req* req, bool convert) noexcept;

// Describe a local tensor object using a DLPack capsule
FALCOR_API ndarray_handle* ndarray_create(
    void* value,
    size_t ndim,
    const size_t* shape,
    PyObject* owner,
    const int64_t* strides,
    dlpack::dtype* dtype,
    int32_t device,
    int32_t device_id
);

/// Increase the reference count of the given tensor object; returns a pointer
/// to the underlying DLtensor
FALCOR_API dlpack::dltensor* ndarray_inc_ref(ndarray_handle*) noexcept;

/// Decrease the reference count of the given tensor object
FALCOR_API void ndarray_dec_ref(ndarray_handle*) noexcept;

/// Wrap a ndarray_handle* into a PyCapsule
FALCOR_API PyObject* ndarray_wrap(ndarray_handle*, int framework, return_value_policy policy) noexcept;

} // namespace detail
} // namespace pybind11

namespace pybind11
{

namespace device
{
#define NB_DEVICE(enum_name, enum_value)                             \
    struct enum_name                                                 \
    {                                                                \
        static constexpr auto name = detail::const_name(#enum_name); \
        static constexpr int32_t value = enum_value;                 \
        static constexpr bool is_device = true;                      \
    }
NB_DEVICE(none, 0);
NB_DEVICE(cpu, 1);
NB_DEVICE(cuda, 2);
NB_DEVICE(cuda_host, 3);
NB_DEVICE(opencl, 4);
NB_DEVICE(vulkan, 7);
NB_DEVICE(metal, 8);
NB_DEVICE(rocm, 10);
NB_DEVICE(rocm_host, 11);
NB_DEVICE(cuda_managed, 13);
NB_DEVICE(oneapi, 14);
#undef NB_DEVICE
} // namespace device

namespace dlpack
{

enum class dtype_code : uint8_t
{
    Int = 0,
    UInt = 1,
    Float = 2,
    Bfloat = 4,
    Complex = 5
};

struct device
{
    int32_t device_type = 0;
    int32_t device_id = 0;
};

struct dtype
{
    uint8_t code = 0;
    uint8_t bits = 0;
    uint16_t lanes = 0;

    bool operator==(const dtype& o) const { return code == o.code && bits == o.bits && lanes == o.lanes; }
    bool operator!=(const dtype& o) const { return !operator==(o); }
};

struct dltensor
{
    void* data = nullptr;
    pybind11::dlpack::device device;
    int32_t ndim = 0;
    pybind11::dlpack::dtype dtype;
    int64_t* shape = nullptr;
    int64_t* strides = nullptr;
    uint64_t byte_offset = 0;
};

} // namespace dlpack

constexpr size_t any = (size_t)-1;

template<size_t... Is>
struct shape
{
    static constexpr size_t size = sizeof...(Is);
};

struct c_contig
{};
struct f_contig
{};
struct any_contig
{};
struct numpy
{};
struct tensorflow
{};
struct pytorch
{};
struct jax
{};

template<typename T>
constexpr dlpack::dtype dtype()
{
    static_assert(
        std::is_floating_point_v<T> || std::is_integral_v<T>, "pybind11::dtype<T>: T must be a floating point or integer variable!"
    );

    dlpack::dtype result;

    if constexpr (std::is_floating_point_v<T>)
        result.code = (uint8_t)dlpack::dtype_code::Float;
    else if constexpr (std::is_signed_v<T>)
        result.code = (uint8_t)dlpack::dtype_code::Int;
    else
        result.code = (uint8_t)dlpack::dtype_code::UInt;

    result.bits = sizeof(T) * 8;
    result.lanes = 1;

    return result;
}

namespace detail
{

enum class ndarray_framework : int
{
    none,
    numpy,
    tensorflow,
    pytorch,
    jax
};

struct ndarray_req
{
    dlpack::dtype dtype;
    uint32_t ndim = 0;
    size_t* shape = nullptr;
    bool req_shape = false;
    bool req_dtype = false;
    char req_order = '\0';
    uint8_t req_device = 0;
};

template<typename T, typename = int>
struct ndarray_arg
{
    static constexpr size_t size = 0;
    static constexpr auto name = descr<0>{};
    static void apply(ndarray_req&) {}
};

template<typename T>
struct ndarray_arg<T, enable_if_t<std::is_floating_point_v<T>>>
{
    static constexpr size_t size = 0;

    static constexpr auto name = const_name("dtype=float") + const_name<sizeof(T) * 8>();

    static void apply(ndarray_req& tr)
    {
        tr.dtype = dtype<T>();
        tr.req_dtype = true;
    }
};

template<typename T>
struct ndarray_arg<T, enable_if_t<std::is_integral_v<T>>>
{
    static constexpr size_t size = 0;

    static constexpr auto name =
        const_name("dtype=") + const_name<std::is_unsigned_v<T>>("u", "") + const_name("int") + const_name<sizeof(T) * 8>();

    static void apply(ndarray_req& tr)
    {
        tr.dtype = dtype<T>();
        tr.req_dtype = true;
    }
};

template<size_t... Is>
struct ndarray_arg<shape<Is...>>
{
    static constexpr size_t size = sizeof...(Is);
    static constexpr auto name =
        const_name("shape=(") + concat(const_name<Is == any>(const_name("*"), const_name<Is>())...) + const_name(")");

    static void apply(ndarray_req& tr)
    {
        size_t i = 0;
        ((tr.shape[i++] = Is), ...);
        tr.ndim = (uint32_t)sizeof...(Is);
        tr.req_shape = true;
    }
};

template<>
struct ndarray_arg<c_contig>
{
    static constexpr size_t size = 0;
    static constexpr auto name = const_name("order='C'");
    static void apply(ndarray_req& tr) { tr.req_order = 'C'; }
};

template<>
struct ndarray_arg<f_contig>
{
    static constexpr size_t size = 0;
    static constexpr auto name = const_name("order='F'");
    static void apply(ndarray_req& tr) { tr.req_order = 'F'; }
};

template<>
struct ndarray_arg<any_contig>
{
    static constexpr size_t size = 0;
    static constexpr auto name = const_name("order='*'");
    static void apply(ndarray_req& tr) { tr.req_order = '\0'; }
};

template<typename T>
struct ndarray_arg<T, enable_if_t<T::is_device>>
{
    static constexpr size_t size = 0;
    static constexpr auto name = const_name("device='") + T::name + const_name("'");
    static void apply(ndarray_req& tr) { tr.req_device = (uint8_t)T::value; }
};

template<typename... Ts>
struct ndarray_info
{
    using scalar_type = void;
    using shape_type = void;
    constexpr static auto name = const_name("ndarray");
    constexpr static ndarray_framework framework = ndarray_framework::none;
};

template<typename T, typename... Ts>
struct ndarray_info<T, Ts...> : ndarray_info<Ts...>
{
    using scalar_type = std::conditional_t<std::is_scalar_v<T>, T, typename ndarray_info<Ts...>::scalar_type>;
};

template<size_t... Is, typename... Ts>
struct ndarray_info<shape<Is...>, Ts...> : ndarray_info<Ts...>
{
    using shape_type = shape<Is...>;
};

template<typename... Ts>
struct ndarray_info<numpy, Ts...> : ndarray_info<Ts...>
{
    constexpr static auto name = const_name("numpy.ndarray");
    constexpr static ndarray_framework framework = ndarray_framework::numpy;
};

template<typename... Ts>
struct ndarray_info<pytorch, Ts...> : ndarray_info<Ts...>
{
    constexpr static auto name = const_name("torch.Tensor");
    constexpr static ndarray_framework framework = ndarray_framework::pytorch;
};

template<typename... Ts>
struct ndarray_info<tensorflow, Ts...> : ndarray_info<Ts...>
{
    constexpr static auto name = const_name("tensorflow.python.framework.ops.EagerTensor");
    constexpr static ndarray_framework framework = ndarray_framework::tensorflow;
};

template<typename... Ts>
struct ndarray_info<jax, Ts...> : ndarray_info<Ts...>
{
    constexpr static auto name = const_name("jaxlib.xla_extension.DeviceArray");
    constexpr static ndarray_framework framework = ndarray_framework::jax;
};

} // namespace detail

template<typename... Args>
class ndarray
{
public:
    using Info = detail::ndarray_info<Args...>;
    using Scalar = typename Info::scalar_type;

    ndarray() = default;

    explicit ndarray(detail::ndarray_handle* handle) : m_handle(handle)
    {
        if (handle)
            m_dltensor = *detail::ndarray_inc_ref(handle);
    }

    ndarray(
        void* value,
        size_t ndim,
        const size_t* shape,
        handle owner = pybind11::handle(),
        const int64_t* strides = nullptr,
        dlpack::dtype dtype = pybind11::dtype<Scalar>(),
        int32_t device_type = device::cpu::value,
        int32_t device_id = 0
    )
    {
        m_handle = detail::ndarray_create(value, ndim, shape, owner.ptr(), strides, &dtype, device_type, device_id);
        m_dltensor = *detail::ndarray_inc_ref(m_handle);
    }

    ~ndarray() { detail::ndarray_dec_ref(m_handle); }

    ndarray(const ndarray& t) : m_handle(t.m_handle), m_dltensor(t.m_dltensor) { detail::ndarray_inc_ref(m_handle); }

    ndarray(ndarray&& t) noexcept : m_handle(t.m_handle), m_dltensor(t.m_dltensor)
    {
        t.m_handle = nullptr;
        t.m_dltensor = dlpack::dltensor();
    }

    ndarray& operator=(ndarray&& t) noexcept
    {
        detail::ndarray_dec_ref(m_handle);
        m_handle = t.m_handle;
        m_dltensor = t.m_dltensor;
        t.m_handle = nullptr;
        t.m_dltensor = dlpack::dltensor();
        return *this;
    }

    ndarray& operator=(const ndarray& t)
    {
        detail::ndarray_inc_ref(t.m_handle);
        detail::ndarray_dec_ref(m_handle);
        m_handle = t.m_handle;
        m_dltensor = t.m_dltensor;
        return *this;
    }

    dlpack::dtype dtype() const { return m_dltensor.dtype; }
    size_t ndim() const { return m_dltensor.ndim; }
    size_t shape(size_t i) const { return m_dltensor.shape[i]; }
    int64_t stride(size_t i) const { return m_dltensor.strides[i]; }
    bool is_valid() const { return m_handle != nullptr; }
    int32_t device_type() const { return m_dltensor.device.device_type; }
    int32_t device_id() const { return m_dltensor.device.device_id; }
    detail::ndarray_handle* handle() const { return m_handle; }

    const Scalar* data() const { return (const Scalar*)((const uint8_t*)m_dltensor.data + m_dltensor.byte_offset); }

    Scalar* data() { return (Scalar*)((uint8_t*)m_dltensor.data + m_dltensor.byte_offset); }

    template<typename... Ts>
    inline auto& operator()(Ts... indices)
    {
        static_assert(
            !std::is_same_v<Scalar, void>,
            "To use pybind11::ndarray::operator(), you must add a scalar type "
            "annotation (e.g. 'float') to the ndarray template parameters."
        );
        static_assert(
            !std::is_same_v<Scalar, void>,
            "To use pybind11::ndarray::operator(), you must add a pybind11::shape<> "
            "annotation to the ndarray template parameters."
        );
        static_assert(sizeof...(Ts) == Info::shape_type::size, "pybind11::ndarray::operator(): invalid number of arguments");

        int64_t counter = 0, index = 0;
        ((index += int64_t(indices) * m_dltensor.strides[counter++]), ...);
        return (Scalar&)*((uint8_t*)m_dltensor.data + m_dltensor.byte_offset + index * sizeof(typename Info::scalar_type));
    }

private:
    detail::ndarray_handle* m_handle = nullptr;
    dlpack::dltensor m_dltensor;
};

namespace detail
{

constexpr descr<0> concat_maybe()
{
    return {};
}

template<size_t N, typename... Ts>
constexpr descr<N, Ts...> concat_maybe(const descr<N, Ts...>& descr)
{
    return descr;
}

template<size_t N, typename... Ts, typename... Args>
constexpr auto concat_maybe(const descr<N, Ts...>& d, const Args&... args)
    -> decltype(std::declval<descr<N + sizeof...(Ts) == 0 ? 0 : (N + 2), Ts...>>() + concat_maybe(args...))
{
    if constexpr (N + sizeof...(Ts) == 0)
        return concat_maybe(args...);
    else
        return d + const_name(", ") + concat_maybe(args...);
}

template<typename... Args>
struct type_caster<ndarray<Args...>>
{
    using Value = ndarray<Args...>;

    PYBIND11_TYPE_CASTER(
        ndarray<Args...>,
        Value::Info::name + const_name("[") + concat_maybe(detail::ndarray_arg<Args>::name...) + const_name("]")
    );

    // BACKPORT
    // bool from_python(handle src, uint8_t flags, cleanup_list*) noexcept
    bool load(handle src, bool convert) noexcept
    {
        constexpr size_t size = (0 + ... + detail::ndarray_arg<Args>::size);
        size_t shape[size + 1];
        detail::ndarray_req req;
        req.shape = shape;
        (detail::ndarray_arg<Args>::apply(req), ...);
        // BACKPORT
        // value = ndarray<Args...>(ndarray_import(src.ptr(), &req, flags & (uint8_t)cast_flags::convert));
        value = ndarray<Args...>(ndarray_import(src.ptr(), &req, convert));
        return value.is_valid();
    }

    // BACKPORT
    // static handle from_cpp(const ndarray<Args...>& tensor, rv_policy policy, cleanup_list*) noexcept
    static handle cast(const ndarray<Args...>& tensor, return_value_policy policy, handle /* parent */) noexcept
    {
        return ndarray_wrap(tensor.handle(), int(Value::Info::framework), policy);
    }
};

} // namespace detail
} // namespace pybind11
