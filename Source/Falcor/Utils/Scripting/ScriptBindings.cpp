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
#include "ScriptBindings.h"
#include "Core/Error.h"
#include "Core/Plugin.h"
#include "Utils/Scripting/Scripting.h"
#include "Utils/Math/Vector.h"
#include "Utils/Math/Matrix.h"
#include <fmt/format.h>
#include <pybind11/embed.h>
#include <pybind11/operators.h>
#include <algorithm>

using namespace pybind11::literals;

namespace pybind11::detail
{
// Casting float16_t <-> float.
template<>
struct type_caster<Falcor::float16_t>
{
public:
    PYBIND11_TYPE_CASTER(Falcor::float16_t, _("float16_t"));
    using float_caster = type_caster<float>;

    bool load(handle src, bool convert)
    {
        float_caster caster;
        if (caster.load(src, convert))
        {
            this->value = Falcor::float16_t(float(caster));
            return true;
        }
        return false;
    }

    static handle cast(Falcor::float16_t src, return_value_policy policy, handle parent)
    {
        return float_caster::cast(float(src), policy, parent);
    }
};
} // namespace pybind11::detail

namespace Falcor::ScriptBindings
{
namespace
{
struct DeferredBinding
{
    std::string name;
    RegisterBindingFunc bindingFunc;
    bool isRegistered = false;

    DeferredBinding(const std::string& name, RegisterBindingFunc bindingFunc) : name(name), bindingFunc(bindingFunc) {}

    void bind(pybind11::module& m)
    {
        if (!isRegistered)
        {
            isRegistered = true;
            bindingFunc(m);
        }
    }
};

static std::vector<DeferredBinding>& getDeferredBindings()
{
    static std::vector<DeferredBinding> deferredBindings;
    return deferredBindings;
}

static std::string getUniqueDeferredBindingName()
{
    static uint32_t id = 0;
    return fmt::format("_DeferredBinding{}", id++);
}

static pybind11::module sModule;
} // namespace

void registerBinding(RegisterBindingFunc f)
{
    if (sModule)
    {
        try
        {
            f(sModule);
        }
        catch (const std::exception& e)
        {
            PyErr_SetString(PyExc_ImportError, e.what());
            reportErrorAndContinue(e.what());
            return;
        }
    }
    else
    {
        registerDeferredBinding(getUniqueDeferredBindingName(), f);
    }
}

void registerDeferredBinding(const std::string& name, RegisterBindingFunc f)
{
    auto& deferredBindings = getDeferredBindings();
    if (std::find_if(
            deferredBindings.begin(), deferredBindings.end(), [&name](const DeferredBinding& binding) { return binding.name == name; }
        ) != deferredBindings.end())
    {
        FALCOR_THROW("A script binding with the name '{}' already exists!", name);
    }
    deferredBindings.emplace_back(name, f);
}

void resolveDeferredBinding(const std::string& name, pybind11::module& m)
{
    auto& deferredBindings = getDeferredBindings();
    auto it = std::find_if(
        deferredBindings.begin(), deferredBindings.end(), [&name](const DeferredBinding& binding) { return binding.name == name; }
    );
    if (it != deferredBindings.end())
        it->bind(m);
}

template<typename VecT, bool WithOperators>
void defineVecType(pybind11::class_<VecT>& vec)
{
    using ScalarT = typename VecT::value_type;

    auto constexpr length = VecT::length();
    static_assert(length >= 2 && length <= 4, "Unsupported number of components");

    vec.def_readwrite("x", &VecT::x);
    vec.def_readwrite("y", &VecT::y);
    if constexpr (length >= 3)
        vec.def_readwrite("z", &VecT::z);
    if constexpr (length >= 4)
        vec.def_readwrite("w", &VecT::w);

    auto initEmpty = []() { return VecT(ScalarT(0)); };
    vec.def(pybind11::init(initEmpty));

    auto initScalar = [](ScalarT c) { return VecT(c); };
    vec.def(pybind11::init(initScalar), "c"_a);

    if constexpr (length == 2)
    {
        auto initVector = [](ScalarT x, ScalarT y) { return VecT(x, y); };
        vec.def(pybind11::init(initVector), "x"_a, "y"_a);
    }
    else if constexpr (length == 3)
    {
        auto initVector = [](ScalarT x, ScalarT y, ScalarT z) { return VecT(x, y, z); };
        vec.def(pybind11::init(initVector), "x"_a, "y"_a, "z"_a);
    }
    else if constexpr (length == 4)
    {
        auto initVector = [](ScalarT x, ScalarT y, ScalarT z, ScalarT w) { return VecT(x, y, z, w); };
        vec.def(pybind11::init(initVector), "x"_a, "y"_a, "z"_a, "w"_a);
    }

    // Initialization from array (to allow implicit conversion from python lists).
    auto initArray = [](std::array<ScalarT, length> a)
    {
        VecT v;
        for (size_t i = 0; i < VecT::length(); ++i)
            v[i] = a[i];
        return v;
    };
    vec.def(pybind11::init(initArray));
    pybind11::implicitly_convertible<std::array<ScalarT, length>, VecT>();

    // Initialization from integer array (to allow implicit conversion from python lists).
    if constexpr (std::is_floating_point_v<ScalarT>)
    {
        auto initIntArray = [](std::array<int64_t, length> a)
        {
            VecT v;
            for (size_t i = 0; i < VecT::length(); ++i)
                v[i] = ScalarT(a[i]);
            return v;
        };
        vec.def(pybind11::init(initIntArray));
        pybind11::implicitly_convertible<std::array<int64_t, length>, VecT>();
    }

    // Casting float16_t <-> float vectors.
    // This allows explicit casts, e.g., float16_t3(c), where c is a float3 in python.
    if constexpr (std::is_same<ScalarT, float16_t>::value)
    {
        using floatN = math::vector<float, length>;
        auto initVector = [](floatN v) { return VecT(v); };
        vec.def(pybind11::init(initVector), "v"_a);
    }
    else if constexpr (std::is_same<ScalarT, float>::value)
    {
        using float16_tN = math::vector<float16_t, length>;
        auto initVector = [](float16_tN v) { return VecT(v); };
        vec.def(pybind11::init(initVector), "v"_a);
    }

    auto repr = [](const VecT& v)
    {
        std::string str =
            math::ScalarTraits<typename VecT::value_type>::name + std::to_string(VecT::length()) + "(" + math::to_string(v[0]);
        for (int i = 1; i < VecT::length(); i++)
            str += ", " + math::to_string(v[i]);
        str += ")";
        return str;
    };
    vec.def("__repr__", repr);

    auto str = [](const VecT& v)
    {
        std::string str = "[" + math::to_string(v[0]);
        for (int i = 1; i < VecT::length(); i++)
            str += ", " + math::to_string(v[i]);
        str += "]";
        return str;
    };
    vec.def("__str__", str);

    vec.def(pybind11::pickle(
        [&](const VecT& v)
        {
            pybind11::tuple t(length);
            for (auto i = 0; i < length; ++i)
                t[i] = v[i];
            return t;
        },
        [&](pybind11::tuple t)
        {
            if (t.size() != length)
                FALCOR_THROW("Invalid state!");
            VecT v;
            for (auto i = 0; i < length; ++i)
                v[i] = t[i].cast<ScalarT>();
            return v;
        }
    ));

    if constexpr (WithOperators)
    {
        vec.def(pybind11::self + pybind11::self);
        vec.def(pybind11::self += pybind11::self);
        vec.def(pybind11::self - pybind11::self);
        vec.def(pybind11::self -= pybind11::self);
        vec.def(pybind11::self * pybind11::self);
        vec.def(pybind11::self *= pybind11::self);
        vec.def(pybind11::self / pybind11::self);
        vec.def(pybind11::self /= pybind11::self);
        vec.def(pybind11::self + ScalarT());
        vec.def(pybind11::self += ScalarT());
        vec.def(pybind11::self - ScalarT());
        vec.def(pybind11::self -= ScalarT());
        vec.def(pybind11::self * ScalarT());
        vec.def(pybind11::self *= ScalarT());
        vec.def(pybind11::self / ScalarT());
        vec.def(pybind11::self /= ScalarT());
    }
}

void initModule(pybind11::module& m)
{
    using namespace pybind11::literals;

    pybind11::class_<float16_t>(m, "float16_t");

    // Declare all vector types.
    // We do this before defining the bindings to allow explicit casts between vector types.

    pybind11::class_<bool2> bool2Type(m, "bool2");
    pybind11::class_<bool3> bool3Type(m, "bool3");
    pybind11::class_<bool4> bool4Type(m, "bool4");

    pybind11::class_<int2> int2Type(m, "int2");
    pybind11::class_<int3> int3Type(m, "int3");
    pybind11::class_<int4> int4Type(m, "int4");

    pybind11::class_<uint2> uint2Type(m, "uint2");
    pybind11::class_<uint3> uint3Type(m, "uint3");
    pybind11::class_<uint4> uint4Type(m, "uint4");

    pybind11::class_<float2> float2Type(m, "float2");
    pybind11::class_<float3> float3Type(m, "float3");
    pybind11::class_<float4> float4Type(m, "float4");

    pybind11::class_<float16_t2> float16_t2Type(m, "float16_t2");
    pybind11::class_<float16_t3> float16_t3Type(m, "float16_t3");
    pybind11::class_<float16_t4> float16_t4Type(m, "float16_t4");

    // bool2, bool3, bool4
    defineVecType<bool2, false>(bool2Type);
    defineVecType<bool3, false>(bool3Type);
    defineVecType<bool4, false>(bool4Type);

    // int2, int3, int4
    defineVecType<int2, true>(int2Type);
    defineVecType<int3, true>(int3Type);
    defineVecType<int4, true>(int4Type);

    // uint2, uint3, uint4
    defineVecType<uint2, true>(uint2Type);
    defineVecType<uint3, true>(uint3Type);
    defineVecType<uint4, true>(uint4Type);

    // float2, float3, float4
    defineVecType<float2, true>(float2Type);
    defineVecType<float3, true>(float3Type);
    defineVecType<float4, true>(float4Type);

    // float16_t2, float16_t3, float16_t4
    defineVecType<float16_t2, false>(float16_t2Type);
    defineVecType<float16_t3, false>(float16_t3Type);
    defineVecType<float16_t4, false>(float16_t4Type);

    // float3x3, float4x4
    // Note: We register these as simple data types without any operations because semantics may change in the future.
    pybind11::class_<float3x3>(m, "float3x3");
    pybind11::class_<float3x4>(m, "float3x4");
    pybind11::class_<float4x4>(m, "float4x4");

    // ObjectID
    pybind11::class_<uint32_t>(m, "ObjectID");

    // Plugins.
    m.def(
        "load_plugin", [](const std::string& name) { PluginManager::instance().loadPluginByName(name); }, "name"_a
    );
    m.def(
        "loadPlugin", [](const std::string& name) { PluginManager::instance().loadPluginByName(name); }, "name"_a
    ); // PYTHONDEPRECATED

#if FALCOR_ENABLE_OBJECT_TRACKING
    m.def("dump_alive_objects", []() { Object::dumpAliveObjects(); });
#endif

    // Bind all deferred bindings.
    for (auto& binding : getDeferredBindings())
        binding.bind(m);
    getDeferredBindings().clear();

    // Retain a handle to the module to add new bindings at runtime.
    sModule = m;

    // Register atexit handler to automatically release the module handle on exit.
    auto atexit = pybind11::module_::import("atexit");
    atexit.attr("register")(pybind11::cpp_function([]() { sModule.release(); }));
}

} // namespace Falcor::ScriptBindings
