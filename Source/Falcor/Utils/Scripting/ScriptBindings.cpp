/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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
#include "Core/Plugin.h"
#include "Utils/Scripting/Scripting.h"
#include "Utils/Math/Float16.h"
#include "Utils/Math/Matrix/Matrix.h"
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
}

namespace Falcor::ScriptBindings
{
    namespace
    {
        struct DeferredBinding
        {
            std::string name;
            RegisterBindingFunc bindingFunc;
            bool isRegistered = false;

            DeferredBinding(const std::string& name, RegisterBindingFunc bindingFunc)
                : name(name)
                , bindingFunc(bindingFunc)
            {}

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
    }

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
                reportError(e.what());
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
        if (std::find_if(deferredBindings.begin(), deferredBindings.end(), [&name](const DeferredBinding& binding) { return binding.name == name; }) != deferredBindings.end())
        {
            throw RuntimeError("A script binding with the name '{}' already exists!", name);
        }
        deferredBindings.emplace_back(name, f);
    }

    void resolveDeferredBinding(const std::string &name, pybind11::module& m)
    {
        auto& deferredBindings = getDeferredBindings();
        auto it = std::find_if(deferredBindings.begin(), deferredBindings.end(), [&name](const DeferredBinding& binding) { return binding.name == name; });
        if (it != deferredBindings.end()) it->bind(m);
    }


    template<typename VecT, bool withOperators>
    void addVecType(pybind11::module& m, const std::string name)
    {
        using ScalarT = typename VecT::value_type;

        auto constexpr length = VecT::length();
        static_assert(length >= 2 && length <= 4, "Unsupported number of components");

        pybind11::class_<VecT> vec(m, name.c_str());

        vec.def_readwrite("x", &VecT::x);
        vec.def_readwrite("y", &VecT::y);
        if constexpr (length >= 3) vec.def_readwrite("z", &VecT::z);
        if constexpr (length >= 4) vec.def_readwrite("w", &VecT::w);

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

        // Casting float16_t <-> float vectors.
        // This allows explicit casts, e.g., float16_t3(c), where c is a float3 in python.
        if constexpr (std::is_same<ScalarT, float16_t>::value)
        {
            using floatN = glm::vec<length, float, glm::defaultp>;
            auto initVector = [](floatN v) { return VecT(v); };
            vec.def(pybind11::init(initVector), "v"_a);
        }
        else if constexpr (std::is_same<ScalarT, float>::value)
        {
            using float16_tN = tfloat16_vec<length>;
            auto initVector = [](float16_tN v) { return VecT(v); };
            vec.def(pybind11::init(initVector), "v"_a);
        }

        auto repr = [](const VecT& v) { return Falcor::to_string(v); };
        vec.def("__repr__", repr);

        auto str = [](const VecT& v) {
            auto tostr = [](const ScalarT& s) {
                if constexpr (std::is_same<ScalarT, float16_t>::value) return to_string(s);
                else return std::to_string(s);
            };
            std::string vec = "[" + tostr(v[0]);
            for (int i = 1; i < VecT::length(); i++)
            {
                vec += ", " + tostr(v[i]);
            }
            vec += "]";
            return vec;
        };
        vec.def("__str__", str);

        vec.def(pybind11::pickle(
            [&] (const VecT &v) {
                pybind11::tuple t(length);
                for (auto i = 0; i < length; ++i) t[i] = v[i];
                return t;
            },
            [&] (pybind11::tuple t) {
                if (t.size() != length) throw RuntimeError("Invalid state!");
                VecT v;
                for (auto i = 0; i < length; ++i) v[i] = t[i].cast<ScalarT>();
                return v;
            }
        ));

        if constexpr (withOperators)
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

        // bool2, bool3, bool4
        addVecType<bool2, false>(m, "bool2");
        addVecType<bool3, false>(m, "bool3");
        addVecType<bool4, false>(m, "bool4");

        // float2, float3, float4
        addVecType<float2, true>(m, "float2");
        addVecType<float3, true>(m, "float3");
        addVecType<float4, true>(m, "float4");

        // int2, int3, int4
        addVecType<int2, true>(m, "int2");
        addVecType<int3, true>(m, "int3");
        addVecType<int4, true>(m, "int4");

        // uint2, uint3, uint4
        addVecType<uint2, true>(m, "uint2");
        addVecType<uint3, true>(m, "uint3");
        addVecType<uint4, true>(m, "uint4");

        // float3x3, float4x4
        // Note: We register these as simple data types without any operations because semantics may change in the future.
        pybind11::class_<rmcv::mat3>(m, "float3x3");
        pybind11::class_<rmcv::mat4>(m, "float4x4");

        // float16_t types
        pybind11::class_<float16_t>(m, "float16_t");
        addVecType<float16_t2, false>(m, "float16_t2");
        addVecType<float16_t3, false>(m, "float16_t3");
        addVecType<float16_t4, false>(m, "float16_t4");

        // ObjectID
        pybind11::class_<uint32_t>(m, "ObjectID");

        // Plugins.
        m.def("loadPlugin", [](const std::string& name) {
            PluginManager::instance().loadPluginByName(name);
        }, "name"_a);

        // Bind all deferred bindings.
        for (auto& binding : getDeferredBindings())
            binding.bind(m);
        getDeferredBindings().clear();

        // Retain a handle to the module to add new bindings at runtime.
        sModule = m;

        // Register atexit handler to automatically release the module handle on exit.
        auto atexit = pybind11::module_::import("atexit");
        atexit.attr("register")(pybind11::cpp_function([]() {
            sModule.release();
        }));
    }

}
