/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include "stdafx.h"
#include "ScriptBindings.h"
#include "pybind11/embed.h"
#include <algorithm>

namespace Falcor::ScriptBindings
{
    namespace
    {
        /** `gRegisterFuncs` is declared as pointer so that we can ensure it can be explicitly
            allocated when registerBinding() is called. (The C++ static objectinitialization fiasco.)
        */
        std::unique_ptr<std::vector<RegisterBindingFunc>> gRegisterFuncs;
    }

    void registerBinding(RegisterBindingFunc f)
    {
        if (Scripting::isRunning())
        {
            try
            {
                auto m = pybind11::module::import("falcor");
                f(m);
                // Re-import falcor
                pybind11::exec("from falcor import *");
            }
            catch (const std::exception& e)
            {
                PyErr_SetString(PyExc_ImportError, e.what());
                logError(e.what());
                return;
            }
        }
        else
        {
            if (!gRegisterFuncs) gRegisterFuncs.reset(new std::vector<RegisterBindingFunc>());
            gRegisterFuncs->push_back(f);
        }
    }

    template<typename VecT, typename...Args>
    VecT makeVec(Args...args)
    {
        return VecT(args...);
    }

    template<typename VecT, typename...Args>
    void addVecType(pybind11::module& m, const std::string name)
    {
        auto ctor = [](Args...components) { return makeVec<VecT>(components...); };
        auto repr = [](const VecT& v) { return Falcor::to_string(v); };
        auto vecStr = [](const VecT& v) {
            std::string vec = "[" + std::to_string(v[0]);
            for (int i = 1; i < v.length(); i++)
            {
                vec += ", " + std::to_string(v[i]);
            }
            vec += "]";
            return vec;
        };
        pybind11::class_<VecT>(m, name.c_str())
            .def(pybind11::init(ctor))
            .def("__repr__", repr)
            .def("__str__", vecStr);
    }

    PYBIND11_EMBEDDED_MODULE(falcor, m)
    {
        // bool2, bool3, bool4
        addVecType<bool2, bool, bool>(m, "bool2");
        addVecType<bool3, bool, bool, bool>(m, "bool3");
        addVecType<bool4, bool, bool, bool, bool>(m, "bool4");

        // float2, float3, float4
        addVecType<float2, float, float>(m, "float2");
        addVecType<float3, float, float, float>(m, "float3");
        addVecType<float4, float, float, float, float>(m, "float4");

        // int2, int3, int4
        addVecType<int2, int32_t, int32_t>(m, "int2");
        addVecType<int3, int32_t, int32_t, int32_t>(m, "int3");
        addVecType<int4, int32_t, int32_t, int32_t, int32_t>(m, "int4");

        // uint2, uint3, uint4
        addVecType<uint2, uint32_t, uint32_t>(m, "uint2");
        addVecType<uint3, uint32_t, uint32_t, uint32_t>(m, "uint3");
        addVecType<uint4, uint32_t, uint32_t, uint32_t, uint32_t>(m, "uint4");

        if (gRegisterFuncs)
        {
            for (auto f : *gRegisterFuncs) f(m);
        }
    }
}
