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
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
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
    ClassesMap sClasses;
    std::unordered_map<std::type_index, std::string> sEnumNames;

    namespace
    {
        /** `gBindFuncs` is declared as pointer so that we can ensure it can be explicitly
         allocated when registerBinding() is called.  (The C++ static objectinitialization fiasco.)
         */
        std::vector<BindComponentFunc>* gBindFuncs = nullptr;
    }

    void registerBinding(BindComponentFunc f)
    {
        if(Scripting::isRunning())
        {
            try
            {
                auto pymod = pybind11::module::import("falcor");
                Module m(pymod);
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
            if (!gBindFuncs) gBindFuncs = new std::vector<BindComponentFunc>();
            gBindFuncs->push_back(f);
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
        auto repr = [](const VecT& v) { return to_string(v); };
        auto vecStr = [](const VecT& v) {
            std::string vec = "[" + to_string(v[0]);
            for (int i = 1; i < v.length(); i++)
            {
                vec += ", " + to_string(v[i]);
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
        // Alias python's True/False to true/false
        m.attr("true") = true;
        m.attr("false") = false;

        // Bind glm vectors to Python
        addVecType<glm::vec2, float, float>(m, "vec2");
        addVecType<glm::vec3, float, float, float>(m, "vec3");
        addVecType<glm::vec4, float, float, float, float>(m, "vec4");
        addVecType<glm::uvec2, uint32_t, uint32_t>(m, "uvec2");
        addVecType<glm::uvec3, uint32_t, uint32_t, uint32_t>(m, "uvec3");
        addVecType<glm::uvec4, uint32_t, uint32_t, uint32_t, uint32_t>(m, "uvec4");
        addVecType<glm::ivec2, int32_t, int32_t>(m, "ivec2");
        addVecType<glm::ivec3, int32_t, int32_t, int32_t>(m, "ivec3");
        addVecType<glm::ivec4, int32_t, int32_t, int32_t, int32_t>(m, "ivec4");

        if (gBindFuncs)
        {
            ScriptBindings::Module fm(m);
            for (auto f : *gBindFuncs) f(fm);
        }
    }
}
