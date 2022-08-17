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
#include "AABB.h"
#include "Utils/Scripting/ScriptBindings.h"
#include <pybind11/operators.h>

namespace Falcor
{
    FALCOR_SCRIPT_BINDING(AABB)
    {
        using namespace pybind11::literals;

        pybind11::class_<AABB> aabb(m, "AABB");

        aabb.def(pybind11::init<>());
        aabb.def(pybind11::init<const float3&>(), "p"_a);
        aabb.def(pybind11::init<const float3&, const float3&>(), "pmin"_a, "pmax"_a);

        aabb.def("__repr__", [] (const AABB& aabb) {
            return
                "AABB(minPoint=" +
                std::string(pybind11::repr(pybind11::cast(aabb.minPoint))) +
                ", maxPoint=" +
                std::string(pybind11::repr(pybind11::cast(aabb.maxPoint))) +
                ")";
        });
        aabb.def("__str__", [] (const AABB& aabb) {
            return
                "[" +
                std::string(pybind11::str(pybind11::cast(aabb.minPoint))) +
                ", " +
                std::string(pybind11::str(pybind11::cast(aabb.maxPoint))) +
                "]";
        });

        aabb.def_readwrite("minPoint", &AABB::minPoint);
        aabb.def_readwrite("maxPoint", &AABB::maxPoint);

        aabb.def_property_readonly("valid", &AABB::valid);
        aabb.def_property_readonly("center", &AABB::center);
        aabb.def_property_readonly("extent", &AABB::extent);
        aabb.def_property_readonly("area", &AABB::area);
        aabb.def_property_readonly("volume", &AABB::volume);
        aabb.def_property_readonly("radius", &AABB::radius);

        aabb.def("invalidate", &AABB::invalidate);
        aabb.def("include", pybind11::overload_cast<const float3&>(&AABB::include), "p"_a);
        aabb.def("include", pybind11::overload_cast<const AABB&>(&AABB::include), "b"_a);
        aabb.def("intersection", &AABB::intersection);

        aabb.def(pybind11::self == pybind11::self);
        aabb.def(pybind11::self != pybind11::self);
        aabb.def(pybind11::self | pybind11::self);
        aabb.def(pybind11::self |= pybind11::self);
        aabb.def(pybind11::self & pybind11::self);
        aabb.def(pybind11::self &= pybind11::self);
    }
}
