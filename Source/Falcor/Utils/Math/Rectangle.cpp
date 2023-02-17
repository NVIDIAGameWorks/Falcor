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
#include "Rectangle.h"
#include "Utils/Scripting/ScriptBindings.h"
#include <pybind11/operators.h>

namespace Falcor
{
FALCOR_SCRIPT_BINDING(Rectangle)
{
    using namespace pybind11::literals;

    pybind11::class_<Rectangle> uvTile(m, "Rectangle");

    uvTile.def(pybind11::init<>());
    uvTile.def(pybind11::init<const float2&>(), "p"_a);
    uvTile.def(pybind11::init<const float2&, const float2&>(), "pmin"_a, "pmax"_a);

    uvTile.def(
        "__repr__",
        [](const Rectangle& uvTile)
        {
            return "uvTile(minPoint=" + std::string(pybind11::repr(pybind11::cast(uvTile.minPoint))) +
                   ", maxPoint=" + std::string(pybind11::repr(pybind11::cast(uvTile.maxPoint))) + ")";
        }
    );
    uvTile.def(
        "__str__",
        [](const Rectangle& uvTile)
        {
            return "[" + std::string(pybind11::str(pybind11::cast(uvTile.minPoint))) + ", " +
                   std::string(pybind11::str(pybind11::cast(uvTile.maxPoint))) + "]";
        }
    );

    uvTile.def_readwrite("minPoint", &Rectangle::minPoint);
    uvTile.def_readwrite("maxPoint", &Rectangle::maxPoint);

    uvTile.def_property_readonly("valid", &Rectangle::valid);
    uvTile.def_property_readonly("center", &Rectangle::center);
    uvTile.def_property_readonly("extent", &Rectangle::extent);
    uvTile.def_property_readonly("area", &Rectangle::area);
    uvTile.def_property_readonly("radius", &Rectangle::radius);

    uvTile.def("invalidate", &Rectangle::invalidate);
    uvTile.def("include", pybind11::overload_cast<const float2&>(&Rectangle::include), "p"_a);
    uvTile.def("include", pybind11::overload_cast<const Rectangle&>(&Rectangle::include), "b"_a);
    uvTile.def("intersection", &Rectangle::intersection);

    uvTile.def(pybind11::self == pybind11::self);
    uvTile.def(pybind11::self != pybind11::self);
    uvTile.def(pybind11::self | pybind11::self);
    uvTile.def(pybind11::self |= pybind11::self);
    uvTile.def(pybind11::self & pybind11::self);
    uvTile.def(pybind11::self &= pybind11::self);
}
} // namespace Falcor
