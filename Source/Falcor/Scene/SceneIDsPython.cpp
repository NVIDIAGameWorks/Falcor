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
#pragma once

#include "SceneIDs.h"
#include "Utils/Scripting/ScriptBindings.h"

#include <fmt/format.h>
#include <pybind11/pybind11.h>

namespace Falcor::scene1
{

#define FALCOR_BIND_SCENEID(typename)                \
    FALCOR_SCRIPT_BINDING(typename)                  \
    {                                                \
        using namespace pybind11::literals;          \
        namespace py = pybind11;                     \
                                                     \
        pybind11::class_<typename> id(m, #typename); \
        id.def(py::init<>());                        \
        id.def("isValid", &typename ::isValid);      \
        id.def("get", &typename ::get);              \
        id.def("__str__", [](const typename& v) { return fmt::format("{}", v); }); \
        id.def("__repr__", [](const typename& v) { return fmt::format(#typename "({})", v); }); \
    }


FALCOR_BIND_SCENEID(NodeID)
FALCOR_BIND_SCENEID(MeshID)
FALCOR_BIND_SCENEID(CurveID)
FALCOR_BIND_SCENEID(CurveOrMeshID)
FALCOR_BIND_SCENEID(SdfDescID)
FALCOR_BIND_SCENEID(SdfGridID)
FALCOR_BIND_SCENEID(MaterialID)
FALCOR_BIND_SCENEID(LightID)
FALCOR_BIND_SCENEID(CameraID)
FALCOR_BIND_SCENEID(VolumeID)
FALCOR_BIND_SCENEID(GlobalGeometryID)

#undef FALCOR_BIND_SCENEID

} // namespace Falcor::scene1
