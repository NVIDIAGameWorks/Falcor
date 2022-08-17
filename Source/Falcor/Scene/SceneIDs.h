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
#pragma once
#include "Utils/ObjectID.h"
#include <cstdint>

namespace Falcor
{
    enum class SceneObjectKind
    {
        kNode,        ///< NodeID, but also for MatrixID for animation.
        kMesh,        ///< MeshID, also curves that tesselate into triangle mesh.
        kCurve,       ///< CurveID
        kCurveOrMesh, ///< Used when the ID in curves is aliased based on tessellation mode.
        kSdfDesc,     ///< The user-facing ID.
        kSdfGrid,     ///< The internal ID, can get deduplicated.
        kMaterial,
        kLight,
        kCamera,
        kVolume,
        kGlobalGeometry, ///< The linearized global ID, current in order: mest, curve, sdf, custom. Not to be confused with geometryID in curves, which is "either Mesh or Curve, depending on tessellation mode".
    };


    using NodeID = ObjectID<SceneObjectKind, SceneObjectKind::kNode, uint32_t>;
    using MeshID = ObjectID<SceneObjectKind, SceneObjectKind::kMesh, uint32_t>;
    using CurveID = ObjectID<SceneObjectKind, SceneObjectKind::kCurve, uint32_t>;
    using CurveOrMeshID = ObjectID<SceneObjectKind, SceneObjectKind::kCurveOrMesh, uint32_t>;
    using SdfDescID = ObjectID<SceneObjectKind, SceneObjectKind::kSdfDesc, uint32_t>;
    using SdfGridID = ObjectID<SceneObjectKind, SceneObjectKind::kSdfGrid, uint32_t>;
    using MaterialID = ObjectID<SceneObjectKind, SceneObjectKind::kMaterial, uint32_t>;
    using LightID = ObjectID<SceneObjectKind, SceneObjectKind::kLight, uint32_t>;
    using CameraID = ObjectID<SceneObjectKind, SceneObjectKind::kCamera, uint32_t>;
    using VolumeID = ObjectID<SceneObjectKind, SceneObjectKind::kVolume, uint32_t>;
    using GlobalGeometryID = ObjectID<SceneObjectKind, SceneObjectKind::kGlobalGeometry, uint32_t>;
}
