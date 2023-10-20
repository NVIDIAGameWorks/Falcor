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
#include "USDHelpers.h"
#include "USDUtils.h"
#include "Scene/SceneBuilder.h"

BEGIN_DISABLE_USD_WARNINGS
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec3d.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/usd/usd/object.h>
#include <pxr/usd/usd/attribute.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/usd/usdSkel/tokens.h>
#include <pxr/usd/usdGeom/basisCurves.h>
#include <pxr/usd/usdGeom/imageable.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
END_DISABLE_USD_WARNINGS

#include <string>
#include <functional>

using namespace pxr;

namespace Falcor
{
inline SceneBuilder::Node makeNode(const std::string& name, NodeID parentId = NodeID::Invalid())
{
    return SceneBuilder::Node{name, float4x4::identity(), float4x4::identity(), float4x4::identity(), parentId};
}

inline SceneBuilder::Node makeNode(
    const std::string& name,
    const float4x4& xform,
    const float4x4& bindTransform,
    NodeID parentId = NodeID::Invalid()
)
{
    return SceneBuilder::Node{name, xform, bindTransform, float4x4::identity(), parentId};
}

using AttributeFrequency = SceneBuilder::Mesh::AttributeFrequency;

inline size_t computeElementCount(AttributeFrequency freq, size_t faceCount, size_t vertexCount)
{
    if (freq == AttributeFrequency::Constant)
    {
        return 1;
    }
    else if (freq == AttributeFrequency::Uniform)
    {
        return faceCount;
    }
    else if (freq == AttributeFrequency::Vertex)
    {
        return vertexCount;
    }
    else if (freq == AttributeFrequency::FaceVarying)
    {
        return 3 * faceCount;
    }
    else
    {
        logError("Unsupported primvar interpolation mode {}.", (uint32_t)freq);
        return 0;
    }
}

// Compute the count of per-face elements, based on interpolation type
inline size_t computePerFaceElementCount(AttributeFrequency freq, size_t faceCount)
{
    if (freq == AttributeFrequency::Uniform)
    {
        return faceCount;
    }
    else if (freq == AttributeFrequency::FaceVarying)
    {
        return 3 * faceCount;
    }
    // Everything else is indexed (vertex, varying), or constant
    return 0;
}

inline AttributeFrequency convertInterpolation(const TfToken& mode)
{
    if (mode == UsdGeomTokens->constant)
    {
        return AttributeFrequency::Constant;
    }
    else if (mode == UsdGeomTokens->uniform)
    {
        return AttributeFrequency::Uniform;
    }
    else if (mode == UsdGeomTokens->vertex || mode == UsdGeomTokens->varying)
    {
        // For our purposes, vertex and varying are synonymous.
        return AttributeFrequency::Vertex;
    }
    else if (mode == UsdGeomTokens->faceVarying)
    {
        return AttributeFrequency::FaceVarying;
    }
    else
    {
        logError("Unknown vertex interpolation mode '{}'.", mode.GetString());
        return AttributeFrequency::None;
    }
}
} // namespace Falcor
