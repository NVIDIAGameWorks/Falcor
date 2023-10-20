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
#include "Core/Error.h"

#include "USDUtils/USDUtils.h"
#include "USDUtils/USDHelpers.h"

BEGIN_DISABLE_USD_WARNINGS
#include <pxr/usd/usdGeom/mesh.h>
END_DISABLE_USD_WARNINGS

namespace Falcor
{

/**
 * Mesh topology information.
 */
struct MeshTopology
{
    MeshTopology() {}
    MeshTopology(pxr::TfToken scheme, pxr::TfToken orient, VtIntArray& faceCounts, VtIntArray& faceIndices)
        : scheme(scheme), orient(orient), faceCounts(faceCounts), faceIndices(faceIndices)
    {}
    pxr::TfToken scheme;         ///< Subdivision scheme, "none", "catmullClark", etc.
    pxr::TfToken orient;         ///< Orientation, nominally "leftHanded" or "rightHanded"
    pxr::VtIntArray faceCounts;  ///< Per-face number of vertices.
    pxr::VtIntArray faceIndices; ///< Per-face-vertex indices.
    pxr::VtIntArray holeIndices; ///< Indices of hole faces (sorted, per USD spec).

    uint32_t getNumFaces() const { return faceCounts.size(); }

    VtVec3iArray getTriangleIndices() const
    {
        FALCOR_ASSERT((faceIndices.size() % 3) == 0);

        VtVec3iArray ret;
        for (uint32_t i = 0; i < faceIndices.size(); i += 3)
        {
            ret.push_back(GfVec3i(faceIndices[i + 0], faceIndices[i + 1], faceIndices[i + 2]));
        }
        return ret;
    }
};

/**
 * A Basic mesh, as represented using USD datatypes.
 */
struct UsdMeshData
{
    MeshTopology topology;     ///< Topology
    pxr::VtVec3fArray points;  ///< Vertex positions
    pxr::VtVec3fArray normals; ///< Shading normals
    pxr::VtVec2fArray uvs;     ///< Texture coordinates
    pxr::TfToken normalInterp; ///< Normal interpolation mode (none, vertex, varying, faceVarying)
    pxr::TfToken uvInterp;     ///< Texture coordinate interpolatoin mode (none, vertex, varying, faceVarying)
};

/**
 * @brief Tessellate a UsdMeshData into triangles
 *
 * @param[in] geomMesh UsdGeomMesh being tessellated; used to extract subdivision and tessellation attributes.
 * @param[in] baseMesh Base mesh to tessellate.
 * @param[in] maxRefinementLevel Maximum subdivision refinement level. Zero indicates no subdivision.
 * @param[out] coarseFaceIndices Index of base face from which each output triangle derives.
 * @return UsdMeshData containing tessellated results; points will be zero-length on failure.
 */
UsdMeshData tessellate(
    const pxr::UsdGeomMesh& geomMesh,
    const UsdMeshData& baseMesh,
    uint32_t maxRefinementLevel,
    pxr::VtIntArray& coarseFaceIndices
);
} // namespace Falcor
