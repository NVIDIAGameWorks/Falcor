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
#include "Subdivision.h"
#include "Core/Assert.h"
#include "Utils/Logger.h"

#include <unordered_map>

#include <pxr/imaging/hd/vertexAdjacency.h>
#include <pxr/imaging/hd/smoothNormals.h>

#include <opensubdiv/far/topologyDescriptor.h>
#include <opensubdiv/far/stencilTableFactory.h>
#include <opensubdiv/sdc/scheme.h>
#include <opensubdiv/sdc/options.h>


using namespace pxr;
using namespace OpenSubdiv;

namespace Falcor
{

namespace
{
struct SubdivVec3f
{
    GfVec3f v;
    void Clear(void* = nullptr)
    {
        v = {0.f, 0.f, 0.f};
    }

    void AddWithWeight(const SubdivVec3f& o, float weight)
    {
        v += o.v * weight;
    }
};

struct SubdivVec2f
{
    GfVec2f v;
    void Clear(void* = nullptr)
    {
        v = {0.f, 0.f};
    }

    void AddWithWeight(const SubdivVec2f& o, float weight)
    {
        v += o.v * weight;
    }
};
struct GfVec2fHash
{
    size_t operator()(const GfVec2f& v) const
    {
        // Simple hash function that multiplies the integer interpretation of each component by a prime and xors the results
        return (*reinterpret_cast<const uint32_t*>(&v[0]) * 7727ULL) ^ (*reinterpret_cast<const uint32_t*>(&v[1]) * 5521ULL);
    }
};

Sdc::Options::FVarLinearInterpolation getFaceVaryingLinearInterpolation(const UsdGeomMesh& geomMesh)
{
    TfToken interp = getAttribute(geomMesh.GetFaceVaryingLinearInterpolationAttr(), UsdGeomTokens->cornersPlus1);

    if (interp == UsdGeomTokens->none) return Sdc::Options::FVarLinearInterpolation::FVAR_LINEAR_NONE;
    else if (interp == UsdGeomTokens->cornersOnly) return Sdc::Options::FVarLinearInterpolation::FVAR_LINEAR_CORNERS_ONLY;
    else if (interp == UsdGeomTokens->cornersPlus1) return Sdc::Options::FVarLinearInterpolation::FVAR_LINEAR_CORNERS_PLUS1;
    else if (interp == UsdGeomTokens->cornersPlus2) return Sdc::Options::FVarLinearInterpolation::FVAR_LINEAR_CORNERS_PLUS2;
    else if (interp == UsdGeomTokens->boundaries) return Sdc::Options::FVarLinearInterpolation::FVAR_LINEAR_BOUNDARIES;
    else if (interp == UsdGeomTokens->all) return Sdc::Options::FVarLinearInterpolation::FVAR_LINEAR_ALL;

    logWarning("Unsupported face varying linear interpolation mode '{}' on '{}'.", interp.GetString(), geomMesh.GetPath().GetString());
    return Sdc::Options::FVarLinearInterpolation::FVAR_LINEAR_CORNERS_PLUS1;
}

Sdc::Options::VtxBoundaryInterpolation getVertexBoundaryInterpolation(const UsdGeomMesh& geomMesh)
{
    TfToken interp = getAttribute(geomMesh.GetInterpolateBoundaryAttr(), UsdGeomTokens->edgeAndCorner);

    if (interp == UsdGeomTokens->none) return Sdc::Options::VtxBoundaryInterpolation::VTX_BOUNDARY_NONE;
    else if (interp == UsdGeomTokens->edgeOnly) return Sdc::Options::VtxBoundaryInterpolation::VTX_BOUNDARY_EDGE_ONLY;
    else if (interp == UsdGeomTokens->edgeAndCorner) return Sdc::Options::VtxBoundaryInterpolation::VTX_BOUNDARY_EDGE_AND_CORNER;

    logWarning("Unsupported vertex boundary interpolation mode '{}' on '{}'.", interp.GetString(), geomMesh.GetPath().GetString());
    return Sdc::Options::VtxBoundaryInterpolation::VTX_BOUNDARY_EDGE_AND_CORNER;
}
} // anonymous namespace

bool refine(const UsdGeomMesh& geomMesh,
            const HdMeshTopology& topology,
            const uint32_t& maxLevel,
            const VtVec3fArray& basePoints,
            const VtVec2fArray& baseUVs,
            const TfToken uvFreq,
            HdMeshTopology& refinedTopology,
            VtVec3fArray& refinedPoints,
            VtVec3fArray& refinedNormals,
            VtVec2fArray& refinedUVs,
            std::unique_ptr<HdMeshUtil>& meshUtil)
{
    if (maxLevel == 0 || basePoints.size() == 0 || topology.GetNumPoints() == 0 || topology.GetNumFaces() == 0)
    {
        return false;
    }

    uint32_t uvInterpolationMode = 0;

    if (uvFreq == UsdGeomTokens->faceVarying) uvInterpolationMode = Far::StencilTableFactory::INTERPOLATE_FACE_VARYING;
    else if (uvFreq == UsdGeomTokens->varying) uvInterpolationMode = Far::StencilTableFactory::INTERPOLATE_VARYING;
    else if (uvFreq == UsdGeomTokens->vertex) uvInterpolationMode = Far::StencilTableFactory::INTERPOLATE_VERTEX;
    else
    {
        logWarning("Unsupported texture coordinate frequency: {}", uvFreq.GetString());
        return false;
    }

    OpenSubdiv::Sdc::SchemeType scheme = Sdc::SCHEME_CATMARK;
    TfToken usdScheme = topology.GetScheme();

    if (usdScheme == UsdGeomTokens->catmullClark)
    {
        scheme = Sdc::SCHEME_CATMARK;
    }
    else if (usdScheme == UsdGeomTokens->loop)
    {
        // Ensure that the input consists solely of triangles.
        VtIntArray courseIndices = topology.GetFaceVertexCounts();
        auto it = std::find_if(courseIndices.begin(), courseIndices.end(), [](int i) { return i != 3; });
        if (it != courseIndices.end())
        {
            logWarning("Cannot apply Loop subdivision to non-triangular base mesh '{}'.", geomMesh.GetPath().GetString());
            return false;
        }
        scheme = Sdc::SCHEME_LOOP;
    }
    else if (usdScheme == UsdGeomTokens->bilinear)
    {
        scheme = Sdc::SCHEME_BILINEAR;
    }
    else if (usdScheme == UsdGeomTokens->none)
    {
        return false;
    }
    else
    {
        logWarning("Unknown subdivision scheme: '{}'.", usdScheme.GetString());
        return false;
    }

    OpenSubdiv::Sdc::Options options;
    options.SetVtxBoundaryInterpolation(getVertexBoundaryInterpolation(geomMesh));
    options.SetFVarLinearInterpolation(getFaceVaryingLinearInterpolation(geomMesh));
    Far::TopologyRefinerFactory<Far::TopologyDescriptor>::Options refinerOptions(scheme, options);

    Far::TopologyDescriptor desc;
    desc.numVertices = topology.GetNumPoints();
    desc.numFaces = topology.GetNumFaces();
    desc.numVertsPerFace = topology.GetFaceVertexCounts().data();
    desc.vertIndicesPerFace  = topology.GetFaceVertexIndices().data();

    Far::TopologyDescriptor::FVarChannel channel;
    std::vector<Far::Index> uvIndices;
    VtVec2fArray indexedUVs;

    if (baseUVs.size() > 0 && uvFreq == UsdGeomTokens->faceVarying)
    {
        // Construct unique set of UVs and indices
        std::unordered_map<GfVec2f, uint32_t, GfVec2fHash> indexMap;
        for (auto& uv : baseUVs)
        {
            auto iter = indexMap.find(uv);
            if (iter == indexMap.end())
            {
                iter = indexMap.insert(std::make_pair(uv, indexedUVs.size())).first;
                indexedUVs.push_back(uv);
            }
            uvIndices.push_back(iter->second);
        }
        channel.numValues = indexedUVs.size();
        channel.valueIndices = uvIndices.data();
        desc.numFVarChannels = 1;
        desc.fvarChannels = &channel;
    }

    std::unique_ptr<Far::TopologyRefiner> refiner(Far::TopologyRefinerFactory<Far::TopologyDescriptor>::Create(desc, refinerOptions));
    refiner->RefineUniform(Far::TopologyRefiner::UniformOptions(maxLevel));

    // Construct a new HdMeshTopogy from the bottom-most refined level.
    uint32_t refinementLevel = refiner->GetMaxLevel();
    const Far::TopologyLevel bottomTopology = refiner->GetLevel(refinementLevel);
    uint32_t bottomFaceCount = bottomTopology.GetNumFaces();
    uint32_t bottomFaceVertexCount = bottomTopology.GetNumFaceVertices();
    uint32_t bottomFaceVaryingCount = bottomTopology.GetNumFVarValues(0);

    // Construct per-face vertex and indices arrays, used to construct refined HdMeshTopology.
    VtIntArray faceVertexCounts = VtIntArray(bottomFaceCount);
    VtIntArray faceVertexIndices = VtIntArray(bottomFaceVertexCount);

    uint32_t faceIdx = 0;
    for (uint32_t f = 0; f < bottomFaceCount; ++f)
    {
        Far::ConstIndexArray faceIndices = bottomTopology.GetFaceVertices(f);
        faceVertexCounts[f] = faceIndices.size();
        for (int i = 0; i < faceVertexCounts[f]; ++i)
        {
            faceVertexIndices[faceIdx++] = faceIndices[i];
        }
    }

    if (faceIdx != bottomFaceVertexCount)
    {
        logError("Face vertex count mismatch while refining '{}'", geomMesh.GetPath().GetString());
        return false;
    }

    // Construct an HdMeshTopology for the refined mesh.
    refinedTopology = HdMeshTopology(topology.GetScheme(), topology.GetOrientation(), faceVertexCounts, faceVertexIndices, refinementLevel);

    // Construct an HdMeshUtil for the refined topology.
    meshUtil = std::make_unique<HdMeshUtil>(&refinedTopology, geomMesh.GetPath());

    // Build point interpolation stencil table.
    Far::StencilTableFactory::Options stencilOptions;
    stencilOptions.interpolationMode = Far::StencilTableFactory::INTERPOLATE_VERTEX;
    stencilOptions.generateIntermediateLevels = false;
    stencilOptions.generateOffsets = false;

    std::unique_ptr<const Far::StencilTable> stencilTable(Far::StencilTableFactory::Create(*refiner, stencilOptions));

    // Compute refined vertex positions.
    refinedPoints.resize(stencilTable->GetNumStencils());
    stencilTable->UpdateValues(reinterpret_cast<const SubdivVec3f*>(basePoints.data()), reinterpret_cast<SubdivVec3f*>(refinedPoints.data()));

    logDebug("After refinement, {} faces, {} vertex indices, {} face varying values, {} points", bottomFaceCount, bottomFaceVertexCount, bottomFaceVaryingCount, refinedPoints.size());

    // Compute refined normals. Note that we ignore any authored normals, as per the USD spec.
    Hd_VertexAdjacency adjacency;
    adjacency.BuildAdjacencyTable(&refinedTopology);
    refinedNormals = Hd_SmoothNormals::ComputeSmoothNormals(&adjacency, (int)refinedPoints.size(), refinedPoints.cdata());

    // Compute refined texcoords, if required.
    if (baseUVs.size() > 0)
    {
        // Create a new stencil table, if required.
        if (uvInterpolationMode != stencilOptions.interpolationMode)
        {
            stencilOptions.interpolationMode = uvInterpolationMode;
            stencilTable.reset(Far::StencilTableFactory::Create(*refiner, stencilOptions));
        }

        refinedUVs.resize(stencilTable->GetNumStencils());
        const GfVec2f* uvData = (uvFreq == UsdGeomTokens->faceVarying ? indexedUVs.cdata() : baseUVs.cdata());
        stencilTable->UpdateValues(reinterpret_cast<const SubdivVec2f*>(uvData), reinterpret_cast<SubdivVec2f*>(refinedUVs.data()));

        if (uvFreq == UsdGeomTokens->faceVarying)
        {
            // Texcoord are face varying. Create flattened array of face-varying UVs, to match Falcor convention.
            VtVec2fArray tmpUVs = std::move(refinedUVs);
            refinedUVs.clear();
            for (uint32_t f = 0; f < bottomFaceCount; ++f)
            {
                const Far::ConstIndexArray uvs = bottomTopology.GetFaceFVarValues(f, 0);
                std::for_each(uvs.begin(), uvs.end(), [&](const auto& idx){refinedUVs.push_back(tmpUVs[idx]);});
            }
        }
    }

    return true;
}
}
