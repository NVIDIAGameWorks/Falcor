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
#include "Tessellation.h"
#include "Core/Error.h"
#include "Utils/Logger.h"
#include "Utils/Math/FNVHash.h"
#include "IndexedVector.h"

#include <opensubdiv/far/topologyDescriptor.h>
#include <opensubdiv/far/stencilTableFactory.h>
#include <opensubdiv/sdc/options.h>
#include <opensubdiv/sdc/scheme.h>

#include <opensubdiv/bfr/refinerSurfaceFactory.h>
#include <opensubdiv/bfr/surface.h>
#include <opensubdiv/bfr/tessellation.h>

using namespace pxr;
using namespace OpenSubdiv;

namespace Falcor
{

namespace
{

struct GfVec2fHash
{
    size_t operator()(const GfVec2f& v) const
    {
        FNVHash64 hash;
        hash.insert(&v, sizeof(v));
        return hash.get();
    }
};

struct GfVec3fHash
{
    size_t operator()(const GfVec3f& v) const
    {
        FNVHash64 hash;
        hash.insert(&v, sizeof(v));
        return hash.get();
    }
};

/**
 * @brief Mesh construction helper class.
 *
 * Aggregates per-face (facet set) indices/positions/normals/uvs into indexed mesh data, sharing position data as possible.
 * Note that normals are assumed to be per-vertex.
 */
class MeshIndexer
{
public:
    MeshIndexer(const TfToken& uvInterp) : mUVInterp(uvInterp) {}

    // Add a face's worth of facets to the mesh. Assumes that the facets are triangles.
    void addFacets(
        const std::vector<int>& indices,
        const std::vector<float>& positions,
        const std::vector<float>& normals,
        const std::vector<float>& uvs
    )
    {
        FALCOR_ASSERT((indices.size() % 3) == 0);

        // Ensure attribute interp == "none" iff attribute data size == 0
        FALCOR_ASSERT((mUVInterp == UsdGeomTokens->none) == (uvs.size() == 0));

        const size_t positionCount = positions.size() / 3;

        // Table to map from given vertex index to mesh vertex index
        std::vector<int> indexMap;

        for (int i = 0, j = 0; i < positionCount; ++i, j += 3)
        {
            GfVec3f pos(positions[j + 0], positions[j + 1], positions[j + 2]);
            uint32_t idx;
            if (mPositionSet.append(pos, idx))
            {
                // If this was the first time this position was seen, also push normals, and uvs if they are per-vertex.
                mNormals.push_back(GfVec3f(normals[j + 0], normals[j + 1], normals[j + 2]));

                if (mUVInterp == UsdGeomTokens->vertex || mUVInterp == UsdGeomTokens->varying)
                {
                    mUVs.push_back(GfVec2f(uvs[2 * i + 0], uvs[2 * i + 1]));
                }
            }
            indexMap.push_back(idx);
        }

        for (const auto& idx : indices)
        {
            mIndices.push_back(indexMap[idx]);

            // If uv attributes are face varying, append to the current index.
            if (mUVInterp == UsdGeomTokens->faceVarying)
            {
                mUVs.push_back(GfVec2f(uvs[2 * idx + 0], uvs[2 * idx + 1]));
            }
        }

        if (mUVInterp == UsdGeomTokens->uniform)
        {
            mUVs.push_back(GfVec2f(uvs[0], uvs[1]));
        }
    }

    VtIntArray getIndices() { return mIndices; }
    VtVec3fArray getPositions() { return mPositions.size() > 0 ? mPositions : mPositionSet.getValues(); }
    VtVec2fArray getUVs() { return mUVs; }
    VtVec3fArray getNormals() { return mNormals; }

    TfToken getUVInterp() const { return mUVInterp; }

private:
    TfToken mUVInterp;
    IndexedVector<GfVec3f, int32_t, GfVec3fHash> mPositionSet;
    VtIntArray mIndices;
    VtVec3fArray mPositions;
    VtVec3fArray mNormals;
    VtVec2fArray mUVs;
};

Sdc::Options::FVarLinearInterpolation getFaceVaryingLinearInterpolation(const UsdGeomMesh& geomMesh)
{
    TfToken interp = getAttribute(geomMesh.GetFaceVaryingLinearInterpolationAttr(), UsdGeomTokens->cornersPlus1);

    if (interp == UsdGeomTokens->none)
        return Sdc::Options::FVarLinearInterpolation::FVAR_LINEAR_NONE;
    else if (interp == UsdGeomTokens->cornersOnly)
        return Sdc::Options::FVarLinearInterpolation::FVAR_LINEAR_CORNERS_ONLY;
    else if (interp == UsdGeomTokens->cornersPlus1)
        return Sdc::Options::FVarLinearInterpolation::FVAR_LINEAR_CORNERS_PLUS1;
    else if (interp == UsdGeomTokens->cornersPlus2)
        return Sdc::Options::FVarLinearInterpolation::FVAR_LINEAR_CORNERS_PLUS2;
    else if (interp == UsdGeomTokens->boundaries)
        return Sdc::Options::FVarLinearInterpolation::FVAR_LINEAR_BOUNDARIES;
    else if (interp == UsdGeomTokens->all)
        return Sdc::Options::FVarLinearInterpolation::FVAR_LINEAR_ALL;

    logWarning("Unsupported face varying linear interpolation mode '{}' on '{}'.", interp.GetString(), geomMesh.GetPath().GetString());
    return Sdc::Options::FVarLinearInterpolation::FVAR_LINEAR_CORNERS_PLUS1;
}

Sdc::Options::VtxBoundaryInterpolation getVertexBoundaryInterpolation(const UsdGeomMesh& geomMesh)
{
    TfToken interp = getAttribute(geomMesh.GetInterpolateBoundaryAttr(), UsdGeomTokens->edgeAndCorner);

    if (interp == UsdGeomTokens->none)
        return Sdc::Options::VtxBoundaryInterpolation::VTX_BOUNDARY_NONE;
    else if (interp == UsdGeomTokens->edgeOnly)
        return Sdc::Options::VtxBoundaryInterpolation::VTX_BOUNDARY_EDGE_ONLY;
    else if (interp == UsdGeomTokens->edgeAndCorner)
        return Sdc::Options::VtxBoundaryInterpolation::VTX_BOUNDARY_EDGE_AND_CORNER;

    logWarning("Unsupported vertex boundary interpolation mode '{}' on '{}'.", interp.GetString(), geomMesh.GetPath().GetString());
    return Sdc::Options::VtxBoundaryInterpolation::VTX_BOUNDARY_EDGE_AND_CORNER;
}

/**
 * Triangulate a mesh without applying any refinement.
 *
 * This is both faster than using the general-case tessellation path, and preserves ordering of vertices so that other constructs that
 * rely on same (e.g., joint indices) will remain compatible post-triangulation.
 *
 * We use simple fan triangulation, which is only guaranteed to produce correct results on convex faces.
 * Note that this is the same basic approach used by USD's HdMeshUtil::ComputeTriangleIndices().
 */
UsdMeshData triangulate(const pxr::UsdGeomMesh& geomMesh, const UsdMeshData& baseMesh, pxr::VtIntArray& coarseFaceIndices)
{
    const std::string& meshName = geomMesh.GetPath().GetString();
    const auto& faceIndices = baseMesh.topology.faceIndices;
    const auto& faceCounts = baseMesh.topology.faceCounts;
    const size_t faceCount = faceCounts.size();

    const bool leftHanded = baseMesh.topology.orient != UsdGeomTokens->rightHanded;

    VtVec3fArray outNormals;
    if (baseMesh.normalInterp == UsdGeomTokens->vertex || baseMesh.normalInterp == UsdGeomTokens->varying)
    {
        outNormals = baseMesh.normals;
    }

    // If there are no input normals, then as per the USD spec, we generate uniform face ("flat") normals.
    bool generateNormals = baseMesh.normals.size() == 0;

    VtVec2fArray outUVs;
    if (baseMesh.uvInterp == UsdGeomTokens->vertex || baseMesh.uvInterp == UsdGeomTokens->varying)
    {
        outUVs = baseMesh.uvs;
    }

    // Sort the hole indices so that determining if a given face is a hole is a constant-time operation.
    VtIntArray sortedHoleIndices = baseMesh.topology.holeIndices;
    std::sort(sortedHoleIndices.begin(), sortedHoleIndices.end());
    const uint32_t holeFaceCount = sortedHoleIndices.size();

    VtIntArray outFaceIndices;

    int next[2] = {1, 2};
    if (leftHanded)
        std::swap(next[0], next[1]);

    // Triangulate each face, f
    // vertIdx is the offset into faceIndices to the first vertex index for the current face
    for (uint32_t f = 0, vertIdx = 0, holeIdx = 0; f < faceCount; vertIdx += faceCounts[f], ++f)
    {
        uint32_t vertexCount = faceCounts[f];

        if (holeIdx < holeFaceCount && sortedHoleIndices[holeIdx] == f)
        {
            // This face is a hole face; skip it.
            ++holeIdx;
            continue;
        }
        else if (vertexCount < 3)
        {
            // Skip degenerate faces
            continue;
        }

        GfVec3f flatNormal = {};
        if (generateNormals)
        {
            // Generate a uniform (per-original-face) normal to use for each triangle we generate.
            const GfVec3f& v0 = baseMesh.points[faceIndices[vertIdx]];
            for (uint32_t j = 0; j < vertexCount - 2; ++j)
            {
                const GfVec3f& v1 = baseMesh.points[faceIndices[vertIdx + j + next[0]]];
                const GfVec3f& v2 = baseMesh.points[faceIndices[vertIdx + j + next[1]]];
                flatNormal += GfCross(v1 - v0, v2 - v0);
            }
            GfNormalize(&flatNormal);
        }

        for (uint32_t v = 0; v < vertexCount - 2; ++v)
        {
            // Append a triplet of face indices, thereby adding a new triangle to the output.
            // As we do so, append copies of any uniform or face-varying attributes to their respective
            // outputs, since they are not indexed.
            // In the case of vertex and varying attributes, we can simply re-use the base mesh values, since
            // they are indexed along with the corresponding vertex.
            outFaceIndices.push_back(faceIndices[vertIdx]);
            outFaceIndices.push_back(faceIndices[vertIdx + v + next[0]]);
            outFaceIndices.push_back(faceIndices[vertIdx + v + next[1]]);

            if (baseMesh.normalInterp == UsdGeomTokens->faceVarying)
            {
                outNormals.push_back(baseMesh.normals[vertIdx]);
                outNormals.push_back(baseMesh.normals[vertIdx + v + next[0]]);
                outNormals.push_back(baseMesh.normals[vertIdx + v + next[1]]);
            }

            if (baseMesh.uvInterp == UsdGeomTokens->faceVarying)
            {
                outUVs.push_back(baseMesh.uvs[vertIdx]);
                outUVs.push_back(baseMesh.uvs[vertIdx + v + next[0]]);
                outUVs.push_back(baseMesh.uvs[vertIdx + v + next[1]]);
            }

            if (generateNormals)
            {
                // Append the generated flat normal to the output
                outNormals.push_back(flatNormal);
            }
            else if (baseMesh.normalInterp == UsdGeomTokens->uniform)
            {
                // Copy the appropriate input uniform normal to the output
                outNormals.push_back(baseMesh.normals[f]);
            }

            if (baseMesh.uvInterp == UsdGeomTokens->uniform)
            {
                // Copy the appropriate input uniform uv to the output
                outUVs.push_back(baseMesh.uvs[f]);
            }

            coarseFaceIndices.push_back(f);
        }
    }

    UsdMeshData tessellatedMesh;
    tessellatedMesh.topology.scheme = UsdGeomTokens->none;
    tessellatedMesh.topology.orient = baseMesh.topology.orient;
    tessellatedMesh.topology.faceIndices = std::move(outFaceIndices);
    tessellatedMesh.topology.faceCounts = VtIntArray(tessellatedMesh.topology.faceIndices.size() / 3, 3);
    tessellatedMesh.normalInterp = generateNormals ? UsdGeomTokens->uniform : baseMesh.normalInterp;
    tessellatedMesh.uvInterp = baseMesh.uvInterp;
    tessellatedMesh.points = baseMesh.points;
    tessellatedMesh.normals = std::move(outNormals);
    tessellatedMesh.uvs = std::move(outUVs);
    return tessellatedMesh;
}
} // anonymous namespace

/**
 * Tessellate a UsdGeomMesh, generating uv and normal attributes as required.
 *
 * If refinement (subdivision) is to be applied, OpenSubdiv is used to evalute the surface and related attributes.
 * If no refinement is to be applied, we use a simple vertex-order-perserving triangulation scheme instead.
 */
UsdMeshData tessellate(
    const pxr::UsdGeomMesh& geomMesh,
    const UsdMeshData& baseMesh,
    uint32_t maxRefinementLevel,
    pxr::VtIntArray& coarseFaceIndices
)
{
    if (baseMesh.points.size() == 0 || baseMesh.topology.getNumFaces() == 0 || baseMesh.topology.faceIndices.size() == 0)
    {
        return UsdMeshData();
    }

    if (baseMesh.topology.scheme == UsdGeomTokens->none || maxRefinementLevel == 0)
    {
        return triangulate(geomMesh, baseMesh, coarseFaceIndices);
    }

    typedef Bfr::RefinerSurfaceFactory<> SurfaceFactory;
    typedef Bfr::Surface<float> Surface;
    typedef Bfr::SurfaceFactoryMeshAdapter::FVarID FVarID;

    OpenSubdiv::Sdc::SchemeType scheme = Sdc::SCHEME_CATMARK;
    TfToken usdScheme = baseMesh.topology.scheme;

    uint32_t tessellationRate = maxRefinementLevel + 1;

    if (usdScheme == UsdGeomTokens->catmullClark)
    {
        scheme = Sdc::SCHEME_CATMARK;
    }
    else if (usdScheme == UsdGeomTokens->loop)
    {
        // Ensure that the input consists solely of triangles.
        auto it = std::find_if(baseMesh.topology.faceCounts.begin(), baseMesh.topology.faceCounts.end(), [](int i) { return i != 3; });
        if (it != baseMesh.topology.faceCounts.end())
        {
            logWarning(
                "Cannot apply Loop subdivision to non-triangular mesh '{}'. Unrefined mesh will be used.", geomMesh.GetPath().GetString()
            );
            return triangulate(geomMesh, baseMesh, coarseFaceIndices);
        }
        scheme = Sdc::SCHEME_LOOP;
    }
    else if (usdScheme == UsdGeomTokens->bilinear)
    {
        scheme = Sdc::SCHEME_BILINEAR;
    }
    else
    {
        logWarning(
            "Unknown subdivision scheme: '{}' on mesh '{}'. Unrefined mesh will be used.",
            usdScheme.GetString(),
            geomMesh.GetPath().GetString()
        );
        return triangulate(geomMesh, baseMesh, coarseFaceIndices);
    }

    // Generate per-vertex normals on refined meshes, as per the USD spec.
    TfToken normalInterp = UsdGeomTokens->vertex;

    if (baseMesh.normals.size() > 0)
    {
        logWarning("Ignoring authored normals on subdivided mesh '{}'.", geomMesh.GetPath().GetString());
    }

    OpenSubdiv::Sdc::Options options;
    options.SetVtxBoundaryInterpolation(getVertexBoundaryInterpolation(geomMesh));
    options.SetFVarLinearInterpolation(getFaceVaryingLinearInterpolation(geomMesh));
    Far::TopologyRefinerFactory<Far::TopologyDescriptor>::Options refinerOptions(scheme, options);

    Far::TopologyDescriptor::FVarChannel channels;

    Far::TopologyDescriptor desc = {};
    desc.numVertices = baseMesh.topology.faceIndices.size();
    desc.numFaces = baseMesh.topology.getNumFaces();
    desc.numVertsPerFace = (const int*)baseMesh.topology.faceCounts.data();
    desc.vertIndicesPerFace = (Far::Index*)baseMesh.topology.faceIndices.data();
    desc.fvarChannels = &channels; // Channels will be initialized below if necessary.

    std::vector<Far::Index> uvIndices;
    VtVec2fArray indexedUVs;

    float const* uvData = (float*)baseMesh.uvs.data();
    TfToken uvInterp = uvData != nullptr ? baseMesh.uvInterp : UsdGeomTokens->none;

    IndexedVector<GfVec2f, Far::Index, GfVec2fHash> indexedUVSet;

    if (baseMesh.uvs.size() > 0 && uvInterp == UsdGeomTokens->faceVarying)
    {
        // OpenSubdiv expects indexed face-varying UVs, whereas ours are unindexed.
        // Construct unique set of UVs and indices.
        for (auto& uv : baseMesh.uvs)
        {
            indexedUVSet.append(uv);
        }
        uvData = (float*)indexedUVSet.getValues().data();
        channels.numValues = indexedUVSet.getValues().size();
        channels.valueIndices = indexedUVSet.getIndices().data();
        ++desc.numFVarChannels;
    }

    std::unique_ptr<Far::TopologyRefiner> refiner(Far::TopologyRefinerFactory<Far::TopologyDescriptor>::Create(desc, refinerOptions));

    SurfaceFactory::Options surfaceOptions;
    SurfaceFactory surfaceFactory(*refiner, surfaceOptions);

    Surface vertexSurface;
    Surface varyingSurface;
    Surface fvarSurface;

    std::vector<float> facePatchPoints;
    std::vector<float> outCoords;
    std::vector<float> outPos;
    std::vector<float> outDu, outDv;
    std::vector<float> outNormals;
    std::vector<float> outUV;
    std::vector<int> outFacets;

    Bfr::Tessellation::Options tessOptions;
    // Facet size 3 => triangulate
    tessOptions.SetFacetSize(3);

    bool leftHanded = baseMesh.topology.orient == UsdGeomTokens->leftHanded;

    // Note that normals on refined meshes are always per-vertex, and generated as part of
    // the subdivision process, as per the USD spec. As such, any authored normals are ignored.
    // Further, we do not need to handle e.g., face-varying or uniform normals here.

    MeshIndexer meshIndexer(uvInterp);

    const uint32_t faceCount = surfaceFactory.GetNumFaces();

    coarseFaceIndices.clear();

    bool createVaryingSurf = false;

    Surface* uvSurface = nullptr;
    if (uvInterp == UsdGeomTokens->faceVarying)
        uvSurface = &fvarSurface;
    else if (uvInterp == UsdGeomTokens->vertex)
        uvSurface = &vertexSurface;
    else if (uvInterp == UsdGeomTokens->varying)
    {
        uvSurface = &varyingSurface;
        createVaryingSurf = true;
    }

    // Face-varying UVs always have ID of 0. It's safe to set this, and to create the face-varying surface,
    // even if UVs aren't provided.
    FVarID fvarID = 0;

    for (uint32_t f = 0; f < faceCount; ++f)
    {
        surfaceFactory.InitSurfaces(
            f, &vertexSurface, &fvarSurface, &fvarID, desc.numFVarChannels, createVaryingSurf ? &varyingSurface : nullptr
        );
        if (!vertexSurface.IsValid())
            continue;
        // Fall back to using vertex surface if the uv surface is invalid for whatever reason.
        if (uvSurface && !uvSurface->IsValid())
            uvSurface = &vertexSurface;

        Bfr::Tessellation tessPattern(vertexSurface.GetParameterization(), tessellationRate, tessOptions);

        const int outCoordCount = tessPattern.GetNumCoords();

        outCoords.resize(outCoordCount * 2);
        tessPattern.GetCoords(outCoords.data());

        outNormals.resize(outCoordCount * 3);
        outPos.resize(outCoordCount * 3);
        outDu.resize(outCoordCount * 3);
        outDv.resize(outCoordCount * 3);

        if (uvSurface)
        {
            const int pointSize = 2;
            facePatchPoints.resize(uvSurface->GetNumPatchPoints() * pointSize);
            outUV.resize(outCoordCount * pointSize);
            uvSurface->PreparePatchPoints(uvData, pointSize, facePatchPoints.data(), pointSize);
            for (int i = 0, j = 0; i < outCoordCount; ++i, j += pointSize)
            {
                uvSurface->Evaluate(&outCoords[i * 2], facePatchPoints.data(), pointSize, &outUV[j]);
            }
        }
        else if (uvInterp == UsdGeomTokens->uniform)
        {
            outUV.resize(2);
            outUV[0] = baseMesh.uvs[f][0];
            outUV[1] = baseMesh.uvs[f][1];
        }

        const int pointSize = 3;
        facePatchPoints.resize(vertexSurface.GetNumPatchPoints() * pointSize);
        vertexSurface.PreparePatchPoints((float*)baseMesh.points.data(), pointSize, facePatchPoints.data(), pointSize);
        {
            float3 du;
            float3 dv;

            // Compute positions and normals
            for (int i = 0, j = 0; i < outCoordCount; ++i, j += pointSize)
            {
                vertexSurface.Evaluate(
                    &outCoords[i * 2],
                    facePatchPoints.data(),
                    pointSize,
                    &outPos[j],
                    reinterpret_cast<float*>(&du),
                    reinterpret_cast<float*>(&dv)
                );
                // Use the partials to construct a normal vector.
                float3 normal = normalize(cross(du, dv));
                if (leftHanded)
                    normal = -normal;
                outNormals[j + 0] = normal.x;
                outNormals[j + 1] = normal.y;
                outNormals[j + 2] = normal.z;
            }
        }

        const int facetCount = tessPattern.GetNumFacets();
        outFacets.resize(facetCount * 3);
        tessPattern.GetFacets(outFacets.data());

        meshIndexer.addFacets(outFacets, outPos, outNormals, outUV);

        // Append the index of each facet's originating coarse face, f.
        for (int i = 0; i < facetCount; ++i)
            coarseFaceIndices.push_back(f);
    }

    UsdMeshData tessellatedMesh;
    tessellatedMesh.topology.scheme = UsdGeomTokens->none;
    tessellatedMesh.topology.orient = baseMesh.topology.orient;
    tessellatedMesh.topology.faceIndices = meshIndexer.getIndices();
    tessellatedMesh.topology.faceCounts = VtIntArray(tessellatedMesh.topology.faceIndices.size() / 3, 3);
    tessellatedMesh.normalInterp = UsdGeomTokens->vertex;
    tessellatedMesh.uvInterp = meshIndexer.getUVInterp();
    tessellatedMesh.points = meshIndexer.getPositions();
    tessellatedMesh.normals = meshIndexer.getNormals();
    tessellatedMesh.uvs = meshIndexer.getUVs();
    return tessellatedMesh;
}
} // namespace Falcor
