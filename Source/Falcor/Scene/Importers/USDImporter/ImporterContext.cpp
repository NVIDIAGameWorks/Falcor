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
#include "ImporterContext.h"
#include "USDHelpers.h"
#include "Core/API/Device.h"
#include "Utils/NumericRange.h"
#include "Scene/Importer.h"
#include "Scene/Curves/CurveConfig.h"
#include "Scene/Material/HairMaterial.h"
#include "Scene/Material/StandardMaterial.h"
#include "Utils/Settings.h"

#include <glm/gtx/matrix_decompose.hpp>

BEGIN_DISABLE_USD_WARNINGS
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/basisCurves.h>
#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usdGeom/primvar.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usd/usdGeom/xformCache.h>
#include <pxr/usd/usdSkel/skeleton.h>
#include <pxr/usd/usdSkel/animation.h>
#include <pxr/usd/usdSkel/root.h>
#include <pxr/usd/usdSkel/binding.h>
#include <pxr/usd/usdSkel/bindingAPI.h>
#include <pxr/usd/usdSkel/tokens.h>
#include <pxr/usd/usdSkel/skeletonQuery.h>
#include <pxr/usd/usdSkel/topology.h>
#include <pxr/usd/usdSkel/animQuery.h>
#include <pxr/imaging/hd/meshTopology.h>
#include <pxr/imaging/hd/meshUtil.h>
#include <pxr/imaging/hd/vertexAdjacency.h>
#include <pxr/imaging/hd/flatNormals.h>
#include <pxr/imaging/hd/smoothNormals.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/usd/usdGeom/scope.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdLux/distantLight.h>
#include <pxr/usd/usdLux/domeLight.h>
#include <pxr/usd/usdLux/rectLight.h>
#include <pxr/usd/usdLux/sphereLight.h>
#include <pxr/usd/usdLux/diskLight.h>
#include <pxr/usd/usdLux/blackbody.h>
END_DISABLE_USD_WARNINGS

namespace Falcor
{
    namespace
    {
        const bool kLoadMeshVertexAnimations = true;

        // Subdivide each bspline curve segment into a single linear swept sphere segments (could be more if memory/perf allows).
        uint32_t kCurveSubdivPerSegment = 1;
        // Skip some hair strands, if necessary for memory/pref reasons.
        uint32_t kCurveKeepOneEveryXStrands = 1;
        // Skip some hair vertices, if necessary for memory/perf reasons.
        uint32_t kCurveKeepOneEveryXVerticesPerStrand = 1;

        // Default curve material parameters.
        const float kDefaultCurveIOR = 1.55f;
        const float kDefaultCurveLongitudinalRoughness = 0.125f;
        const float kDefaultCurveAzimuthalRoughness = 0.3f;
        const float kDefaultScaleAngleDegree = 1.f;

        // Data required to instantiate a UsdGeomSubset
        struct GeomSubset
        {
            uint32_t triIdx;                ///< Starting offset into point index array
            uint32_t triCount;              ///< Triangle count
            size_t normalIdx;               ///< Starting offset into normal array
            size_t texCrdsIdx;              ///< Starting offset into texture coodinate array
            std::string id;                 ///< Name; equal to GeomSubset name, or name of UsdMesh if no subsets or if the catch-all subset
            UsdShadeMaterial material;      ///< Bound material
        };

        // MeshGeomData represents mesh data using a single array for each of points, point indices, normals, and optionally texture coordinates.
        // Points and per-vertex primvars are indexed, while per-face (uniform/varying/facevarying) primvars are flattened.
        // The indices and any per-face primvar data for a subset are stored contiguously in the respective arrays, with
        // the starting index into each stored with each GeomSubset.
        struct MeshGeomData
        {
            std::vector<uint32_t> triangulatedIndices;              // Triangle point indices
            std::vector<float3> points;                             // Vertex positions (always indexed)
            std::vector<float3> normals;                            // Normals; indexed iff normalInterp is vertex or varying
            std::vector<float4> tangents;                           // Tangents; optional and always indexed
            std::vector<float2> texCrds;                            // Texture coordinates, indexed iff texCrdsInterp is vertex or varying
            std::vector<float> curveRadii;                          // Curve widths (always indexed; available if mesh was generated for a curve tessellated into triangles)
            std::vector<uint4> jointIndices;                        // Bone indices
            std::vector<float4> jointWeights;                       // Bone weights corresponding to the bones referenced in jointIndices
            std::vector<GeomSubset> geomSubsets;                    // Geometry subset data; one entry for entire mesh if no subsets are present
            AttributeFrequency normalInterp;                        // Normal interpolation mode
            AttributeFrequency texCrdsInterp;                       // Texture coordinate interpolation mode
            size_t numReferencedPoints = 0;                         // Number of elements of points that are referenced by the point indices.
        };

        // CurveGeomData represents curve data using a single array for each of points, radius, tangents, normals and optionally texture coordinates.
        // All are indexed per vertex.
        struct CurveGeomData
        {
            uint32_t degree;                                        // Polynomial degree of curve; linear (1) by default
            std::vector<uint32_t> indices;                          // First point indices of curve segments
            std::vector<float3> points;                             // Vertex positions
            std::vector<float> radius;                              // Radius of spheres at curve ends
            std::vector<float2> texCrds;                            // Texture coordinates
            std::string id;                                         // Name; equal to name of UsdGeomBasisCurves
            UsdShadeMaterial material;                              // Bound material
        };

        // Shuffle triangle data into GeomSubset order.
        // N specifies the number of data values per face; 1 corresponds to uniform, 3 corresponds to faceVarying
        template <size_t N, class T>
        void remapGeomSubsetData(T& inData, const VtIntArray& faceMap, VtIntArray subsetOffset, const VtIntArray& primitiveParams)
        {
            // Note that subsetOffset is passed by value; we modify the local copy as part of bookkeeping
            FALCOR_ASSERT(N == 1 || N == 3);

            T outData(inData.size());

            for (size_t i = 0; i < inData.size() / N; ++i)
            {
                int coarseFaceParam = primitiveParams[i];
                int faceIdx = HdMeshUtil::DecodeFaceIndexFromCoarseFaceParam(coarseFaceParam);
                int geomSubset = faceMap[faceIdx];
                // Check for unassigned faces to ensure we don't fall over in the face of invalid input
                if (geomSubset >= 0)
                {
                    size_t offset = subsetOffset[geomSubset];
                    for (size_t j = 0; j < N; ++j)
                    {
                        outData[offset * N + j] = inData[i * N + j];
                    }
                    ++subsetOffset[geomSubset];
                }
            }
            inData = std::move(outData);
        }

        template <class T>
        void remapGeomSubsetData(T& inData, AttributeFrequency freq, const VtIntArray& faceMap, VtIntArray& subsetOffset, const VtIntArray& primitiveParams)
        {
            // Reshuffle data into subset order, if necessary (i.e., if the count depends on the number of faces)
            if (freq == AttributeFrequency::FaceVarying)
            {
                remapGeomSubsetData<3>(inData, faceMap, subsetOffset, primitiveParams);
            }
            else if (freq == AttributeFrequency::Uniform)
            {
                remapGeomSubsetData<1>(inData, faceMap, subsetOffset, primitiveParams);
            }
        }

        // Generate triangulated versions of any data items whose count depends on the number of faces.
        // Safe to call for any AttributeFrequency, as non-face-varying data will be passed through unchanged.
        template <class T>
        void convertTriangulatedFaceData(T& inData, AttributeFrequency freq, HdType hdType, size_t faceCount, const VtIntArray& primitiveParams, HdMeshUtil& meshUtil)
        {
            // If the number of data items depends on the number of faces, generate triangulated versions.
            // Otherwise, don't modify inData.
            if (freq == AttributeFrequency::FaceVarying)
            {
                VtValue triangulatedData;
                meshUtil.ComputeTriangulatedFaceVaryingPrimvar((const void*)inData.cdata(), (int)inData.size(), hdType, &triangulatedData);
                inData = triangulatedData.Get<T>();
            }
            else if (freq == AttributeFrequency::Uniform)
            {
                T triangulatedData(faceCount);
                for (size_t i = 0; i < faceCount; ++i)
                {
                    int coarseFaceParam = primitiveParams[i];
                    int coarseFaceIdx = HdMeshUtil::DecodeFaceIndexFromCoarseFaceParam(coarseFaceParam);
                    triangulatedData[i] = inData[coarseFaceIdx];
                }
                inData = std::move(triangulatedData);
            }
        }

        /** Remap a mesh's joint indices to correctly refer to its skeleton bones.
            A mesh may reference a subset of bones in a skeleton. In these cases, we must translate the local indices to be skeleton-based.
            At the same time an offset is applied based on where the skeleton was added to the SceneBuilder scene graph.
        */
        void remapMeshJointIndices(ImporterContext& ctx, const Skeleton::SubSkeleton& subskel, const UsdPrim& prim, VtIntArray& meshJointIndices)
        {
            auto jointMapIt = subskel.skinnedMeshes.find(prim);
            FALCOR_ASSERT(jointMapIt != subskel.skinnedMeshes.end());
            for (size_t i = 0; i < meshJointIndices.size(); i++)
            {
                // If there's no mapping table, index doesn't need to be remapped
                uint32_t skeletonBasedIndex = jointMapIt->second.empty() ? meshJointIndices[i] : jointMapIt->second[meshJointIndices[i]];

                // Add offset based on where skeleton was inserted into scene
                meshJointIndices[i] = subskel.nodeOffset + skeletonBasedIndex;
            }
        }

        // Convert a UsdGeomMesh, and any GeomSubsets, into a MeshGeomData
        bool convertMeshGeomData(const UsdGeomMesh& usdMesh, const UsdTimeCode& timeCode, ImporterContext& ctx, MeshGeomData& geomOut)
        {
            std::string meshName = usdMesh.GetPath().GetString();

            UsdGeomPrimvarsAPI primvarApi(usdMesh);

            // Gather mesh metadata
            TfToken scheme = getAttribute(usdMesh.GetSubdivisionSchemeAttr(), UsdGeomTokens->catmullClark);
            TfToken orient = getAttribute(usdMesh.GetOrientationAttr(), UsdGeomTokens->rightHanded);

            UsdAttribute pointsAttr = usdMesh.GetPointsAttr();
            if (!pointsAttr)
            {
                logWarning("Mesh '{}' does not specify vertices. Ignoring.", meshName);
                return false;
            }
            VtVec3fArray usdPoints;
            pointsAttr.Get(&usdPoints, timeCode);

            UsdAttribute faceCountsAttr = usdMesh.GetFaceVertexCountsAttr();
            if (!faceCountsAttr)
            {
                logWarning("Mesh '{}' has no faces. Ignoring.", meshName);
                return false;
            }
            VtIntArray usdFaceCounts;
            faceCountsAttr.Get(&usdFaceCounts, timeCode);

            UsdAttribute faceIndicesAttr = usdMesh.GetFaceVertexIndicesAttr();
            if (!faceIndicesAttr)
            {
                logWarning("Mesh '{}' does not specify face indices. Ignoring.", meshName);
                return false;
            }

            VtIntArray usdFaceIndices;
            faceIndicesAttr.Get(&usdFaceIndices, timeCode);

            VtIntArray usdHoleIndices;
            UsdAttribute holeIndicesAttr = usdMesh.GetHoleIndicesAttr();
            if (holeIndicesAttr)
            {
                holeIndicesAttr.Get(&usdHoleIndices, timeCode);
            }

            // Construct a HdMeshUtil to perform triangulation for us
            HdMeshTopology topology(scheme, orient, usdFaceCounts, usdFaceIndices, usdHoleIndices);
            HdMeshUtil meshUtil(&topology, usdMesh.GetPath());

            VtVec3iArray triangleIndices;
            VtIntArray primitiveParams;
            meshUtil.ComputeTriangleIndices(&triangleIndices, &primitiveParams);

            Hd_VertexAdjacency adjacency;
            adjacency.BuildAdjacencyTable(&topology);
            geomOut.numReferencedPoints = adjacency.GetNumPoints();

            // Get normals
            VtVec3fArray usdNormals;

            // As per the USD docs, ignore authored normals unless interpolation is set to "none".
            if (scheme == UsdGeomTokens->none)
            {
                // If both normals and primvars:normals are specified, the latter has precedence.
                const TfToken normalsVarName("primvars:normals");
                UsdGeomPrimvar normalsPrimvar(primvarApi.GetPrimvar(normalsVarName));
                if (normalsPrimvar.HasValue())
                {
                    // Get flattened version of the normals
                    normalsPrimvar.ComputeFlattened(&usdNormals, timeCode);
                    geomOut.normalInterp = convertInterpolation(normalsPrimvar.GetInterpolation());
                }
                else if (usdMesh.GetNormalsAttr().IsAuthored())
                {
                    // Normals specified via the attribute cannot be indexed, so there is no need to flatten them.
                    usdMesh.GetNormalsAttr().Get(&usdNormals, timeCode);
                    geomOut.normalInterp = convertInterpolation(usdMesh.GetNormalsInterpolation());
                }
                else
                {
                    // No authored normals. Generate faceted normals for this non-suddivided mesh.
                    usdNormals = Hd_FlatNormals::ComputeFlatNormals(&topology, usdPoints.cdata());
                    geomOut.normalInterp = AttributeFrequency::Uniform;
                }
            }
            else
            {
                // Generate smooth normals for this suddivided mesh
                usdNormals = Hd_SmoothNormals::ComputeSmoothNormals(&adjacency, (int)usdPoints.size(), usdPoints.cdata());
                geomOut.normalInterp = AttributeFrequency::Vertex;
            }

            FALCOR_ASSERT(usdNormals.size() > 0);

            // Generated triangulated normals, if necessary
            convertTriangulatedFaceData(usdNormals, geomOut.normalInterp, HdType::HdTypeFloatVec3, triangleIndices.size(), primitiveParams, meshUtil);

            // Get texture coordinates, if any.
            const TfToken uvVarName("primvars:st");
            UsdGeomPrimvar uvPrimvar(primvarApi.GetPrimvar(uvVarName));
            VtVec2fArray usdUVs;
            if (!uvPrimvar)
            {
                // "st_0" seems a common choice of texcoord primvar name when there are multiple uv sets, so check for it as well
                const TfToken uvVarName0("primvars:st_0");
                uvPrimvar = primvarApi.GetPrimvar(uvVarName0);
            }
            if (uvPrimvar)
            {
                geomOut.texCrdsInterp = convertInterpolation(uvPrimvar.GetInterpolation());

                // Get flattened version of the texture coordinates, and generate triangulated version, if necessary
                uvPrimvar.ComputeFlattened(&usdUVs, timeCode);
                convertTriangulatedFaceData(usdUVs, geomOut.texCrdsInterp, HdType::HdTypeFloatVec2, triangleIndices.size(), primitiveParams, meshUtil);
            }

            // Load skinning data

            // Mesh must have skinning data, and a corresponding skeleton.
            UsdGeomPrimvar jointIndicesPrimvar(primvarApi.GetPrimvar(UsdSkelTokens->primvarsSkelJointIndices));
            UsdGeomPrimvar jointWeightsPrimvar(primvarApi.GetPrimvar(UsdSkelTokens->primvarsSkelJointWeights));
            if (jointIndicesPrimvar && jointWeightsPrimvar)
            {
                auto skelMapIt = ctx.meshSkelMap.find(usdMesh.GetPrim());
                if (skelMapIt == ctx.meshSkelMap.end())
                {
                    logWarning("Mesh '{}' has skinning data but no skeleton. Skinning data will not be loaded.", meshName);
                }
                else
                {
                    FALCOR_ASSERT(jointIndicesPrimvar.GetElementSize() == jointWeightsPrimvar.GetElementSize());
                    const uint32_t elementSize = uint32_t(jointIndicesPrimvar.GetElementSize());

                    // Only support up to 4 bones per vertex
                    if (elementSize > Scene::kMaxBonesPerVertex)
                    {
                        logWarning("Mesh '{}' contains more than {} bones per vertex ({}). Ignoring extra data.", meshName, Scene::kMaxBonesPerVertex, elementSize);
                    }

                    // Skinning data can only be constant or vertex. Supporting vertex only first
                    if (jointIndicesPrimvar.GetInterpolation() == UsdGeomTokens->vertex && jointWeightsPrimvar.GetInterpolation() == UsdGeomTokens->vertex)
                    {
                        VtIntArray jointIndices;
                        VtFloatArray jointWeights;
                        jointIndicesPrimvar.Get(&jointIndices, UsdTimeCode::EarliestTime());
                        jointWeightsPrimvar.Get(&jointWeights, UsdTimeCode::EarliestTime());

                        // Should be a set of bone influences per point when interpolation is "vertex"
                        FALCOR_ASSERT(usdPoints.size() == jointIndices.size() / elementSize);

                        // Get the skeleton bound to this mesh and remap indices
                        const auto& skelIdx = skelMapIt->second;
                        const Skeleton::SubSkeleton& subskel = ctx.skeletons[skelIdx.first].subskeletons[skelIdx.second];
                        remapMeshJointIndices(ctx, subskel, usdMesh.GetPrim(), jointIndices);

                        // Load per vertex bone influences
                        geomOut.jointIndices.resize(usdPoints.size(), uint4(NodeID::kInvalidID));
                        geomOut.jointWeights.resize(usdPoints.size(), float4(0.0f));
                        for (uint32_t i = 0; i < usdPoints.size(); i++)
                        {
                            uint4& indices = geomOut.jointIndices[i];
                            float4& weights = geomOut.jointWeights[i];

                            const uint32_t elementsToRead = std::min(Scene::kMaxBonesPerVertex, elementSize);
                            for (uint32_t j = 0; j < elementsToRead; j++)
                            {
                                uint32_t sourceIdx = i * elementSize + j;

                                if (jointWeights[sourceIdx] > 0.0f)
                                {
                                    indices[j] = jointIndices[sourceIdx];
                                    weights[j] = jointWeights[sourceIdx];
                                }
                            }

                            // Normalize weights in case the sum isn't 1
                            float sum = weights[0] + weights[1] + weights[2] + weights[3];
                            weights /= sum;
                        }
                    }
                    else
                    {
                        logWarning("Skinning data for mesh '{}' must be per-vertex. \"constant\" interpolation is not supported. Ignoring primitive.", meshName);
                        return false;
                    }
                }
            }

            // We now have triangulated points and normals, as well as texture coordinates, if specified.

            std::vector<UsdGeomSubset> geomSubsets = UsdGeomSubset::GetAllGeomSubsets(usdMesh);
            uint32_t subsetCount = (uint32_t)geomSubsets.size();

            if (subsetCount == 0)
            {
                // No subsets; create a single mesh
                geomOut.geomSubsets.resize(1);
                geomOut.geomSubsets[0].triIdx = 0;
                geomOut.geomSubsets[0].triCount = (uint32_t)triangleIndices.size();
                geomOut.geomSubsets[0].id = meshName;
                geomOut.geomSubsets[0].material = ctx.getBoundMaterial(usdMesh);
            }
            else
            {
                // Falcor doesn't support submeshes or the like. As such, we create a separate mesh for each GeomSubset;
                // SceneBuilder's mesh optimization code remove any redundant points.
                // First, gather the face indices for each subset. Per the USD docs, each face belongs to exactly
                // one subset, either explicitly, or to the implied "catch-all" subset.

                // Construct mapping from face idx to subset idx
                VtIntArray faceMap(topology.GetNumFaces(), -1);
                for (uint32_t i = 0; i < subsetCount; ++i)
                {
                    const UsdGeomSubset& geomSubset = geomSubsets[i];
                    UsdAttribute indicesAttr = geomSubset.GetIndicesAttr();
                    if (indicesAttr)
                    {
                        VtIntArray subGeomIndices;
                        indicesAttr.Get(&subGeomIndices, UsdTimeCode::EarliestTime());
                        for (int faceIdx : subGeomIndices)
                        {
                            if (faceIdx > faceMap.size() || faceMap[faceIdx] != -1)
                            {
                                logError("Invalid GeomSubset '{}' specified for '{}', ignoring.", geomSubset.GetPath().GetString(), meshName);
                                continue;
                            }
                            faceMap[faceIdx] = (int)i;
                        }
                    }
                }

                // Unmapped faces are assigned to a separate catchall subset
                VtIntArray unassignedIndices = UsdGeomSubset::GetUnassignedIndices(geomSubsets, topology.GetNumFaces());
                if (unassignedIndices.size() > 0)
                {
                    for (int idx : unassignedIndices)
                    {
                        FALCOR_ASSERT(faceMap[idx] == -1);
                        faceMap[idx] = subsetCount;
                    }
                    ++subsetCount;
                }

                // Compute the number of triangle faces per subset
                VtIntArray trianglesPerSubset(subsetCount, 0);
                for (int coarseFaceParam : primitiveParams)
                {
                    int faceIdx = HdMeshUtil::DecodeFaceIndexFromCoarseFaceParam(coarseFaceParam);
                    int geomSubset = faceMap[faceIdx];
                    if (geomSubset >= 0)
                    {
                        trianglesPerSubset[geomSubset]++;
                    }
                }

                // Compute face index offset per subset
                VtIntArray subsetOffset(subsetCount, 0);
                for (uint32_t i = 1; i < subsetCount; ++i)
                {
                    subsetOffset[i] = subsetOffset[i - 1] + trianglesPerSubset[i - 1];
                }

                // Reshuffle face indices into subset order
                remapGeomSubsetData(triangleIndices, AttributeFrequency::Uniform, faceMap, subsetOffset, primitiveParams);

                // Reshuffle normals
                remapGeomSubsetData(usdNormals, geomOut.normalInterp, faceMap, subsetOffset, primitiveParams);

                // Reshuffle texcoords
                remapGeomSubsetData(usdUVs, geomOut.texCrdsInterp, faceMap, subsetOffset, primitiveParams);

                // Count the number of non-degenerate subsets
                subsetCount = (uint32_t)std::count_if(trianglesPerSubset.begin(), trianglesPerSubset.end(), [](int val) {return val > 0; });

                // Initialize vector of GeomSubsets
                geomOut.geomSubsets.resize(subsetCount);
                size_t subsetIdx = 0;
                size_t normalIdx = 0;
                size_t texCrdsIdx = 0;
                for (size_t i = 0; i < trianglesPerSubset.size(); ++i)
                {
                    if (trianglesPerSubset[i] > 0)
                    {
                        GeomSubset& subset = geomOut.geomSubsets[subsetIdx];
                        subset.triIdx = subsetOffset[i];
                        subset.triCount = trianglesPerSubset[i];
                        subset.normalIdx = normalIdx;
                        subset.texCrdsIdx = texCrdsIdx;
                        if (unassignedIndices.size() > 0 && i == trianglesPerSubset.size() - 1)
                        {
                            // This subset is the catchall; use the id and material associated with the base gprim
                            subset.id = meshName;
                            subset.material = ctx.getBoundMaterial(usdMesh);
                        }
                        else
                        {
                            subset.id = geomSubsets[i].GetPath().GetString();
                            subset.material = ctx.getBoundMaterial(geomSubsets[i]);
                        }
                        normalIdx += computePerFaceElementCount(geomOut.normalInterp, subset.triCount);
                        texCrdsIdx += computePerFaceElementCount(geomOut.texCrdsInterp, subset.triCount);
                        ++subsetIdx;
                    }
                }
            }

            // Copy points
            static_assert(sizeof(GfVec3f) == sizeof(float3));
            geomOut.points.resize(usdPoints.size());
            std::memcpy(geomOut.points.data(), usdPoints.data(), usdPoints.size() * sizeof(GfVec3f));

            // Copy indices
            geomOut.triangulatedIndices.resize(3 * triangleIndices.size());
            std::memcpy(geomOut.triangulatedIndices.data(), triangleIndices.data(), geomOut.triangulatedIndices.size() * sizeof(uint32_t));

            // Copy normals
            if (usdNormals.size() > 0)
            {
                static_assert(sizeof(GfVec3f) == sizeof(float3));
                geomOut.normals.resize(usdNormals.size());
                std::memcpy(geomOut.normals.data(), usdNormals.cdata(), usdNormals.size() * sizeof(GfVec3f));
            }
            else
            {
                // There should always be normals, either authored or computed.
                logError("Mesh '{}' has no normals.", meshName);
            }

            // Copy texture coordinates
            if (usdUVs.size() > 0)
            {
                static_assert(sizeof(GfVec2f) == sizeof(float2));
                geomOut.texCrds.resize(usdUVs.size());
                std::memcpy(geomOut.texCrds.data(), usdUVs.cdata(), geomOut.texCrds.size() * sizeof(GfVec2f));
            }
            return true;
        }

        bool extractCurveData(const UsdGeomBasisCurves& usdCurve, const UsdTimeCode& timeCode, VtVec3fArray& usdPoints, VtIntArray& usdCurveVertexCounts, VtFloatArray& usdCurveWidths, VtVec2fArray& usdUVs)
        {
            std::string curveName = usdCurve.GetPath().GetString();

            UsdGeomPrimvarsAPI primvarApi(usdCurve);

            TfToken basis = getAttribute(usdCurve.GetBasisAttr(), UsdGeomTokens->bspline);
            TfToken curveType = getAttribute(usdCurve.GetTypeAttr(), UsdGeomTokens->cubic);
            TfToken wrap = getAttribute(usdCurve.GetWrapAttr(), UsdGeomTokens->nonperiodic);

            UsdAttribute extentAttr = usdCurve.GetExtentAttr();
            if (!extentAttr)
            {
                logWarning("Curve '{}' has no AABB. Ignoring.", curveName);
                return false;
            }
            VtVec3fArray usdExtent;
            extentAttr.Get(&usdExtent, timeCode);

            UsdAttribute pointsAttr = usdCurve.GetPointsAttr();
            if (!pointsAttr)
            {
                logWarning("Curve '{}' does not specify control points. Ignoring.", curveName);
                return false;
            }
            pointsAttr.Get(&usdPoints, timeCode);

            UsdAttribute curveVertexCountsAttr = usdCurve.GetCurveVertexCountsAttr();
            if (!curveVertexCountsAttr)
            {
                logWarning("Curve '{}' has no vertices. Ignoring.", curveName);
                return false;
            }
            curveVertexCountsAttr.Get(&usdCurveVertexCounts, timeCode);

            UsdAttribute widthsAttr = usdCurve.GetWidthsAttr();
            if (!widthsAttr)
            {
                logWarning("Curve '{}' has no width attribute. Ignoring.", curveName);
                return false;
            }
            widthsAttr.Get(&usdCurveWidths, timeCode);

            // Get curve texture coordinates, if any.
            const TfToken uvVarName("primvars:st");
            UsdGeomPrimvar uvPrimvar(primvarApi.GetPrimvar(uvVarName));
            if (!uvPrimvar)
            {
                // "st_0" seems a common choice of texcoord primvar name when there are multiple uv sets, so check for it as well
                const TfToken uvVarName0("primvars:st_0");
                uvPrimvar = primvarApi.GetPrimvar(uvVarName0);
            }
            if (uvPrimvar)
            {
                // Get flattened version of the texture coordinates, and generate triangulated version, if necessary
                uvPrimvar.ComputeFlattened(&usdUVs, timeCode);
            }

            return true;
        }

        // Convert a UsdGeomBasisCurves into a CurveGeomData (curve primitive).
        bool convertToCurveGeomData(const UsdGeomBasisCurves& usdCurve, const UsdTimeCode& timeCode, ImporterContext& ctx, CurveGeomData& geomOut)
        {
            std::string curveName = usdCurve.GetPath().GetString();

            VtVec3fArray usdPoints;
            VtIntArray usdCurveVertexCounts;
            VtFloatArray usdCurveWidths;
            VtVec2fArray usdUVs;
            if (!extractCurveData(usdCurve, timeCode, usdPoints, usdCurveVertexCounts, usdCurveWidths, usdUVs))
            {
                return false;
            }

            size_t strandCount = usdCurveVertexCounts.size();
            size_t vertexCount = std::accumulate(usdCurveVertexCounts.begin(), usdCurveVertexCounts.end(), 0);
            FALCOR_ASSERT(vertexCount == usdPoints.size());
            FALCOR_ASSERT(vertexCount == usdCurveWidths.size());
            FALCOR_ASSERT(vertexCount == usdUVs.size() || usdUVs.size() == 0);

            const float2* pUsdUVs = usdUVs.empty() ? nullptr : (float2*)usdUVs.data();

            uint32_t subdivPerSegment                = gpFramework->getSettings().getAttribute(curveName, "curves:subdivPerSegment", kCurveSubdivPerSegment);
            uint32_t keepOneEveryXStrands            = gpFramework->getSettings().getAttribute(curveName, "curves:keepOneEveryXStrands", kCurveKeepOneEveryXStrands);
            uint32_t keepOneEveryXVerticesPerStrand  = gpFramework->getSettings().getAttribute(curveName, "curves:keepOneEveryXVerticesPerStrand", kCurveKeepOneEveryXVerticesPerStrand);

            // Perceptually, it is a good practice to increase width of hair strands if we render less of them than anticipated.
            float widthScale = std::sqrt((float)keepOneEveryXStrands);

            // Convert to linear swept sphere segments.
            CurveTessellation::SweptSphereResult result = CurveTessellation::convertToLinearSweptSphere(strandCount, reinterpret_cast<const uint32_t*>(usdCurveVertexCounts.data()),
                (float3*)usdPoints.data(), usdCurveWidths.data(), pUsdUVs, 1,
                subdivPerSegment, keepOneEveryXStrands, keepOneEveryXVerticesPerStrand, widthScale, rmcv::identity<rmcv::mat4x4>());

            // Copy data.
            geomOut.id = curveName;
            geomOut.degree = result.degree;
            geomOut.indices = std::move(result.indices);
            geomOut.points = std::move(result.points);
            geomOut.radius = std::move(result.radius);
            geomOut.material = ctx.getBoundMaterial(usdCurve);

            // Copy texture coordinates.
            if (result.texCrds.size() > 0)
            {
                geomOut.texCrds = std::move(result.texCrds);
            }
            else
            {
                logWarning("Curve '{}' has no texture coordinates.", curveName);
            }

            return true;
        }

        // Convert a UsdGeomBasisCurves into a MeshGeomData (mesh).
        bool convertToMeshGeomData(const UsdGeomBasisCurves& usdCurve, const UsdTimeCode& timeCode, ImporterContext& ctx, CurveTessellationMode tessellationMode, MeshGeomData& geomOut)
        {
            std::string curveName = usdCurve.GetPath().GetString();

            VtVec3fArray usdPoints;
            VtIntArray usdCurveVertexCounts;
            VtFloatArray usdCurveWidths;
            VtVec2fArray usdUVs;
            if (!extractCurveData(usdCurve, timeCode, usdPoints, usdCurveVertexCounts, usdCurveWidths, usdUVs))
            {
                return false;
            }

            size_t strandCount = usdCurveVertexCounts.size();
            size_t vertexCount = std::accumulate(usdCurveVertexCounts.begin(), usdCurveVertexCounts.end(), 0);
            FALCOR_ASSERT(vertexCount == usdPoints.size());
            FALCOR_ASSERT(vertexCount == usdCurveWidths.size());
            FALCOR_ASSERT(vertexCount == usdUVs.size() || usdUVs.size() == 0);

            const float2* pUsdUVs = usdUVs.empty() ? nullptr : (float2*)usdUVs.data();

            uint32_t subdivPerSegment                = gpFramework->getSettings().getAttribute(curveName, "curves:subdivPerSegment", kCurveSubdivPerSegment);
            uint32_t keepOneEveryXStrands            = gpFramework->getSettings().getAttribute(curveName, "curves:keepOneEveryXStrands", kCurveKeepOneEveryXStrands);
            uint32_t keepOneEveryXVerticesPerStrand  = gpFramework->getSettings().getAttribute(curveName, "curves:keepOneEveryXVerticesPerStrand", kCurveKeepOneEveryXVerticesPerStrand);

            // Perceptually, it is a good practice to increase width of hair strands if we render less of them than anticipated.
            float widthScale = std::sqrt((float)keepOneEveryXStrands);

            // Tessellation into mesh.
            CurveTessellation::MeshResult result;

            if (tessellationMode == CurveTessellationMode::PolyTube)
            {
                result = CurveTessellation::convertToPolytube(strandCount, reinterpret_cast<const uint32_t*>(usdCurveVertexCounts.data()), (float3*)usdPoints.data(), usdCurveWidths.data(), pUsdUVs, subdivPerSegment, keepOneEveryXStrands, keepOneEveryXVerticesPerStrand, widthScale, 4);
            }
            else
            {
                FALCOR_UNREACHABLE();
            }

            // Copy data.
            geomOut.geomSubsets.resize(1);
            geomOut.geomSubsets[0].triIdx = 0;
            geomOut.geomSubsets[0].triCount = (uint32_t)result.faceVertexIndices.size() / 3;
            geomOut.geomSubsets[0].id = curveName;
            geomOut.geomSubsets[0].material = ctx.getBoundMaterial(usdCurve);

            geomOut.points = std::move(result.vertices);
            geomOut.numReferencedPoints = geomOut.points.size();
            geomOut.triangulatedIndices = std::move(result.faceVertexIndices);
            geomOut.normals = std::move(result.normals);
            geomOut.normalInterp = AttributeFrequency::Vertex;
            geomOut.tangents = std::move(result.tangents);
            geomOut.curveRadii = std::move(result.radii);

            // Copy texture coordinates.
            if (result.texCrds.size() > 0)
            {
                geomOut.texCrds = std::move(result.texCrds);
                geomOut.texCrdsInterp = AttributeFrequency::Vertex;
            }
            else
            {
                logWarning("Curve '{}' has no texture coordinates.", curveName);
            }

            return true;
        }

        // Create a SceneBuilder::Mesh from the given GeomSubset of the given MeshGeomData
        bool createSceneBuilderMesh(const UsdPrim& meshPrim, const MeshGeomData& geomData, const GeomSubset& subset, ImporterContext& ctx, SceneBuilder::Mesh& sbMesh)
        {
            sbMesh.name = subset.id;
            sbMesh.faceCount = subset.triCount;
            sbMesh.indexCount = subset.triCount * 3UL;
            sbMesh.vertexCount = (uint32_t)geomData.points.size();
            sbMesh.pIndices = geomData.triangulatedIndices.data() + subset.triIdx * 3ULL;
            sbMesh.positions.pData = geomData.points.data();
            sbMesh.positions.frequency = AttributeFrequency::Vertex;
            sbMesh.normals.pData = (geomData.normals.size() > 0 ? geomData.normals.data() + subset.normalIdx : nullptr);
            sbMesh.normals.frequency = geomData.normalInterp;
            if (geomData.tangents.size() > 0)
            {
                sbMesh.tangents.pData = geomData.tangents.data() + subset.normalIdx;
                sbMesh.useOriginalTangentSpace = true;
            }
            sbMesh.tangents.frequency = AttributeFrequency::Vertex;
            sbMesh.texCrds.pData = (geomData.texCrds.size() > 0 ? geomData.texCrds.data() + subset.texCrdsIdx : nullptr);
            sbMesh.texCrds.frequency = geomData.texCrdsInterp;
            sbMesh.curveRadii.pData = (geomData.curveRadii.size() > 0 ? geomData.curveRadii.data() : nullptr);
            sbMesh.curveRadii.frequency = AttributeFrequency::Vertex;
            sbMesh.boneIDs.pData = geomData.jointIndices.size() > 0 ? geomData.jointIndices.data() : nullptr;
            sbMesh.boneIDs.frequency = AttributeFrequency::Vertex;
            sbMesh.boneWeights.pData = geomData.jointWeights.size() > 0 ? geomData.jointWeights.data() : nullptr;
            sbMesh.boneWeights.frequency = AttributeFrequency::Vertex;
            sbMesh.topology = Vao::Topology::TriangleList;

            auto skelMapIt = ctx.meshSkelMap.find(meshPrim);
            if (skelMapIt != ctx.meshSkelMap.end())
            {
                // Look up node containing the skeleton's world transform.
                FALCOR_ASSERT(sbMesh.boneIDs.pData != nullptr && sbMesh.boneWeights.pData != nullptr);
                sbMesh.skeletonNodeId = ctx.skeletons[skelMapIt->second.first].nodeID;
            }

            sbMesh.pMaterial = ctx.resolveMaterial(meshPrim, subset.material, sbMesh.name);

            if (sbMesh.pIndices == nullptr || sbMesh.indexCount == 0)
            {
                logWarning("Gprim '{}' has no indices. Ignoring.", meshPrim.GetPath().GetString());
                return false;
            }
            if (sbMesh.positions.pData == nullptr)
            {
                logWarning("Gprim '{}' has no position data. Ignoring.", meshPrim.GetPath().GetString());
                return false;
            }
            return true;
        }

        bool createSceneBuilderCurve(const UsdPrim& curvePrim, const CurveGeomData& curveData, ImporterContext& ctx, SceneBuilder::Curve& sbCurve)
        {
            sbCurve.name = curveData.id;
            sbCurve.indexCount = (uint32_t)curveData.indices.size();
            sbCurve.vertexCount = (uint32_t)curveData.points.size();
            sbCurve.pIndices = curveData.indices.data();
            sbCurve.positions.pData = curveData.points.data();
            sbCurve.radius.pData = curveData.radius.data();
            sbCurve.texCrds.pData = (curveData.texCrds.size() > 0 ? curveData.texCrds.data() : nullptr);

            Material::SharedPtr pMaterial = ctx.resolveMaterial(curvePrim, curveData.material, sbCurve.name);

            // It is possible the material was found in the scene and is assigned to other non-curve geometry.
            // We'll issue a warning if there is a material type mismatch.
            FALCOR_ASSERT(pMaterial);
            if (pMaterial->getType() != MaterialType::Hair)
            {
                logWarning("Material '{}' assigned to curve '{}' is of non-hair type.", pMaterial->getName(), sbCurve.name);
            }

            sbCurve.pMaterial = pMaterial;

            if (sbCurve.pIndices == nullptr || sbCurve.indexCount == 0)
            {
                logWarning("Gprim '{}' has no indices. Ignoring.", curvePrim.GetPath().GetString());
                return false;
            }
            if (sbCurve.positions.pData == nullptr)
            {
                logWarning("Gprim '{}' has no position data. Ignoring.", curvePrim.GetPath().GetString());
                return false;
            }
            return true;
        }

        void verifyMeshGeomData(const MeshGeomData& geomData, const std::filesystem::path& path, const std::string& primName)
        {
            // Basic sanity checks
            if (geomData.normals.size() > 0)
            {
                size_t expectedNormals = computeElementCount(geomData.normalInterp, geomData.triangulatedIndices.size() / 3, geomData.numReferencedPoints);
                if (expectedNormals != geomData.normals.size())
                {
                    throw ImporterError(path, "Conversion error on USD prim '{}'. Expected {} normals, created {}.", primName, expectedNormals, geomData.normals.size());
                }
            }

            if (geomData.texCrds.size() > 0)
            {
                size_t expectedTexcoords = computeElementCount(geomData.texCrdsInterp, geomData.triangulatedIndices.size() / 3, geomData.points.size());
                if (expectedTexcoords != geomData.texCrds.size())
                {
                    throw ImporterError(path, "Conversion error on USD prim '{}'. Expected {} texcoords, created {}.", primName, expectedTexcoords, geomData.texCrds.size());
                }
            }
        }

        bool processMesh(Mesh& mesh, ImporterContext& ctx)
        {
            FALCOR_ASSERT(mesh.prim.IsA<UsdGeomMesh>() || mesh.prim.IsA<UsdGeomBasisCurves>());
            FALCOR_ASSERT(isRenderable(UsdGeomImageable(mesh.prim)));

            std::string primName = mesh.prim.GetPath().GetString();

            // First, convert USD data to Falcor-friendly data, based on the underlying prim type.
            MeshGeomData geomData;

            if (mesh.prim.IsA<UsdGeomMesh>())
            {
                if (!convertMeshGeomData(UsdGeomMesh(mesh.prim), UsdTimeCode(mesh.timeSamples[0]), ctx, geomData))
                {
                    return false;
                }
            }
            else
            {
                return false;
            }

            verifyMeshGeomData(geomData, ctx.path, primName);

            // Finally, create a SceneBuilder::Mesh for each geomSubset in the MeshGeomData.
            mesh.processedMeshes.reserve(geomData.geomSubsets.size());
            if (isTimeSampled(UsdGeomPointBased(mesh.prim)))
            {
                if (gpFramework->getSettings().getOption("usdImporter:loadMeshVertexAnimations", kLoadMeshVertexAnimations))
                {
                    mesh.attributeIndices.resize(geomData.geomSubsets.size());
                }
                else
                {
                    logWarning("Scene contains time-sampled mesh data, but their loading is disabled. See ImporterContext::kLoadMeshVertexAnimations");
                }
            }

            for (size_t i = 0; i < geomData.geomSubsets.size(); ++i)
            {
                // Create separate mesh for each GeomSubset
                SceneBuilder::Mesh sbMesh;
                if (!createSceneBuilderMesh(mesh.prim, geomData, geomData.geomSubsets[i], ctx, sbMesh))
                {
                    continue;
                }

                auto pAttributeIndices = mesh.attributeIndices.empty() ? nullptr : &mesh.attributeIndices[i];
                mesh.processedMeshes.push_back(ctx.builder.processMesh(sbMesh, pAttributeIndices));
            }

            return true;
        }

        bool processMeshKeyframe(Mesh& mesh, uint32_t subsetIdx, uint32_t sampleIdx, ImporterContext& ctx)
        {
            MeshGeomData geomData;

            if (mesh.prim.IsA<UsdGeomMesh>())
            {
                UsdGeomMesh geomMesh(mesh.prim);

                if (!convertMeshGeomData(geomMesh, UsdTimeCode(mesh.timeSamples[sampleIdx]), ctx, geomData))
                {
                    return false;
                }
            }
            else
            {
                return false;
            }

            // Convert geom data to keyframe data for the mesh at a particular sample
            FALCOR_ASSERT(geomData.geomSubsets.size() == mesh.processedMeshes.size());
            FALCOR_ASSERT(!mesh.meshIDs.empty()); // Mesh should have been added to builder already
            for (size_t i = 0; i < geomData.geomSubsets.size(); ++i)
            {
                // Create mesh to set up data according to subsets
                SceneBuilder::Mesh sbMesh;
                if (!createSceneBuilderMesh(mesh.prim, geomData, geomData.geomSubsets[i], ctx, sbMesh))
                {
                    continue;
                }

                if (sbMesh.tangents.pData == nullptr)
                {
                    ctx.builder.generateTangents(sbMesh, geomData.tangents);
                }

                auto& indices = mesh.attributeIndices[i];

                if (!(mesh.processedMeshes[i].staticData.size() == indices.size()))
                {
                    throw ImporterError(ctx.path, "Keyframe {} for mesh '{}' does not match vertex count of original mesh.", sampleIdx, mesh.prim.GetName().GetString());
                }

                // Fill vertex data
                mesh.cachedMeshes[i].timeSamples = mesh.timeSamples;
                mesh.cachedMeshes[i].meshID = mesh.meshIDs[i];
                for (auto& t : mesh.cachedMeshes[i].timeSamples) t /= ctx.timeCodesPerSecond; // Convert to seconds

                std::vector<PackedStaticVertexData>& keyframeData = mesh.cachedMeshes[i].vertexData[sampleIdx];
                keyframeData.reserve(indices.size());
                for (size_t j = 0; j < indices.size(); j++)
                {
                    SceneBuilder::Mesh::Vertex v = sbMesh.getVertex(indices[j]);

                    StaticVertexData data;
                    data.position = v.position;
                    data.normal = v.normal;
                    data.tangent = v.tangent;
                    data.texCrd = v.texCrd;
                    keyframeData.emplace_back(data);
                }
            }

            return true;
        }

        bool processCurve(Curve& curve, ImporterContext& ctx)
        {
            UsdGeomBasisCurves geomCurve(curve.curvePrim);
            std::string primName(geomCurve.GetPath().GetString());

            // Extract time samples.
            // Assume the point attribute has a complete set of time samples (shared by other attributes such as vertex count).
            geomCurve.GetPointsAttr().GetTimeSamples(&(curve.timeSamples));

            if (!gpFramework->getSettings().getAttribute(primName, "usdImporter:enableMotion", true))
                curve.timeSamples.clear();

            // Remove frames with timeCode < 1.
            // TODO: We can remove this once the assets do not have these redundant keyframes.
            while (curve.timeSamples.size() > 0 && curve.timeSamples.front() < 1.0)
            {
                curve.timeSamples.erase(curve.timeSamples.begin());
            }

            uint32_t timeSampleCount = (uint32_t)curve.timeSamples.size();
            std::vector<UsdTimeCode> timeCodes;
            if (timeSampleCount == 0)
            {
                timeCodes.push_back(UsdTimeCode::EarliestTime());
                curve.timeSamples.resize(1);
                curve.timeSamples[0] = 0;
            }
            else
            {
                for (uint32_t i = 0; i < timeSampleCount; i++) timeCodes.push_back(UsdTimeCode(curve.timeSamples[i]));
            }

            for (size_t i = 0; i < timeCodes.size(); i++)
            {
                CurveGeomData curveData;
                if (!convertToCurveGeomData(geomCurve, timeCodes[i], ctx, curveData))
                {
                    return false;
                }

                SceneBuilder::Curve sbCurve;
                if (createSceneBuilderCurve(curve.curvePrim, curveData, ctx, sbCurve))
                {
                    curve.processedCurves.push_back(ctx.builder.processCurve(sbCurve));
                }

                // Compute keyframe time in seconds.
                curve.timeSamples[i] /= ctx.timeCodesPerSecond;
            }

            bool processFirstKeyframeMesh = curve.tessellationMode == CurveTessellationMode::PolyTube;
            if (processFirstKeyframeMesh)
            {
                MeshGeomData geomData;
                if (!convertToMeshGeomData(geomCurve, UsdTimeCode::EarliestTime(), ctx, curve.tessellationMode, geomData))
                {
                    return false;
                }

                verifyMeshGeomData(geomData, ctx.path, primName);

                // Finally, create a SceneBuilder::Mesh for the (single) geomSubset in the MeshGeomData.
                FALCOR_ASSERT(geomData.geomSubsets.size() == 1);

                SceneBuilder::Mesh sbMesh;
                if (createSceneBuilderMesh(curve.curvePrim, geomData, geomData.geomSubsets[0], ctx, sbMesh))
                {
                    sbMesh.mergeDuplicateVertices = false;
                    curve.processedMesh = ctx.builder.processMesh(sbMesh);
                }
            }

            return true;
        }

        void addSkeletonsToSceneBuilder(ImporterContext& ctx, TimeReport& timeReport)
        {
            for (auto& skel : ctx.skeletons)
            {
                skel.nodeID = ctx.builder.addNode(makeNode(skel.name, skel.parentID));

                // Add bone nodes and any animations
                for (auto& subskel : skel.subskeletons)
                {
                    auto& bones = subskel.bones;
                    auto& animations = subskel.animations;

                    // This skeleton's nodes will start at the next node in the scene graph
                    subskel.nodeOffset = ctx.builder.getNodeCount();

                    // Add skeleton bones and their animations.
                    for (size_t i = 0; i < bones.size(); i++)
                    {
                        if (bones[i].parent == NodeID::Invalid())
                        {
                            // If this is the root bone, connect it to the skeleton base
                            bones[i].parent = skel.nodeID;
                        }
                        else
                        {
                            // Shift node ID so they still hook up to the right node after being added to the SceneBuilder
                            bones[i].parent = NodeID{ bones[i].parent.get() + subskel.nodeOffset };
                        }

                        NodeID nodeID = ctx.builder.addNode(bones[i]);

                        // FIXME: Handle unanimated case
                        if (animations.size() > 0)
                        {
                            // Animation and bones array correspond 1:1, so set the Animation nodeId to wherever the bone node index is
                            FALCOR_ASSERT_EQ(animations[i]->getNodeID().get() + subskel.nodeOffset, nodeID.get());
                            animations[i]->setNodeID(nodeID);
                            ctx.builder.addAnimation(animations[i]);
                        }
                    }
                }
            }
        }

        void addMeshesToSceneBuilder(ImporterContext& ctx, TimeReport& timeReport)
        {
            // Process collected mesh tasks.
            NumericRange<size_t> meshRange(0, ctx.meshTasks.size());
            std::for_each(std::execution::par, meshRange.begin(), meshRange.end(),
                [&](size_t i)
                {
                    FALCOR_ASSERT(ctx.meshTasks[i].sampleIdx == 0);
                    processMesh(ctx.meshes[ctx.meshTasks[i].meshId], ctx);
                }
            );

            // Add processed meshes to scene builder.
            // This is done sequentially after being processed in parallel to ensure a deterministic ordering.
            for (auto& mesh : ctx.meshes)
            {
                FALCOR_ASSERT(mesh.meshIDs.empty());
                for (auto& m : mesh.processedMeshes)
                {
                    mesh.meshIDs.push_back(ctx.builder.addProcessedMesh(m));
                }
            }

            if (gpFramework->getSettings().getOption("usdImporter:loadMeshVertexAnimations", kLoadMeshVertexAnimations))
            {
                // Allocate storage for mesh keyframe output
                for (auto& m : ctx.meshes)
                {
                    if (m.timeSamples.size() > 1)
                    {
                        m.cachedMeshes.resize(m.processedMeshes.size());
                        for (auto& c : m.cachedMeshes)
                        {
                            c.vertexData.resize(m.timeSamples.size());
                        }
                    }
                }

                // Process time-sampled mesh keyframes
                NumericRange<size_t> keyframeRange(0, ctx.meshKeyframeTasks.size());
                std::for_each(std::execution::par, keyframeRange.begin(), keyframeRange.end(),
                    [&](size_t i)
                    {
                        auto& task = ctx.meshKeyframeTasks[i];
                        processMeshKeyframe(ctx.meshes[task.meshId], task.meshId, task.sampleIdx, ctx);
                    }
                );

                // Gather keyframe data from all meshes
                size_t totalMeshes = 0;
                for (auto& m : ctx.meshes) totalMeshes += m.cachedMeshes.size();
                std::vector<CachedMesh> cachedMeshes;
                cachedMeshes.reserve(totalMeshes);

                for (auto& m : ctx.meshes)
                {
                    for (auto& c : m.cachedMeshes)
                    {
                        cachedMeshes.push_back(std::move(c));
                    }
                }
                ctx.builder.setCachedMeshes(std::move(cachedMeshes));
            }

            timeReport.measure("Process meshes");

            // Helper function to add all submeshes associated with the given UsdGeomMesh to SceneBuilder
            auto addSubmeshes = [&](const UsdPrim& meshPrim, const std::string& name, const rmcv::mat4& xform, const rmcv::mat4& bindXform, NodeID parentId)
            {
                auto subNodeId = ctx.builder.addNode(makeNode(name, xform, bindXform, parentId));
                FALCOR_ASSERT(meshPrim.IsA<UsdGeomMesh>());
                const auto& mesh = ctx.getMesh(meshPrim);
                for (MeshID meshID : mesh.meshIDs)
                {
                    ctx.builder.addMeshInstance(subNodeId, meshID);
                }
            };

            // Add individual instances of prepocessed geometry to scene builder. Each mesh may contain more than one submesh, each corresponding to a UsdGeomSubset.
            for (const auto& instance : ctx.geomInstances)
            {
                addSubmeshes(instance.prim, instance.name, rmcv::mat4(1.f), instance.bindTransform, instance.parentID);
            }

            // Add instances of prototypes to scene builder. Because SceneBuilder only supports instanced meshes, and not
            // general instancing, we effectively replicate each Prototype's subgraph. We could in theory collapse the subgraph
            // if all of the transformations are static, but time-sampled transformations require us to use a more general approach.
            for (const auto& inst : ctx.prototypeInstances)
            {
                std::vector<std::pair<PrototypeInstance, NodeID>> protoInstanceStack = { std::make_pair(inst, inst.parentID) };
                while (!protoInstanceStack.empty())
                {
                    PrototypeInstance protoInstance = protoInstanceStack.back().first;
                    NodeID parentID = protoInstanceStack.back().second;
                    protoInstanceStack.pop_back();

                    // Add root node for this prototype instance, parented appropriately.
                    NodeID rootNodeID = ctx.builder.addNode(makeNode(protoInstance.name, protoInstance.xform, rmcv::mat4(1.f), parentID));

                    // If there are keyframes, create an animation from them targeting the root node we just created.
                    // This could be more cleanly addressed if a Falcor::Animation could target more than one node.
                    if (protoInstance.keyframes.size() > 0)
                    {
                        Animation::SharedPtr pAnimation = Animation::create(protoInstance.name, rootNodeID, protoInstance.keyframes.back().time);
                        for (const auto& keyframe : protoInstance.keyframes)
                        {
                            pAnimation->addKeyframe(keyframe);
                        }
                        ctx.builder.addAnimation(pAnimation);
                    }

                    // Get a reference to the prototype
                    if (!ctx.hasPrototype(protoInstance.protoPrim))
                    {
                        logError("Cannot create instance of '{}'; no prototype exists.", protoInstance.protoPrim.GetPath().GetString());
                        continue;
                    }

                    const PrototypeGeom& protoGeom = ctx.getPrototypeGeom(protoInstance.protoPrim);

                    // Parent IDs of nodes in the PrototypeGeom are relative to the ID of the prototype's root, zero.
                    // Get the current builder node count, which will be the SceneBuilder node ID of the prototype root node.
                    // We use this value to map from protoGeom node IDs to SceneBuilder node IDs.
                    NodeID protoRootID{ ctx.builder.getNodeCount() };

                    for (const auto& node : protoGeom.nodes)
                    {
                        SceneBuilder::Node builderNode = node;
                        builderNode.parent = (node.parent == NodeID::Invalid()) ? rootNodeID : NodeID{ node.parent.get() + protoRootID.get() };
                        ctx.builder.addNode(builderNode);
                    }

                    // Create any animations from the stashed keyframes, each targeting the proper node.
                    // Again, this could be simplified if Animations could target multiple nodes.
                    for (const auto& animation : protoGeom.animations)
                    {
                        // Use the name of the target node as the animation name. Note that this will result in multiple animations with the same name
                        // if the prototype is instantiated multiple times.
                        std::string animationName = protoGeom.nodes[animation.targetNodeID.get()].name;
                        NodeID targetNodeID{ animation.targetNodeID.get() + protoRootID.get() };
                        Animation::SharedPtr pAnimation = Animation::create(animationName, targetNodeID, animation.keyframes.back().time);
                        for (const auto& keyframe : animation.keyframes)
                        {
                            pAnimation->addKeyframe(keyframe);
                        }
                        ctx.builder.addAnimation(pAnimation);
                    }

                    // Add all of the prototype's geom instances (currently limited to meshes)
                    for (const auto& meshInstance : protoGeom.geomInstances)
                    {
                        addSubmeshes(meshInstance.prim, protoInstance.name + "/" + meshInstance.name, meshInstance.xform, rmcv::mat4(1.f), NodeID{ meshInstance.parentID.get() + protoRootID.get() });
                    }

                    // Push child prototype instances, and the current nodeID as the parent, onto the stack. Do so in reverse order to maintain traversal ordering.
                    for (auto it = protoGeom.prototypeInstances.rbegin(); it != protoGeom.prototypeInstances.rend(); ++it)
                    {
                        protoInstanceStack.push_back(std::make_pair(*it, NodeID{ it->parentID.get() + protoRootID.get() }));
                    }
                }
            }

            timeReport.measure("Create instances");
        }

        // Note that this function can also add meshes to scene builder (depending on curve tessellation mode).
        void addCurvesToSceneBuilder(ImporterContext& ctx, TimeReport& timeReport)
        {
            // Process collected curves.
            NumericRange<size_t> range(0, ctx.curves.size());
            std::for_each(std::execution::par, range.begin(), range.end(),
                [&](size_t i) { processCurve(ctx.curves[i], ctx); }
            );

            // Add processed curves or meshes (of the first keyframe) to scene builder.
            // This is done sequentially after being processed in parallel to ensure a deterministic ordering.
            for (auto& curve : ctx.curves)
            {
                if (curve.tessellationMode == CurveTessellationMode::LinearSweptSphere)
                {
                    FALCOR_ASSERT(!curve.processedCurves.empty());
                    curve.geometryID = CurveOrMeshID{ ctx.builder.addProcessedCurve(curve.processedCurves[0]) };
                }
                else
                {
                    curve.geometryID = CurveOrMeshID{ ctx.builder.addProcessedMesh(curve.processedMesh) };
                }
            }

            // Add curve vertex cache (only has positions) to scene builder.
            for (auto& curve : ctx.curves) ctx.addCachedCurve(curve);
            ctx.builder.setCachedCurves(std::move(ctx.cachedCurves));

            timeReport.measure("Process curves");

            // Add instances to scene builder.
            for (const auto& instance : ctx.curveInstances)
            {
                auto nodeId = ctx.builder.addNode(makeNode(instance.name, instance.xform, rmcv::mat4(1.f), instance.parentID));
                const auto& curve = ctx.getCurve(instance.prim);

                if (curve.tessellationMode == CurveTessellationMode::LinearSweptSphere)
                {
                    ctx.builder.addCurveInstance(nodeId, CurveID{ curve.geometryID });
                }
                else
                {
                    ctx.builder.addMeshInstance(nodeId, MeshID{ curve.geometryID });
                }
            }

            timeReport.measure("Create curve instances");
        }

        float3 getLightIntensity(const UsdLuxLight& light)
        {
            float exposure = getAttribute(light.GetExposureAttr(), 0.f);
            float intens = getAttribute(light.GetIntensityAttr(), 1.f);
            GfVec3f blackbodyRGB(1.f, 1.f, 1.f);
            if (getAttribute(light.GetEnableColorTemperatureAttr(), false))
            {
                float temperature = getAttribute(light.GetColorTemperatureAttr(), 6500.f);
                blackbodyRGB = UsdLuxBlackbodyTemperatureAsRgb(temperature);
            }
            GfVec3f color = getAttribute(light.GetColorAttr(), GfVec3f(1.f, 1.f, 1.f));
            return std::pow(2.f, exposure) * intens * toGlm(blackbodyRGB) * toGlm(color);
        }

        Falcor::Animation::Keyframe createKeyframe(const UsdGeomXformCommonAPI& xformAPI, double timeCode, double timeCodesPerSecond)
        {
            Falcor::Animation::Keyframe keyframe;

            GfVec3d trans;
            GfVec3f rot;
            GfVec3f scale;
            GfVec3f pivot;
            UsdGeomXformCommonAPI::RotationOrder rotOrder;

            xformAPI.GetXformVectors(&trans, &rot, &scale, &pivot, &rotOrder, timeCode);
            keyframe.translation = toGlm(trans);
            keyframe.scaling = toGlm(scale);
            keyframe.time = timeCode / timeCodesPerSecond;

            if (pivot != GfVec3f(0.f, 0.f, 0.f))
            {
                logWarning("Ignoring non-zero pivot extracted from '{}'.", xformAPI.GetPath().GetString());
            }

            glm::quat qx = glm::angleAxis(glm::radians(rot[0]), glm::vec3(1.f, 0.f, 0.f));
            glm::quat qy = glm::angleAxis(glm::radians(rot[1]), glm::vec3(0.f, 1.f, 0.f));
            glm::quat qz = glm::angleAxis(glm::radians(rot[2]), glm::vec3(0.f, 0.f, 1.f));

            switch (rotOrder)
            {
            case UsdGeomXformCommonAPI::RotationOrder::RotationOrderXYZ:
                keyframe.rotation = qz * qy * qx;
                break;
            case UsdGeomXformCommonAPI::RotationOrder::RotationOrderXZY:
                keyframe.rotation = qy * qz * qx;
                break;
            case UsdGeomXformCommonAPI::RotationOrder::RotationOrderYXZ:
                keyframe.rotation = qz * qx * qy;
                break;
            case UsdGeomXformCommonAPI::RotationOrder::RotationOrderYZX:
                keyframe.rotation = qx * qz * qy;
                break;
            case UsdGeomXformCommonAPI::RotationOrder::RotationOrderZXY:
                keyframe.rotation = qy * qx * qz;
                break;
            case UsdGeomXformCommonAPI::RotationOrder::RotationOrderZYX:
                keyframe.rotation = qx * qy * qz;
                break;
            default:
                FALCOR_UNREACHABLE();
            }

            return keyframe;
        }
    }

    void PrototypeGeom::addAnimation(const UsdGeomXformable& xformable)
    {
        logDebug("Creating prototype-internal animation for '{}'.", xformable.GetPath().GetString());

        FALCOR_ASSERT(xformable.TransformMightBeTimeVarying());

        std::vector<double> times;
        xformable.GetTimeSamples(&times);

        auto xformAPI = UsdGeomXformCommonAPI(xformable);

        AnimationKeyframes animation;

        for (double t : times)
        {
            Animation::Keyframe keyframe = createKeyframe(xformAPI, t, timeCodesPerSecond);
            animation.keyframes.push_back(keyframe);
        }

        NodeID nodeID{ nodes.size() };
        animation.targetNodeID = nodeID;
        animations.push_back(std::move(animation));
        nodes.push_back(makeNode(xformable.GetPath().GetString(), nodeStack.back()));
        nodeStack.push_back(nodeID);
    }

    void ImporterContext::createEnvMap(const UsdPrim& lightPrim)
    {
        const UsdAttribute texAttribute = UsdLuxDomeLight(lightPrim).GetTextureFileAttr();
        if (!texAttribute)
        {
            logWarning("No texture attribute specified for dome light '{}'. Ignoring.", lightPrim.GetPath().GetString());
            return;
        }

        SdfAssetPath path;
        texAttribute.Get(&path, UsdTimeCode::EarliestTime());

        UsdLuxDomeLight domeLight(lightPrim);
        float exposure = 0.f;
        const UsdAttribute exposureAttr = domeLight.GetExposureAttr();
        if (exposureAttr)
        {
            exposureAttr.Get(&exposure, UsdTimeCode::EarliestTime());
        }

        float intens = 1.f;
        const UsdAttribute intensAttr = domeLight.GetIntensityAttr();
        if (intensAttr)
        {
            intensAttr.Get(&intens, UsdTimeCode::EarliestTime());
        }

        float3 color = { 1.f, 1.f, 1.f };
        const UsdAttribute colorAttr = domeLight.GetColorAttr();
        if (colorAttr)
        {
            GfVec3f vec;
            colorAttr.Get(&vec, UsdTimeCode::EarliestTime());
            color = float3(vec[0], vec[1], vec[2]);
        }

        std::string envMapPath(path.GetResolvedPath());

        if (envMapPath.empty())
        {
            logError("Failed to resolve environment map path '{}' for light '{}'.", path.GetAssetPath(), lightPrim.GetPath().GetString());
            return;
        }

        EnvMap::SharedPtr pEnvMap = EnvMap::createFromFile(envMapPath);

        if (pEnvMap == nullptr)
        {
            logError("Failed to create environment map for light '{}'.", lightPrim.GetPath().GetString());
            return;
        }

        pEnvMap->setIntensity(std::pow(2.f, exposure) * intens);
        pEnvMap->setTint(color);

        // The local-to-world transform includes a 90-degree rotation about X to account for the difference
        // between USD dome light orientation (up=+Z) and Falcor env map orientation (up=+Y), as well as
        // a 90-degree around Y in world space to account for differences if longitudinal mapping.
        // FIXME: This assumes a static local to world xform. Should support rigid body animation
        rmcv::mat4 xform = rmcv::eulerAngleY(glm::radians(90.f)) * getLocalToWorldXform(UsdGeomXformable(lightPrim)) * rmcv::eulerAngleX(glm::radians(90.f));

        float3 rotation;
        // Extract rotation from the computed transform
        rmcv::extractEulerAngleXYZ(xform, rotation.x, rotation.y, rotation.z);
        pEnvMap->setRotation(glm::degrees(rotation));
        builder.setEnvMap(pEnvMap);
    }

    void ImporterContext::addLight(const UsdPrim& lightPrim, Light::SharedPtr pLight, NodeID parentId)
    {
        NodeID nodeId = builder.addNode(makeNode(lightPrim.GetName(), parentId));
        pLight->setNodeID(nodeId);
        pLight->setHasAnimation(builder.isNodeAnimated(parentId));
        builder.addLight(pLight);
    }

    void ImporterContext::createDistantLight(const UsdPrim& lightPrim)
    {
        UsdLuxDistantLight distantLight(lightPrim);

        float3 intensity = getLightIntensity(distantLight);
        float angle = getAttribute(distantLight.GetAngleAttr(), 0.f);

        DistantLight::SharedPtr pLight = DistantLight::create(lightPrim.GetName());
        pLight->setIntensity(intensity);
        pLight->setWorldDirection(float3(0.f, 0.f, -1.f));
        pLight->setAngle(float(0.5 * angle * M_PI / 180));

        addLight(lightPrim, pLight, nodeStack.back());
    }

    void ImporterContext::createRectLight(const UsdPrim& lightPrim)
    {
        UsdLuxRectLight rectLight(lightPrim);

        float3 intensity = getLightIntensity(rectLight);

        float width = getAttribute(rectLight.GetWidthAttr(), 1.f);
        float height = getAttribute(rectLight.GetHeightAttr(), 1.f);

        AnalyticAreaLight::SharedPtr pLight = RectLight::create(lightPrim.GetName());
        pLight->setIntensity(intensity);
        // Scale width and height to account for the fact that Falcor's 'unit' rect light extends
        // from (-1,-1) to (1,1), vs. USD's (-0.5,-0.5) to (0.5,0.5).
        // Flip Z to emit along -Z axis, flip X to preserve handedness.
        pLight->setScaling(float3(-width / 2.f, height / 2.f, -1.f));

        addLight(lightPrim, pLight, nodeStack.back());
    }

    void ImporterContext::createSphereLight(const UsdPrim& lightPrim)
    {
        UsdLuxSphereLight sphereLight(lightPrim);

        float3 intensity = getLightIntensity(sphereLight);

        float radius = getAttribute(sphereLight.GetRadiusAttr(), 0.5f);

        AnalyticAreaLight::SharedPtr pLight = SphereLight::create(lightPrim.GetName());
        pLight->setIntensity(intensity);
        pLight->setScaling(float3(radius, radius, radius));

        addLight(lightPrim, pLight, nodeStack.back());
    }

    void ImporterContext::createDiskLight(const UsdPrim& lightPrim)
    {
        UsdLuxDiskLight diskLight(lightPrim);

        float3 intensity = getLightIntensity(diskLight);

        float radius = getAttribute(diskLight.GetRadiusAttr(), 0.5f);

        AnalyticAreaLight::SharedPtr pLight = DiscLight::create(lightPrim.GetName());
        pLight->setIntensity(intensity);

        // Flip Z to emit along -Z axis, flip X to preserve handedness. Equivalent to a scale by (radius, radius, 1), followed by a 180 degree rotation around Y.
        pLight->setScaling(float3(-radius, radius, -1.f));

        addLight(lightPrim, pLight, nodeStack.back());
    }

    // DEMO21 Opera -- tdavidovic: I see no problem with tessellating lights
    void ImporterContext::createMeshedDiskLight(const UsdPrim& lightPrim)
    {
        UsdLuxDiskLight diskLight(lightPrim);

        float3 intensity = getLightIntensity(diskLight);

        float radius = getAttribute(diskLight.GetRadiusAttr(), 0.5f);

        // Demo hack: Replace disc analytic light with mesh, bind magic IES material
        VtArray<GfVec3f> points;
        VtArray<GfVec3f> normals;
        VtArray<int> vertexCounts;
        VtArray<int> vertexIndices;

        const uint32_t vertexCount = 32;
        for (uint32_t i = 0; i < vertexCount; ++i)
        {
            float theta = (float)(2. * M_PI * i) / vertexCount;
            float x = radius * cosf(theta);
            float y = radius * sinf(theta);
            GfVec3f v(x, y, 0.f);
            points.push_back(v);

            vertexCounts.push_back(3);
            vertexIndices.push_back((i + 1) % vertexCount);
            vertexIndices.push_back(i);
            vertexIndices.push_back(vertexCount - 1);
        }

        // Disc center
        points.push_back(GfVec3f(0.f, 0.f, 0.f));

        // Constant normal
        normals.push_back(GfVec3f(0.f, 0.f, -1.f));

        SdfPath newPath(lightPrim.GetPath().AppendPath(SdfPath("mesh")));
        UsdGeomMesh mesh = UsdGeomMesh::Define(pStage, newPath);
        logWarning("Replacing analytic disc light '" + mesh.GetPath().GetString() + "' with mesh + IES material.");

        mesh.CreateSubdivisionSchemeAttr(VtValue(UsdGeomTokens->none));
        mesh.CreatePointsAttr(VtValue(points));
        mesh.SetNormalsInterpolation(UsdGeomTokens->constant);
        mesh.CreateNormalsAttr(VtValue(normals));
        mesh.CreateFaceVertexCountsAttr(VtValue(vertexCounts));
        mesh.CreateFaceVertexIndicesAttr(VtValue(vertexIndices));

        UsdShadeMaterial material = UsdShadeMaterial::Define(pStage, newPath.AppendPath(SdfPath("IES")));
        UsdShadeMaterialBindingAPI(mesh).Bind(material);

        auto materialName = material.GetPath().GetString();
        auto pMaterial = StandardMaterial::create(materialName);
        pMaterial->setEmissiveColor(intensity);
        pMaterial->setLightProfileEnabled(true);

        std::lock_guard<std::mutex> lock(materialMutex);
        builder.addMaterial(std::move(pMaterial));
        localDict[materialName] = materialName;

        addMesh(mesh.GetPrim());
        addGeomInstance(lightPrim.GetName(), mesh.GetPrim(), float4x4(), float4x4());
    }

    bool ImporterContext::createCamera(const UsdPrim& cameraPrim)
    {
        const UsdGeomCamera camera(cameraPrim);
        GfCamera cam = camera.GetCamera(0.f);

        Camera::SharedPtr pCamera = Camera::create(cameraPrim.GetName());

        NodeID nodeId = builder.addNode(makeNode(cameraPrim.GetName(), nodeStack.back()));
        pCamera->setNodeID(nodeId);
        pCamera->setHasAnimation(builder.isNodeAnimated(nodeId));

        float focalDistance = std::max(1.f, cam.GetFocusDistance());
        rmcv::mat4 view(toRMCV(cam.GetTransform()));

        float3 pos = view * float4(0.f, 0.f, 0.f, 1.f);
        float3 target = view * float4(0.f, 0.f, -focalDistance, 1.f);
        float3 up = view * float4(0.f, 1.f, 0.f, 0.f);

        pCamera->setPosition(pos);
        pCamera->setTarget(target);
        pCamera->setUpVector(up);

        float fl = cam.GetFocalLength();
        // Focal length, per the USD spec, is supposed to be specified in tenths of a USD world unit.
        // However, it seems to always be specified in mm in OV assets.
        pCamera->setFocalLength(fl);

        float fstop = cam.GetFStop();

        // Check the custom OV depthOfField attribute
        UsdAttribute dofAttrib(cameraPrim.GetAttribute(TfToken("depthOfField")));
        if (dofAttrib)
        {
            bool useDof;
            if (dofAttrib.Get<bool>(&useDof) && !useDof)
            {
                fstop = 0.f;
            }
        }

        if (fstop > 0.f)
        {
            // The aperture radius is specified in Falcor world unit (meters).
            pCamera->setApertureRadius(.001f * 0.5f * fl / fstop);
        }

        pCamera->setFocalDistance(metersPerUnit * focalDistance);
        GfRange1f clipRange = cam.GetClippingRange();
        pCamera->setDepthRange(clipRange.GetMin() * metersPerUnit, clipRange.GetMax() * metersPerUnit);

        // If film width is given, deduce the height from the aspect ratio.
        // Otherwise set film height and deduce the width from the aspect ratio.
        if (camera.GetPrim().GetAttribute(UsdGeomTokens->horizontalAperture).IsAuthored())
        {
            pCamera->setFrameWidth(cam.GetHorizontalAperture());
        }
        else
        {
            pCamera->setFrameHeight(cam.GetVerticalAperture());
        }

        builder.addCamera(pCamera);
        return true;
    }

    bool ImporterContext::createPointInstanceKeyframes(const UsdGeomPointInstancer& instancer, std::vector<std::vector<Animation::Keyframe>>& keyframes)
    {
        logDebug("Creating PointInstancer keyframes for '{}'.", instancer.GetPath().GetString());

        UsdAttribute posAttr(instancer.GetPositionsAttr());

        std::vector<double> times;
        if (!posAttr.GetTimeSamples(&times) || times.size() < 2)
        {
            // No need to create an animation.
            return false;
        }

        std::vector<UsdTimeCode> timeCodes;
        timeCodes.resize(times.size());
        for (uint32_t i = 0; i < times.size(); ++i)
        {
            timeCodes[i] = times[i];
        }

        std::vector<VtMatrix4dArray> instXforms;
        if (!instancer.ComputeInstanceTransformsAtTimes(&instXforms, timeCodes, UsdTimeCode::EarliestTime(), UsdGeomPointInstancer::ProtoXformInclusion::ExcludeProtoXform))
        {
            logError("Error occurred computing sampled point instancer transforms for '{}'. Ignoring prim.", instancer.GetPath().GetString());
            return false;
        }

        // instXforms is a vector of length equal to the number of time codes.
        // Each element of the vector holds an array of size equal to the number of instances.
        // We need to, in effect, transpose this layout when constructing the vector of Animations.
        keyframes.resize(instXforms[0].size());
        FALCOR_ASSERT(instXforms.size() == times.size());

        // For each time sample
        for (uint32_t i = 0; i < instXforms.size(); ++i)
        {
            const auto& matrices = instXforms[i];
            double time = times[i];

            // For each instance
            for (uint32_t j = 0; j < matrices.size(); ++j)
            {
                const auto& matrix = matrices[j];
                rmcv::mat4 glmMat = toRMCV(matrix);
                Animation::Keyframe keyframe;
                float3 skew;
                float4 persp;
                rmcv::decompose(glmMat, keyframe.scaling, keyframe.rotation, keyframe.translation, skew, persp);
                keyframe.time = time / timeCodesPerSecond;
                keyframes[j].push_back(keyframe);
            }
        }

        return true;
    }

    NodeID ImporterContext::createAnimation(const UsdGeomXformable& xformable)
    {
        logDebug("Creating animation for '{}'.", xformable.GetPath().GetString());
        FALCOR_ASSERT(xformable.TransformMightBeTimeVarying());

        std::vector<double> times;
        xformable.GetTimeSamples(&times);

        auto xformAPI = UsdGeomXformCommonAPI(xformable);

        // Gather keyframes
        auto pAnimation = Animation::create(xformable.GetPath().GetString(), NodeID::Invalid(), times.back() / timeCodesPerSecond);

        for (double t : times)
        {
            Animation::Keyframe keyframe = createKeyframe(xformAPI, t, timeCodesPerSecond);
            pAnimation->addKeyframe(keyframe);
        }

        NodeID nodeID = builder.addNode(makeNode(xformable.GetPath().GetString(), nodeStack.back()));
        pAnimation->setNodeID(nodeID);
        builder.addAnimation(pAnimation);
        return nodeID;
    }

    void ImporterContext::addCachedCurve(Curve& curve)
    {
        CachedCurve cachedCurve;
        cachedCurve.tessellationMode = curve.tessellationMode;
        cachedCurve.geometryID = curve.geometryID;

        cachedCurve.timeSamples.resize(curve.timeSamples.size());
        std::memcpy(cachedCurve.timeSamples.data(), curve.timeSamples.data(), cachedCurve.timeSamples.size() * sizeof(double));

        // Make sure topology doesn't change across keyframes.
        const auto& refIndexData = curve.processedCurves[0].indexData;
        bool isSameTopology = true;
        for (size_t i = 1; i < cachedCurve.timeSamples.size(); i++)
        {
            const auto& indexData = curve.processedCurves[i].indexData;
            if (indexData.size() != refIndexData.size())
            {
                isSameTopology = false;
                break;
            }

            NumericRange<size_t> range(0, indexData.size());
            std::for_each(std::execution::par, range.begin(), range.end(),
                [&](size_t j)
                {
                    isSameTopology |= (indexData[j] == refIndexData[j]);
                }
            );
            if (!isSameTopology) break;
        }
        if (!isSameTopology)
        {
            throw ImporterError(path, "The topology/indexing of curves changes across keyframes. Only dynamic vertex positions are supported.");
        }

        cachedCurve.indexData.resize(refIndexData.size());
        std::memcpy(cachedCurve.indexData.data(), refIndexData.data(), cachedCurve.indexData.size() * sizeof(uint32_t));

        cachedCurve.vertexData.resize(curve.processedCurves.size());
        for (size_t i = 0; i < cachedCurve.vertexData.size(); i++)
        {
            cachedCurve.vertexData[i].resize(curve.processedCurves[i].staticData.size());
            for (size_t j = 0; j < cachedCurve.vertexData[i].size(); j++)
            {
                cachedCurve.vertexData[i][j].position = curve.processedCurves[i].staticData[j].position;
            }

            // Deallocate memory.
            if (i > 0)
            {
                curve.processedCurves[i].staticData = std::vector<StaticCurveVertexData>();
            }
        }

        cachedCurves.push_back(cachedCurve);
    }

    ImporterContext::ImporterContext(const std::filesystem::path& path, UsdStageRefPtr pStage, SceneBuilder& builder, const Dictionary& dict, TimeReport& timeReport, bool useInstanceProxies /*= false*/)
        : path(path)
        , pStage(pStage)
        , builder(builder)
        , dict(dict)
        , timeReport(timeReport)
        , useInstanceProxies(useInstanceProxies)
    {
        mpPreviewSurfaceConverter = std::make_unique<PreviewSurfaceConverter>();
    }


    Material::SharedPtr ImporterContext::resolveMaterial(const UsdPrim& prim, const UsdShadeMaterial& material, const std::string& primName)
    {
        Material::SharedPtr pMaterial;

        if (material)
        {
            // Note that this call will block if another thread is in the process of converting the same material.
            pMaterial = mpPreviewSurfaceConverter->convert(material, primName, gpDevice->getRenderContext());
        }
        if (!pMaterial)
        {
            logInfo("No material bound to '{}'. Using a default material.", primName);
            pMaterial = getDefaultMaterial(prim);
        }
        return pMaterial;
    }

    Material::SharedPtr ImporterContext::getDefaultMaterial(const UsdPrim& prim)
    {
        Material::SharedPtr pMaterial;
        float3 defaultColor;

        if (prim.IsA<UsdGeomBasisCurves>())
        {
            defaultColor = float3(0.8f, 0.4f, 0.05f);
        }
        else if (prim.IsA<UsdGeomMesh>())
        {
            defaultColor = float3(0.7f, 0.7f, 0.7f);
        }
        else
        {
            throw ImporterError(path, "Conversion error on USD prim '{}'. Cannot create default material for non-mesh, non-curve prim.", prim.GetPath().GetString());
        }

        // If there is a displayColor attribute associated with this prim, use it as the default color.
        UsdAttribute displayColorAttr;
        if (prim.IsA<UsdGeomGprim>())
        {
            displayColorAttr = UsdGeomGprim(prim).GetDisplayColorAttr();
        }
        else
        {
            displayColorAttr = UsdGeomGprim(prim.GetParent()).GetDisplayColorAttr();
        }
        if (displayColorAttr)
        {
            VtVec3fArray displayColor;
            displayColorAttr.Get(&displayColor, UsdTimeCode::EarliestTime());
            if (displayColor.size() > 0)
            {
                // Ignore any interpolation modes and simply use the first color
                GfVec3f c = displayColor[0];
                defaultColor = float3(c[0], c[1], c[2]);
            }
        }

        {
            std::lock_guard<std::mutex> lock(materialMutex);

            if (prim.IsA<UsdGeomMesh>())
            {
                auto it = defaultMaterialMap.find(defaultColor);
                if (it != defaultMaterialMap.end())
                {
                    return it->second;
                }
                StandardMaterial::SharedPtr pDefaultMaterial = StandardMaterial::create("default-mesh-" + std::to_string(defaultMaterialMap.size()));
                pDefaultMaterial->setBaseColor(float4(defaultColor, 1.f));
                pDefaultMaterial->setRoughness(0.3f);
                pDefaultMaterial->setMetallic(0.f);
                pDefaultMaterial->setIndexOfRefraction(1.5f);
                pDefaultMaterial->setDoubleSided(true);
                defaultMaterialMap[defaultColor] = pDefaultMaterial;
                return pDefaultMaterial;
            }
            else if (prim.IsA<UsdGeomBasisCurves>())
            {
                auto it = defaultCurveMaterialMap.find(defaultColor);
                if (it != defaultCurveMaterialMap.end())
                {
                    return it->second;
                }
                HairMaterial::SharedPtr pDefaultCurveMaterial = HairMaterial::create("default-curve-" + std::to_string(defaultCurveMaterialMap.size()));
                pDefaultCurveMaterial->setBaseColor(float4(defaultColor, 1.f));
                pDefaultCurveMaterial->setSpecularParams(float4(kDefaultCurveLongitudinalRoughness, kDefaultCurveAzimuthalRoughness, kDefaultScaleAngleDegree, 0.f));
                pDefaultCurveMaterial->setIndexOfRefraction(kDefaultCurveIOR);
                defaultCurveMaterialMap[defaultColor] = pDefaultCurveMaterial;
                return pDefaultCurveMaterial;
            }
            else
            {
                // Non-curve, non-mesh prims should result in an exception above.
                FALCOR_UNREACHABLE();
            }
        }
        return nullptr;
    }


    void ImporterContext::addMesh(const UsdPrim& prim)
    {
        // It's possible that the same mesh gprim may be appear both as a regular Mesh and as part of a prototype.
        // Make sure to only add it once to meshes and meshMap.
        if (geomMap.find(prim) == geomMap.end())
        {
            Mesh mesh{ prim };

            // Mesh will be added at the next index
            size_t index = meshes.size();

            // Get time samples from the points attribute
            UsdGeomPointBased geomPointBased(prim);
            geomPointBased.GetPointsAttr().GetTimeSamples(&mesh.timeSamples);

            if (!gpFramework->getSettings().getAttribute(prim.GetPath().GetString(), "usdImporter:enableMotion", true))
                mesh.timeSamples.clear();

            if (mesh.timeSamples.size() == 0)
            {
                mesh.timeSamples.push_back(0.0);
                meshTasks.push_back(MeshProcessingTask{ (uint32_t)index, 0 });
            }
            else
            {
                // Create mesh processing tasks, one for each time sample
                for (uint32_t i = 0; i < mesh.timeSamples.size(); ++i)
                {
                    MeshProcessingTask task{ (uint32_t)index, i };

                    // Add the default mesh (or first time-sample) to mesh tasks
                    if (i == 0) meshTasks.push_back(task);

                    // Add task for processing keyframe
                    if(mesh.timeSamples.size() > 1) meshKeyframeTasks.push_back(task);
                }
            }

            meshes.push_back(std::move(mesh));
            geomMap.emplace(prim, index);
        }
    }

    void ImporterContext::createPrototype(const UsdPrim& rootPrim)
    {
        PrototypeGeom proto(rootPrim, timeCodesPerSecond);
        UsdGeomXformCache xformCache(UsdTimeCode::EarliestTime());

        logDebug("Creating prototype '{}'.", rootPrim.GetPath().GetString());

        // Traverse prototype prim, ignoring everything but transforms and gprims
        Usd_PrimFlagsPredicate pred = UsdPrimDefaultPredicate;
        UsdPrimRange range = UsdPrimRange::PreAndPostVisit(rootPrim, pred);
        for (auto it = range.begin(); it != range.end(); ++it)
        {
            UsdPrim prim = *it;
            std::string primName = prim.GetPath().GetString();

            if (!it.IsPostVisit())
            {
                // Pre visits

                // If this prim has an xform associated with it, push it onto the xform stack
                if (prim.IsA<UsdGeomXformable>())
                {
                    proto.pushNode(UsdGeomXformable(prim));
                }

                if (prim.IsA<UsdGeomImageable>() && !isRenderable(UsdGeomImageable(prim)))
                {
                    logDebug("Pruning non-renderable prim '{}'.", primName);
                    it.PruneChildren();
                    continue;
                }

                if (prim.IsInstance())
                {
                    const UsdPrim protoPrim(prim.GetMaster());

                    if (!protoPrim.IsValid())
                    {
                        logError("No valid prototype prim for instance '{}'.", primName);
                    }
                    else
                    {
                        logDebug("Adding instance '{}' of '{}' to prototype '{}'.", primName, protoPrim.GetPath().GetString(), rootPrim.GetPath().GetString());
                        PrototypeInstance inst{ primName, protoPrim, proto.nodeStack.back() };
                        proto.addPrototypeInstance(inst);
                    }
                }
                else if (prim.IsA<UsdGeomPointInstancer>())
                {
                    logDebug("Processing instanced PointInstancer '{}'.", primName);
                    createPointInstances(prim, &proto);
                    it.PruneChildren();
                }
                else if (prim.IsA<UsdGeomMesh>())
                {
                    logDebug("Adding mesh '{}'.", primName);
                    addMesh(prim);
                    proto.addGeomInstance(primName, prim, rmcv::mat4(1.f), rmcv::mat4(1.f));
                }
                else if (prim.IsA<UsdSkelRoot>() ||
                    prim.IsA<UsdSkelSkeleton>() ||
                    prim.IsA<UsdSkelAnimation>())
                {
                    continue;
                }
                else if (prim.IsA<UsdGeomBasisCurves>())
                {
                    logDebug("Ignoring unsupported basis curves '{}' encountered in prototype prim.", primName);
                    it.PruneChildren();
                }
                else if (prim.IsA<UsdLuxLight>())
                {
                    logWarning("Ignoring light '{}' encountered in prototype prim.", primName);
                    it.PruneChildren();
                }
                else if (prim.IsA<UsdGeomXform>())
                {
                    logDebug("Processing xform '{}'.", primName);
                    // Processing of this UsdGeomXformable performed above
                }
                else if (prim.IsA<UsdShadeMaterial>() ||
                    prim.IsA<UsdShadeShader>() ||
                    prim.IsA<UsdGeomSubset>())
                {
                    // No processing to do; ignore without issuing a warning.
                    it.PruneChildren();
                }
                else if (prim.IsA<UsdGeomScope>())
                {
                    logDebug("Processing scope '{}'.", primName);
                }
                else if (!prim.GetTypeName().GetString().empty())
                {
                    logWarning("Ignoring prim '{}' of unsupported type {} while traversing prototype prim.", primName, prim.GetTypeName().GetString());
                    it.PruneChildren();
                }
            }
            else
            {
                // Post visits
                if (prim.IsA<UsdGeomXformable>())
                {
                    // Pop node
                    proto.popNode();
                }
            }
        }

        // Add the prototype
        size_t index = prototypeGeoms.size();
        prototypeGeoms.push_back(proto);
        prototypeGeomMap.emplace(proto.protoPrim, index);
    }

    bool ImporterContext::hasPrototype(const UsdPrim& protoPrim) const
    {
        return prototypeGeomMap.find(protoPrim) != prototypeGeomMap.end();
    }

    void ImporterContext::addGeomInstance(const std::string& name, const UsdPrim& prim, const rmcv::mat4& xform, const rmcv::mat4& bindXform)
    {
        geomInstances.push_back(GeomInstance{ name, prim, xform, bindXform, nodeStack.back()});
    }

    void ImporterContext::addPrototypeInstance(const PrototypeInstance& protoInst)
    {
        prototypeInstances.push_back(protoInst);
    }


    void ImporterContext::createPointInstances(const UsdPrim& prim, PrototypeGeom* proto)
    {
        std::string primName = prim.GetPath().GetString();
        UsdGeomPointInstancer instancer(prim);

        SdfPathVector prototypePaths;
        if (!instancer.GetPrototypesRel().GetForwardedTargets(&prototypePaths))
        {
            logError("Error occurred gathering prototypes for point instancer '{}'.", primName);
        }

        UsdAttribute protoIndicesAttr(instancer.GetProtoIndicesAttr());
        if (!protoIndicesAttr.IsDefined())
        {
            logError("Point instancer '{}' has no prototype indices. Ignoring prim.", primName);
            return;
        }

        // We make use of the the same machinery used to create general instances, namely
        // traversal of a Prototype prim to create an underlying prototype, followed by adding instances.

        // Prototypes prims are gathered during iteration over the prototype paths.  Most often, prototypes are
        // specified as children of the PointInstancer, but this isn't guaranteed.
        std::vector<UsdPrim> protoPrims;

        for (auto path : prototypePaths)
        {
            logDebug("Processing point instancer prototype '{}'.", path.GetString());
            UsdPrim protoPrim(pStage->GetPrimAtPath(path));

            if (!protoPrim.IsDefined())
            {
                logError("Point instancer '{}' references nonexistent prim '{}'. Ignoring.", primName, path.GetString());
                continue;
            }

            protoPrims.push_back(protoPrim);

            // Create a prototype from this prim if one doesn't already exist.
            if (!hasPrototype(protoPrim))
            {
                // Start a new node stack for this traversal.
                pushNodeStack();

                createPrototype(protoPrim);

                // Pop the node stack we created, restoring the one being used for the in-progress traversal.
                popNodeStack();
            }
        }

        VtIntArray protoIndices;
        protoIndicesAttr.Get(&protoIndices, UsdTimeCode::EarliestTime());

        std::vector<std::vector<Animation::Keyframe>> keyframes;
        VtMatrix4dArray instXforms;

        if (createPointInstanceKeyframes(instancer, keyframes))
        {
            if (protoIndices.size() != keyframes.size())
            {
                logError("Point instancer '{}' has {} prototype indices but {} sampled transforms.", primName, protoIndices.size(), keyframes.size());
                return;
            }
        }
        else
        {
            // Compute 4x4 transforms for each instance at the earliest time sample.
            // The prototype xform is included in its definition, so we exclude it in the computed instance xforms.
            if (!instancer.ComputeInstanceTransformsAtTime(&instXforms, UsdTimeCode::EarliestTime(), UsdTimeCode::EarliestTime(), UsdGeomPointInstancer::ProtoXformInclusion::ExcludeProtoXform))
            {
                logError("Error occurred computing point instancer transforms for '{}'. Ignoring prim.", primName);
                return;
            }
            if (protoIndices.size() != instXforms.size())
            {
                logError("Point instancer '{}' has {} prototype indices but {} transforms.", primName, protoIndices.size(), instXforms.size());
                return;
            }
        }

        // Create instances from the prototypes.
        for (size_t i = 0; i < protoIndices.size(); ++i)
        {
            UsdPrim& protoPrim(protoPrims[protoIndices[i]]);
            std::string instanceName(protoPrim.GetPath().GetString() + "_" + std::to_string(i));
            PrototypeInstance protoInst = {instanceName, protoPrim};
            if (keyframes.size() > 0)
            {
                protoInst.keyframes = std::move(keyframes[i]);
            }
            else
            {
                protoInst.xform = toRMCV(instXforms[i]);
            }

            if (proto)
            {
                protoInst.parentID = proto->nodeStack.back();
                proto->addPrototypeInstance(protoInst);
            }
            else
            {
                protoInst.parentID = nodeStack.back();
                addPrototypeInstance(protoInst);
            }
        }
    }

    void ImporterContext::addCurve(const UsdPrim& curvePrim)
    {
        if (curves.size() >= std::numeric_limits<uint32_t>::max())
        {
            throw RuntimeError("Curve count exceeds the maximum");
        }

        uint32_t index = (uint32_t)curves.size();

        CurveTessellationMode mode = CurveTessellationMode::LinearSweptSphere;

        if (is_set(builder.getFlags(), SceneBuilder::Flags::TessellateCurvesIntoPolyTubes))
        {
            mode = CurveTessellationMode::PolyTube;
        }

        // Example of Attribute
        {
            if(auto strMode = gpFramework->getSettings().getAttribute<std::string>(curvePrim.GetPath().GetString(), "curves:mode"))
            {
                if(strMode == "lss")
                    mode = CurveTessellationMode::LinearSweptSphere;
                else if(strMode == "polytube")
                    mode = CurveTessellationMode::PolyTube;
            }
        }

        curves.push_back(Curve{ curvePrim, mode });
        curveMap.emplace(curvePrim, index);
    }

    void ImporterContext::createSkeleton(const UsdPrim& prim)
    {
        UsdSkelRoot skelRoot(prim);
        skelCache.Populate(skelRoot);

        std::vector<UsdSkelBinding> bindings;
        skelCache.ComputeSkelBindings(skelRoot, &bindings);

        if (bindings.empty())
        {
            logWarning("SkelRoot '{}' does not contain any skeleton bindings.", prim.GetName().GetString());
            return;
        }

        skeletons.push_back(Skeleton());
        Skeleton& skelData = skeletons.back();
        skelData.name = prim.GetName().GetString();
        skelData.parentID = nodeStack.back();

        // Iterate over Skeletons under this SkelRoot
        for (auto& binding : bindings)
        {
            UsdSkelSkeletonQuery skelQuery = skelCache.GetSkelQuery(binding.GetSkeleton());
            UsdSkelAnimQuery animQuery = skelQuery.GetAnimQuery();

            if (!animQuery)
            {
                // FIXME: An associated animation shouldn't be required.
                logWarning("SkelRoot '{}' contains a skeleton swithout an associated animation, which is not supported. Ignoring.", skelData.name);
                continue;
            }

            VtArray<TfToken> joints = skelQuery.GetJointOrder();

            skelData.subskeletons.emplace_back();
            Skeleton::SubSkeleton& subskeleton = skelData.subskeletons.back();

            // Record helpful data about each mesh affected by this Skeleton
            for (const auto& skinningQuery : binding.GetSkinningTargets())
            {
                UsdPrim prim = skinningQuery.GetPrim();
                if (prim.IsA<UsdGeomMesh>())
                {
                    // A mesh's joint indices may only refer to a subset of the skeleton.
                    // Use USD's JointMapper to reliably determine if we need to remap,
                    // but it isn't clear how to actually use the class to do what we want...
                    auto pMapper = skinningQuery.GetJointMapper();
                    if (pMapper && !pMapper->IsIdentity())
                    {
                        // Create lookup table for Skeleton joints by name
                        std::unordered_map<std::string, uint32_t> skelJointToIndex;
                        for (uint32_t i = 0; i < uint32_t(joints.size()); i++)
                        {
                            skelJointToIndex.emplace(joints[i].GetString(), i);
                        }

                        VtArray<TfToken> meshJoints;
                        skinningQuery.GetJointOrder(&meshJoints);

                        // Create mapping. For every joint referenced in the mesh, look it up in the skeleton
                        std::vector<uint32_t> meshToSkeletonJoints(meshJoints.size());
                        for (uint32_t i = 0; i < uint32_t(meshJoints.size()); i++)
                        {
                            meshToSkeletonJoints[i] = skelJointToIndex.at(meshJoints[i]);
                        }
                        subskeleton.skinnedMeshes.emplace(prim, std::move(meshToSkeletonJoints));
                    }
                    else
                    {
                        // Joints do not need remapping
                        subskeleton.skinnedMeshes.emplace(prim, std::vector<uint32_t>());
                    }

                    meshSkelMap.emplace(prim, std::make_pair(skeletons.size() - 1, skelData.subskeletons.size() - 1));
                }
                else
                {
                    logWarning("Cannot apply skinning to non-mesh prim '{}' of type {}.", prim.GetName().GetString(), prim.GetTypeName().GetString());
                }
            }

            // Load bone data
            UsdSkelTopology skelTopo = skelQuery.GetTopology();

            VtArray<GfMatrix4d> bindTransform;
            skelQuery.GetJointWorldBindTransforms(&bindTransform);

            VtArray<GfMatrix4d> restTransform;
            skelQuery.ComputeJointLocalTransforms(&restTransform, UsdTimeCode::EarliestTime(), true);

            for (size_t i = 0; i < joints.size(); i++)
            {
                SceneBuilder::Node n;
                n.name = joints[i].GetString();
                n.transform = toRMCV(restTransform[i]);
                n.localToBindPose = rmcv::inverse(toRMCV(bindTransform[i]));
                n.parent = skelTopo.IsRoot(i) ? NodeID::Invalid() : NodeID{ skelTopo.GetParent(i) };
                subskeleton.bones.push_back(n);
            }

            std::vector<double> times;
            animQuery.GetJointTransformTimeSamples(&times);

            // Init Animations
            for (uint32_t i = 0; i < uint32_t(subskeleton.bones.size()); i++)
            {
                subskeleton.animations.push_back(Animation::create(subskeleton.bones[i].name, NodeID{ i }, times.back() / timeCodesPerSecond));
            }

            // Load animation data
            VtArray<GfVec3f> trans;
            VtArray<GfQuatf> rot;
            VtArray<GfVec3h> scales;

            for (double t : times)
            {
                // The API samples transforms for all joints at once
                animQuery.ComputeJointLocalTransformComponents(&trans, &rot, &scales, t);

                for (size_t i = 0; i < subskeleton.bones.size(); i++)
                {
                    Animation::Keyframe keyframe;
                    keyframe.time = t / timeCodesPerSecond;
                    keyframe.translation = toGlm(trans[i]);
                    keyframe.rotation = glm::quat(rot[i].GetReal(), toGlm(rot[i].GetImaginary()));
                    keyframe.scaling = toGlm(scales[i]);

                    subskeleton.animations[i]->addKeyframe(keyframe);
                }
            }
        }
    }

    rmcv::mat4 ImporterContext::getLocalToWorldXform(const UsdGeomXformable& prim, UsdTimeCode timeCode)
    {
        rmcv::mat4 localToWorld = toRMCV(prim.ComputeLocalToWorldTransform(timeCode));
        return rootXform * localToWorld;
    }

    void ImporterContext::pushNode(const UsdGeomXformable& prim)
    {
        NodeID nodeID;
        if (prim.TransformMightBeTimeVarying())
        {
            nodeID = createAnimation(prim);
        }
        else
        {
            // The node stack should at least contain the root node.
            FALCOR_ASSERT(nodeStack.size() > 0);
            rmcv::mat4 localTransform;
            bool resets = getLocalTransform(prim, localTransform);
            SceneBuilder::Node node;
            node.name = prim.GetPath().GetString();
            node.transform = localTransform;
            node.parent = resets ? getRootNodeID() : nodeStack.back();
            nodeID = builder.addNode(node);
        }
        nodeStack.push_back(nodeID);
    }

    rmcv::mat4 ImporterContext::getGeomBindTransform(const UsdPrim& usdPrim) const
    {
        GfMatrix4d bindXform(1.0);
        UsdGeomPrimvar geomBindTransform(UsdGeomPrimvarsAPI(usdPrim).GetPrimvar(UsdSkelTokens->primvarsSkelGeomBindTransform));
        if (geomBindTransform.HasValue())
        {
            geomBindTransform.Get(&bindXform, UsdTimeCode::EarliestTime());
        }

        return toRMCV(bindXform);
    }

    void ImporterContext::pushNodeStack(NodeID rootNode)
    {
        nodeStackStartDepth.push_back(nodeStack.size());
        nodeStack.push_back(rootNode);
    }

    void ImporterContext::popNodeStack()
    {
        // We emulate using a separate stack by popping the identity matrix
        nodeStack.pop_back();
        size_t origDepth = nodeStackStartDepth.back();
        if (origDepth != nodeStack.size())
        {
            throw ImporterError(path, "USDImporter transform stack error detected.");
        }
        nodeStackStartDepth.pop_back();
    }

    void ImporterContext::setRootXform(const rmcv::mat4& xform)
    {
        if (rootXformNodeId != NodeID::Invalid())
        {
            throw RuntimeError("ImporterContext::setRootXform() - Root xform has already been set.");
        }

        if (!nodeStack.empty())
        {
            throw RuntimeError("ImporterContext::setRootXform() - node stack must be empty.");
        }

        rootXform = xform;

        // Create a root node that contains the stage-to-world transformation
        SceneBuilder::Node node;
        node.name = "StageRoot";
        node.transform = rootXform;
        rootXformNodeId = builder.addNode(node);
        pushNodeStack(rootXformNodeId);
    }

    void ImporterContext::addCurveInstance(const std::string& name, const UsdPrim& curvePrim, const rmcv::mat4& xform, NodeID parentID)
    {
        curveInstances.push_back(GeomInstance{ name, curvePrim, xform, rmcv::mat4(1.f), parentID });
    }

    void ImporterContext::finalize()
    {
        addSkeletonsToSceneBuilder(*this, timeReport);
        addMeshesToSceneBuilder(*this, timeReport);
        addCurvesToSceneBuilder(*this, timeReport);
    }
}
