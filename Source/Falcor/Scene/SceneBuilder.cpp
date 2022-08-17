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
#include "SceneBuilder.h"
#include "SceneCache.h"
#include "Importer.h"
#include "Curves/CurveConfig.h"
#include "Material/StandardMaterial.h"
#include "Utils/Logger.h"
#include "Utils/Math/Common.h"
#include "Utils/Image/TextureAnalyzer.h"
#include "Utils/Timing/TimeReport.h"
#include "Utils/Scripting/ScriptBindings.h"
#include "Utils/Math/MathHelpers.h"
#include <mikktspace.h>
#include <filesystem>
#include <cmath>

namespace Falcor
{
    namespace
    {
        // Large mesh groups are split in order to reduce the size of the largest BLAS.
        // The target is max 16M triangles per BLAS (= approx 0.5GB post-compaction). Note that this is not a strict limit.
        const size_t kMaxTrianglesPerBLAS = 1ull << 24;

        // Texture coordinates for textured emissive materials are quantized for performance reasons.
        // We'll log a warning if the maximum quantization error exceeds this value.
        const float kMaxTexelError = 0.5f;

        int largestAxis(const float3& v)
        {
            if (v.x >= v.y && v.x >= v.z) return 0;
            else if (v.y >= v.z) return 1;
            else return 2;
        }

        class MikkTSpaceWrapper
        {
        public:
            static std::vector<float4> generateTangents(const SceneBuilder::Mesh& mesh)
            {
                if (!mesh.normals.pData || !mesh.positions.pData || !mesh.texCrds.pData || !mesh.pIndices)
                {
                    logWarning("Can't generate tangent space. The mesh '{}' doesn't have positions/normals/texCrd/indices.", mesh.name);
                    return {};
                }

                // Generate new tangent space.
                SMikkTSpaceInterface mikktspace = {};
                mikktspace.m_getNumFaces = [](const SMikkTSpaceContext* pContext) { return ((MikkTSpaceWrapper*)(pContext->m_pUserData))->getFaceCount(); };
                mikktspace.m_getNumVerticesOfFace = [](const SMikkTSpaceContext* pContext, int32_t face) { return 3; };
                mikktspace.m_getPosition = [](const SMikkTSpaceContext* pContext, float position[], int32_t face, int32_t vert) { ((MikkTSpaceWrapper*)(pContext->m_pUserData))->getPosition(position, face, vert); };
                mikktspace.m_getNormal = [](const SMikkTSpaceContext* pContext, float normal[], int32_t face, int32_t vert) { ((MikkTSpaceWrapper*)(pContext->m_pUserData))->getNormal(normal, face, vert); };
                mikktspace.m_getTexCoord = [](const SMikkTSpaceContext* pContext, float texCrd[], int32_t face, int32_t vert) { ((MikkTSpaceWrapper*)(pContext->m_pUserData))->getTexCrd(texCrd, face, vert); };
                mikktspace.m_setTSpaceBasic = [](const SMikkTSpaceContext* pContext, const float tangent[], float sign, int32_t face, int32_t vert) { ((MikkTSpaceWrapper*)(pContext->m_pUserData))->setTangent(tangent, sign, face, vert); };

                MikkTSpaceWrapper wrapper(mesh);
                SMikkTSpaceContext context = {};
                context.m_pInterface = &mikktspace;
                context.m_pUserData = &wrapper;

                if (genTangSpaceDefault(&context) == false)
                {
                    throw RuntimeError("MikkTSpace failed to generate tangents for the mesh '{}'.", mesh.name);
                }

                return wrapper.mTangents;
            }

        private:
            MikkTSpaceWrapper(const SceneBuilder::Mesh& mesh)
                : mMesh(mesh)
            {
                FALCOR_ASSERT(mesh.indexCount > 0);
                mTangents.resize(mesh.indexCount, float4(0));

                FALCOR_ASSERT_EQ(mesh.indexCount, mMesh.faceCount * 3);
                mPositions.resize(mMesh.faceCount * 3);
                switch (mMesh.positions.frequency)
                {
                case SceneBuilder::Mesh::AttributeFrequency::Constant:
                {
                    std::fill_n(mPositions.begin(), mPositions.size(), mMesh.positions.pData[0]);
                    break;
                }
                case SceneBuilder::Mesh::AttributeFrequency::Uniform:
                {
                    for (uint32_t i = 0; i < mMesh.faceCount; ++i)
                        std::fill_n(mPositions.begin() + i * 3, 3, mMesh.positions.pData[i]);
                    break;
                }
                case SceneBuilder::Mesh::AttributeFrequency::Vertex:
                {
                    FALCOR_ASSERT_EQ(mesh.indexCount, mPositions.size());
                    for (size_t fvarIdx = 0; fvarIdx < mPositions.size(); ++fvarIdx)
                        mPositions[fvarIdx] = mMesh.positions.pData[mMesh.pIndices[fvarIdx]];
                    break;
                }
                case SceneBuilder::Mesh::AttributeFrequency::FaceVarying:
                {
                    memcpy(mPositions.data(), mMesh.positions.pData, mPositions.size() * sizeof(float3));
                    break;
                }
                default:
                    FALCOR_UNREACHABLE();
                }

            }
            const SceneBuilder::Mesh& mMesh;
            std::vector<float4> mTangents;
            std::vector<float3> mPositions;
            int32_t getFaceCount() const { return (int32_t)mMesh.faceCount; }
            void getPosition(float position[], int32_t face, int32_t vert) const { FALCOR_ASSERT_LT(face * 3 + vert, (ssize_t)mPositions.size()); memcpy(position, mPositions.data() + (face * 3 + vert), sizeof(float3)); }
            void getNormal(float normal[], int32_t face, int32_t vert) { *reinterpret_cast<float3*>(normal) = mMesh.getNormal(face, vert); }
            void getTexCrd(float texCrd[], int32_t face, int32_t vert) { *reinterpret_cast<float2*>(texCrd) = mMesh.getTexCrd(face, vert); }

            void setTangent(const float tangent[], float sign, int32_t face, int32_t vert)
            {
                float3 T = *reinterpret_cast<const float3*>(tangent);
                mTangents[face * 3 + vert] = float4(glm::normalize(T), sign);
            }
        };

        void validateVertex(const SceneBuilder::Mesh::Vertex& v, size_t& invalidCount, size_t& zeroCount)
        {
            auto isInvalid = [](const auto& x)
            {
                return glm::any(glm::isinf(x) || glm::isnan(x));
            };
            auto isZero = [](const auto& x)
            {
                return glm::length(x) < 1e-6f;
            };

            if (isInvalid(v.position) || isInvalid(v.normal) || isInvalid(v.tangent) || isInvalid(v.texCrd) || isInvalid(v.boneWeights)) invalidCount++;
            if (isZero(v.normal) || isZero(v.tangent.xyz())) zeroCount++;
        }

        bool compareVertices(const SceneBuilder::Mesh::Vertex& lhs, const SceneBuilder::Mesh::Vertex& rhs, float threshold = 1e-6f)
        {
            using namespace glm;
            if (lhs.position != rhs.position) return false; // Position need to be exact to avoid cracks
            if (lhs.tangent.w != rhs.tangent.w) return false;
            if (lhs.curveRadius != rhs.curveRadius) return false;
            if (lhs.boneIDs != rhs.boneIDs) return false;
            if (any(greaterThan(abs(lhs.normal - rhs.normal), float3(threshold)))) return false;
            if (any(greaterThan(abs(lhs.tangent.xyz - rhs.tangent.xyz), float3(threshold)))) return false;
            if (any(greaterThan(abs(lhs.texCrd - rhs.texCrd), float2(threshold)))) return false;
            if (any(greaterThan(abs(lhs.boneWeights - rhs.boneWeights), float4(threshold)))) return false;
            return true;
        }

        std::vector<uint32_t> compact16BitIndices(const std::vector<uint32_t>& indices)
        {
            if (indices.empty()) return {};
            size_t sz = div_round_up(indices.size(), (std::size_t)2); // Storing two 16-bit indices per dword.
            std::vector<uint32_t> indexData(sz);
            uint16_t* pIndices = reinterpret_cast<uint16_t*>(indexData.data());
            for (size_t i = 0; i < indices.size(); i++)
            {
                FALCOR_ASSERT(indices[i] < (1u << 16));
                pIndices[i] = static_cast<uint16_t>(indices[i]);
            }
            return indexData;
        }

        SceneCache::Key computeSceneCacheKey(const std::filesystem::path& path, SceneBuilder::Flags buildFlags)
        {
            SceneBuilder::Flags cacheFlags = buildFlags & (~(SceneBuilder::Flags::UseCache | SceneBuilder::Flags::RebuildCache));
            SHA1 sha1;
            auto pathStr = path.string();
            sha1.update(pathStr.data(), pathStr.size());
            sha1.update(&cacheFlags, sizeof(cacheFlags));
            return sha1.finalize();

        }
    }

    SceneBuilder::SceneBuilder(Flags flags)
        : mFlags(flags)
    {
        mpFence = GpuFence::create();
        mSceneData.pMaterials = MaterialSystem::create();
    }

    SceneBuilder::SharedPtr SceneBuilder::create(Flags flags)
    {
        return SharedPtr(new SceneBuilder(flags));
    }

    SceneBuilder::SharedPtr SceneBuilder::create(const std::filesystem::path& path, Flags buildFlags, const InstanceMatrices& instances)
    {
        std::filesystem::path fullPath;
        if (!findFileInDataDirectories(path, fullPath))
        {
            throw ImporterError(path, "Can't find scene file '{}'.", path);
        }

        auto pBuilder = create(buildFlags);

        // We can only use scene cache if not using instances.
        bool sceneCacheSupported = instances.empty();

        // Compute scene cache key based on absolute scene path and build flags.
        pBuilder->mSceneCacheKey = computeSceneCacheKey(fullPath, buildFlags);

        // Determine if scene cache should be written after import.
        bool useCache = is_set(buildFlags, Flags::UseCache);
        bool rebuildCache = is_set(buildFlags, Flags::RebuildCache);
        pBuilder->mWriteSceneCache = sceneCacheSupported && (useCache || rebuildCache);

        // Try to load scene cache if supported, available and requested.
        if (sceneCacheSupported && useCache && !rebuildCache && SceneCache::hasValidCache(pBuilder->mSceneCacheKey))
        {
            try
            {
                pBuilder->mpScene = Scene::create(SceneCache::readCache(pBuilder->mSceneCacheKey));
                return pBuilder;
            }
            catch (const std::exception& e)
            {
                throw ImporterError(fullPath, "Failed to load scene cache: {}", e.what());
            }
        }

        pBuilder->import(path, instances);
        return pBuilder;
    }

    void SceneBuilder::import(const std::filesystem::path& path, const InstanceMatrices& instances, const Dictionary& dict)
    {
        mSceneData.path = path;
        Importer::import(path, *this, instances, dict);
    }

    Scene::SharedPtr SceneBuilder::getScene()
    {
        if (mpScene) return mpScene;

        // Finish loading textures. This blocks until all textures are loaded and assigned.
        mpMaterialTextureLoader.reset();

        // If no meshes were added, we create a dummy mesh to keep the scene generation working.
        // Scenes with no meshes can be useful for example when using volumes in isolation.
        if (mMeshes.empty())
        {
            logWarning("Scene contains no meshes. Creating a dummy mesh.");
            // Add a dummy (degenerate) mesh.
            auto dummyMesh = TriangleMesh::createDummy();
            auto dummyMaterial = StandardMaterial::create("Dummy");
            auto meshID = addTriangleMesh(dummyMesh, dummyMaterial);
            Node dummyNode = { "Dummy", rmcv::identity<rmcv::mat4>(), rmcv::identity<rmcv::mat4>() };
            NodeID nodeID = addNode(dummyNode);
            addMeshInstance(nodeID, meshID);
        }

        // Post-process the scene data.
        TimeReport timeReport;

        // Prepare displacement maps. This either removes them (if requested in build flags)
        // or makes sure that normal maps are removed if displacement is in use.
        prepareDisplacementMaps();

        prepareSceneGraph();
        prepareMeshes();
        removeUnusedMeshes();
        flattenStaticMeshInstances();
        pretransformStaticMeshes();
        unifyTriangleWinding();
        optimizeSceneGraph();
        calculateMeshBoundingBoxes();
        createMeshGroups();
        optimizeGeometry();
        sortMeshes();
        createGlobalBuffers();
        createCurveGlobalBuffers();
        collectVolumeGrids();
        removeDuplicateSDFGrids();

        timeReport.measure("Post processing geometry");

        optimizeMaterials();
        removeDuplicateMaterials();
        quantizeTexCoords();

        timeReport.measure("Optimizing materials");

        // Prepare scene resources.
        createSceneGraph();
        createMeshData();
        createMeshBoundingBoxes();
        createCurveData();
        calculateCurveBoundingBoxes();

        // Create instance data.
        uint32_t tlasInstanceIndex = 0;
        createMeshInstanceData(tlasInstanceIndex);
        createCurveInstanceData(tlasInstanceIndex);
        // Adjust instance indices of SDF grid instances.
        for (auto& sdfInstanceData : mSceneData.sdfGridInstances) sdfInstanceData.instanceIndex = tlasInstanceIndex++;

        mSceneData.useCompressedHitInfo = is_set(mFlags, Flags::UseCompressedHitInfo);

        // Write scene cache if requested.
        if (mWriteSceneCache)
        {
            SceneCache::writeCache(mSceneData, mSceneCacheKey);
            timeReport.measure("Writing cache");
        }

        // Create the scene object.
        mpScene = Scene::create(std::move(mSceneData));
        mSceneData = {};

        timeReport.measure("Creating resources");
        timeReport.printToLog();

        return mpScene;
    }

    // Meshes

    MeshID SceneBuilder::addMesh(const Mesh& mesh)
    {
        return addProcessedMesh(processMesh(mesh));
    }

    MeshID SceneBuilder::addTriangleMesh(const TriangleMesh::SharedPtr& pTriangleMesh, const Material::SharedPtr& pMaterial)
    {
        checkArgument(pTriangleMesh != nullptr, "'pTriangleMesh' is missing");
        checkArgument(pMaterial != nullptr, "'pMaterial' is missing");

        Mesh mesh;

        const auto& indices = pTriangleMesh->getIndices();
        const auto& vertices = pTriangleMesh->getVertices();

        mesh.name = pTriangleMesh->getName();
        mesh.faceCount = (uint32_t)(indices.size() / 3);
        mesh.vertexCount = (uint32_t)vertices.size();
        mesh.indexCount = (uint32_t)indices.size();
        mesh.pIndices = indices.data();
        mesh.topology = Vao::Topology::TriangleList;
        mesh.isFrontFaceCW = pTriangleMesh->getFrontFaceCW();
        mesh.pMaterial = pMaterial;

        std::vector<float3> positions(vertices.size());
        std::vector<float3> normals(vertices.size());
        std::vector<float2> texCoords(vertices.size());
        std::transform(vertices.begin(), vertices.end(), positions.begin(), [] (const auto& v) { return v.position; });
        std::transform(vertices.begin(), vertices.end(), normals.begin(), [] (const auto& v) { return v.normal; });
        std::transform(vertices.begin(), vertices.end(), texCoords.begin(), [] (const auto& v) { return v.texCoord; });

        mesh.positions = { positions.data(), SceneBuilder::Mesh::AttributeFrequency::Vertex };
        mesh.normals = { normals.data(), SceneBuilder::Mesh::AttributeFrequency::Vertex };
        mesh.texCrds = { texCoords.data(), SceneBuilder::Mesh::AttributeFrequency::Vertex };

        return addMesh(mesh);
    }

    SceneBuilder::ProcessedMesh SceneBuilder::processMesh(const Mesh& mesh_, MeshAttributeIndices* pAttributeIndices) const
    {
        // This function preprocesses a mesh into the final runtime representation.
        // Note the function needs to be thread safe. The following steps are performed:
        //  - Error checking
        //  - Compute tangent space if needed
        //  - Merge identical vertices, compute new indices (optional)
        //  - Validate final vertex data
        //  - Compact vertices/indices into runtime format

        // Copy the mesh desc so we can update it. The caller retains the ownership of the data.
        Mesh mesh = mesh_;
        ProcessedMesh processedMesh;

        processedMesh.name = mesh.name;
        processedMesh.topology = mesh.topology;
        processedMesh.pMaterial = mesh.pMaterial;
        processedMesh.isFrontFaceCW = mesh.isFrontFaceCW;
        processedMesh.skeletonNodeId = mesh.skeletonNodeId;

        // Error checking.
        auto throw_on_missing_element = [&](const std::string& element)
        {
            throw RuntimeError("Error when adding the mesh '{}' to the scene. The mesh is missing {}.", mesh.name, element);
        };

        auto missing_element_warning = [&](const std::string& element)
        {
            logWarning("The mesh '{}' is missing the element {}. This is not an error, the element will be filled with zeros which may result in incorrect rendering.", mesh.name, element);
        };

        if (mesh.topology != Vao::Topology::TriangleList) throw RuntimeError("Error when adding the mesh '{}' to the scene. Only triangle list topology is supported.", mesh.name);
        if (mesh.pMaterial == nullptr) throw_on_missing_element("material");

        if (mesh.faceCount == 0) throw_on_missing_element("faces");
        if (mesh.vertexCount == 0) throw_on_missing_element("vertices");
        if (mesh.indexCount == 0 || !mesh.pIndices) throw_on_missing_element("indices");
        if (mesh.indexCount != mesh.faceCount * 3) throw RuntimeError("Error when adding the mesh '{}' to the scene. Unexpected face/vertex count.", mesh.name);

        if (mesh.positions.pData == nullptr) throw_on_missing_element("positions");
        if (mesh.normals.pData == nullptr) missing_element_warning("normals");
        if (mesh.texCrds.pData == nullptr) missing_element_warning("texture coordinates");

        if (mesh.hasBones())
        {
            if (mesh.boneIDs.pData == nullptr) throw_on_missing_element("bone IDs");
            if (mesh.boneWeights.pData == nullptr) throw_on_missing_element("bone weights");
        }

        // Generate tangent space if that's required.
        std::vector<float4> tangents;
        if (!(is_set(mFlags, Flags::UseOriginalTangentSpace) || mesh.useOriginalTangentSpace) || !mesh.tangents.pData)
        {
            generateTangents(mesh, tangents);
        }

        // Pretransform the texture coordinates, rather than transforming them at runtime.
        std::vector<float2> transformedTexCoords;
        if (mesh.texCrds.pData != nullptr)
        {
            const rmcv::mat4 xform = mesh.pMaterial->getTextureTransform().getMatrix();
            if (xform != rmcv::identity<rmcv::mat4>())
            {
                size_t texCoordCount = mesh.getAttributeCount(mesh.texCrds);
                transformedTexCoords.resize(texCoordCount);
                // The given matrix transforms the texture (e.g., scaling > 1 enlarges the texture).
                // Because we're transforming the input coordinates, apply the inverse.
                const rmcv::mat4 invXform = rmcv::inverse(xform);
                // Because texture transforms are 2D and affine, we only need apply the corresponding 3x2 matrix
                rmcv::matrix<2,3, float> coordTransform;
                coordTransform.setCol(0, invXform.getCol(0).xy);
                coordTransform.setCol(1, invXform.getCol(0).xy);
                coordTransform.setCol(2, invXform.getCol(0).xy);

                for (size_t i = 0; i < texCoordCount; ++i)
                {
                    transformedTexCoords[i] = coordTransform * float3(mesh.texCrds.pData[i], 1.f);
                }
                mesh.texCrds.pData = transformedTexCoords.data();
            }
        }

        // Build new vertex/index buffers by merging identical vertices.
        // The search is based on the topology defined by the original index buffer.
        //
        // A linked-list of vertices is built for each original vertex index.
        // We iterate over all vertices and first check if a vertex is identical to any of the other vertices
        // using the same original vertex index. If not, a new vertex is inserted and added to the list.
        // The 'heads' array point to the first vertex in each list, and each vertex has an associated next-pointer.
        // This ensures that adding to the linked lists do not require any dynamic memory allocation.
        //
        const uint32_t invalidIndex = 0xffffffff;
        std::vector<std::pair<Mesh::Vertex, uint32_t>> vertices;
        std::vector<uint32_t> indices(mesh.indexCount);

        if (pAttributeIndices)
        {
            pAttributeIndices->reserve(mesh.vertexCount);
        }

        if (mesh.mergeDuplicateVertices)
        {
            vertices.reserve(mesh.vertexCount);

            std::vector<uint32_t> heads(mesh.vertexCount, invalidIndex);

            for (uint32_t face = 0; face < mesh.faceCount; face++)
            {
                for (uint32_t vert = 0; vert < 3; vert++)
                {
                    const Mesh::Vertex v = mesh.getVertex(face, vert);
                    const uint32_t origIndex = mesh.pIndices[face * 3 + vert];

                    // Iterate over vertex list to check if it already exists.
                    FALCOR_ASSERT(origIndex < heads.size());
                    uint32_t index = heads[origIndex];
                    bool found = false;

                    while (index != invalidIndex)
                    {
                        if (compareVertices(v, vertices[index].first))
                        {
                            found = true;
                            break;
                        }
                        index = vertices[index].second;
                    }

                    // Insert new vertex if we couldn't find it.
                    if (!found)
                    {
                        FALCOR_ASSERT(vertices.size() < std::numeric_limits<uint32_t>::max());
                        index = (uint32_t)vertices.size();
                        vertices.push_back({ v, heads[origIndex] });

                        if (pAttributeIndices)
                        {
                            pAttributeIndices->push_back(mesh.getAttributeIndices(face, vert));
                            FALCOR_ASSERT(vertices.size() == pAttributeIndices->size());
                        }

                        heads[origIndex] = index;
                    }

                    // Store new vertex index.
                    indices[face * 3 + vert] = index;
                }
            }
        }
        else
        {
            vertices = { mesh.vertexCount, std::make_pair(Mesh::Vertex{}, invalidIndex) };

            for (uint32_t face = 0; face < mesh.faceCount; face++)
            {
                for (uint32_t vert = 0; vert < 3; vert++)
                {
                    const Mesh::Vertex v = mesh.getVertex(face, vert);
                    const uint32_t index = mesh.getAttributeIndex(mesh.positions, face, vert);

                    FALCOR_ASSERT(index < vertices.size());
                    vertices[index].first = v;

                    if (pAttributeIndices)
                    {
                        pAttributeIndices->push_back(mesh.getAttributeIndices(face, vert));
                    }
                }
            }

            if (pAttributeIndices)
            {
                FALCOR_ASSERT(vertices.size() == pAttributeIndices->size());
            }

            indices.assign(mesh.pIndices, mesh.pIndices + mesh.indexCount);
        }

        FALCOR_ASSERT(vertices.size() > 0);
        FALCOR_ASSERT(indices.size() == mesh.indexCount);
        if (vertices.size() != mesh.vertexCount)
        {
            logDebug("Mesh with name '{}' had original vertex count {}, new vertex count {}.", mesh.name, mesh.vertexCount, vertices.size());
        }

        // Validate vertex data to check for invalid numbers and missing tangent frame.
        size_t invalidCount = 0;
        size_t zeroCount = 0;
        for (const auto& v : vertices)
        {
            validateVertex(v.first, invalidCount, zeroCount);
        }
        if (invalidCount > 0) logWarning("The mesh '{}' has inf/nan vertex attributes at {} vertices. Please fix the asset.", mesh.name, invalidCount);
        if (zeroCount > 0) logWarning("The mesh '{}' has zero-length normals/tangents at {} vertices. Please fix the asset.", mesh.name, zeroCount);

        // If the non-indexed vertices build flag is set, we will de-index the data below.
        const bool isIndexed = !is_set(mFlags, Flags::NonIndexedVertices);
        const uint32_t vertexCount = isIndexed ? (uint32_t)vertices.size() : mesh.indexCount;

        // Copy indices into processed mesh.
        if (isIndexed)
        {
            processedMesh.indexCount = indices.size();
            processedMesh.use16BitIndices = (vertices.size() <= (1u << 16)) && !(is_set(mFlags, Flags::Force32BitIndices));

            if (!processedMesh.use16BitIndices) processedMesh.indexData = std::move(indices);
            else processedMesh.indexData = compact16BitIndices(indices);
        }

        // Copy vertices into processed mesh.
        processedMesh.staticData.reserve(vertexCount);
        if (mesh.hasBones()) processedMesh.skinningData.reserve(vertexCount);

        for (uint32_t i = 0; i < vertexCount; i++)
        {
            uint32_t index = isIndexed ? i : indices[i];
            FALCOR_ASSERT(index < vertices.size());
            const Mesh::Vertex& v = vertices[index].first;

            StaticVertexData s;
            s.position = v.position;
            s.normal = v.normal;
            s.texCrd = v.texCrd;
            s.tangent = v.tangent;
            s.curveRadius = v.curveRadius;
            processedMesh.staticData.push_back(s);

            if (mesh.hasBones())
            {
                SkinningVertexData s;
                s.boneWeight = v.boneWeights;
                s.boneID = v.boneIDs;
                s.staticIndex = i; // This references the local vertex here and gets updated in addProcessedMesh().
                s.bindMatrixID = 0; // This will be initialized in createMeshData().
                s.skeletonMatrixID = 0; // This will be initialized in createMeshData().
                processedMesh.skinningData.push_back(s);
            }
        }

        return processedMesh;
    }

    void SceneBuilder::generateTangents(Mesh& mesh, std::vector<float4>& tangents) const
    {
        tangents = MikkTSpaceWrapper::generateTangents(mesh);
        if (!tangents.empty())
        {
            FALCOR_ASSERT(tangents.size() == mesh.indexCount);
            mesh.tangents.pData = tangents.data();
            mesh.tangents.frequency = Mesh::AttributeFrequency::FaceVarying;
        }
        else
        {
            mesh.tangents.pData = nullptr;
            mesh.tangents.frequency = Mesh::AttributeFrequency::None;
        }
    }

    MeshID SceneBuilder::addProcessedMesh(const ProcessedMesh& mesh)
    {
        const bool isIndexed = !is_set(mFlags, Flags::NonIndexedVertices);

        MeshSpec spec;

        // Add the mesh to the scene.
        spec.name = mesh.name;
        spec.topology = mesh.topology;
        spec.materialId = addMaterial(mesh.pMaterial);
        spec.isFrontFaceCW = mesh.isFrontFaceCW;
        spec.skeletonNodeID = mesh.skeletonNodeId;

        spec.vertexCount = (uint32_t)mesh.staticData.size();
        spec.staticVertexCount = (uint32_t)mesh.staticData.size();
        spec.skinningVertexCount = (uint32_t)mesh.skinningData.size();

        spec.indexData = std::move(mesh.indexData);
        spec.staticData = std::move(mesh.staticData);
        spec.skinningData = std::move(mesh.skinningData);

        if (isIndexed)
        {
            spec.indexCount = (uint32_t)mesh.indexCount;
            spec.use16BitIndices = mesh.use16BitIndices;
        }

        if (!spec.skinningData.empty())
        {
            FALCOR_ASSERT(spec.skinningVertexCount > 0);
            spec.hasSkinningData = true;
            spec.prevVertexCount = spec.skinningVertexCount;
        }

        mMeshes.push_back(spec);

        if (mMeshes.size() > std::numeric_limits<uint32_t>::max())
        {
            throw RuntimeError("Trying to build a scene that exceeds supported number of meshes");
        }

        return MeshID(mMeshes.size() - 1);
    }

    void SceneBuilder::setCachedMeshes(std::vector<CachedMesh>&& cachedMeshes)
    {
        mSceneData.cachedMeshes = std::move(cachedMeshes);
    }

    void SceneBuilder::addCustomPrimitive(uint32_t userID, const AABB& aabb)
    {
        // Currently each custom primitive has exactly one AABB. This may change in the future.
        FALCOR_ASSERT(mSceneData.customPrimitiveDesc.size() == mSceneData.customPrimitiveAABBs.size());
        if (mSceneData.customPrimitiveAABBs.size() > std::numeric_limits<uint32_t>::max())
        {
            throw RuntimeError("Custom primitive count exceeds the maximum");
        }

        CustomPrimitiveDesc desc = {};
        desc.userID = userID;
        desc.aabbOffset = (uint32_t)mSceneData.customPrimitiveAABBs.size();

        mSceneData.customPrimitiveDesc.push_back(desc);
        mSceneData.customPrimitiveAABBs.push_back(aabb);
    }

    // Curves

    CurveID SceneBuilder::addCurve(const Curve& curve)
    {
        return addProcessedCurve(processCurve(curve));
    }

    SceneBuilder::ProcessedCurve SceneBuilder::processCurve(const Curve& curve) const
    {
        ProcessedCurve processedCurve;

        processedCurve.name = curve.name;
        processedCurve.topology = Vao::Topology::LineStrip;
        processedCurve.pMaterial = curve.pMaterial;

        // Error checking.
        auto throw_on_missing_element = [&](const std::string& element)
        {
            throw RuntimeError("Error when adding the curve '{}' to the scene. The curve is missing {}.", curve.name, element);
        };

        auto missing_element_warning = [&](const std::string& element)
        {
            logWarning("The curve '{}' is missing the element {}. This is not an error, the element will be filled with zeros which may result in incorrect rendering.", curve.name, element);
        };

        if (curve.pMaterial == nullptr) throw_on_missing_element("material");

        if (curve.vertexCount == 0) throw_on_missing_element("vertices");
        if (curve.indexCount == 0) throw_on_missing_element("indices");

        if (curve.positions.pData == nullptr) throw_on_missing_element("positions");
        if (curve.radius.pData == nullptr) throw_on_missing_element("radius");
        if (curve.texCrds.pData == nullptr) missing_element_warning("texture coordinates");

        // Copy indices and vertices into processed curve.
        processedCurve.indexData.assign(curve.pIndices, curve.pIndices + curve.indexCount);

        processedCurve.staticData.reserve(curve.vertexCount);
        for (uint32_t i = 0; i < curve.vertexCount; i++)
        {
            StaticCurveVertexData s;
            s.position = curve.positions.pData[i];
            s.radius = curve.radius.pData[i];

            if (curve.texCrds.pData != nullptr)
            {
                s.texCrd = curve.texCrds.pData[i];
            }
            else
            {
                s.texCrd = float2(0.f);
            }

            processedCurve.staticData.push_back(s);
        }

        return processedCurve;
    }

    CurveID SceneBuilder::addProcessedCurve(const ProcessedCurve& curve)
    {
        CurveSpec spec;

        // Add the curve to the scene.
        spec.name = curve.name;
        spec.topology = curve.topology;
        spec.materialId = addMaterial(curve.pMaterial);

        spec.vertexCount = (uint32_t)curve.staticData.size();
        spec.staticVertexCount = (uint32_t)curve.staticData.size();

        spec.indexData = std::move(curve.indexData);
        spec.staticData = std::move(curve.staticData);

        spec.indexCount = (uint32_t)curve.indexData.size();

        mCurves.push_back(spec);

        if (mCurves.size() > std::numeric_limits<uint32_t>::max())
        {
            throw RuntimeError("Trying to build a scene that exceeds supported number of curves.");
        }

        return CurveID(mCurves.size() - 1);
    }

    // SDFs

    SdfDescID SceneBuilder::addSDFGrid(const SDFGrid::SharedPtr& pSDFGrid, const Material::SharedPtr& pMaterial)
    {
        checkArgument(pSDFGrid != nullptr, "'pSDFGrid' is missing");
        checkArgument(pMaterial != nullptr, "'pMaterial' is missing");

        Scene::SDFGridDesc desc;
        desc.materialID = addMaterial(pMaterial);
        desc.sdfGridID = SdfGridID(mSceneData.sdfGrids.size());
        mSceneData.sdfGrids.push_back(pSDFGrid);

        mSceneData.sdfGridDesc.push_back(desc);
        mSceneData.sdfGridMaxLODCount = glm::max(bitScanReverse(pSDFGrid->getGridWidth()) + 1, mSceneData.sdfGridMaxLODCount);
        return SdfDescID(mSceneData.sdfGridDesc.size() - 1);
    }

    // Materials

    MaterialID SceneBuilder::addMaterial(const Material::SharedPtr& pMaterial)
    {
        checkArgument(pMaterial != nullptr, "'pMaterial' is missing");
        return mSceneData.pMaterials->addMaterial(pMaterial);
    }

    void SceneBuilder::loadMaterialTexture(const Material::SharedPtr& pMaterial, Material::TextureSlot slot, const std::filesystem::path& path)
    {
        checkArgument(pMaterial != nullptr, "'pMaterial' is missing");
        if (!mpMaterialTextureLoader)
        {
            mpMaterialTextureLoader.reset(new MaterialTextureLoader(mSceneData.pMaterials->getTextureManager(), !is_set(mFlags, Flags::AssumeLinearSpaceTextures)));
        }
        mpMaterialTextureLoader->loadTexture(pMaterial, slot, path);
    }

    void SceneBuilder::waitForMaterialTextureLoading()
    {
        mpMaterialTextureLoader.reset();
    }

    // GridVolumes

    GridVolume::SharedPtr SceneBuilder::getGridVolume(const std::string& name) const
    {
        for (const auto& pGridVolume : mSceneData.gridVolumes)
        {
            if (pGridVolume->getName() == name) return pGridVolume;
        }
        return nullptr;
    }

    VolumeID SceneBuilder::addGridVolume(const GridVolume::SharedPtr& pGridVolume, NodeID nodeID)
    {
        FALCOR_ASSERT(pGridVolume);
        checkArgument(nodeID == NodeID::Invalid() || nodeID.get() < mSceneGraph.size(), "'nodeID' ({}) is out of range", nodeID);

        if (nodeID != NodeID::Invalid())
        {
            pGridVolume->setHasAnimation(true);
            pGridVolume->setNodeID(nodeID);
        }

        mSceneData.gridVolumes.push_back(pGridVolume);
        FALCOR_ASSERT(mSceneData.gridVolumes.size() <= std::numeric_limits<uint32_t>::max());
        return VolumeID{ mSceneData.gridVolumes.size() - 1 };
    }

    // Lights

    Light::SharedPtr SceneBuilder::getLight(const std::string& name) const
    {
        for (const auto& pLight : mSceneData.lights)
        {
            if (pLight->getName() == name) return pLight;
        }
        return nullptr;
    }

    LightID SceneBuilder::addLight(const Light::SharedPtr& pLight)
    {
        checkArgument(pLight != nullptr, "'pLight' is missing");
        mSceneData.lights.push_back(pLight);
        FALCOR_ASSERT(mSceneData.lights.size() <= std::numeric_limits<uint32_t>::max());
        return LightID(mSceneData.lights.size() - 1);
    }

    void SceneBuilder::loadLightProfile(const std::string& filename, bool normalize)
    {
        mSceneData.pLightProfile = LightProfile::createFromIesProfile(std::filesystem::path(filename), normalize);
    }

    // Cameras

    CameraID SceneBuilder::addCamera(const Camera::SharedPtr& pCamera)
    {
        checkArgument(pCamera != nullptr, "'pCamera' is missing");
        mSceneData.cameras.push_back(pCamera);
        FALCOR_ASSERT(mSceneData.cameras.size() <= std::numeric_limits<uint32_t>::max());
        return CameraID(mSceneData.cameras.size() - 1);
    }

    Camera::SharedPtr SceneBuilder::getSelectedCamera() const
    {
        return mSceneData.selectedCamera < mSceneData.cameras.size() ? mSceneData.cameras[mSceneData.selectedCamera] : nullptr;
    }

    void SceneBuilder::setSelectedCamera(const Camera::SharedPtr& pCamera)
    {
        auto it = std::find(mSceneData.cameras.begin(), mSceneData.cameras.end(), pCamera);
        mSceneData.selectedCamera = it != mSceneData.cameras.end() ? (uint32_t)std::distance(mSceneData.cameras.begin(), it) : 0;
    }

    // Animations

    void SceneBuilder::addAnimation(const Animation::SharedPtr& pAnimation)
    {
        checkArgument(pAnimation != nullptr, "'pAnimation' is missing");
        mSceneData.animations.push_back(pAnimation);
    }

    Animation::SharedPtr SceneBuilder::createAnimation(Animatable::SharedPtr pAnimatable, const std::string& name, double duration)
    {
        checkArgument(pAnimatable != nullptr, "'pAnimatable' is missing");

        NodeID nodeID = pAnimatable->getNodeID();

        if (nodeID != NodeID::Invalid() && isNodeAnimated(nodeID))
        {
            logWarning("Animatable object is already animated.");
            return nullptr;
        }
        if (nodeID == NodeID::Invalid()) nodeID = addNode(Node{ name, rmcv::identity<rmcv::mat4>(), rmcv::identity<rmcv::mat4>() });

        pAnimatable->setNodeID(nodeID);
        pAnimatable->setHasAnimation(true);
        pAnimatable->setIsAnimated(true);

        auto animation = Animation::create(name, nodeID, duration);
        addAnimation(animation);
        return animation;
    }

    // Scene graph

    NodeID SceneBuilder::addNode(const Node& node)
    {
        // Validate node.
        auto validateMatrix = [&](rmcv::mat4 m, const char* field)
        {
            if (!isMatrixValid(m))
            {
                throw RuntimeError("Node '{}' {} matrix has inf/nan values", node.name, field);
            }
            // Check the assumption that transforms are affine. Note that glm is column-major.
            if (!isMatrixAffine(m))
            {
                logWarning("SceneBuilder::addNode() - Node '{}' {} matrix is not affine. Setting last row to (0,0,0,1).", node.name, field);
                m[3] = rmcv::vec4(0, 0, 0, 1);
            }
            return m;
        };

        InternalNode internalNode(node);
        internalNode.transform = validateMatrix(node.transform, "transform");
        internalNode.localToBindPose = validateMatrix(node.localToBindPose, "localToBindPose");

        static_assert(NodeID::kInvalidID >= std::numeric_limits<uint32_t>::max());
        if (node.parent.isValid() && node.parent.get() >= mSceneGraph.size()) throw RuntimeError("Node parent is out of range");
        if (mSceneGraph.size() >= std::numeric_limits<NodeID::IntType>::max()) throw RuntimeError("Scene graph is too large");

        // Add node to scene graph.
        NodeID newNodeID{ mSceneGraph.size() };
        mSceneGraph.push_back(internalNode);
        if (node.parent.isValid()) mSceneGraph[node.parent.get()].children.push_back(newNodeID);

        return newNodeID;
    }

    void SceneBuilder::addMeshInstance(NodeID nodeID, MeshID meshID)
    {
        checkArgument(nodeID.get() < mSceneGraph.size(), "'nodeID' ({}) is out of range", nodeID);
        checkArgument(meshID.get() < mMeshes.size(), "'meshID' ({}) is out of range", meshID);

        mSceneGraph[nodeID.get()].meshes.push_back(meshID);
        mMeshes[meshID.get()].instances.push_back(nodeID);
    }

    void SceneBuilder::addCurveInstance(NodeID nodeID, CurveID curveID)
    {
        checkArgument(nodeID.get() < mSceneGraph.size(), "'nodeID' ({}) is out of range", nodeID);
        checkArgument(curveID.get() < mCurves.size(), "'curveID' ({}) is out of range", curveID);

        mSceneGraph[nodeID.get()].curves.push_back(curveID);
        mCurves[curveID.get()].instances.push_back(nodeID);
    }

    void SceneBuilder::addSDFGridInstance(NodeID nodeID, SdfDescID sdfGridID)
    {
        checkArgument(nodeID.get() < mSceneGraph.size(), "'nodeID' ({}) is out of range", nodeID);
        checkArgument(sdfGridID.get() < mSceneData.sdfGridDesc.size(), "'sdfGridID' ({}) is out of range", sdfGridID);

        Scene::SDFGridDesc& desc = mSceneData.sdfGridDesc[sdfGridID.get()];
        mSceneGraph[nodeID.get()].sdfGrids.push_back(desc.sdfGridID);
        desc.instances.push_back(nodeID);

        GeometryInstanceData instance(GeometryType::SDFGrid);
        // sdfGridID refers to the ID returned by SceneBuilder::addSDFGrid and does not necessarily reflect an ID in SceneData::sdfGrids.
        instance.geometryID = desc.sdfGridID.getSlang();
        instance.materialID = desc.materialID.getSlang();
        instance.globalMatrixID = nodeID.getSlang();
        instance.geometryIndex = 0;

        mSceneData.sdfGridInstances.push_back(instance);
    }

    bool SceneBuilder::doesNodeHaveAnimation(NodeID nodeID) const
    {
        FALCOR_ASSERT(nodeID != NodeID::Invalid() && nodeID.get() < mSceneGraph.size());
        for (const auto& pAnimation : mSceneData.animations)
        {
            if (pAnimation->getNodeID() == nodeID) return true;
        }

        return false;
    }

    bool SceneBuilder::isNodeAnimated(NodeID nodeID) const
    {
        while (nodeID != NodeID::Invalid())
        {
            if (doesNodeHaveAnimation(nodeID)) return true;
            nodeID = mSceneGraph[nodeID.get()].parent;
        }

        return false;
    }

    void SceneBuilder::setNodeInterpolationMode(NodeID nodeID, Animation::InterpolationMode interpolationMode, bool enableWarping)
    {
        FALCOR_ASSERT(nodeID.get() < mSceneGraph.size());

        while (nodeID != NodeID::Invalid())
        {
            for (const auto& pAnimation : mSceneData.animations)
            {
                if (pAnimation->getNodeID() == nodeID)
                {
                    pAnimation->setInterpolationMode(interpolationMode);
                    pAnimation->setEnableWarping(enableWarping);
                }
            }
            nodeID = mSceneGraph[nodeID.get()].parent;
        }
    }

    // Internal

    void SceneBuilder::updateLinkedObjects(NodeID nodeID, NodeID newNodeID)
    {
        // Helper function to update all objects linked from a node to point to newNodeID.
        // This is useful when modifying the graph.

        FALCOR_ASSERT(nodeID != NodeID::Invalid() && nodeID.get() < mSceneGraph.size());
        FALCOR_ASSERT(newNodeID != NodeID::Invalid() && newNodeID.get() < mSceneGraph.size());
        const auto& node = mSceneGraph[nodeID.get()];

        for (auto childID : node.children)
        {
            FALCOR_ASSERT(childID.get() < mSceneGraph.size());
            FALCOR_ASSERT(mSceneGraph[childID.get()].parent == nodeID);
            mSceneGraph[childID.get()].parent = newNodeID;
        }
        for (auto meshID : node.meshes)
        {
            FALCOR_ASSERT(meshID.get() < mMeshes.size());
            auto& mesh = mMeshes[meshID.get()];
            std::replace(mesh.instances.begin(), mesh.instances.end(), nodeID, newNodeID);
        }
        for (auto curveID : node.curves)
        {
            FALCOR_ASSERT(curveID.get() < mCurves.size());
            auto& curve = mCurves[curveID.get()];
            std::replace(curve.instances.begin(), curve.instances.end(), nodeID, newNodeID);
        }
        for (auto sdfGridID : node.sdfGrids)
        {
            FALCOR_ASSERT(sdfGridID.get() < mSceneData.sdfGridDesc.size());
            auto& sdfGridDesc = mSceneData.sdfGridDesc[sdfGridID.get()];
            std::replace(sdfGridDesc.instances.begin(), sdfGridDesc.instances.end(), nodeID, newNodeID);
        }
        for (auto pObject : node.animatable)
        {
            FALCOR_ASSERT(pObject);
            FALCOR_ASSERT(pObject->getNodeID() == nodeID);
            pObject->setNodeID(newNodeID);
        }
    }

    bool SceneBuilder::collapseNodes(NodeID parentNodeID, NodeID childNodeID)
    {
        // Collapses the nodes from parent...child node into the parent node if possible.
        // The transform of the parent node is updated to account for the combined transform.
        // The prerequisite for this is that the parent..child-1 nodes have no other children and that
        // all the nodes are static. The function returns false if the necessary conditions are not met.

        // Check that nodes are valid.
        if (parentNodeID == NodeID::Invalid() || childNodeID == NodeID::Invalid()) return false;
        FALCOR_ASSERT(parentNodeID.get() < mSceneGraph.size() && childNodeID.get() < mSceneGraph.size());

        if (mSceneGraph[parentNodeID.get()].dontOptimize || mSceneGraph[childNodeID.get()].dontOptimize) return false;
        if (doesNodeHaveAnimation(childNodeID)) return false;

        // Compute the combined transform.
        auto& child = mSceneGraph[childNodeID.get()];
        rmcv::mat4 transform = child.transform;
        NodeID prevNodeID = childNodeID;
        NodeID nodeID = child.parent;

        while (nodeID != NodeID::Invalid())
        {
            FALCOR_ASSERT(nodeID.get() < mSceneGraph.size());
            const auto& node = mSceneGraph[nodeID.get()];

            // Check that node is a static interior node with a single child.
            if (node.children.size() > 1 ||
                node.hasObjects() ||
                doesNodeHaveAnimation(nodeID) ||
                mSceneGraph[nodeID.get()].dontOptimize) return false;

            FALCOR_ASSERT(node.children.size() == 1);
            FALCOR_ASSERT(node.children[0] == prevNodeID);

            // Update the transform and step to the parent.
            transform = node.transform * transform;

            if (nodeID == parentNodeID) break;
            prevNodeID = nodeID;
            nodeID = mSceneGraph[nodeID.get()].parent;
        }

        if (nodeID == NodeID::Invalid()) return false; // We didn't find the parent.

        // Update all linked objects to point to the new parent node.
        updateLinkedObjects(childNodeID, parentNodeID);

        // Update the parent node to the child's data.
        // The new parent node will hold the combined transform.
        auto& parent = mSceneGraph[parentNodeID.get()];
        auto oldParentID = parent.parent;

        parent = std::move(child);
        parent.parent = oldParentID;
        parent.transform = transform;

        // Reset the now unused nodes below the parent to a valid empty state.
        // TODO: Run a separate optimization pass to compact the node list.
        nodeID = childNodeID;
        while (nodeID != parentNodeID)
        {
            auto& node = mSceneGraph[nodeID.get()];
            nodeID = node.parent;
            node = InternalNode();
        }

        return true;
    }

    bool SceneBuilder::mergeNodes(NodeID dstNodeID, NodeID srcNodeID)
    {
        // This function merges the source node into the destination node.
        // The prerequisite for this to work is that the two nodes are static
        // and have identical transforms and parent nodes (or no parents).
        // The function returns false if the necessary conditions are not met.

        // Check that nodes are valid and compatible for merging.
        if (dstNodeID == NodeID::Invalid() || srcNodeID == NodeID::Invalid()) return false;
        FALCOR_ASSERT(dstNodeID.get() < mSceneGraph.size() && srcNodeID.get() < mSceneGraph.size());

        auto& dst = mSceneGraph[dstNodeID.get()];
        auto& src = mSceneGraph[srcNodeID.get()];

        if (mSceneGraph[dstNodeID.get()].dontOptimize || mSceneGraph[srcNodeID.get()].dontOptimize) return false;
        if (doesNodeHaveAnimation(dstNodeID) || doesNodeHaveAnimation(srcNodeID)) return false;

        if (dst.parent != src.parent ||
            dst.transform != src.transform ||
            dst.localToBindPose != src.localToBindPose) return false;

        // Update all linked objects to point to the dest node.
        updateLinkedObjects(srcNodeID, dstNodeID);

        // Merge the source node into the dest node.
        dst.children.insert(dst.children.end(), src.children.begin(), src.children.end());
        dst.meshes.insert(dst.meshes.end(), src.meshes.begin(), src.meshes.end());
        dst.curves.insert(dst.curves.end(), src.curves.begin(), src.curves.end());
        dst.sdfGrids.insert(dst.sdfGrids.end(), src.sdfGrids.begin(), src.sdfGrids.end());
        dst.animatable.insert(dst.animatable.end(), src.animatable.begin(), src.animatable.end());

        // Reset the now unused source node to a valid empty state.
        src = InternalNode();

        return true;
    }

    void SceneBuilder::prepareDisplacementMaps()
    {
        for (const auto& pMaterial : mSceneData.pMaterials->getMaterials())
        {
            if (pMaterial->getTexture(Material::TextureSlot::Displacement) != nullptr)
            {
                // Remove displacement maps if requested by scene flags.
                if (is_set(mFlags, Flags::DontUseDisplacement))
                {
                    pMaterial->clearTexture(Material::TextureSlot::Displacement);
                }

                // Remove normal maps for materials using displacement.
                if (pMaterial->hasTextureSlot(Material::TextureSlot::Normal))
                {
                    pMaterial->clearTexture(Material::TextureSlot::Normal);
                }
            }
        }
    }

    void SceneBuilder::prepareSceneGraph()
    {
        // This function validates and prepares the scene graph for use by later passes.
        // It appends pointers to all animatable objects to their respective scene graph nodes.

        auto addAnimatable = [this](Animatable* pObject, const std::string& name) {
            if (auto nodeID = pObject->getNodeID(); nodeID != NodeID::Invalid())
            {
                if (nodeID.get() >= mSceneGraph.size()) throw RuntimeError("Invalid node ID in animatable object named '{}'", name);
                mSceneGraph[nodeID.get()].animatable.push_back(pObject);
            }
        };

        for (const auto& light : mSceneData.lights) addAnimatable(light.get(), light->getName());
        for (const auto& camera : mSceneData.cameras) addAnimatable(camera.get(), camera->getName());
        for (const auto& gridVolume : mSceneData.gridVolumes) addAnimatable(gridVolume.get(), gridVolume->getName());

        for (const auto& mesh : mMeshes)
        {
            if (mesh.skeletonNodeID != NodeID::Invalid())
            {
                mSceneGraph[mesh.skeletonNodeID.get()].dontOptimize = true;
            }
        }
    }

    void SceneBuilder::prepareMeshes()
    {
        // Initialize any mesh properties that depend on the scene modifications to be finished.

        // Set mesh properties related to vertex animations
        for (auto& m : mMeshes) m.isAnimated = false;
        for (auto& cache : mSceneData.cachedMeshes)
        {
            auto& mesh = mMeshes[cache.meshID.get()];
            FALCOR_ASSERT(!mesh.isDynamic());
            mesh.isAnimated = true;
            mesh.prevVertexCount = mesh.staticVertexCount;
        }
        for (auto& cache : mSceneData.cachedCurves)
        {
            if (cache.tessellationMode != CurveTessellationMode::LinearSweptSphere)
            {
                auto& mesh = mMeshes[cache.geometryID.get()];
                FALCOR_ASSERT(!mesh.isDynamic());
                mesh.isAnimated = true;
                mesh.prevVertexCount = mesh.staticVertexCount;
            }
        }
    }

    void SceneBuilder::removeUnusedMeshes()
    {
        // If the scene contained meshes that are not referenced by the scene graph,
        // those will be removed here and warnings logged.

        // First count number of unused meshes.
        size_t unusedCount = 0;
        for (MeshID meshID{ 0 }; meshID.get() < (uint32_t)mMeshes.size(); ++meshID)
        {
            auto& mesh = mMeshes[meshID.get()];
            if (mesh.instances.empty())
            {
                logWarning("Mesh with ID {} named '{}' is not referenced by any scene graph nodes.", meshID, mesh.name);
                unusedCount++;
            }
        }

        // Rebuild mesh list and scene graph only if one or more meshes need to be removed.
        if (unusedCount > 0)
        {
            logWarning("Scene has {} unused meshes that will be removed.", unusedCount);

            const size_t meshCount = mMeshes.size();
            MeshList meshes;
            meshes.reserve(meshCount);

            for (MeshID meshID{ 0 }; meshID.get() < (uint32_t)meshCount; ++meshID)
            {
                auto& mesh = mMeshes[meshID.get()];
                if (mesh.instances.empty()) continue; // Skip unused meshes

                // Get new mesh ID.
                const MeshID newMeshID(meshes.size());

                // Update the mesh IDs in the scene graph nodes.
                for (const auto& nodeID : mesh.instances)
                {
                    FALCOR_ASSERT(nodeID.get() < mSceneGraph.size());
                    auto& node = mSceneGraph[nodeID.get()];
                    std::replace(node.meshes.begin(), node.meshes.end(), meshID, newMeshID);
                }

                // Update the mesh IDs of cached meshes.
                for (auto &cachedMesh : mSceneData.cachedMeshes)
                {
                    if (cachedMesh.meshID == meshID) cachedMesh.meshID = newMeshID;
                }
                for (auto& cache : mSceneData.cachedCurves)
                {
                    if (cache.tessellationMode != CurveTessellationMode::LinearSweptSphere)
                    {
                        if (cache.geometryID == CurveOrMeshID{ meshID }) cache.geometryID = CurveOrMeshID{ newMeshID };
                    }
                }

                meshes.push_back(std::move(mesh));
            }

            mMeshes = std::move(meshes);

            // Validate scene graph.
            FALCOR_ASSERT(mMeshes.size() == meshCount - unusedCount);
            for (const auto& node : mSceneGraph)
            {
                for (MeshID meshID : node.meshes) FALCOR_ASSERT_LT(meshID.get(), mMeshes.size());
            }
        }
    }

    void SceneBuilder::flattenStaticMeshInstances()
    {
        // This function optionally flattens all instanced non-skinned mesh instances to
        // separate non-instanced meshes by duplicating mesh data and composing transformations.
        // The pass is disabled by default. Can lead to a large increase in memory use.

        if (!is_set(mFlags, Flags::FlattenStaticMeshInstances))
        {
            return;
        }

        size_t flattenedInstanceCount = 0;
        std::vector<MeshSpec> newMeshes;

        for (MeshID meshID{ 0 }; meshID.get() < (uint32_t)mMeshes.size(); ++meshID)
        {
            auto& mesh = mMeshes[meshID.get()];

            // Skip non-instanced and dynamic meshes.
            if (mesh.instances.size() == 1 || mesh.isDynamic())
            {
                continue;
            }

            FALCOR_ASSERT(!mesh.instances.empty());
            FALCOR_ASSERT(mesh.skinningData.empty() && mesh.skinningVertexCount == 0);

            // i is the current index into mesh.instances in the loop below.
            // It is only incremented when skipping over an instance that is not being flattened.
            uint32_t i = 0;
            // Because we're deleting elements from mesh.instances below, stash the original size of the vector.
            const size_t instCount = mesh.instances.size();
            for (uint32_t instNum = 0; instNum < instCount; ++instNum)
            {
                auto nodeID = mesh.instances.at(i);
                // Skip animated/skinned instances.
                if (isNodeAnimated(nodeID))
                {
                    ++i;
                    continue;
                }
                MeshSpec meshCopy;
                // newMesh will point to the mesh representing the instance we are flattening.
                MeshSpec* newMesh;
                if (mesh.instances.size() > 1)
                {
                    // There is more than once instance, either static or dynamic.
                    // Create a copy of the mesh. This can be expensive.
                    meshCopy = mesh;
                    meshCopy.name = mesh.name + "[" + std::to_string(instNum) + "]";
                    // Make newMesh point to the copy
                    newMesh = &meshCopy;
                }
                else
                {
                    // This is now the only instance of the mesh. Re-use it, rather than
                    // making an (potentially expensive) copy.
                    newMesh = &mesh;
                }

                // Compute the object->world transform for the node.
                FALCOR_ASSERT(nodeID != NodeID::Invalid());

                rmcv::mat4 transform = rmcv::identity<rmcv::mat4>();
                while (nodeID != NodeID::Invalid())
                {
                    FALCOR_ASSERT_LT(nodeID.get(), mSceneGraph.size());
                    transform = mSceneGraph[nodeID.get()].transform * transform;

                    nodeID = mSceneGraph[nodeID.get()].parent;
                }

                flattenedInstanceCount++;

                // Unlink original instance from its previous transform node.
                auto& prevNode = mSceneGraph[mesh.instances[i].get()];
                auto it = std::find(prevNode.meshes.begin(), prevNode.meshes.end(), meshID);
                FALCOR_ASSERT(it != prevNode.meshes.end());
                prevNode.meshes.erase(it);

                // Link mesh to new top-level node.
                NodeID newNodeID      = addNode(Node{newMesh->name, transform, rmcv::identity<rmcv::mat4>()});
                InternalNode& newNode = mSceneGraph[newNodeID.get()];

                // Clear the copied list of instance parents, and replace with the new, single instance parent.
                newMesh->instances.clear();
                newMesh->instances.push_back(newNodeID);

                if (mesh.instances.size() > 1)
                {
                    // newMesh is a new copy of the original.
                    // Add it to the new node
                    MeshID newMeshID(mMeshes.size() + newMeshes.size());
                    newNode.meshes.push_back(newMeshID);
                    // Remove the instance from the current mesh
                    mesh.instances.erase(mesh.instances.begin() + i);
                    // Add to vector of meshes to be appended to mMeshes
                    newMeshes.push_back(*newMesh);
                }
                else
                {
                    // Re-using the original mesh; add it to the new node.
                    newNode.meshes.push_back(meshID);
                }
            }
        }

        if (mMeshes.size() == 0)
        {
            mMeshes = std::move(newMeshes);
        }
        else
        {
            mMeshes.reserve(mMeshes.size() + newMeshes.size());
            std::move(newMeshes.begin(), newMeshes.end(), std::back_inserter(mMeshes));
        }

        if (flattenedInstanceCount > 0) logInfo("Flattened {} static instances.", flattenedInstanceCount);
    }

    void SceneBuilder::optimizeSceneGraph()
    {
        // This function optimizes the scene graph to flatten transform hierarchies
        // where possible by merging nodes.
        if (is_set(mFlags, Flags::DontOptimizeGraph)) return;

        // Iterate over all nodes to collapse sub-trees of static nodes.
        size_t removedNodes = 0;
        for (NodeID nodeID{ 0 }; nodeID.get() < mSceneGraph.size(); ++nodeID)
        {
            const auto& node = mSceneGraph[nodeID.get()];
            if (collapseNodes(node.parent, nodeID)) removedNodes++;
        }

        if (removedNodes > 0) logInfo("Optimized scene graph by removing {} internal static nodes.", removedNodes);

        // Merge identical static nodes.
        // We build a set of unique nodes. If a node is identical to one of the
        // existing nodes, its contents are merged into the matching node.

        auto lessThan = [](const rmcv::mat4& lhs, const rmcv::mat4& rhs) {
            return rmcv::lex_lt(lhs, rhs);
        };

        // Comparison for strict weak ordering of scene graph nodes w r t to the fields we care about.
        auto cmp = [this, lessThan](NodeID lhsID, NodeID rhsID) {
            const auto& lhs = mSceneGraph[lhsID.get()];
            const auto& rhs = mSceneGraph[rhsID.get()];
            if (lhs.parent != rhs.parent) return lhs.parent < rhs.parent;
            if (lhs.transform != rhs.transform) return lessThan(lhs.transform, rhs.transform);
            if (lhs.localToBindPose != rhs.localToBindPose) return lessThan(lhs.localToBindPose, rhs.localToBindPose);
            return false;
        };

        std::set<NodeID, decltype(cmp)> uniqueStaticNodes(cmp); // In C++20 we can drop the constructor argument.

        size_t mergedNodesCount = 0;
        for (NodeID nodeID{ 0 }; nodeID.get() < mSceneGraph.size(); ++nodeID)
        {
            const auto& node = mSceneGraph[nodeID.get()];

            // Skip over unused or animated nodes.
            if (node.children.empty() && !node.hasObjects()) continue;
            if (doesNodeHaveAnimation(nodeID)) continue;
            if (mSceneGraph[nodeID.get()].dontOptimize) continue;

            // Look for an identical node and merge current node into it if found.
            auto it = uniqueStaticNodes.find(nodeID);
            if (it != uniqueStaticNodes.end())
            {
                bool merged = mergeNodes(*it, nodeID);
                if (!merged) throw RuntimeError("Unexpectedly failed to merge nodes");
                mergedNodesCount++;
            }
            else
            {
                uniqueStaticNodes.insert(nodeID);
            }
        }

        if (mergedNodesCount > 0) logInfo("Optimized scene graph by merging {} identical static nodes.", mergedNodesCount);
    }

    void SceneBuilder::pretransformStaticMeshes()
    {
        // This function transforms all static, non-instanced meshes to world space.
        // A new identity transform node is inserted in the scene graph, linking all transformed meshes.
        // This step is a prerequisite for the ray tracing optimizations we do later.

        // Add an identity transform node.
        NodeID identityNodeID = addNode(Node{ "Identity", rmcv::identity<rmcv::mat4>(), rmcv::identity<rmcv::mat4>() });
        auto& identityNode = mSceneGraph[identityNodeID.get()];

        size_t transformedMeshCount = 0;
        for (MeshID meshID{ 0 }; meshID.get() < (uint32_t)mMeshes.size(); ++meshID)
        {
            auto& mesh = mMeshes[meshID.get()];

            // Skip instanced/animated/skinned meshes.
            FALCOR_ASSERT(!mesh.instances.empty());
            if (mesh.instances.size() > 1 || isNodeAnimated(mesh.instances[0]) || mesh.isDynamic()) continue;

            FALCOR_ASSERT(mesh.skinningData.empty());
            mesh.isStatic = true;

            // Compute the object->world transform for the node.
            auto nodeID = mesh.instances[0];
            FALCOR_ASSERT(nodeID != NodeID::Invalid());

            rmcv::mat4 transform = rmcv::identity<rmcv::mat4>();
            while (nodeID != NodeID::Invalid())
            {
                FALCOR_ASSERT_LT(nodeID.get(), mSceneGraph.size());
                transform = mSceneGraph[nodeID.get()].transform * transform;

                nodeID = mSceneGraph[nodeID.get()].parent;
            }

            // Flip triangle winding flag if the transform flips the coordinate system handedness (negative determinant).
            bool flippedWinding = rmcv::determinant((rmcv::mat3)transform) < 0.f;
            if (flippedWinding) mesh.isFrontFaceCW = !mesh.isFrontFaceCW;

            // Transform vertices to world space if not already identity transform.
            if (transform != rmcv::identity<rmcv::mat4>())
            {
                FALCOR_ASSERT(!mesh.staticData.empty());
                FALCOR_ASSERT((size_t)mesh.vertexCount == mesh.staticData.size());

                rmcv::mat3 invTranspose3x3 = (rmcv::mat3)rmcv::transpose(rmcv::inverse(transform));
                rmcv::mat3 transform3x3 = (rmcv::mat3)transform;

                for (auto& v : mesh.staticData)
                {
                    float4 p = transform * float4(v.position, 1.f);
                    v.position = p.xyz;
                    v.normal = glm::normalize(invTranspose3x3 * v.normal);
                    v.tangent.xyz = glm::normalize(transform3x3 * float3(v.tangent.xyz)); // TODO: This cast shouldn't be necessary
                    // TODO: We should flip the sign of v.tangent.w if flippedWinding is true.
                    // Leaving that out for now for consistency with the shader code that needs the same fix.

                    v.curveRadius = glm::length(transform3x3 * float3(v.curveRadius, 0.f, 0.f));
                }

                transformedMeshCount++;
            }

            // Unlink mesh from its previous transform node.
            // TODO: This will leave some nodes unused. We could run a separate pass to compact the node list.
            FALCOR_ASSERT(mesh.instances.size() == 1);
            auto& prevNode = mSceneGraph[mesh.instances[0].get()];
            auto it = std::find(prevNode.meshes.begin(), prevNode.meshes.end(), meshID);
            FALCOR_ASSERT(it != prevNode.meshes.end());
            prevNode.meshes.erase(it);

            // Link mesh to the identity transform node.
            identityNode.meshes.push_back(meshID);
            mesh.instances[0] = identityNodeID;
        }

        if (transformedMeshCount > 0) logInfo("Pre-transformed {} static meshes to world space.", transformedMeshCount);
    }

    void SceneBuilder::flipTriangleWinding(MeshSpec& mesh)
    {
        FALCOR_ASSERT(mesh.topology == Vao::Topology::TriangleList);

        // Abort if mesh is non-indexed. Implement this code path when/if needed.
        // Note that both static and dynamic vertices have to be swapped for dynamic meshes.
        if (mesh.indexCount == 0)
        {
            throw RuntimeError("SceneBuilder::flipTriangleWinding() is not implemented for non-indexed meshes");
        }

        // Flip winding of indexed mesh by swapping vertex index 0 and 1 for each triangle.
        FALCOR_ASSERT(!mesh.indexData.empty());
        FALCOR_ASSERT(mesh.indexCount % 3 == 0);

        if (mesh.use16BitIndices)
        {
            FALCOR_ASSERT(mesh.indexCount <= mesh.indexData.size() * 2);
            uint16_t* indices = reinterpret_cast<uint16_t*>(mesh.indexData.data());
            for (size_t i = 0; i < mesh.indexCount; i += 3) std::swap(indices[i], indices[i + 1]);
        }
        else
        {
            FALCOR_ASSERT(mesh.indexCount == mesh.indexData.size());
            uint32_t* indices = mesh.indexData.data();
            for (size_t i = 0; i < mesh.indexCount; i += 3) std::swap(indices[i], indices[i + 1]);
        }

        mesh.isFrontFaceCW = !mesh.isFrontFaceCW;
    }

    void SceneBuilder::updateSDFGridID(SdfGridID oldID, SdfGridID newID)
    {
        // This is a helper function to update all the references to a specific SDF grid ID
        // to a new SDF grid ID.

        for (Scene::SDFGridDesc& sdfGridDesc : mSceneData.sdfGridDesc)
        {
            if (sdfGridDesc.sdfGridID == oldID) sdfGridDesc.sdfGridID = newID;
        }

        for (GeometryInstanceData& sdfGridInstance : mSceneData.sdfGridInstances)
        {
            if (SdfGridID::fromSlang(sdfGridInstance.geometryID) == oldID)
            {
                sdfGridInstance.geometryID = newID.getSlang();
                InternalNode& node = mSceneGraph[sdfGridInstance.globalMatrixID];
                std::replace(node.sdfGrids.begin(), node.sdfGrids.end(), oldID, newID);
            }
        }
    }

    void SceneBuilder::unifyTriangleWinding()
    {
        // This function makes the triangle winding for all meshes consistent in object space,
        // so that a triangle is front facing if its vertices appear counter-clockwise from the ray origin
        // in a right-handed coordinate system (Falcor's default).
        //
        // The reason to do this is so that we can pack meshes into BLASes without having to separate them
        // by clockwise and counter-clockwise winding. This is a requirement to support backface culling
        // when ray tracing, since all meshes in a BLAS must have the same winding.
        //
        // Note that this pass needs to run *after* pre-transformation of static meshes to world space,
        // as those transforms may flip the winding.

        size_t flippedMeshCount = 0;
        for (uint32_t meshID = 0; meshID < (uint32_t)mMeshes.size(); meshID++)
        {
            auto& mesh = mMeshes[meshID];

            // Skip meshes that are already front face counter-clockwise.
            if (mesh.isFrontFaceCW == false) continue;

            flipTriangleWinding(mesh);
            FALCOR_ASSERT(!mesh.isFrontFaceCW);

            flippedMeshCount++;
        }

        if (flippedMeshCount > 0) logInfo("Flipped triangle winding for {} out of {} meshes.", flippedMeshCount, mMeshes.size());
    }

    void SceneBuilder::calculateMeshBoundingBoxes()
    {
        for (auto& mesh : mMeshes)
        {
            FALCOR_ASSERT(!mesh.staticData.empty());
            FALCOR_ASSERT((size_t)mesh.vertexCount == mesh.staticData.size());

            AABB meshBB;
            for (auto& v : mesh.staticData)
            {
                meshBB.include(v.position);
            }

            mesh.boundingBox = meshBB;
        }
    }

    void SceneBuilder::createMeshGroups()
    {
        FALCOR_ASSERT(mMeshGroups.empty());

        // This function sorts meshes into groups based on their properties.
        // The scene will build one BLAS per mesh group for raytracing.
        // Note that a BLAS may be referenced by multiple TLAS instances.
        //
        // The sorting criteria are:
        //  - Non-instanced static meshes are all placed in the same group (BLAS).
        //    The vertices are pre-transformed in the BLAS and the TLAS instance has an identity transform.
        //    This ensures fast traversal for the static parts of a scene independent of the scene hierarchy.
        //  - Non-instanced dynamic meshes (skinned and/or animated) are sorted into groups (BLASes) with the same transform.
        //    The idea is that all parts of a dynamic object that move together go in the same BLAS and the TLAS instance applies the transform.
        //  - Instanced meshes are sorted into groups (BLASes) with identical instances.
        //    The idea is that all parts of an instanced object go in the same BLAS and the TLAS instances apply the transforms.
        //    Note that dynamic (skinned) meshes currently cannot be instanced due to limitations in the scene structures. See #1118.
        // TODO: Add build flag to turn off pre-transformation to world space.

        // Classify non-instanced meshes.
        // The non-instanced dynamic meshes are grouped based on what global matrix ID their transform is.
        // The non-instanced static meshes are placed in the same group.

        using meshList = std::vector<MeshID>;
        std::unordered_map<NodeID, meshList> nodeToMeshList;
        meshList staticMeshes;
        meshList staticDisplacedMeshes;
        meshList dynamicDisplacedMeshes;
        size_t nonInstancedMeshCount = 0;

        for (MeshID meshID{ 0 }; meshID.get() < (uint32_t)mMeshes.size(); ++meshID)
        {
            auto& mesh = mMeshes[meshID.get()];
            if (mesh.instances.size() > 1) continue; // Only processing non-instanced meshes here

            FALCOR_ASSERT(mesh.instances.size() == 1);
            NodeID nodeID = mesh.instances[0];

            // Mark displaced meshes.
            const auto& pMaterial = mSceneData.pMaterials->getMaterial(mesh.materialId);
            if (pMaterial->isDisplaced()) mesh.isDisplaced = true;

            if (mesh.isStatic && mesh.isDisplaced) staticDisplacedMeshes.push_back(meshID);
            else if (mesh.isStatic) staticMeshes.push_back(meshID);
            else if (!mesh.isStatic && mesh.isDisplaced) dynamicDisplacedMeshes.push_back(meshID);
            else nodeToMeshList[nodeID].push_back(meshID);
            nonInstancedMeshCount++;
        }

        // Validate that mesh counts add up.
        size_t nonInstancedDynamicMeshCount = 0;
        for (const auto& it : nodeToMeshList) nonInstancedDynamicMeshCount += it.second.size();
        FALCOR_ASSERT(staticMeshes.size() + staticDisplacedMeshes.size() + dynamicDisplacedMeshes.size() + nonInstancedDynamicMeshCount == nonInstancedMeshCount);

        // Classify instanced meshes.
        // The instanced meshes are grouped based on their lists of instances.
        // Meshes with an identical set of instances can be placed together in a BLAS.

        // It's important the instance lists are ordered and unique, so using std::set to describe them.
        // TODO: Maybe we should just change MeshSpec::instances to be a std::set in the first place?
        using instances = std::set<NodeID>;
        std::map<instances, meshList> instancesToMeshList;
        std::map<instances, meshList> displacedInstancesToMeshList;
        size_t instancedMeshCount = 0;

        for (MeshID meshID{ 0 }; meshID.get() < (uint32_t)mMeshes.size(); ++meshID)
        {
            auto& mesh = mMeshes[meshID.get()];
            if (mesh.instances.size() <= 1) continue; // Only processing instanced meshes here

            // Mark displaced meshes.
            const auto& pMaterial = mSceneData.pMaterials->getMaterial(mesh.materialId);
            if (pMaterial->isDisplaced()) mesh.isDisplaced = true;

            instances inst(mesh.instances.begin(), mesh.instances.end());
            if (mesh.isDisplaced) displacedInstancesToMeshList[inst].push_back(meshID);
            else instancesToMeshList[inst].push_back(meshID);
            instancedMeshCount++;
        }

        // Validate that each mesh is only indexed once.
        std::set<MeshID> instancedMeshes;
        size_t instancedCount = 0;
        for (const auto& it : instancesToMeshList)
        {
            instancedMeshes.insert(it.second.begin(), it.second.end());
            instancedCount += it.second.size();
        }
        std::set<MeshID> displacedInstancedMeshes;
        size_t displacedInstancedCount = 0;
        for (const auto& it : displacedInstancesToMeshList)
        {
            displacedInstancedMeshes.insert(it.second.begin(), it.second.end());
            displacedInstancedCount += it.second.size();
        }
        if ((instancedCount + displacedInstancedCount) != instancedMeshCount ||
            (instancedMeshes.size() + displacedInstancedMeshes.size()) != instancedMeshCount) throw RuntimeError("Error in instanced mesh grouping logic");

        logInfo("Found {} static non-instanced meshes, arranged in 1 mesh group.", staticMeshes.size());
        logInfo("Found {} displaced non-instanced meshes, arranged in 1 mesh group.", staticDisplacedMeshes.size());
        logInfo("Found {} dynamic non-instanced meshes, arranged in {} mesh groups.", nonInstancedDynamicMeshCount, nodeToMeshList.size());
        logInfo("Found {} instanced meshes, arranged in {} mesh groups.", instancedMeshCount, instancesToMeshList.size());

        // Build final result. Format is a list of Mesh ID's per mesh group.

        auto addMeshes = [this](const meshList& meshes, bool isStatic, bool isDisplaced, bool splitGroup)
        {
            if (!splitGroup)
            {
                mMeshGroups.push_back({ meshes, isStatic, isDisplaced });
            }
            else
            {
                for (const auto& meshID : meshes) mMeshGroups.push_back(MeshGroup{ meshList({ meshID }), isStatic, isDisplaced });
            }
        };

        // All static non-instanced meshes go in a single group or individual groups depending on config.
        if (!staticMeshes.empty())
        {
            addMeshes(staticMeshes, true, false, is_set(mFlags, Flags::RTDontMergeStatic));
        }

        // Non-instanced dynamic meshes were sorted above so just copy each list.
        for (const auto& it : nodeToMeshList)
        {
            addMeshes(it.second, false, false, is_set(mFlags, Flags::RTDontMergeDynamic));
        }

        // Instanced static and dynamic meshes are grouped based on instance lists.
        for (const auto& it : instancesToMeshList)
        {
            addMeshes(it.second, false, false, is_set(mFlags, Flags::RTDontMergeInstanced));
        }

        // All static displaced meshes go in a single group or individual groups depending on config.
        if (!staticDisplacedMeshes.empty())
        {
            addMeshes(staticDisplacedMeshes, true, true, is_set(mFlags, Flags::RTDontMergeStatic));
        }

        // All dynamic displaced meshes go in a single group or individual groups depending on config.
        if (!dynamicDisplacedMeshes.empty())
        {
            addMeshes(dynamicDisplacedMeshes, false, true, is_set(mFlags, Flags::RTDontMergeDynamic));
        }

        // Instanced displaced meshes are grouped based on instance lists.
        for (const auto& it : displacedInstancesToMeshList)
        {
            addMeshes(it.second, false, true, is_set(mFlags, Flags::RTDontMergeInstanced));
        }
    }

    std::pair<std::optional<MeshID>, std::optional<MeshID>> SceneBuilder::splitMesh(const MeshID meshID, const int axis, const float pos)
    {
        // Splits a mesh by an axis-aligned plane.
        // Each triangle is placed on either the left or right side of the plane with respect to its centroid.
        // Individual triangles are not split, so the resulting meshes will in general have overlapping bounding boxes.
        // If all triangles are already on either side, no split is necessary and the original mesh is retained.

        FALCOR_ASSERT_LT(meshID.get(), mMeshes.size());
        FALCOR_ASSERT(axis >= 0 && axis <= 2);
        const auto& mesh = mMeshes[meshID.get()];

        // Check if mesh is supported.
        if (mesh.isDynamic())
        {
            throw RuntimeError("Cannot split mesh '{}', only non-dynamic meshes supported", mesh.name);
        }
        if (mesh.topology != Vao::Topology::TriangleList)
        {
            throw RuntimeError("Cannot split mesh '{}', only triangle list topology supported", mesh.name);
        }

        // Early out if mesh is fully on either side of the splitting plane.
        if (mesh.boundingBox.maxPoint[axis] < pos) return { meshID, std::nullopt };
        else if (mesh.boundingBox.minPoint[axis] >= pos) return { std::nullopt, meshID };

        // Setup mesh specs.
        auto createSpec = [](const MeshSpec& mesh, const std::string& name)
        {
            MeshSpec spec;
            spec.name = name;
            spec.topology = mesh.topology;
            spec.materialId = mesh.materialId;
            spec.isStatic = mesh.isStatic;
            spec.isFrontFaceCW = mesh.isFrontFaceCW;
            spec.instances = mesh.instances;
            FALCOR_ASSERT(mesh.isDynamic() == false);
            FALCOR_ASSERT(mesh.skinningVertexCount == 0);
            return spec;
        };

        MeshSpec leftMesh = createSpec(mesh, mesh.name + ".0");
        MeshSpec rightMesh = createSpec(mesh, mesh.name + ".1");

        if (mesh.indexCount > 0) splitIndexedMesh(mesh, leftMesh, rightMesh, axis, pos);
        else splitNonIndexedMesh(mesh, leftMesh, rightMesh, axis, pos);

        // Check that no triangles were added or removed.
        FALCOR_ASSERT(leftMesh.getTriangleCount() + rightMesh.getTriangleCount() == mesh.getTriangleCount());

        // It is possible all triangles ended up on either side of the splitting plane.
        // In that case, there is no need to modify the original mesh and we'll just return.
        if (leftMesh.getTriangleCount() == 0) return { std::nullopt, meshID };
        else if (rightMesh.getTriangleCount() == 0) return { meshID, std::nullopt };

        logDebug(
            "Mesh '{}' with {} triangles was split into two meshes with '{}' and '{}' triangles, respectively.",
            mesh.name, mesh.getTriangleCount(), leftMesh.getTriangleCount(), rightMesh.getTriangleCount()
        );

        // Store new meshes.
        // The left mesh replaces the existing mesh.
        // The right mesh is appended at the end of the mesh list and linked to the instances.
        FALCOR_ASSERT(leftMesh.vertexCount > 0 && rightMesh.vertexCount > 0);
        mMeshes[meshID.get()] = std::move(leftMesh);

        MeshID rightMeshID(mMeshes.size());
        for (auto nodeID : mesh.instances)
        {
            mSceneGraph.at(nodeID.get()).meshes.push_back(rightMeshID);
        }
        mMeshes.push_back(std::move(rightMesh));

        return { meshID, rightMeshID };
    }

    void SceneBuilder::splitIndexedMesh(const MeshSpec& mesh, MeshSpec& leftMesh, MeshSpec& rightMesh, const int axis, const float pos)
    {
        FALCOR_ASSERT(mesh.indexCount > 0 && !mesh.indexData.empty());

        const uint32_t invalidIdx = uint32_t(-1);
        std::vector<uint32_t> leftIndexMap(mesh.indexCount, invalidIdx);
        std::vector<uint32_t> rightIndexMap(mesh.indexCount, invalidIdx);

        // Iterate over the triangles.
        const size_t triangleCount = mesh.getTriangleCount();
        for (size_t i = 0; i < triangleCount * 3; i += 3)
        {
            const uint32_t indices[3] = { mesh.getIndex(i + 0), mesh.getIndex(i + 1), mesh.getIndex(i + 2) };

            auto addVertex = [&](const uint32_t vtxIndex, MeshSpec& dstMesh, std::vector<uint32_t>& indexMap)
            {
                if (indexMap[vtxIndex] != invalidIdx) return indexMap[vtxIndex];

                uint32_t dstIndex = (uint32_t)dstMesh.staticData.size();
                dstMesh.staticData.push_back(mesh.staticData[vtxIndex]);
                indexMap[vtxIndex] = dstIndex;
                return dstIndex;
            };
            auto addTriangleToMesh = [&](MeshSpec& dstMesh, std::vector<uint32_t>& indexMap)
            {
                for (size_t j = 0; j < 3; j++)
                {
                    uint32_t dstIdx = addVertex(indices[j], dstMesh, indexMap);
                    dstMesh.indexData.push_back(dstIdx);
                }
            };

            // Compute the centroid and add the triangle to the left or right side.
            float centroid = 0.f;
            for (size_t j = 0; j < 3; j++)
            {
                centroid += mesh.staticData[indices[j]].position[axis];
            };
            centroid /= 3.f;

            if (centroid < pos) addTriangleToMesh(leftMesh, leftIndexMap);
            else addTriangleToMesh(rightMesh, rightIndexMap);
        }

        auto finalizeMesh = [this](MeshSpec& m)
        {
            m.indexCount = (uint32_t)m.indexData.size();
            m.vertexCount = (uint32_t)m.staticData.size();
            m.staticVertexCount = m.vertexCount;

            m.use16BitIndices = (m.vertexCount <= (1u << 16)) && !(is_set(mFlags, Flags::Force32BitIndices));
            if (m.use16BitIndices) m.indexData = compact16BitIndices(m.indexData);

            m.boundingBox = AABB();
            for (auto& v : m.staticData) m.boundingBox.include(v.position);
        };

        finalizeMesh(leftMesh);
        finalizeMesh(rightMesh);
    }

    void SceneBuilder::splitNonIndexedMesh(const MeshSpec& mesh, MeshSpec& leftMesh, MeshSpec& rightMesh, const int axis, const float pos)
    {
        FALCOR_ASSERT(mesh.indexCount == 0 && mesh.indexData.empty());
        throw RuntimeError("SceneBuilder::splitNonIndexedMesh() not implemented");
    }

    size_t SceneBuilder::countTriangles(const MeshGroup& meshGroup) const
    {
        size_t triangleCount = 0;
        for (auto meshID : meshGroup.meshList)
        {
            triangleCount += mMeshes[meshID.get()].getTriangleCount();
        }
        return triangleCount;
    }

    AABB SceneBuilder::calculateBoundingBox(const MeshGroup& meshGroup) const
    {
        AABB bb;
        for (auto meshID : meshGroup.meshList)
        {
            bb.include(mMeshes[meshID.get()].boundingBox);
        }
        return bb;
    }

    bool SceneBuilder::needsSplit(const MeshGroup& meshGroup, size_t& triangleCount) const
    {
        FALCOR_ASSERT(!meshGroup.meshList.empty());

        for (auto meshID : meshGroup.meshList)
        {
            const auto& mesh = mMeshes[meshID.get()];

            // Groups with dynamic meshes are not supported.
            if (mesh.isDynamic())
            {
                return false;
            }
        }

        triangleCount = countTriangles(meshGroup);

        if (triangleCount <= kMaxTrianglesPerBLAS)
        {
            return false;
        }
        else if (meshGroup.meshList.size() == 1)
        {
            // Issue warning if single mesh exceeds the triangle count limit.
            // TODO: Implement mesh splitting to handle this case.
            const auto& mesh = mMeshes[meshGroup.meshList[0].get()];
            FALCOR_ASSERT(mesh.getTriangleCount() == triangleCount);
            logWarning("Mesh '{}' has {} triangles, expect extraneous GPU memory usage.", mesh.name, triangleCount);

            return false;
        }
        FALCOR_ASSERT(meshGroup.meshList.size() > 1);
        FALCOR_ASSERT(triangleCount > kMaxTrianglesPerBLAS);

        return true;
    }

    SceneBuilder::MeshGroupList SceneBuilder::splitMeshGroupSimple(MeshGroup& meshGroup) const
    {
        // This function partitions a mesh group into smaller groups based on triangle count.
        // Note that the meshes are *not* reordered and individual meshes are not split,
        // so it is still possible to get large spatial overlaps between groups.

        // Early out if splitting is not needed or possible.
        size_t triangleCount = 0;
        if (!needsSplit(meshGroup, triangleCount)) return MeshGroupList{ std::move(meshGroup) };

        // Each new group holds at least one mesh, or if multiple, up to the target number of triangles.
        FALCOR_ASSERT(triangleCount > 0);
        size_t targetGroupCount = div_round_up(triangleCount, kMaxTrianglesPerBLAS);
        size_t targetTrianglesPerGroup = triangleCount / targetGroupCount;

        triangleCount = 0;
        MeshGroupList groups;

        for (auto meshID : meshGroup.meshList)
        {
            // Start new group on first iteration or if triangle count would exceed the target.
            size_t meshTris = mMeshes[meshID.get()].getTriangleCount();
            if (triangleCount == 0 || triangleCount + meshTris > targetTrianglesPerGroup)
            {
                groups.push_back({ std::vector<MeshID>(), meshGroup.isStatic });
                triangleCount = 0;
            }

            // Add mesh to group.
            groups.back().meshList.push_back(meshID);
            triangleCount += meshTris;
        }

        FALCOR_ASSERT(!groups.empty());
        return groups;
    }

    SceneBuilder::MeshGroupList SceneBuilder::splitMeshGroupMedian(MeshGroup& meshGroup) const
    {
        // This function implements a recursive top-down BVH builder to partition a mesh group
        // into smaller groups by splitting at the median in terms of triangle count.
        // Note that individual meshes are not split, so it is still possible to get large spatial overlaps between groups.

        // Early out if splitting is not needed or possible.
        size_t triangleCount = 0;
        if (!needsSplit(meshGroup, triangleCount)) return MeshGroupList{ std::move(meshGroup) };

        // Sort the meshes by centroid along the largest axis.
        AABB bb = calculateBoundingBox(meshGroup);
        const int axis = largestAxis(bb.extent());
        auto compareCentroids = [this, axis](MeshID leftMeshID, MeshID rightMeshID)
        {
            return mMeshes[leftMeshID.get()].boundingBox.center()[axis] < mMeshes[rightMeshID.get()].boundingBox.center()[axis];
        };

        std::vector<MeshID> meshes = std::move(meshGroup.meshList);
        std::sort(meshes.begin(), meshes.end(), compareCentroids);

        // Find the median mesh in terms of triangle count.
        size_t triangles = 0;
        auto countTriangles = [&](MeshID meshID)
        {
            triangles += mMeshes[meshID.get()].getTriangleCount();
            return triangles > triangleCount / 2;
        };

        auto splitIter = std::find_if(meshes.begin(), meshes.end(), countTriangles);

        // If all meshes ended up on either side, fall back on splitting at the middle mesh.
        if (splitIter == meshes.begin() || splitIter == meshes.end())
        {
            FALCOR_ASSERT(meshes.size() >= 2);
            splitIter = meshes.begin() + meshes.size() / 2;
        }
        FALCOR_ASSERT(splitIter != meshes.begin() && splitIter != meshes.end());

        // Recursively split the left and right mesh groups.
        MeshGroup leftGroup{ std::vector<MeshID>(meshes.begin(), splitIter), meshGroup.isStatic };
        MeshGroup rightGroup{ std::vector<MeshID>(splitIter, meshes.end()), meshGroup.isStatic };
        FALCOR_ASSERT(!leftGroup.meshList.empty() && !rightGroup.meshList.empty());

        MeshGroupList leftList = splitMeshGroupMedian(leftGroup);
        MeshGroupList rightList = splitMeshGroupMedian(rightGroup);

        // Move elements into a single list and return.
        leftList.insert(
            leftList.end(),
            std::make_move_iterator(rightList.begin()),
            std::make_move_iterator(rightList.end()));

        return leftList;
    }

    SceneBuilder::MeshGroupList SceneBuilder::splitMeshGroupMidpointMeshes(MeshGroup& meshGroup)
    {
        // This function recursively splits a mesh group at the midpoint along the largest axis.
        // Individual meshes that straddle the splitting plane are split into two halves.
        // This will ensure minimal spatial overlaps between groups.

        // Early out if splitting is not needed or possible.
        size_t triangleCount = 0;
        if (!needsSplit(meshGroup, triangleCount)) return MeshGroupList{ std::move(meshGroup) };

        // Find the midpoint along the largest axis.
        AABB bb = calculateBoundingBox(meshGroup);
        const int axis = largestAxis(bb.extent());
        const float pos = bb.center()[axis];

        // Partition all meshes by the splitting plane.
        std::vector<MeshID> leftMeshes, rightMeshes;

        for (auto meshID : meshGroup.meshList)
        {
            auto result = splitMesh(meshID, axis, pos);
            if (auto leftMeshID = result.first) leftMeshes.push_back(*leftMeshID);
            if (auto rightMeshID = result.second) rightMeshes.push_back(*rightMeshID);
        }

        // If either side contains all meshes, do not split further.
        if (leftMeshes.empty() || rightMeshes.empty()) return MeshGroupList{ std::move(meshGroup) };

        // Recursively split the left and right mesh groups.
        MeshGroup leftGroup{ std::move(leftMeshes), meshGroup.isStatic };
        MeshGroup rightGroup{ std::move(rightMeshes), meshGroup.isStatic };

        MeshGroupList leftList = splitMeshGroupMidpointMeshes(leftGroup);
        MeshGroupList rightList = splitMeshGroupMidpointMeshes(rightGroup);

        // Move elements into a single list and return.
        leftList.insert(
            leftList.end(),
            std::make_move_iterator(rightList.begin()),
            std::make_move_iterator(rightList.end()));

        return leftList;
    }

    void SceneBuilder::optimizeGeometry()
    {
        // This function optimizes the geometry for raytracing performance and memory usage.
        //
        // There is a max triangles per group limit to reduce the worst-case memory requirements for BLAS builds.
        // If the limit is exceeded, the geometry is split into multiple groups (BLASes).
        // Splitting has performance implications for the traversal due to spatial overlap between the BLASes.
        //
        // To reduce the perf impact we may perform these steps:
        //  - Split large mesh groups (BLASes) into multiple smaller ones.
        //  - Split large meshes into smaller to reduce spatial overlap between BLASes.
        //  - Sort meshes into BLASes based on spatial locality.

        MeshGroupList optimizedGroups;

        for (auto& meshGroup : mMeshGroups)
        {
            //auto groups = splitMeshGroupSimple(meshGroup);
            //auto groups = splitMeshGroupMedian(meshGroup);
            auto groups = splitMeshGroupMidpointMeshes(meshGroup);

            if (groups.size() > 1) logWarning("SceneBuilder::optimizeGeometry() performance warning - Mesh group was split into {} groups.", groups.size());

            optimizedGroups.insert(
                optimizedGroups.end(),
                std::make_move_iterator(groups.begin()),
                std::make_move_iterator(groups.end()));
        }

        mMeshGroups = std::move(optimizedGroups);
    }

    void SceneBuilder::sortMeshes()
    {
        // This function sorts meshes by the order they are used in the mesh groups.
        // This is required because at runtime we assume geometries within a mesh group (BLAS)
        // to use consecutive indices (e.g. mesh IDs).

        // Generate a mapping from old to new mesh IDs.
        std::unordered_map<MeshID, MeshID> old2newMeshMap;
        MeshID newMeshID{ 0 };
        for (const auto &meshGroup : mMeshGroups) {
            for (MeshID meshID : meshGroup.meshList) {
                old2newMeshMap[meshID] = newMeshID++;
            }
        }

        // Sort meshes by new IDs.
        FALCOR_ASSERT_EQ(old2newMeshMap.size(), mMeshes.size());
        std::vector<MeshSpec> sortedMeshes(mMeshes.size());
        for (MeshID i{ 0 }; i.get() < mMeshes.size(); ++i) {
            sortedMeshes[old2newMeshMap[i].get()] = std::move(mMeshes[i.get()]);
        }
        mMeshes = std::move(sortedMeshes);

        // Remap mesh lists in mesh groups.
        for (auto &meshGroup : mMeshGroups) {
            auto &meshList = meshGroup.meshList;
            for (size_t i = 0 ; i < meshList.size(); ++i) {
                meshList[i] = old2newMeshMap[meshList[i]];
            }
        }

        // Remap cached meshes.
        for (auto &cachedMesh : mSceneData.cachedMeshes)
        {
            cachedMesh.meshID = old2newMeshMap[cachedMesh.meshID];
        }
        for (auto& cache : mSceneData.cachedCurves)
        {
            if (cache.tessellationMode != CurveTessellationMode::LinearSweptSphere)
            {
                cache.geometryID = CurveOrMeshID{ old2newMeshMap[MeshID{ cache.geometryID }] };
            }
        }
    }

    void SceneBuilder::createGlobalBuffers()
    {
        FALCOR_ASSERT(mSceneData.meshIndexData.empty());
        FALCOR_ASSERT(mSceneData.meshStaticData.empty());
        FALCOR_ASSERT(mSceneData.meshSkinningData.empty());

        const bool isIndexed = !is_set(mFlags, Flags::NonIndexedVertices);

        // Count total number of vertex and index data elements.
        size_t totalIndexDataCount = 0;
        size_t totalStaticVertexCount = 0;
        size_t totalSkinningVertexCount = 0;

        for (const auto& mesh : mMeshes)
        {
            totalIndexDataCount += mesh.indexData.size();
            totalStaticVertexCount += mesh.staticData.size();
            totalSkinningVertexCount += mesh.skinningData.size();
            mSceneData.prevVertexCount += mesh.prevVertexCount;
        }

        // Check the range. We currently use 32-bit offsets.
        if (totalIndexDataCount > std::numeric_limits<uint32_t>::max() ||
            totalStaticVertexCount > std::numeric_limits<uint32_t>::max() ||
            totalSkinningVertexCount > std::numeric_limits<uint32_t>::max())
        {
            throw RuntimeError("Trying to build a scene that exceeds supported mesh data size.");
        }

        mSceneData.meshIndexData.reserve(totalIndexDataCount);
        mSceneData.meshStaticData.reserve(totalStaticVertexCount);
        mSceneData.meshSkinningData.reserve(totalSkinningVertexCount);

        // Copy all vertex and index data into the global buffers.
        for (auto& mesh : mMeshes)
        {
            mesh.staticVertexOffset = (uint32_t)mSceneData.meshStaticData.size();
            mesh.skinningVertexOffset = (uint32_t)mSceneData.meshSkinningData.size();
            mesh.prevVertexOffset = mesh.skinningVertexOffset;

            // Insert the static vertex data in the global array.
            // The vertices are automatically converted to their packed format in this step.
            mSceneData.meshStaticData.insert(mSceneData.meshStaticData.end(), mesh.staticData.begin(), mesh.staticData.end());

            if (isIndexed)
            {
                mesh.indexOffset = (uint32_t)mSceneData.meshIndexData.size();
                mSceneData.meshIndexData.insert(mSceneData.meshIndexData.end(), mesh.indexData.begin(), mesh.indexData.end());
            }

            if (mesh.isSkinned())
            {
                FALCOR_ASSERT(!mesh.skinningData.empty());
                mSceneData.meshSkinningData.insert(mSceneData.meshSkinningData.end(), mesh.skinningData.begin(), mesh.skinningData.end());

                // Patch vertex index references.
                for (uint32_t i = 0; i < mesh.skinningData.size(); ++i)
                {
                    mSceneData.meshSkinningData[mesh.skinningVertexOffset + i].staticIndex += mesh.staticVertexOffset;
                }
            }

            // Free the mesh local data.
            mesh.indexData.clear();
            mesh.staticData.clear();
            mesh.skinningData.clear();
        }

        // Initialize offsets for prev vertex data for vertex-animated meshes
        uint32_t prevOffset = (uint32_t)mSceneData.meshSkinningData.size();
        for (auto& cache : mSceneData.cachedMeshes)
        {
            auto& mesh = mMeshes[cache.meshID.get()];
            mesh.prevVertexOffset = prevOffset;
            prevOffset += mesh.prevVertexCount;
        }
        for (auto& cache : mSceneData.cachedCurves)
        {
            if (cache.tessellationMode != CurveTessellationMode::LinearSweptSphere)
            {
                auto& mesh = mMeshes[cache.geometryID.get()];
                mesh.prevVertexOffset = prevOffset;
                prevOffset += mesh.prevVertexCount;
            }
        }
    }

    void SceneBuilder::createCurveGlobalBuffers()
    {
        FALCOR_ASSERT(mSceneData.curveIndexData.empty());
        FALCOR_ASSERT(mSceneData.curveStaticData.empty());

        // Count total number of curve vertex and index data elements.
        size_t totalIndexDataCount = 0;
        size_t totalStaticCurveVertexCount = 0;

        for (const auto& curve : mCurves)
        {
            totalIndexDataCount += curve.indexData.size();
            totalStaticCurveVertexCount += curve.staticData.size();
        }

        // Check the range. We currently use 32-bit offsets.
        if (totalIndexDataCount > std::numeric_limits<uint32_t>::max() ||
            totalStaticCurveVertexCount > std::numeric_limits<uint32_t>::max())
        {
            throw RuntimeError("Trying to build a scene that exceeds supported curve data size.");
        }

        mSceneData.curveIndexData.reserve(totalIndexDataCount);
        mSceneData.curveStaticData.reserve(totalStaticCurveVertexCount);

        // Copy all curve vertex and index data into the curve global buffers.
        for (auto& curve : mCurves)
        {
            curve.staticVertexOffset = (uint32_t)mSceneData.curveStaticData.size();
            mSceneData.curveStaticData.insert(mSceneData.curveStaticData.end(), curve.staticData.begin(), curve.staticData.end());

            curve.indexOffset = (uint32_t)mSceneData.curveIndexData.size();
            mSceneData.curveIndexData.insert(mSceneData.curveIndexData.end(), curve.indexData.begin(), curve.indexData.end());

            // Free the curve local data.
            curve.indexData.clear();
            curve.staticData.clear();
        }
    }

    void SceneBuilder::optimizeMaterials()
    {
        // This passes optimizes the materials by analyzing the material textures
        // and replacing constant textures by uniform material parameters.
        // NOTE: This code has to be updated if the texture usage changes.

        if (is_set(mFlags, Flags::DontOptimizeMaterials)) return;

        mSceneData.pMaterials->optimizeMaterials();
    }

    void SceneBuilder::removeDuplicateMaterials()
    {
        // This pass identifies materials with identical set of parameters.
        // It should run after optimizeMaterials() as materials with different
        // textures may be reduced to identical materials after optimization,
        // increasing the likelihood of finding duplicates here.

        if (is_set(mFlags, Flags::DontMergeMaterials)) return;

        std::vector<MaterialID> idMap;
        size_t removed = mSceneData.pMaterials->removeDuplicateMaterials(idMap);

        // Reassign material IDs.
        if (removed > 0)
        {
            for (auto& mesh : mMeshes)
            {
                mesh.materialId = idMap[mesh.materialId.get()];
            }

            for (auto& curve : mCurves)
            {
                curve.materialId = idMap[curve.materialId.get()];
            }

            for (auto& sdfGridInstance : mSceneData.sdfGridInstances)
            {
                sdfGridInstance.materialID = idMap[sdfGridInstance.materialID].getSlang();
            }
        }
    }

    void SceneBuilder::collectVolumeGrids()
    {
        // Collect grids from volumes.
        std::set<Grid::SharedPtr> uniqueGrids;
        for (auto& pGridVolume : mSceneData.gridVolumes)
        {
            auto grids = pGridVolume->getAllGrids();
            uniqueGrids.insert(grids.begin(), grids.end());
        }
        mSceneData.grids = std::vector<Grid::SharedPtr>(uniqueGrids.begin(), uniqueGrids.end());
    }

    void SceneBuilder::quantizeTexCoords()
    {
        // Match texture coordinate quantization for textured emissives to format of PackedEmissiveTriangle.
        // This is to avoid mismatch when sampling and evaluating emissive triangles.
        // Note that non-emissive meshes are unmodified and use full precision texcoords.
        for (auto& mesh : mMeshes)
        {
            const auto& pMaterial = mSceneData.pMaterials->getMaterial(mesh.materialId)->toBasicMaterial();
            if (pMaterial && pMaterial->getEmissiveTexture() != nullptr)
            {
                // Quantize texture coordinates to fp16. Also track the bounds and max error.
                float2 minTexCrd = float2(std::numeric_limits<float>::infinity());
                float2 maxTexCrd = float2(-std::numeric_limits<float>::infinity());
                float2 maxError = float2(0);

                for (uint32_t i = 0; i < mesh.staticVertexCount; ++i)
                {
                    auto& v = mSceneData.meshStaticData[mesh.staticVertexOffset + i];
                    float2 texCrd = v.texCrd;
                    minTexCrd = min(minTexCrd, texCrd);
                    maxTexCrd = max(maxTexCrd, texCrd);
                    v.texCrd = f16tof32(f32tof16(texCrd));
                    maxError = max(maxError, abs(v.texCrd - texCrd));
                }

                // Issue warning if quantization errors are too large.
                float2 maxAbsCrd = max(abs(minTexCrd), abs(maxTexCrd));
                if (maxAbsCrd.x > HLF_MAX || maxAbsCrd.y > HLF_MAX)
                {
                    logWarning("Texture coordinates for emissive textured mesh '{}' are outside the representable range, expect rendering errors.", mesh.name);
                }
                else
                {
                    // Compute maximum quantization error in texels.
                    // The texcoords are used for all texture channels so taking the maximum dimensions.
                    uint2 maxTexDim = pMaterial->getMaxTextureDimensions();
                    maxError *= maxTexDim;
                    float maxTexelError = std::max(maxError.x, maxError.y);

                    if (maxTexelError > kMaxTexelError)
                    {
                        logWarning(
                            "Texture coordinates for emissive textured mesh '{}' have a large quantization error of {} texels."
                            "The coordinate range is [{},{}] x [{},{}] for maximum texture dimensions ({},{}).",
                            mesh.name, maxTexelError,
                            minTexCrd.x, maxTexCrd.x, minTexCrd.y, maxTexCrd.y, maxTexDim.x, maxTexDim.y
                        );
                    }
                }
            }
        }
    }

    void SceneBuilder::removeDuplicateSDFGrids()
    {
        // Removes duplicate SDF grids.

        std::vector<SDFGrid::SharedPtr> uniqueSDFGrids;
        std::unordered_set<SdfGridID> removedIDs;

        for (SdfGridID i{ 0 }; i.get() < mSceneData.sdfGrids.size(); ++i)
        {
            if (removedIDs.count(i) > 0)
                continue;

            const SDFGrid::SharedPtr& pSDFGridA = mSceneData.sdfGrids[i.get()];
            uniqueSDFGrids.push_back(pSDFGridA);

            if (removedIDs.size() > 0)
                updateSDFGridID(i, SdfGridID{ i.get() - (uint32_t)removedIDs.size() });

            for (SdfGridID j{ i.get() + 1 }; j.get() < mSceneData.sdfGrids.size(); ++j)
            {
                const SDFGrid::SharedPtr& pSDFGridB = mSceneData.sdfGrids[j.get()];

                if (pSDFGridA == pSDFGridB)
                {
                    removedIDs.insert(j);
                    updateSDFGridID(j, i);
                }
            }
        }

        mSceneData.sdfGrids = std::move(uniqueSDFGrids);
    }

    void SceneBuilder::createMeshData()
    {
        FALCOR_ASSERT(mSceneData.meshDesc.empty());

        auto& meshData = mSceneData.meshDesc;
        meshData.resize(mMeshes.size());

        // Setup all mesh data.
        for (uint32_t meshID = 0; meshID < mMeshes.size(); meshID++)
        {
            const auto& mesh = mMeshes[meshID];
            meshData[meshID].materialID = mesh.materialId.getSlang();
            meshData[meshID].vbOffset = mesh.staticVertexOffset;
            meshData[meshID].ibOffset = mesh.indexOffset;
            meshData[meshID].vertexCount = mesh.vertexCount;
            meshData[meshID].indexCount = mesh.indexCount;
            meshData[meshID].skinningVbOffset = mesh.hasSkinningData ? mesh.skinningVertexOffset : 0;
            meshData[meshID].prevVbOffset = mesh.isDynamic() ? mesh.prevVertexOffset : 0;
            FALCOR_ASSERT(mesh.skinningVertexCount == 0 || mesh.skinningVertexCount == mesh.staticVertexCount);

            mSceneData.meshNames.push_back(mesh.name);

            uint32_t meshFlags = 0;
            meshFlags |= mesh.use16BitIndices ? (uint32_t)MeshFlags::Use16BitIndices : 0;
            meshFlags |= mesh.isSkinned() ? (uint32_t)MeshFlags::IsSkinned : 0;
            meshFlags |= mesh.isFrontFaceCW ? (uint32_t)MeshFlags::IsFrontFaceCW : 0;
            meshFlags |= mesh.isDisplaced ? (uint32_t)MeshFlags::IsDisplaced : 0;
            meshFlags |= mesh.isAnimated ? (uint32_t)MeshFlags::IsAnimated : 0;
            meshData[meshID].flags = meshFlags;

            if (mesh.use16BitIndices) mSceneData.has16BitIndices = true;
            else mSceneData.has32BitIndices = true;

            if (mesh.isSkinned())
            {
                // Dynamic (skinned) meshes can only be instanced if an explicit skeleton transform node is specified.
                FALCOR_ASSERT(mesh.instances.size() == 1 || mesh.skeletonNodeID != NodeID::Invalid());

                for (uint32_t i = 0; i < mesh.vertexCount; i++)
                {
                    SkinningVertexData& s = mSceneData.meshSkinningData[mesh.skinningVertexOffset + i];

                    // The bind matrix is per mesh, so just take it from the first instance
                    s.bindMatrixID = mesh.instances[0].getSlang();

                    // If a skeleton's world transform node is not explicitly set, it is the same transform as the instance (Assimp behavior)
                    s.skeletonMatrixID = mesh.skeletonNodeID == NodeID::Invalid() ? mesh.instances[0].getSlang() : mesh.skeletonNodeID.getSlang();
                }
            }
        }
    }

    void SceneBuilder::createMeshInstanceData(uint32_t& tlasInstanceIndex)
    {
        // Setup all mesh instances.
        //
        // Mesh instances are added in the same order as the meshes in the mesh groups.
        // For ray tracing, one BLAS per mesh group is created and the mesh instances
        // can therefore be directly indexed by [InstanceID() + GeometryIndex()].
        // This avoids the need to have a lookup table from hit IDs to mesh instance.
        //
        // If the mesh group is instanced, then a complete set of such mesh instances
        // are created for each instance. The final mesh instances are thus ordered by:
        //
        //  1. Mesh group
        //  2. Mesh group instance
        //  3. Mesh within the mesh group
        //
        // Example:
        //
        //  MeshGroup 0 has 3 meshes with 2 instance nodes => 6 mesh instances added
        //  MeshGroup 1 has 2 meshes with 1 instance node  => 2 mesh instances added
        //
        //  Final layout:
        //
        //  |-----------------------------------------------------------------------|
        //  | Inst 0 | Inst 1 | Inst 2 | Inst 3 | Inst 4 | Inst 5 | Inst 6 | Inst 7 |
        //  | Mesh 0 | Mesh 1 | Mesh 2 | Mesh 0 | Mesh 1 | Mesh 2 | Mesh 0 | Mesh 1 |
        //  |-----------------------------------------------------------------------|
        //  <--      MeshGroup 0    --><--     MeshGroup 0     --><-- MeshGroup 1 -->
        //
        FALCOR_ASSERT(mSceneData.meshInstanceData.empty());
        FALCOR_ASSERT(mSceneData.meshIdToInstanceIds.empty());
        FALCOR_ASSERT(mSceneData.meshGroups.empty());

        auto& instanceData = mSceneData.meshInstanceData;
        size_t drawCount = 0;

        for (const auto& meshGroup : mMeshGroups)
        {
            const auto& meshList = meshGroup.meshList;

            // If mesh group is instanced, all meshes have identical lists of instances.
            // This is a requirement for ray tracing and ensured by createMeshGroups().
            // For non-instanced static mesh groups, we allow the meshes to have different nodes.
            // This case is handled by pre-transforming the vertices in the BLAS build.
            FALCOR_ASSERT(!meshList.empty());
            const auto& firstMesh = mMeshes[meshList[0].get()];
            size_t instanceCount = firstMesh.instances.size();

            FALCOR_ASSERT(instanceCount > 0);
            for (size_t instanceIdx = 0; instanceIdx < instanceCount; instanceIdx++)
            {
                uint32_t blasGeometryIndex = 0;
                for (const MeshID meshID : meshList)
                {
                    const auto& mesh = mMeshes[meshID.get()];

                    // Figure out node ID to use for the current mesh instance.
                    // If mesh group is non-instanced, just use the current mesh's node.
                    // If instanced, then all meshes have identical lists of instances.
                    // But there is a subtle issue: the lists may be permuted differently depending on the order
                    // in which mesh instances were added. Therefore, use the node ID from the first mesh to get
                    // a consistent ordering across all meshes. This is a requirement for the TLAS build.
                    FALCOR_ASSERT(instanceCount == mesh.instances.size());
                    NodeID nodeID = instanceCount == 1
                        ? mesh.instances[0] // non-instanced => use per-mesh transform.
                        : firstMesh.instances[instanceIdx]; // instanced => get transform from the first mesh.

                    GeometryType geomType = GeometryType::TriangleMesh;
                    if (meshGroup.isDisplaced) geomType = GeometryType::DisplacedTriangleMesh;

                    GeometryInstanceData instance(geomType);
                    instance.globalMatrixID = nodeID.getSlang();
                    instance.materialID = mesh.materialId.getSlang();
                    instance.geometryID = meshID.getSlang();
                    instance.vbOffset = mesh.staticVertexOffset;
                    instance.ibOffset = mesh.indexOffset;
                    instance.flags |= mesh.use16BitIndices ? (uint32_t)GeometryInstanceFlags::Use16BitIndices : 0;
                    instance.flags |= mesh.isDynamic() ? (uint32_t)GeometryInstanceFlags::IsDynamic : 0;
                    instance.instanceIndex = tlasInstanceIndex;
                    instance.geometryIndex = blasGeometryIndex;
                    instanceData.push_back(instance);

                    blasGeometryIndex++;
                }

                tlasInstanceIndex++;
            }

            drawCount += instanceCount * meshList.size();
        }

        // Create mapping of mesh IDs to their instance IDs.
        mSceneData.meshIdToInstanceIds.resize(mMeshes.size());
        for (uint32_t instanceID = 0; instanceID < (uint32_t)instanceData.size(); instanceID++)
        {
            const auto& instance = instanceData[instanceID];
            mSceneData.meshIdToInstanceIds[instance.geometryID].push_back(instanceID);
        }

        // Setup mesh groups. This just copies our final list.
        mSceneData.meshGroups = mMeshGroups;

        FALCOR_ASSERT(drawCount <= std::numeric_limits<uint32_t>::max());
        mSceneData.meshDrawCount = (uint32_t)drawCount;
    }

    void SceneBuilder::createCurveData()
    {
        auto& curveData = mSceneData.curveDesc;
        curveData.resize(mCurves.size());

        for (uint32_t curveID = 0; curveID < mCurves.size(); curveID++)
        {
            // Curve data.
            const auto& curve = mCurves[curveID];
            curveData[curveID].materialID = curve.materialId.getSlang();
            curveData[curveID].degree = curve.degree;
            curveData[curveID].vbOffset = curve.staticVertexOffset;
            curveData[curveID].ibOffset = curve.indexOffset;
            curveData[curveID].vertexCount = curve.vertexCount;
            curveData[curveID].indexCount = curve.indexCount;
        }
    }

    void SceneBuilder::createCurveInstanceData(uint32_t& tlasInstanceIndex)
    {
        auto& instanceData = mSceneData.curveInstanceData;

        uint32_t blasGeometryIndex = 0;
        size_t maxInstanceCount = 0;

        for (uint32_t curveID = 0; curveID < mCurves.size(); curveID++)
        {
            const auto& curve = mCurves[curveID];
            size_t instanceCount = curve.instances.size();
            if (instanceCount > 1) throw RuntimeError("Instanced curves are currently not supported!");
            maxInstanceCount = std::max(maxInstanceCount, instanceCount);
            for (size_t instanceIdx = 0; instanceIdx < instanceCount; instanceIdx++)
            {
                GeometryInstanceData instance(GeometryType::Curve);
                instance.globalMatrixID = curve.instances[instanceIdx].getSlang();
                instance.materialID = curve.materialId.getSlang();
                instance.geometryID = curveID;
                instance.vbOffset = curve.staticVertexOffset;
                instance.ibOffset = curve.indexOffset;
                instance.instanceIndex = tlasInstanceIndex + (uint32_t)instanceIdx;
                instance.geometryIndex = blasGeometryIndex;
                instanceData.push_back(instance);
            }

            blasGeometryIndex++;
        }

        tlasInstanceIndex += (uint32_t)maxInstanceCount;
    }

    void SceneBuilder::createSceneGraph()
    {
        mSceneData.sceneGraph.resize(mSceneGraph.size());

        for (size_t i = 0; i < mSceneGraph.size(); i++)
        {
            FALCOR_ASSERT(mSceneGraph[i].parent.get() <= std::numeric_limits<uint32_t>::max());
            mSceneData.sceneGraph[i] = Scene::Node(mSceneGraph[i].name, mSceneGraph[i].parent, mSceneGraph[i].transform, mSceneGraph[i].meshBind, mSceneGraph[i].localToBindPose);
        }
    }

    void SceneBuilder::createMeshBoundingBoxes()
    {
        mSceneData.meshBBs.resize(mMeshes.size());

        for (size_t i = 0; i < mMeshes.size(); i++)
        {
            const auto& mesh = mMeshes[i];
            mSceneData.meshBBs[i] = mesh.boundingBox;
        }
    }

    void SceneBuilder::calculateCurveBoundingBoxes()
    {
        // Calculate curve bounding boxes.
        mSceneData.curveBBs.resize(mCurves.size());
        for (size_t i = 0; i < mCurves.size(); i++)
        {
            const auto& curve = mCurves[i];
            AABB curveBB;

            const auto* staticData = &mSceneData.curveStaticData[curve.staticVertexOffset];
            for (uint32_t v = 0; v < curve.vertexCount; v++)
            {
                float radius = staticData[v].radius;
                curveBB.include(staticData[v].position - float3(radius));
                curveBB.include(staticData[v].position + float3(radius));
            }

            mSceneData.curveBBs[i] = curveBB;
        }
    }

    FALCOR_SCRIPT_BINDING(SceneBuilder)
    {
        using namespace pybind11::literals;

        FALCOR_SCRIPT_BINDING_DEPENDENCY(Scene)
        FALCOR_SCRIPT_BINDING_DEPENDENCY(TriangleMesh)
        FALCOR_SCRIPT_BINDING_DEPENDENCY(Material)
        FALCOR_SCRIPT_BINDING_DEPENDENCY(Light)
        FALCOR_SCRIPT_BINDING_DEPENDENCY(Transform)
        FALCOR_SCRIPT_BINDING_DEPENDENCY(EnvMap)
        FALCOR_SCRIPT_BINDING_DEPENDENCY(Animation)
        FALCOR_SCRIPT_BINDING_DEPENDENCY(AABB)
        FALCOR_SCRIPT_BINDING_DEPENDENCY(GridVolume)

        pybind11::enum_<SceneBuilder::Flags> flags(m, "SceneBuilderFlags");
        flags.value("Default", SceneBuilder::Flags::Default);
        flags.value("DontMergeMaterials", SceneBuilder::Flags::DontMergeMaterials);
        flags.value("UseOriginalTangentSpace", SceneBuilder::Flags::UseOriginalTangentSpace);
        flags.value("AssumeLinearSpaceTextures", SceneBuilder::Flags::AssumeLinearSpaceTextures);
        flags.value("DontMergeMeshes", SceneBuilder::Flags::DontMergeMeshes);
        flags.value("UseSpecGlossMaterials", SceneBuilder::Flags::UseSpecGlossMaterials);
        flags.value("UseMetalRoughMaterials", SceneBuilder::Flags::UseMetalRoughMaterials);
        flags.value("NonIndexedVertices", SceneBuilder::Flags::NonIndexedVertices);
        flags.value("Force32BitIndices", SceneBuilder::Flags::Force32BitIndices);
        flags.value("RTDontMergeStatic", SceneBuilder::Flags::RTDontMergeStatic);
        flags.value("RTDontMergeDynamic", SceneBuilder::Flags::RTDontMergeDynamic);
        flags.value("RTDontMergeInstanced", SceneBuilder::Flags::RTDontMergeInstanced);
        flags.value("FlattenStaticMeshInstances", SceneBuilder::Flags::FlattenStaticMeshInstances);
        flags.value("DontOptimizeGraph", SceneBuilder::Flags::DontOptimizeGraph);
        flags.value("DontOptimizeMaterials", SceneBuilder::Flags::DontOptimizeMaterials);
        flags.value("DontUseDisplacement", SceneBuilder::Flags::DontUseDisplacement);
        flags.value("UseCompressedHitInfo", SceneBuilder::Flags::UseCompressedHitInfo);
        flags.value("TessellateCurvesIntoPolyTubes", SceneBuilder::Flags::TessellateCurvesIntoPolyTubes);
        flags.value("UseCache", SceneBuilder::Flags::UseCache);
        flags.value("RebuildCache", SceneBuilder::Flags::RebuildCache);
        ScriptBindings::addEnumBinaryOperators(flags);

        pybind11::class_<SceneBuilder, SceneBuilder::SharedPtr> sceneBuilder(m, "SceneBuilder");
        sceneBuilder.def_property_readonly("flags", &SceneBuilder::getFlags);
        sceneBuilder.def_property_readonly("materials", &SceneBuilder::getMaterials);
        sceneBuilder.def_property_readonly("gridVolumes", &SceneBuilder::getGridVolumes);
        sceneBuilder.def_property_readonly("volumes", &SceneBuilder::getGridVolumes); // PYTHONDEPRECATED
        sceneBuilder.def_property_readonly("lights", &SceneBuilder::getLights);
        sceneBuilder.def_property_readonly("cameras", &SceneBuilder::getCameras);
        sceneBuilder.def_property_readonly("animations", &SceneBuilder::getAnimations);
        sceneBuilder.def_property("renderSettings", pybind11::overload_cast<>(&SceneBuilder::getRenderSettings, pybind11::const_), &SceneBuilder::setRenderSettings);
        sceneBuilder.def_property("envMap", &SceneBuilder::getEnvMap, &SceneBuilder::setEnvMap);
        sceneBuilder.def_property("selectedCamera", &SceneBuilder::getSelectedCamera, &SceneBuilder::setSelectedCamera);
        sceneBuilder.def_property("cameraSpeed", &SceneBuilder::getCameraSpeed, &SceneBuilder::setCameraSpeed);
        sceneBuilder.def("importScene", [] (SceneBuilder* pSceneBuilder, const std::filesystem::path& path, const pybind11::dict& dict, const std::vector<Transform>& instances) {
            SceneBuilder::InstanceMatrices instanceMatrices;
            for (const auto& instance : instances)
            {
                instanceMatrices.push_back(instance.getMatrix());
            }
            pSceneBuilder->import(path, instanceMatrices, Dictionary(dict));
        }, "path"_a, "dict"_a = pybind11::dict(), "instances"_a = std::vector<Transform>());
        sceneBuilder.def("addTriangleMesh", &SceneBuilder::addTriangleMesh, "triangleMesh"_a, "material"_a);
        sceneBuilder.def("addSDFGrid", &SceneBuilder::addSDFGrid, "sdfGrid"_a, "material"_a);
        sceneBuilder.def("addMaterial", &SceneBuilder::addMaterial, "material"_a);
        sceneBuilder.def("getMaterial", &SceneBuilder::getMaterial, "name"_a);
        sceneBuilder.def("loadMaterialTexture", &SceneBuilder::loadMaterialTexture, "material"_a, "slot"_a, "path"_a);
        sceneBuilder.def("waitForMaterialTextureLoading", &SceneBuilder::waitForMaterialTextureLoading);
        sceneBuilder.def("addGridVolume", &SceneBuilder::addGridVolume, "gridVolume"_a, "nodeID"_a = NodeID::kInvalidID);
        sceneBuilder.def("addVolume", &SceneBuilder::addGridVolume, "gridVolume"_a, "nodeID"_a = NodeID::kInvalidID); // PYTHONDEPRECATED
        sceneBuilder.def("getGridVolume", &SceneBuilder::getGridVolume, "name"_a);
        sceneBuilder.def("getVolume", &SceneBuilder::getGridVolume, "name"_a); // PYTHONDEPRECATED
        sceneBuilder.def("addLight", &SceneBuilder::addLight, "light"_a);
        sceneBuilder.def("getLight", &SceneBuilder::getLight, "name"_a);
        sceneBuilder.def("loadLightProfile", &SceneBuilder::loadLightProfile, "filename"_a, "normalize"_a = true);
        sceneBuilder.def("addCamera", &SceneBuilder::addCamera, "camera"_a);
        sceneBuilder.def("addAnimation", &SceneBuilder::addAnimation, "animation"_a);
        sceneBuilder.def("createAnimation", &SceneBuilder::createAnimation, "animatable"_a, "name"_a, "duration"_a);
        sceneBuilder.def("addNode", [] (SceneBuilder* pSceneBuilder, const std::string& name, const Transform& transform, NodeID parent) {
            checkArgument(pSceneBuilder, "'pSceneBuilder' is missing");
            SceneBuilder::Node node;
            node.name = name;
            node.transform = transform.getMatrix();
            node.parent = parent;
            return pSceneBuilder->addNode(node);
        }, "name"_a, "transform"_a = Transform(), "parent"_a = NodeID::kInvalidID);
        sceneBuilder.def("addMeshInstance", &SceneBuilder::addMeshInstance);
        sceneBuilder.def("addSDFGridInstance", &SceneBuilder::addSDFGridInstance);
        sceneBuilder.def("addCustomPrimitive", &SceneBuilder::addCustomPrimitive);
    }
}
