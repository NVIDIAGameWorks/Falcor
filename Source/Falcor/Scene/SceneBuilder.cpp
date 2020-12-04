/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include "stdafx.h"
#include "SceneBuilder.h"
#include "Importer.h"
#include "Utils/Math/MathConstants.slangh"
#include "Utils/Timing/TimeReport.h"
#include <mikktspace.h>
#include <filesystem>

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
                    logWarning("Can't generate tangent space. The mesh '" + mesh.name + "' doesn't have positions/normals/texCrd/indices.");
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
                    logError("Failed to generate MikkTSpace tangents for the mesh '" + mesh.name + "'.");
                    return {};
                }

                return wrapper.mTangents;
            }

        private:
            MikkTSpaceWrapper(const SceneBuilder::Mesh& mesh)
                : mMesh(mesh)
            {
                assert(mesh.indexCount > 0);
                mTangents.resize(mesh.indexCount, float4(0));
            }
            const SceneBuilder::Mesh& mMesh;
            std::vector<float4> mTangents;
            int32_t getFaceCount() const { return (int32_t)mMesh.faceCount; }
            void getPosition(float position[], int32_t face, int32_t vert) { *reinterpret_cast<float3*>(position) = mMesh.getPosition(face, vert); }
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
            size_t sz = div_round_up(indices.size(), 2ull); // Storing two 16-bit indices per dword.
            std::vector<uint32_t> indexData(sz);
            uint16_t* pIndices = reinterpret_cast<uint16_t*>(indexData.data());
            for (size_t i = 0; i < indices.size(); i++)
            {
                assert(indices[i] < (1u << 16));
                pIndices[i] = static_cast<uint16_t>(indices[i]);
            }
            return indexData;
        }
    }

    SceneBuilder::SceneBuilder(Flags flags) : mFlags(flags) {}

    SceneBuilder::SharedPtr SceneBuilder::create(Flags flags)
    {
        return SharedPtr(new SceneBuilder(flags));
    }

    SceneBuilder::SharedPtr SceneBuilder::create(const std::string& filename, Flags buildFlags, const InstanceMatrices& instances)
    {
        auto pBuilder = create(buildFlags);
        return pBuilder->import(filename, instances) ? pBuilder : nullptr;
    }

    bool SceneBuilder::import(const std::string& filename, const InstanceMatrices& instances, const Dictionary& dict)
    {
        bool success = Importer::import(filename, *this, instances, dict);
        mFilename = filename;
        return success;
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
            auto dummyMaterial = Material::create("Dummy");
            auto meshID = addTriangleMesh(dummyMesh, dummyMaterial);
            Node dummyNode = { "Dummy", glm::identity<glm::mat4>(), glm::identity<glm::mat4>() };
            auto nodeID = addNode(dummyNode);
            addMeshInstance(nodeID, meshID);
        }

        // Post-process the scene data.
        TimeReport timeReport;

        removeUnusedMeshes();
        pretransformStaticMeshes();
        calculateMeshBoundingBoxes();
        createMeshGroups();
        optimizeGeometry();
        createGlobalBuffers();
        createCurveGlobalBuffers();
        removeDuplicateMaterials();
        collectVolumeGrids();
        quantizeTexCoords();

        timeReport.measure("Post processing meshes");

        // Create the scene object and assign resources.
        mpScene = Scene::create();
        mpScene->mRenderSettings = mRenderSettings;
        mpScene->mCameras = mCameras;
        mpScene->mSelectedCamera = (uint32_t)(mpSelectedCamera ? std::distance(mCameras.begin(), std::find(mCameras.begin(), mCameras.end(), mpSelectedCamera)) : 0);
        mpScene->mCameraSpeed = mCameraSpeed;
        mpScene->mLights = mLights;
        mpScene->mMaterials = mMaterials;
        mpScene->mVolumes = mVolumes;
        mpScene->mCustomPrimitiveAABBs = mCustomPrimitiveAABBs;
        mpScene->mGrids = mGrids;
        mpScene->mGridIDs = mGridIDs;
        mpScene->mpEnvMap = mpEnvMap;
        mpScene->mFilename = mFilename;

        // Prepare scene resources.
        createNodeList();

        uint32_t drawCount = createMeshData();
        createMeshVao(drawCount);
        createMeshBoundingBoxes();

        if (!mCurves.empty())
        {
            createCurveData();
            createCurveVao();
            calculateCurveBoundingBoxes();
            mapCurvesToProceduralPrimitives(Scene::kCurveIntersectionTypeID);
        }

        createRaytracingAABBData();

        mpScene->mpAnimationController = AnimationController::create(mpScene.get(), mBuffersData.staticData, mBuffersData.dynamicData, mAnimations);

        // Finalize the scene object. This is where the final setup is done.
        mpScene->finalize();

        timeReport.measure("Creating resources");
        timeReport.printToLog();

        return mpScene;
    }

    // Meshes

    uint32_t SceneBuilder::addMesh(const Mesh& mesh)
    {
        return addProcessedMesh(processMesh(mesh));
    }

    uint32_t SceneBuilder::addTriangleMesh(const TriangleMesh::SharedPtr& pTriangleMesh, const Material::SharedPtr& pMaterial)
    {
        Mesh mesh;

        const auto& indices = pTriangleMesh->getIndices();
        const auto& vertices = pTriangleMesh->getVertices();

        mesh.name = pTriangleMesh->getName();
        mesh.faceCount = (uint32_t)(indices.size() / 3);
        mesh.vertexCount = (uint32_t)vertices.size();
        mesh.indexCount = (uint32_t)indices.size();
        mesh.pIndices = indices.data();
        mesh.topology = Vao::Topology::TriangleList;
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

    SceneBuilder::ProcessedMesh SceneBuilder::processMesh(const Mesh& mesh_) const
    {
        // This function preprocesses a mesh into the final runtime representation.
        // Note the function needs to be thread safe. The following steps are performed:
        //  - Error checking
        //  - Compute tangent space if needed
        //  - Merge identical vertices, compute new indices
        //  - Validate final vertex data
        //  - Compact vertices/indices into runtime format

        // Copy the mesh desc so we can update it. The caller retains the ownership of the data.
        Mesh mesh = mesh_;
        ProcessedMesh processedMesh;

        processedMesh.name = mesh.name;
        processedMesh.topology = mesh.topology;
        processedMesh.pMaterial = mesh.pMaterial;
        processedMesh.isFrontFaceCW = mesh.isFrontFaceCW;

        // Error checking.
        auto throw_on_missing_element = [&](const std::string& element)
        {
            throw std::runtime_error("Error when adding the mesh '" + mesh.name + "' to the scene.\nThe mesh is missing " + element + ".");
        };

        auto missing_element_warning = [&](const std::string& element)
        {
            logWarning("The mesh '" + mesh.name + "' is missing the element " + element + ". This is not an error, the element will be filled with zeros which may result in incorrect rendering.");
        };

        if (mesh.topology != Vao::Topology::TriangleList) throw std::runtime_error("Error when adding the mesh '" + mesh.name + "' to the scene.\nOnly triangle list topology is supported.");
        if (mesh.pMaterial == nullptr) throw_on_missing_element("material");

        if (mesh.faceCount == 0) throw_on_missing_element("faces");
        if (mesh.vertexCount == 0) throw_on_missing_element("vertices");
        if (mesh.indexCount == 0 || !mesh.pIndices) throw_on_missing_element("indices");
        if (mesh.indexCount != mesh.faceCount * 3) throw std::runtime_error("Error when adding the mesh '" + mesh.name + "' to the scene.\nUnexpected face/vertex count.");

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
            tangents = MikkTSpaceWrapper::generateTangents(mesh);
            if (!tangents.empty())
            {
                assert(tangents.size() == mesh.indexCount);
                mesh.tangents.pData = tangents.data();
                mesh.tangents.frequency = Mesh::AttributeFrequency::FaceVarying;
            }
            else
            {
                mesh.tangents.pData = nullptr;
                mesh.tangents.frequency = Mesh::AttributeFrequency::None;
            }
        }

        // Pretransform the texture coordinates, rather than tranforming them at runtime.
        std::vector<float2> transformedTexCoords;
        if (mesh.texCrds.pData != nullptr)
        {
            const glm::mat4 xform = mesh.pMaterial->getTextureTransform().getMatrix();
            if (xform != glm::identity<glm::mat4>())
            {
                size_t texCoordCount = mesh.getAttributeCount(mesh.texCrds);
                transformedTexCoords.resize(texCoordCount);
                // The given matrix transforms the texture (e.g., scaling > 1 enlarges the texture).
                // Because we're transforming the input coordinates, apply the inverse.
                const float4x4 invXform = glm::inverse(xform);
                // Because texture transforms are 2D and affine, we only need apply the corresponding 3x2 matrix
                glm::mat3x2 coordTransform;
                coordTransform[0] = invXform[0].xy;
                coordTransform[1] = invXform[1].xy;
                coordTransform[2] = invXform[3].xy;

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
        vertices.reserve(mesh.vertexCount);
        std::vector<uint32_t> indices(mesh.indexCount);
        std::vector<uint32_t> heads(mesh.vertexCount, invalidIndex);

        for (uint32_t face = 0; face < mesh.faceCount; face++)
        {
            for (uint32_t vert = 0; vert < 3; vert++)
            {
                const Mesh::Vertex v = mesh.getVertex(face, vert);
                const uint32_t origIndex = mesh.pIndices[face * 3 + vert];

                // Iterate over vertex list to check if it already exists.
                assert(origIndex < heads.size());
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
                    assert(vertices.size() < std::numeric_limits<uint32_t>::max());
                    index = (uint32_t)vertices.size();
                    vertices.push_back({ v, heads[origIndex] });
                    heads[origIndex] = index;
                }

                // Store new vertex index.
                indices[face * 3 + vert] = index;
            }
        }

        assert(vertices.size() > 0);
        assert(indices.size() == mesh.indexCount);
        if (vertices.size() != mesh.vertexCount)
        {
            logDebug("Mesh with name '" + mesh.name + "' had original vertex count " + std::to_string(mesh.vertexCount) + ", new vertex count " + std::to_string(vertices.size()));
        }

        // Validate vertex data to check for invalid numbers and missing tangent frame.
        size_t invalidCount = 0;
        size_t zeroCount = 0;
        for (const auto& v : vertices)
        {
            validateVertex(v.first, invalidCount, zeroCount);
        }
        if (invalidCount > 0) logWarning("The mesh '" + mesh.name + "' has inf/nan vertex attributes at " + std::to_string(invalidCount) + " vertices. Please fix the asset.");
        if (zeroCount > 0) logWarning("The mesh '" + mesh.name + "' has zero-length normals/tangents at " + std::to_string(zeroCount) + " vertices. Please fix the asset.");

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
        if (mesh.hasBones()) processedMesh.dynamicData.reserve(vertexCount);

        for (uint32_t i = 0; i < vertexCount; i++)
        {
            uint32_t index = isIndexed ? i : indices[i];
            assert(index < vertices.size());
            const Mesh::Vertex& v = vertices[index].first;

            StaticVertexData s;
            s.position = v.position;
            s.normal = v.normal;
            s.texCrd = v.texCrd;
            s.tangent = v.tangent;
            processedMesh.staticData.push_back(s);

            if (mesh.hasBones())
            {
                DynamicVertexData d;
                d.boneWeight = v.boneWeights;
                d.boneID = v.boneIDs;
                d.staticIndex = i; // This references the local vertex here and gets updated in addProcessedMesh().
                d.globalMatrixID = 0; // This will be initialized in createMeshData().
                processedMesh.dynamicData.push_back(d);
            }
        }

        return processedMesh;
    }

    uint32_t SceneBuilder::addProcessedMesh(const ProcessedMesh& mesh)
    {
        const bool isIndexed = !is_set(mFlags, Flags::NonIndexedVertices);

        MeshSpec spec;

        // Add the mesh to the scene.
        spec.name = mesh.name;
        spec.topology = mesh.topology;
        spec.materialId = addMaterial(mesh.pMaterial);
        spec.isFrontFaceCW = mesh.isFrontFaceCW;

        spec.vertexCount = (uint32_t)mesh.staticData.size();
        spec.staticVertexCount = (uint32_t)mesh.staticData.size();
        spec.dynamicVertexCount = (uint32_t)mesh.dynamicData.size();

        spec.indexData = std::move(mesh.indexData);
        spec.staticData = std::move(mesh.staticData);
        spec.dynamicData = std::move(mesh.dynamicData);

        if (isIndexed)
        {
            spec.indexCount = (uint32_t)mesh.indexCount;
            spec.use16BitIndices = mesh.use16BitIndices;
        }

        if (!spec.dynamicData.empty())
        {
            spec.hasDynamicData = true;
        }

        mMeshes.push_back(spec);

        if (mMeshes.size() > std::numeric_limits<uint32_t>::max())
        {
            throw std::exception("Trying to build a scene that exceeds supported number of meshes");
        }

        return (uint32_t)(mMeshes.size() - 1);
    }

    void SceneBuilder::addCustomPrimitive(uint32_t typeID, const AABB& aabb)
    {
        uint32_t instanceIdx = 0;
        auto it = mProceduralPrimInstanceCount.find(typeID);
        if (it != mProceduralPrimInstanceCount.end())
        {
            instanceIdx = it->second++;
        }
        else
        {
            mProceduralPrimInstanceCount[typeID] = 1;
        }

        pushProceduralPrimitive(typeID, instanceIdx, (uint32_t)mCustomPrimitiveAABBs.size(), 1);
        mCustomPrimitiveAABBs.push_back(aabb);
    }

    // Curves

    uint32_t SceneBuilder::addCurve(const Curve& curve)
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
            throw std::runtime_error("Error when adding the curve '" + curve.name + "' to the scene.\nThe curve is missing " + element + ".");
        };

        auto missing_element_warning = [&](const std::string& element)
        {
            logWarning("The curve '" + curve.name + "' is missing the element " + element + ". This is not an error, the element will be filled with zeros which may result in incorrect rendering.");
        };

        if (curve.pMaterial == nullptr) throw_on_missing_element("material");

        if (curve.vertexCount == 0) throw_on_missing_element("vertices");
        if (curve.indexCount == 0) throw_on_missing_element("indices");

        if (curve.positions.pData == nullptr) throw_on_missing_element("positions");
        if (curve.radius.pData == nullptr) throw_on_missing_element("radius");
        if (curve.tangents.pData == nullptr) throw_on_missing_element("tangents");
        if (curve.normals.pData == nullptr) missing_element_warning("normals");
        if (curve.texCrds.pData == nullptr) missing_element_warning("texture coordinates");

        // Copy indices and vertices into processed curve.
        processedCurve.indexData.assign(curve.pIndices, curve.pIndices + curve.indexCount);

        processedCurve.staticData.reserve(curve.vertexCount);
        for (uint32_t i = 0; i < curve.vertexCount; i++)
        {
            StaticCurveVertexData s;
            s.position = curve.positions.pData[i];
            s.radius = curve.radius.pData[i];
            s.tangent = curve.tangents.pData[i];

            if (curve.normals.pData != nullptr)
            {
                s.normal = curve.normals.pData[i];
            }
            else
            {
                s.normal = float3(0.f);
            }

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

    uint32_t SceneBuilder::addProcessedCurve(const ProcessedCurve& curve)
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
            throw std::exception("Trying to build a scene that exceeds supported number of curves.");
        }

        return (uint32_t)(mCurves.size() - 1);
    }

    // Materials

    uint32_t SceneBuilder::addMaterial(const Material::SharedPtr& pMaterial)
    {
        assert(pMaterial);

        // Reuse previously added materials
        if (auto it = std::find(mMaterials.begin(), mMaterials.end(), pMaterial); it != mMaterials.end())
        {
            return (uint32_t)std::distance(mMaterials.begin(), it);
        }

        mMaterials.push_back(pMaterial);
        assert(mMaterials.size() <= std::numeric_limits<uint32_t>::max());
        return (uint32_t)mMaterials.size() - 1;
    }

    Material::SharedPtr SceneBuilder::getMaterial(const std::string& name) const
    {
        for (const auto& pMaterial : mMaterials)
        {
            if (pMaterial->getName() == name) return pMaterial;
        }
        return nullptr;
    }

    void SceneBuilder::loadMaterialTexture(const Material::SharedPtr& pMaterial, Material::TextureSlot slot, const std::string& filename)
    {
        if (!mpMaterialTextureLoader) mpMaterialTextureLoader.reset(new MaterialTextureLoader(!is_set(mFlags, Flags::AssumeLinearSpaceTextures)));
        mpMaterialTextureLoader->loadTexture(pMaterial, slot, filename);
    }

    // Volumes

    Volume::SharedPtr SceneBuilder::getVolume(const std::string& name) const
    {
        for (const auto& pVolume : mVolumes)
        {
            if (pVolume->getName() == name) return pVolume;
        }
        return nullptr;
    }

    uint32_t SceneBuilder::addVolume(const Volume::SharedPtr& pVolume)
    {
        assert(pVolume);

        mVolumes.push_back(pVolume);
        assert(mVolumes.size() <= std::numeric_limits<uint32_t>::max());
        return (uint32_t)mVolumes.size() - 1;
    }

    // Lights

    uint32_t SceneBuilder::addLight(const Light::SharedPtr& pLight)
    {
        assert(pLight);
        mLights.push_back(pLight);
        assert(mLights.size() <= std::numeric_limits<uint32_t>::max());
        return (uint32_t)mLights.size() - 1;
    }

    // Cameras

    uint32_t SceneBuilder::addCamera(const Camera::SharedPtr& pCamera)
    {
        assert(pCamera);
        mCameras.push_back(pCamera);
        assert(mCameras.size() <= std::numeric_limits<uint32_t>::max());
        return (uint32_t)mCameras.size() - 1;
    }

    void SceneBuilder::setSelectedCamera(const Camera::SharedPtr& pCamera)
    {
        auto it = std::find(mCameras.begin(), mCameras.end(), pCamera);
        if (it != mCameras.end()) mpSelectedCamera = pCamera;
    }

    // Animations

    void SceneBuilder::addAnimation(const Animation::SharedPtr& pAnimation)
    {
        mAnimations.push_back(pAnimation);
    }

    Animation::SharedPtr SceneBuilder::createAnimation(Animatable::SharedPtr pAnimatable, const std::string& name, double duration)
    {
        assert(pAnimatable);

        uint32_t nodeID = pAnimatable->getNodeID();

        if (nodeID != kInvalidNode && isNodeAnimated(nodeID))
        {
            logWarning("Animatable object is already animated.");
            return nullptr;
        }
        if (nodeID == kInvalidNode) nodeID = addNode(Node{ name, glm::identity<glm::mat4>(), glm::identity<glm::mat4>() });

        pAnimatable->setNodeID(nodeID);
        pAnimatable->setHasAnimation(true);
        pAnimatable->setIsAnimated(true);

        auto animation = Animation::create(name, nodeID, duration);
        addAnimation(animation);
        return animation;
    }

    // Scene graph

    uint32_t SceneBuilder::addNode(const Node& node)
    {
        assert(node.parent == kInvalidNode || node.parent < mSceneGraph.size());

        assert(mSceneGraph.size() <= std::numeric_limits<uint32_t>::max());
        uint32_t newNodeID = (uint32_t)mSceneGraph.size();
        mSceneGraph.push_back(InternalNode(node));
        if (node.parent != kInvalidNode) mSceneGraph[node.parent].children.push_back(newNodeID);
        return newNodeID;
    }

    void SceneBuilder::addMeshInstance(uint32_t nodeID, uint32_t meshID)
    {
        if (nodeID >= mSceneGraph.size()) throw std::runtime_error("SceneBuilder::addMeshInstance() - nodeID " + std::to_string(nodeID) + " is out of range");
        if (meshID >= mMeshes.size()) throw std::runtime_error("SceneBuilder::addMeshInstance() - meshID " + std::to_string(meshID) + " is out of range");

        mSceneGraph[nodeID].meshes.push_back(meshID);
        mMeshes[meshID].instances.push_back(nodeID);
    }

    void SceneBuilder::addCurveInstance(uint32_t nodeID, uint32_t curveID)
    {
        if (nodeID >= mSceneGraph.size()) throw std::runtime_error("SceneBuilder::addCurveInstance() - nodeID " + std::to_string(nodeID) + " is out of range");
        if (curveID >= mCurves.size()) throw std::runtime_error("SceneBuilder::addCurveInstance() - curveID " + std::to_string(curveID) + " is out of range");

        mSceneGraph[nodeID].curves.push_back(curveID);
        mCurves[curveID].instances.push_back(nodeID);
    }

    bool SceneBuilder::isNodeAnimated(uint32_t nodeID) const
    {
        assert(nodeID < mSceneGraph.size());

        while (nodeID != kInvalidNode)
        {
            for (const auto& pAnimation : mAnimations)
            {
                if (pAnimation->getNodeID() == nodeID) return true;
            }
            nodeID = mSceneGraph[nodeID].parent;
        }

        return false;
    }

    void SceneBuilder::setNodeInterpolationMode(uint32_t nodeID, Animation::InterpolationMode interpolationMode, bool enableWarping)
    {
        assert(nodeID < mSceneGraph.size());

        while (nodeID != kInvalidNode)
        {
            for (const auto& pAnimation : mAnimations)
            {
                if (pAnimation->getNodeID() == nodeID)
                {
                    pAnimation->setInterpolationMode(interpolationMode);
                    pAnimation->setEnableWarping(enableWarping);
                }
            }
            nodeID = mSceneGraph[nodeID].parent;
        }
    }

    // Internal

    void SceneBuilder::removeUnusedMeshes()
    {
        // If the scene contained meshes that are not referenced by the scene graph,
        // those will be removed here and warnings logged.

        size_t unusedCount = 0;
        for (uint32_t meshID = 0; meshID < (uint32_t)mMeshes.size(); meshID++)
        {
            auto& mesh = mMeshes[meshID];
            if (mesh.instances.empty())
            {
                logWarning("Mesh with ID " + std::to_string(meshID) + " named '" + mesh.name + "' is not referenced by any scene graph nodes.");
                unusedCount++;
            }
        }

        if (unusedCount > 0)
        {
            logWarning("Scene has " + std::to_string(unusedCount) + " unused meshes that will be removed.");

            const size_t meshCount = mMeshes.size();
            MeshList meshes;
            meshes.reserve(meshCount);

            for (uint32_t meshID = 0; meshID < (uint32_t)meshCount; meshID++)
            {
                auto& mesh = mMeshes[meshID];
                if (mesh.instances.empty()) continue; // Skip unused meshes

                // Get new mesh ID.
                const uint32_t newMeshID = (uint32_t)meshes.size();

                // Update scene graph nodes meshIDs.
                for (const auto nodeID : mesh.instances)
                {
                    assert(nodeID < mSceneGraph.size());
                    auto& node = mSceneGraph[nodeID];
                    std::replace(node.meshes.begin(), node.meshes.end(), meshID, newMeshID);
                }

                meshes.push_back(std::move(mesh));
            }

            mMeshes = std::move(meshes);

            // Validate scene graph.
            assert(mMeshes.size() == meshCount - unusedCount);
            for (const auto& node : mSceneGraph)
            {
                for (uint32_t meshID : node.meshes) assert(meshID < mMeshes.size());
            }
        }
    }

    void SceneBuilder::pretransformStaticMeshes()
    {
        // Add an identity transform node.
        uint32_t identityNodeID = addNode(Node{ "Identity", glm::identity<glm::mat4>(), glm::identity<glm::mat4>() });
        auto& identityNode = mSceneGraph[identityNodeID];

        size_t transformedMeshCount = 0;
        for (uint32_t meshID = 0; meshID < (uint32_t)mMeshes.size(); meshID++)
        {
            auto& mesh = mMeshes[meshID];

            // Skip instanced/animated/skinned meshes.
            assert(!mesh.instances.empty());
            if (mesh.instances.size() > 1 || isNodeAnimated(mesh.instances[0]) || mesh.hasDynamicData) continue;

            assert(mesh.dynamicData.empty() && mesh.dynamicVertexCount == 0);
            mesh.isStatic = true;

            // Compute the object->world transform for the node.
            auto nodeID = mesh.instances[0];
            assert(nodeID != kInvalidNode);

            glm::mat4 transform = glm::identity<glm::mat4>();
            while (nodeID != kInvalidNode)
            {
                assert(nodeID < mSceneGraph.size());
                transform = mSceneGraph[nodeID].transform * transform;

                nodeID = mSceneGraph[nodeID].parent;
            }

            // Flip triangle winding if the transform flips the coordinate system handedness (negative determinant).
            bool flippedWinding = glm::determinant((glm::mat3)transform) < 0.f;
            if (flippedWinding) mesh.isFrontFaceCW = !mesh.isFrontFaceCW;

            // Transform vertices to world space if not already identity transform.
            if (transform != glm::identity<glm::mat4>())
            {
                assert(!mesh.staticData.empty());
                assert((size_t)mesh.vertexCount == mesh.staticData.size());

                glm::mat3 invTranspose3x3 = (glm::mat3)glm::transpose(glm::inverse(transform));
                glm::mat3 transform3x3 = (glm::mat3)transform;

                for (auto& v : mesh.staticData)
                {
                    float4 p = transform * float4(v.position, 1.f);
                    v.position = p.xyz;
                    v.normal = glm::normalize(invTranspose3x3 * v.normal);
                    v.tangent.xyz = glm::normalize(transform3x3 * v.tangent.xyz);
                    // TODO: We should flip the sign of v.tangent.w if flippedWinding is true.
                    // Leaving that out for now for consistency with the shader code that needs the same fix.
                }

                transformedMeshCount++;
            }

            // Unlink mesh from its previous transform node.
            // TODO: This will leave some nodes unused. We could run a separate pass to compact the node list.
            assert(mesh.instances.size() == 1);
            auto& prevNode = mSceneGraph[mesh.instances[0]];
            auto it = std::find(prevNode.meshes.begin(), prevNode.meshes.end(), meshID);
            assert(it != prevNode.meshes.end());
            prevNode.meshes.erase(it);

            // Link mesh to the identity transform node.
            identityNode.meshes.push_back(meshID);
            mesh.instances[0] = identityNodeID;
        }

        logDebug("Pre-transformed " + std::to_string(transformedMeshCount) + " static meshes to world space");
    }

    void SceneBuilder::calculateMeshBoundingBoxes()
    {
        for (auto& mesh : mMeshes)
        {
            assert(!mesh.staticData.empty());
            assert((size_t)mesh.vertexCount == mesh.staticData.size());

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
        assert(mMeshGroups.empty());

        // This function sorts meshes into groups based on their properties.
        // The scene will build one BLAS per mesh group for raytracing.
        // Note that a BLAS may be referenced by multiple TLAS instances.
        //
        // The sorting criteria are:
        //  - Instanced meshes are placed in unique groups (BLASes).
        //    The TLAS instances will apply different transforms but all refer to the same BLAS.
        //  - Non-instanced dynamic meshes (skinned and/or animated) are sorted into groups with the same transform.
        //    The idea is that all parts of a dynamic object that move together go in the same BLAS and the TLAS instance applies the transform.
        //  - Non-instanced static meshes are all placed in the same group.
        //    The vertices are pre-transformed in the BLAS and the TLAS instance has an identity transform.
        //    This ensures fast traversal for the static parts of a scene independent of the scene hierarchy.
        // TODO: Add build flag to turn off pre-transformation to world space.

        // Classify non-instanced meshes.
        // The non-instanced dynamic meshes are grouped based on what global matrix ID their transform is.
        // The non-instanced static meshes are placed in the same group.
        std::unordered_map<uint32_t, std::vector<uint32_t>> nodeToMeshList;
        std::vector<uint32_t> staticMeshes;

        for (uint32_t meshID = 0; meshID < (uint32_t)mMeshes.size(); meshID++)
        {
            const auto& mesh = mMeshes[meshID];
            if (mesh.instances.size() > 1) continue; // Only processing non-instanced meshes here

            assert(mesh.instances.size() == 1);
            uint32_t nodeID = mesh.instances[0];

            if (mesh.isStatic) staticMeshes.push_back(meshID);
            else nodeToMeshList[nodeID].push_back(meshID);
        }

        // Build final result. Format is a list of Mesh ID's per mesh group.

        // All static non-instanced meshes go in a single group or individual groups depending on config.
        if (!staticMeshes.empty())
        {
            if (!is_set(mFlags, Flags::RTDontMergeStatic))
            {
                mMeshGroups.push_back({ staticMeshes, true });
            }
            else
            {
                for (const auto& meshID : staticMeshes) mMeshGroups.push_back(MeshGroup{ std::vector<uint32_t>({ meshID }), true });
            }
        }

        // Non-instanced dynamic meshes were sorted above so just copy each list.
        for (const auto& it : nodeToMeshList)
        {
            if (!is_set(mFlags, Flags::RTDontMergeDynamic))
            {
                mMeshGroups.push_back({ it.second, false });
            }
            else
            {
                for (const auto& meshID : it.second) mMeshGroups.push_back(MeshGroup{ std::vector<uint32_t>({ meshID }), false });
            }
        }

        // Instanced static and dynamic meshes always go in their own groups.
        for (uint32_t meshID = 0; meshID < (uint32_t)mMeshes.size(); meshID++)
        {
            const auto& mesh = mMeshes[meshID];
            if (mesh.instances.size() == 1) continue; // Only processing instanced meshes here
            mMeshGroups.push_back({ std::vector<uint32_t>({ meshID }), false });
        }
    }

    std::pair<std::optional<uint32_t>, std::optional<uint32_t>> SceneBuilder::splitMesh(const uint32_t meshID, const int axis, const float pos)
    {
        // Splits a mesh by an axis-aligned plane.
        // Each triangle is placed on either the left or right side of the plane with respect to its centroid.
        // Individual triangles are not split, so the resulting meshes will in general have overlapping bounding boxes.
        // If all triangles are already on either side, no split is necessary and the original mesh is retained.

        assert(meshID < mMeshes.size());
        assert(axis >= 0 && axis <= 2);
        const auto& mesh = mMeshes[meshID];

        // Check if mesh is supported.
        if (mesh.dynamicVertexCount > 0 || !mesh.dynamicData.empty())
        {
            throw std::exception(("Cannot split mesh '" + mesh.name + "', only non-dynamic meshes supported").c_str());
        }
        if (mesh.topology != Vao::Topology::TriangleList)
        {
            throw std::exception(("Cannot split mesh '" + mesh.name + "', only triangle list topology supported").c_str());
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
            assert(mesh.hasDynamicData == false);
            assert(mesh.dynamicVertexCount == 0);
            return spec;
        };

        MeshSpec leftMesh = createSpec(mesh, mesh.name + ".0");
        MeshSpec rightMesh = createSpec(mesh, mesh.name + ".1");

        if (mesh.indexCount > 0) splitIndexedMesh(mesh, leftMesh, rightMesh, axis, pos);
        else splitNonIndexedMesh(mesh, leftMesh, rightMesh, axis, pos);

        // Check that no triangles were added or removed.
        assert(leftMesh.getTriangleCount() + rightMesh.getTriangleCount() == mesh.getTriangleCount());

        // It is possible all triangles ended up on either side of the splitting plane.
        // In that case, there is no need to modify the original mesh and we'll just return.
        if (leftMesh.getTriangleCount() == 0) return { std::nullopt, meshID };
        else if (rightMesh.getTriangleCount() == 0) return { meshID, std::nullopt };

        logDebug("Mesh '" + mesh.name + "' with " + std::to_string(mesh.getTriangleCount()) + " triangles was split into two meshes with " + std::to_string(leftMesh.getTriangleCount()) + " and " + std::to_string(rightMesh.getTriangleCount()) + " triangles, respectively.");

        // Store new meshes.
        // The left mesh replaces the existing mesh.
        // The right mesh is appended at the end of the mesh list and linked to the instances.
        assert(leftMesh.vertexCount > 0 && rightMesh.vertexCount > 0);
        mMeshes[meshID] = std::move(leftMesh);

        uint32_t rightMeshID = (uint32_t)mMeshes.size();
        for (auto nodeID : mesh.instances)
        {
            mSceneGraph.at(nodeID).meshes.push_back(rightMeshID);
        }
        mMeshes.push_back(std::move(rightMesh));

        return { meshID, rightMeshID };
    }

    void SceneBuilder::splitIndexedMesh(const MeshSpec& mesh, MeshSpec& leftMesh, MeshSpec& rightMesh, const int axis, const float pos)
    {
        assert(mesh.indexCount > 0 && !mesh.indexData.empty());

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
        assert(mesh.indexCount == 0 && mesh.indexData.empty());
        throw std::exception("SceneBuilder::splitNonIndexedMesh() not implemented");
    }

    size_t SceneBuilder::countTriangles(const MeshGroup& meshGroup) const
    {
        size_t triangleCount = 0;
        for (auto meshID : meshGroup.meshList)
        {
            triangleCount += mMeshes[meshID].getTriangleCount();
        }
        return triangleCount;
    }

    AABB SceneBuilder::calculateBoundingBox(const MeshGroup& meshGroup) const
    {
        AABB bb;
        for (auto meshID : meshGroup.meshList)
        {
            bb.include(mMeshes[meshID].boundingBox);
        }
        return bb;
    }

    bool SceneBuilder::needsSplit(const MeshGroup& meshGroup, size_t& triangleCount) const
    {
        assert(!meshGroup.meshList.empty());
        triangleCount = countTriangles(meshGroup);

        if (triangleCount <= kMaxTrianglesPerBLAS)
        {
            return false;
        }
        else if (meshGroup.meshList.size() == 1)
        {
            // Issue warning if single mesh exceeds the triangle count limit.
            // TODO: Implement mesh splitting to handle this case.
            const auto& mesh = mMeshes[meshGroup.meshList[0]];
            assert(mesh.getTriangleCount() == triangleCount);
            logWarning("Mesh '" + mesh.name + "' has " + std::to_string(triangleCount) + " triangles, expect extraneous GPU memory usage.");

            return false;
        }
        assert(meshGroup.meshList.size() > 1);
        assert(triangleCount > kMaxTrianglesPerBLAS);

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
        assert(triangleCount > 0);
        size_t targetGroupCount = div_round_up(triangleCount, kMaxTrianglesPerBLAS);
        size_t targetTrianglesPerGroup = triangleCount / targetGroupCount;

        triangleCount = 0;
        MeshGroupList groups;

        for (auto meshID : meshGroup.meshList)
        {
            // Start new group on first iteration or if triangle count would exceed the target.
            size_t meshTris = mMeshes[meshID].getTriangleCount();
            if (triangleCount == 0 || triangleCount + meshTris > targetTrianglesPerGroup)
            {
                groups.push_back({ std::vector<uint32_t>(), meshGroup.isStatic });
                triangleCount = 0;
            }

            // Add mesh to group.
            groups.back().meshList.push_back(meshID);
            triangleCount += meshTris;
        }

        assert(!groups.empty());
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
        auto compareCentroids = [this, axis](uint32_t leftMeshID, uint32_t rightMeshID)
        {
            return mMeshes[leftMeshID].boundingBox.center()[axis] < mMeshes[rightMeshID].boundingBox.center()[axis];
        };

        std::vector<uint32_t> meshes = std::move(meshGroup.meshList);
        std::sort(meshes.begin(), meshes.end(), compareCentroids);

        // Find the median mesh in terms of triangle count.
        size_t triangles = 0;
        auto countTriangles = [&](uint32_t meshID)
        {
            triangles += mMeshes[meshID].getTriangleCount();
            return triangles > triangleCount / 2;
        };

        auto splitIter = std::find_if(meshes.begin(), meshes.end(), countTriangles);

        // If all meshes ended up on either side, fall back on splitting at the middle mesh.
        if (splitIter == meshes.begin() || splitIter == meshes.end())
        {
            assert(meshes.size() >= 2);
            splitIter = meshes.begin() + meshes.size() / 2;
        }
        assert(splitIter != meshes.begin() && splitIter != meshes.end());

        // Recursively split the left and right mesh groups.
        MeshGroup leftGroup{ std::vector<uint32_t>(meshes.begin(), splitIter), meshGroup.isStatic };
        MeshGroup rightGroup{ std::vector<uint32_t>(splitIter, meshes.end()), meshGroup.isStatic };
        assert(!leftGroup.meshList.empty() && !rightGroup.meshList.empty());

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
        std::vector<uint32_t> leftMeshes, rightMeshes;

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

            if (groups.size() > 1) logWarning("SceneBuilder::optimizeGeometry() performance warning - Mesh group was split into " + std::to_string(groups.size()) + " groups");

            optimizedGroups.insert(
                optimizedGroups.end(),
                std::make_move_iterator(groups.begin()),
                std::make_move_iterator(groups.end()));
        }

        mMeshGroups = std::move(optimizedGroups);
    }

    void SceneBuilder::createGlobalBuffers()
    {
        assert(mBuffersData.indexData.empty());
        assert(mBuffersData.staticData.empty());
        assert(mBuffersData.dynamicData.empty());

        const bool isIndexed = !is_set(mFlags, Flags::NonIndexedVertices);

        // Count total number of vertex and index data elements.
        size_t totalIndexDataCount = 0;
        size_t totalStaticVertexCount = 0;
        size_t totalDynamicVertexCount = 0;

        for (const auto& mesh : mMeshes)
        {
            totalIndexDataCount += mesh.indexData.size();
            totalStaticVertexCount += mesh.staticData.size();
            totalDynamicVertexCount += mesh.dynamicData.size();
        }

        // Check the range. We currently use 32-bit offsets.
        if (totalIndexDataCount > std::numeric_limits<uint32_t>::max() ||
            totalStaticVertexCount > std::numeric_limits<uint32_t>::max() ||
            totalDynamicVertexCount > std::numeric_limits<uint32_t>::max())
        {
            throw std::exception("Trying to build a scene that exceeds supported mesh data size.");
        }

        mBuffersData.indexData.reserve(totalIndexDataCount);
        mBuffersData.staticData.reserve(totalStaticVertexCount);
        mBuffersData.dynamicData.reserve(totalDynamicVertexCount);

        // Copy all vertex and index data into the global buffers.
        for (auto& mesh : mMeshes)
        {
            mesh.staticVertexOffset = (uint32_t)mBuffersData.staticData.size();
            mesh.dynamicVertexOffset = (uint32_t)mBuffersData.dynamicData.size();

            // Insert the static vertex data in the global array.
            // The vertices are automatically converted to their packed format in this step.
            mBuffersData.staticData.insert(mBuffersData.staticData.end(), mesh.staticData.begin(), mesh.staticData.end());

            if (isIndexed)
            {
                mesh.indexOffset = (uint32_t)mBuffersData.indexData.size();
                mBuffersData.indexData.insert(mBuffersData.indexData.end(), mesh.indexData.begin(), mesh.indexData.end());
            }

            if (!mesh.dynamicData.empty())
            {
                mBuffersData.dynamicData.insert(mBuffersData.dynamicData.end(), mesh.dynamicData.begin(), mesh.dynamicData.end());

                // Patch vertex index references.
                for (uint32_t i = 0; i < mesh.dynamicData.size(); ++i)
                {
                    mBuffersData.dynamicData[mesh.dynamicVertexOffset + i].staticIndex += mesh.staticVertexOffset;
                }
            }

            // Free the mesh local data.
            mesh.indexData.clear();
            mesh.staticData.clear();
            mesh.dynamicData.clear();
        }
    }

    void SceneBuilder::createCurveGlobalBuffers()
    {
        assert(mCurveBuffersData.indexData.empty());
        assert(mCurveBuffersData.staticData.empty());

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
            throw std::exception("Trying to build a scene that exceeds supported curve data size.");
        }

        mCurveBuffersData.indexData.reserve(totalIndexDataCount);
        mCurveBuffersData.staticData.reserve(totalStaticCurveVertexCount);

        // Copy all curve vertex and index data into the curve global buffers.
        for (auto& curve : mCurves)
        {
            curve.staticVertexOffset = (uint32_t)mCurveBuffersData.staticData.size();
            mCurveBuffersData.staticData.insert(mCurveBuffersData.staticData.end(), curve.staticData.begin(), curve.staticData.end());

            curve.indexOffset = (uint32_t)mCurveBuffersData.indexData.size();
            mCurveBuffersData.indexData.insert(mCurveBuffersData.indexData.end(), curve.indexData.begin(), curve.indexData.end());

            // Free the curve local data.
            curve.indexData.clear();
            curve.staticData.clear();
        }
    }

    void SceneBuilder::removeDuplicateMaterials()
    {
        if (is_set(mFlags, Flags::DontMergeMaterials)) return;

        std::vector<Material::SharedPtr> uniqueMaterials;
        std::vector<uint32_t> idMap(mMaterials.size());

        // Find unique set of materials.
        for (uint32_t id = 0; id < mMaterials.size(); ++id)
        {
            const auto& pMaterial = mMaterials[id];
            auto it = std::find_if(uniqueMaterials.begin(), uniqueMaterials.end(), [&pMaterial] (const auto& m) { return *m == *pMaterial; });
            if (it == uniqueMaterials.end())
            {
                idMap[id] = (uint32_t)uniqueMaterials.size();
                uniqueMaterials.push_back(pMaterial);
            }
            else
            {
                logInfo("Removing duplicate material '" + pMaterial->getName() + "' (duplicate of '" + (*it)->getName() + "')");
                idMap[id] = (uint32_t)std::distance(uniqueMaterials.begin(), it);
            }
        }

        // Reassign material IDs.
        for (auto& mesh : mMeshes)
        {
            mesh.materialId = idMap[mesh.materialId];
        }

        mMaterials = uniqueMaterials;
    }

    void SceneBuilder::collectVolumeGrids()
    {
        // Collect grids from volumes.
        std::set<Grid::SharedPtr> uniqueGrids;
        for (auto& volume : mVolumes)
        {
            auto grids = volume->getAllGrids();
            uniqueGrids.insert(grids.begin(), grids.end());
        }
        mGrids = GridList(uniqueGrids.begin(), uniqueGrids.end());

        // Setup grid -> id map.
        for (size_t i = 0; i < mGrids.size(); ++i)
        {
            mGridIDs.emplace(mGrids[i], (uint32_t)i);
        }
    }

    void SceneBuilder::quantizeTexCoords()
    {
        // Match texture coordinate quantization for textured emissives to format of PackedEmissiveTriangle.
        // This is to avoid mismatch when sampling and evaluating emissive triangles.
        // Note that non-emissive meshes are unmodified and use full precision texcoords.
        for (auto& mesh : mMeshes)
        {
            const auto& pMaterial = mMaterials[mesh.materialId];
            if (pMaterial->getEmissiveTexture() != nullptr)
            {
                // Quantize texture coordinates to fp16. Also track the bounds and max error.
                float2 minTexCrd = float2(std::numeric_limits<float>::infinity());
                float2 maxTexCrd = float2(-std::numeric_limits<float>::infinity());
                float2 maxError = float2(0);

                for (uint32_t i = 0; i < mesh.staticVertexCount; ++i)
                {
                    auto& v = mBuffersData.staticData[mesh.staticVertexOffset + i];
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
                    logWarning("Texture coordinates for emissive textured mesh '" + mesh.name + "' are outside the representable range, expect rendering errors.");
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
                        std::ostringstream oss;
                        oss << "Texture coordinates for emissive textured mesh '" << mesh.name << "' have a large quantization error of " << maxTexelError << " texels. "
                            << "The coordinate range is [" << minTexCrd.x << ", " << maxTexCrd.x << "] x [" << minTexCrd.y << ", " << maxTexCrd.y << "] for maximum texture dimensions ("
                            << maxTexDim.x << ", " << maxTexDim.y << ").";
                        logWarning(oss.str());
                    }
                }
            }
        }
    }

    void SceneBuilder::createMeshVao(uint32_t drawCount)
    {
        for (auto& mesh : mMeshes) assert(mesh.topology == mMeshes[0].topology);

        // Create the index buffer.
        size_t ibSize = sizeof(uint32_t) * mBuffersData.indexData.size();
        if (ibSize > std::numeric_limits<uint32_t>::max())
        {
            throw std::exception("Index buffer size exceeds 4GB");
        }

        Buffer::SharedPtr pIB = nullptr;
        if (ibSize > 0)
        {
            ResourceBindFlags ibBindFlags = Resource::BindFlags::Index | ResourceBindFlags::ShaderResource;
            pIB = Buffer::create(ibSize, ibBindFlags, Buffer::CpuAccess::None, mBuffersData.indexData.data());
        }

        // Create the vertex data structured buffer.
        const size_t vertexCount = (uint32_t)mBuffersData.staticData.size();
        size_t staticVbSize = sizeof(PackedStaticVertexData) * vertexCount;
        if (staticVbSize > std::numeric_limits<uint32_t>::max())
        {
            throw std::exception("Vertex buffer size exceeds 4GB");
        }

        ResourceBindFlags vbBindFlags = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Vertex;
        Buffer::SharedPtr pStaticBuffer = Buffer::createStructured(sizeof(PackedStaticVertexData), (uint32_t)vertexCount, vbBindFlags, Buffer::CpuAccess::None, nullptr, false);

        Vao::BufferVec pVBs(Scene::kVertexBufferCount);
        pVBs[Scene::kStaticDataBufferIndex] = pStaticBuffer;

        // Create the draw ID buffer.
        // This is only needed when rasterizing the scene.
        ResourceFormat drawIDFormat = drawCount <= (1 << 16) ? ResourceFormat::R16Uint : ResourceFormat::R32Uint;

        Buffer::SharedPtr pDrawIDBuffer = nullptr;
        if (drawIDFormat == ResourceFormat::R16Uint)
        {
            assert(drawCount <= (1 << 16));
            std::vector<uint16_t> drawIDs(drawCount);
            for (uint32_t i = 0; i < drawCount; i++) drawIDs[i] = i;
            pDrawIDBuffer = Buffer::create(drawCount * sizeof(uint16_t), ResourceBindFlags::Vertex, Buffer::CpuAccess::None, drawIDs.data());
        }
        else if (drawIDFormat == ResourceFormat::R32Uint)
        {
            std::vector<uint32_t> drawIDs(drawCount);
            for (uint32_t i = 0; i < drawCount; i++) drawIDs[i] = i;
            pDrawIDBuffer = Buffer::create(drawCount * sizeof(uint32_t), ResourceBindFlags::Vertex, Buffer::CpuAccess::None, drawIDs.data());
        }
        else should_not_get_here();

        assert(pDrawIDBuffer);
        pVBs[Scene::kDrawIdBufferIndex] = pDrawIDBuffer;

        // Create vertex layout.
        // The layout only initializes the vertex data and draw ID layout. The skinning data doesn't get passed into the vertex shader.
        VertexLayout::SharedPtr pLayout = VertexLayout::create();

        // Add the packed static vertex data layout.
        VertexBufferLayout::SharedPtr pStaticLayout = VertexBufferLayout::create();
        pStaticLayout->addElement(VERTEX_POSITION_NAME, offsetof(PackedStaticVertexData, position), ResourceFormat::RGB32Float, 1, VERTEX_POSITION_LOC);
        pStaticLayout->addElement(VERTEX_PACKED_NORMAL_TANGENT_NAME, offsetof(PackedStaticVertexData, packedNormalTangent), ResourceFormat::RGB32Float, 1, VERTEX_PACKED_NORMAL_TANGENT_LOC);
        pStaticLayout->addElement(VERTEX_TEXCOORD_NAME, offsetof(PackedStaticVertexData, texCrd), ResourceFormat::RG32Float, 1, VERTEX_TEXCOORD_LOC);
        pLayout->addBufferLayout(Scene::kStaticDataBufferIndex, pStaticLayout);

        // Add the draw ID layout.
        VertexBufferLayout::SharedPtr pInstLayout = VertexBufferLayout::create();
        pInstLayout->addElement(INSTANCE_DRAW_ID_NAME, 0, drawIDFormat, 1, INSTANCE_DRAW_ID_LOC);
        pInstLayout->setInputClass(VertexBufferLayout::InputClass::PerInstanceData, 1);
        pLayout->addBufferLayout(Scene::kDrawIdBufferIndex, pInstLayout);

        // Create the VAO objects.
        // Note that the global index buffer can be mixed 16/32-bit format.
        // For drawing the meshes we need separate VAOs for these cases.
        assert(mpScene && mpScene->mpVao == nullptr);
        mpScene->mpVao = Vao::create(mMeshes[0].topology, pLayout, pVBs, pIB, ResourceFormat::R32Uint);
        mpScene->mpVao16Bit = Vao::create(mMeshes[0].topology, pLayout, pVBs, pIB, ResourceFormat::R16Uint);
    }

    uint32_t SceneBuilder::createMeshData()
    {
        assert(mpScene->mMeshDesc.empty());
        assert(mpScene->mMeshInstanceData.empty());
        assert(mpScene->mMeshHasDynamicData.empty());
        assert(mpScene->mMeshIdToInstanceIds.empty());
        assert(mpScene->mMeshGroups.empty());

        auto& meshData = mpScene->mMeshDesc;
        auto& instanceData = mpScene->mMeshInstanceData;
        meshData.resize(mMeshes.size());
        mpScene->mMeshHasDynamicData.resize(mMeshes.size());
        size_t drawCount = 0;

        // Setup all mesh data.
        for (uint32_t meshID = 0; meshID < mMeshes.size(); meshID++)
        {
            const auto& mesh = mMeshes[meshID];
            meshData[meshID].materialID = mesh.materialId;
            meshData[meshID].vbOffset = mesh.staticVertexOffset;
            meshData[meshID].ibOffset = mesh.indexOffset;
            meshData[meshID].vertexCount = mesh.vertexCount;
            meshData[meshID].indexCount = mesh.indexCount;
            meshData[meshID].dynamicVbOffset = mesh.hasDynamicData ? mesh.dynamicVertexOffset : 0;
            assert(mesh.dynamicVertexCount == 0 || mesh.dynamicVertexCount == mesh.staticVertexCount);

            mpScene->mMeshNames.push_back(mesh.name);

            uint32_t meshFlags = 0;
            meshFlags |= mesh.use16BitIndices ? (uint32_t)MeshFlags::Use16BitIndices : 0;
            meshFlags |= mesh.hasDynamicData ? (uint32_t)MeshFlags::HasDynamicData : 0;
            meshFlags |= mesh.isFrontFaceCW ? (uint32_t)MeshFlags::IsFrontFaceCW : 0;
            meshData[meshID].flags = meshFlags;

            if (mesh.use16BitIndices) mpScene->mHas16BitIndices = true;
            else mpScene->mHas32BitIndices = true;

            if (mesh.hasDynamicData)
            {
                assert(mesh.instances.size() == 1);
                mpScene->mMeshHasDynamicData[meshID] = true;

                for (uint32_t i = 0; i < mesh.vertexCount; i++)
                {
                    mBuffersData.dynamicData[mesh.dynamicVertexOffset + i].globalMatrixID = (uint32_t)mesh.instances[0];
                }
            }
        }

        // Setup all mesh instances.
        // Mesh instances are added in the order they appear in the mesh groups.
        // For ray tracing, one BLAS per mesh group is created and the mesh instances
        // can therefore be directly indexed by [InstanceID() + GeometryIndex()].
        // This avoids the need to have a lookup table from hit IDs to mesh instance.
        for (const auto& meshGroup : mMeshGroups)
        {
            assert(!meshGroup.meshList.empty());
            for (const uint32_t meshID : meshGroup.meshList)
            {
                const auto& mesh = mMeshes[meshID];
                drawCount += mesh.instances.size();

                for (const auto& nodeID : mesh.instances)
                {
                    instanceData.push_back({});
                    auto& meshInstance = instanceData.back();
                    meshInstance.globalMatrixID = nodeID;
                    meshInstance.materialID = mesh.materialId;
                    meshInstance.meshID = meshID;
                    meshInstance.vbOffset = mesh.staticVertexOffset;
                    meshInstance.ibOffset = mesh.indexOffset;

                    uint32_t instanceFlags = 0;
                    instanceFlags |= mesh.use16BitIndices ? (uint32_t)MeshInstanceFlags::Use16BitIndices : 0;
                    instanceFlags |= mesh.hasDynamicData ? (uint32_t)MeshInstanceFlags::HasDynamicData : 0;
                    meshInstance.flags = instanceFlags;
                }
            }
        }

        // Create mapping of mesh IDs to their instanc IDs.
        mpScene->mMeshIdToInstanceIds.resize(mMeshes.size());
        for (uint32_t instanceID = 0; instanceID < (uint32_t)instanceData.size(); instanceID++)
        {
            const auto& instance = instanceData[instanceID];
            mpScene->mMeshIdToInstanceIds[instance.meshID].push_back(instanceID);
        }

        // Setup mesh groups. This just copies our final list.
        mpScene->mMeshGroups = mMeshGroups;

        assert(drawCount <= std::numeric_limits<uint32_t>::max());
        return (uint32_t)drawCount;
    }

    void SceneBuilder::createCurveVao()
    {
        // Create the index buffer.
        size_t ibSize = sizeof(uint32_t) * mCurveBuffersData.indexData.size();
        if (ibSize > std::numeric_limits<uint32_t>::max())
        {
            throw std::exception("Curve index buffer size exceeds 4GB");
        }

        Buffer::SharedPtr pIB = nullptr;
        if (ibSize > 0)
        {
            ResourceBindFlags ibBindFlags = Resource::BindFlags::Index | ResourceBindFlags::ShaderResource;
            pIB = Buffer::create(ibSize, ibBindFlags, Buffer::CpuAccess::None, mCurveBuffersData.indexData.data());
        }

        // Create the vertex data as structured buffers.
        const size_t vertexCount = (uint32_t)mCurveBuffersData.staticData.size();
        size_t staticVbSize = sizeof(StaticCurveVertexData) * vertexCount;
        if (staticVbSize > std::numeric_limits<uint32_t>::max())
        {
            throw std::exception("Curve vertex buffer exceeds 4GB");
        }

        ResourceBindFlags vbBindFlags = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Vertex;
        // Also upload the curve vertex data.
        Buffer::SharedPtr pStaticBuffer = Buffer::createStructured(sizeof(StaticCurveVertexData), (uint32_t)vertexCount, vbBindFlags, Buffer::CpuAccess::None, mCurveBuffersData.staticData.data(), false);

        // Curves do not need DrawIDBuffer.
        Vao::BufferVec pVBs(Scene::kVertexBufferCount - 1);
        pVBs[Scene::kStaticDataBufferIndex] = pStaticBuffer;

        // Create vertex layout.
        // The layout only initializes the vertex data layout. The skinning data doesn't get passed into the vertex shader.
        VertexLayout::SharedPtr pLayout = VertexLayout::create();

        // Add the packed static vertex data layout.
        VertexBufferLayout::SharedPtr pStaticLayout = VertexBufferLayout::create();
        pStaticLayout->addElement(CURVE_VERTEX_POSITION_NAME, offsetof(StaticCurveVertexData, position), ResourceFormat::RGB32Float, 1, CURVE_VERTEX_POSITION_LOC);
        pStaticLayout->addElement(CURVE_VERTEX_RADIUS_NAME, offsetof(StaticCurveVertexData, radius), ResourceFormat::R32Float, 1, CURVE_VERTEX_RADIUS_LOC);
        pStaticLayout->addElement(CURVE_VERTEX_TANGENT_NAME, offsetof(StaticCurveVertexData, tangent), ResourceFormat::RGB32Float, 1, CURVE_VERTEX_TANGENT_LOC);
        pStaticLayout->addElement(CURVE_VERTEX_NORMAL_NAME, offsetof(StaticCurveVertexData, normal), ResourceFormat::RGB32Float, 1, CURVE_VERTEX_NORMAL_LOC);
        pStaticLayout->addElement(CURVE_VERTEX_TEXCOORD_NAME, offsetof(StaticCurveVertexData, texCrd), ResourceFormat::RG32Float, 1, CURVE_VERTEX_TEXCOORD_LOC);
        pLayout->addBufferLayout(Scene::kStaticDataBufferIndex, pStaticLayout);

        // Create the VAO objects.
        assert(mpScene && mpScene->mpCurveVao == nullptr);
        mpScene->mpCurveVao = Vao::create(Vao::Topology::LineStrip, pLayout, pVBs, pIB, ResourceFormat::R32Uint);
    }

    void SceneBuilder::createCurveData()
    {
        auto& curveData = mpScene->mCurveDesc;
        auto& instanceData = mpScene->mCurveInstanceData;
        curveData.resize(mCurves.size());

        for (uint32_t curveID = 0; curveID < mCurves.size(); curveID++)
        {
            // Curve data.
            const auto& curve = mCurves[curveID];
            curveData[curveID].materialID = curve.materialId;
            curveData[curveID].degree = curve.degree;
            curveData[curveID].vbOffset = curve.staticVertexOffset;
            curveData[curveID].ibOffset = curve.indexOffset;
            curveData[curveID].vertexCount = curve.vertexCount;
            curveData[curveID].indexCount = curve.indexCount;

            // Curve instance data.
            for (const auto& instance : curve.instances)
            {
                instanceData.push_back({});
                auto& curveInstance = instanceData.back();
                curveInstance.globalMatrixID = instance;
                curveInstance.materialID = curve.materialId;
                curveInstance.curveID = curveID;
                curveInstance.vbOffset = curve.staticVertexOffset;
                curveInstance.ibOffset = curve.indexOffset;
            }
        }
    }

    void SceneBuilder::mapCurvesToProceduralPrimitives(uint32_t typeID)
    {
        // Clear any previously mapped curves.
        mProceduralPrimitives.resize(mCustomPrimitiveAABBs.size());

        uint32_t offset = (uint32_t)mCustomPrimitiveAABBs.size(); // Start curve AABBs at end of user defined AABBs.

        // Add curves to mProceduralPrimitives.
        for (uint32_t curveID = 0; curveID < mCurves.size(); curveID++)
        {
            const auto& curve = mCurves[curveID];
            assert(curve.instances.size() == 1); // Assume static curves.
            for (const auto& instID : curve.instances)
            {
                pushProceduralPrimitive(typeID, curveID, offset, curve.indexCount);
            }
            offset += curve.indexCount;
        }
    }

    void SceneBuilder::createRaytracingAABBData()
    {
        if (mProceduralPrimitives.empty()) return;

        uint32_t totalAABBCount = (uint32_t)mCustomPrimitiveAABBs.size();
        for (uint32_t i = 0; i < mCurves.size(); i++) totalAABBCount += mCurves[i].indexCount;

        mpScene->mRtAABBRaw.resize(totalAABBCount);
        uint32_t offset = 0;

        // Add all user-defined AABBs
        for (auto& aabb : mCustomPrimitiveAABBs)
        {
            D3D12_RAYTRACING_AABB& rtAabb = mpScene->mRtAABBRaw[offset++];
            rtAabb.MinX = aabb.minPoint.x;
            rtAabb.MinY = aabb.minPoint.y;
            rtAabb.MinZ = aabb.minPoint.z;
            rtAabb.MaxX = aabb.maxPoint.x;
            rtAabb.MaxY = aabb.maxPoint.y;
            rtAabb.MaxZ = aabb.maxPoint.z;
        }

        // Compute AABBs of curve segments.
        for (const auto& curve : mCurves)
        {
            const auto* indexData = &mCurveBuffersData.indexData[curve.indexOffset];
            const auto* staticData = &mCurveBuffersData.staticData[curve.staticVertexOffset];
            for (uint32_t j = 0; j < curve.indexCount; j++)
            {
                AABB curveSegBB;
                uint32_t v = indexData[j];

                for (uint32_t k = 0; k <= curve.degree; k++)
                {
                    curveSegBB.include(staticData[v + k].position - float3(staticData[v + k].radius));
                    curveSegBB.include(staticData[v + k].position + float3(staticData[v + k].radius));
                }

                D3D12_RAYTRACING_AABB& aabb = mpScene->mRtAABBRaw[offset++];
                aabb.MinX = curveSegBB.minPoint.x;
                aabb.MinY = curveSegBB.minPoint.y;
                aabb.MinZ = curveSegBB.minPoint.z;
                aabb.MaxX = curveSegBB.maxPoint.x;
                aabb.MaxY = curveSegBB.maxPoint.y;
                aabb.MaxZ = curveSegBB.maxPoint.z;
            }
        }

        // Set custom prim metadata
        mpScene->mProceduralPrimData = mProceduralPrimitives;
    }

    void SceneBuilder::createNodeList()
    {
        mpScene->mSceneGraph.resize(mSceneGraph.size());

        for (size_t i = 0; i < mSceneGraph.size(); i++)
        {
            assert(mSceneGraph[i].parent <= std::numeric_limits<uint32_t>::max());
            mpScene->mSceneGraph[i] = Scene::Node(mSceneGraph[i].name, (uint32_t)mSceneGraph[i].parent, mSceneGraph[i].transform, mSceneGraph[i].localToBindPose);
        }
    }

    void SceneBuilder::createMeshBoundingBoxes()
    {
        mpScene->mMeshBBs.resize(mMeshes.size());

        for (size_t i = 0; i < mMeshes.size(); i++)
        {
            const auto& mesh = mMeshes[i];
            mpScene->mMeshBBs[i] = mesh.boundingBox;
        }
    }

    void SceneBuilder::calculateCurveBoundingBoxes()
    {
        // Calculate curve bounding boxes.
        mpScene->mCurveBBs.resize(mCurves.size());
        for (size_t i = 0; i < mCurves.size(); i++)
        {
            const auto& curve = mCurves[i];
            AABB curveBB;

            const auto* staticData = &mCurveBuffersData.staticData[curve.staticVertexOffset];
            for (uint32_t v = 0; v < curve.vertexCount; v++)
            {
                float radius = staticData[v].radius;
                curveBB.include(staticData[v].position - float3(radius));
                curveBB.include(staticData[v].position + float3(radius));
            }

            mpScene->mCurveBBs[i] = curveBB;
        }
    }

    void SceneBuilder::pushProceduralPrimitive(uint32_t typeID, uint32_t instanceIdx, uint32_t AABBOffset, uint32_t AABBCount)
    {
        ProceduralPrimitiveData data;
        data.typeID = typeID;
        data.instanceIdx = instanceIdx;
        data.AABBOffset = AABBOffset;
        data.AABBCount = AABBCount;

        mProceduralPrimitives.push_back(data);
    }

    SCRIPT_BINDING(SceneBuilder)
    {
        SCRIPT_BINDING_DEPENDENCY(Scene)
        SCRIPT_BINDING_DEPENDENCY(TriangleMesh)
        SCRIPT_BINDING_DEPENDENCY(Material)
        SCRIPT_BINDING_DEPENDENCY(Light)
        SCRIPT_BINDING_DEPENDENCY(Transform)
        SCRIPT_BINDING_DEPENDENCY(EnvMap)
        SCRIPT_BINDING_DEPENDENCY(Animation)

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
        ScriptBindings::addEnumBinaryOperators(flags);

        pybind11::class_<SceneBuilder, SceneBuilder::SharedPtr> sceneBuilder(m, "SceneBuilder");
        sceneBuilder.def_property_readonly("flags", &SceneBuilder::getFlags);
        sceneBuilder.def_property_readonly("materials", &SceneBuilder::getMaterials);
        sceneBuilder.def_property_readonly("volumes", &SceneBuilder::getVolumes);
        sceneBuilder.def_property_readonly("lights", &SceneBuilder::getLights);
        sceneBuilder.def_property_readonly("cameras", &SceneBuilder::getCameras);
        sceneBuilder.def_property_readonly("animations", &SceneBuilder::getAnimations);
        sceneBuilder.def_property("renderSettings", pybind11::overload_cast<void>(&SceneBuilder::getRenderSettings, pybind11::const_), &SceneBuilder::setRenderSettings);
        sceneBuilder.def_property("envMap", &SceneBuilder::getEnvMap, &SceneBuilder::setEnvMap);
        sceneBuilder.def_property("selectedCamera", &SceneBuilder::getSelectedCamera, &SceneBuilder::setSelectedCamera);
        sceneBuilder.def_property("cameraSpeed", &SceneBuilder::getCameraSpeed, &SceneBuilder::setCameraSpeed);
        sceneBuilder.def("importScene", [] (SceneBuilder* pSceneBuilder, const std::string& filename, const pybind11::dict& dict, const std::vector<Transform>& instances) {
            SceneBuilder::InstanceMatrices instanceMatrices;
            for (const auto& instance : instances)
            {
                instanceMatrices.push_back(instance.getMatrix());
            }
            return pSceneBuilder->import(filename, instanceMatrices, Dictionary(dict));
        }, "filename"_a, "dict"_a = pybind11::dict(), "instances"_a = std::vector<Transform>());
        sceneBuilder.def("addTriangleMesh", &SceneBuilder::addTriangleMesh, "triangleMesh"_a, "material"_a);
        sceneBuilder.def("addMaterial", &SceneBuilder::addMaterial, "material"_a);
        sceneBuilder.def("getMaterial", &SceneBuilder::getMaterial, "name"_a);
        sceneBuilder.def("loadMaterialTexture", &SceneBuilder::loadMaterialTexture, "material"_a, "slot"_a, "filename"_a);
        sceneBuilder.def("addVolume", &SceneBuilder::addVolume, "volume"_a);
        sceneBuilder.def("getVolume", &SceneBuilder::getVolume, "name"_a);
        sceneBuilder.def("addLight", &SceneBuilder::addLight, "light"_a);
        sceneBuilder.def("addCamera", &SceneBuilder::addCamera, "camera"_a);
        sceneBuilder.def("addAnimation", &SceneBuilder::addAnimation, "animation"_a);
        sceneBuilder.def("createAnimation", &SceneBuilder::createAnimation, "animatable"_a, "name"_a, "duration"_a);
        sceneBuilder.def("addNode", [] (SceneBuilder* pSceneBuilder, const std::string& name, const Transform& transform, uint32_t parent) {
            SceneBuilder::Node node;
            node.name = name;
            node.transform = transform.getMatrix();
            node.parent = parent;
            return pSceneBuilder->addNode(node);
        }, "name"_a, "transform"_a = Transform(), "parent"_a = SceneBuilder::kInvalidNode);
        sceneBuilder.def("addMeshInstance", &SceneBuilder::addMeshInstance);
    }
}
