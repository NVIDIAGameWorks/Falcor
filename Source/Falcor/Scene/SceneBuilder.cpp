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
#include "../Externals/mikktspace/mikktspace.h"
#include <filesystem>

namespace Falcor
{
    namespace
    {
        // Texture coordinates for textured emissive materials are quantized for performance reasons.
        // We'll log a warning if the maximum quantization error exceeds this value.
        const float kMaxTexelError = 0.5f;

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
    }

    SceneBuilder::SceneBuilder(Flags flags) : mFlags(flags) {};

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

    uint32_t SceneBuilder::addNode(const Node& node)
    {
        assert(node.parent == kInvalidNode || node.parent < mSceneGraph.size());

        assert(mSceneGraph.size() <= std::numeric_limits<uint32_t>::max());
        uint32_t newNodeID = (uint32_t)mSceneGraph.size();
        mSceneGraph.push_back(InternalNode(node));
        if(node.parent != kInvalidNode) mSceneGraph[node.parent].children.push_back(newNodeID);
        mDirty = true;
        return newNodeID;
    }

    bool SceneBuilder::isNodeAnimated(uint32_t nodeID) const
    {
        assert(nodeID < mSceneGraph.size());

        while (nodeID != kInvalidNode)
        {
            for (const auto& animation : mAnimations)
            {
                if (animation->getChannel(nodeID) != Animation::kInvalidChannel) return true;
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
            for (const auto& animation : mAnimations)
            {
                if (uint32_t channelID = animation->getChannel(nodeID); channelID != Animation::kInvalidChannel)
                {
                    animation->setInterpolationMode(channelID, interpolationMode, enableWarping);
                }
            }
            nodeID = mSceneGraph[nodeID].parent;
        }
    }

    void SceneBuilder::addMeshInstance(uint32_t nodeID, uint32_t meshID)
    {
        assert(meshID < mMeshes.size());
        mSceneGraph.at(nodeID).meshes.push_back(meshID);
        mMeshes.at(meshID).instances.push_back(nodeID);
        mDirty = true;
    }

    uint32_t SceneBuilder::addMesh(const Mesh& meshDesc)
    {
        logInfo("Adding mesh with name '" + meshDesc.name + "'");

        // Copy the mesh desc so we can update it. The caller retains the ownership of the data.
        Mesh mesh = meshDesc;

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
        if (!is_set(mFlags, Flags::UseOriginalTangentSpace) || !mesh.tangents.pData)
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
            logInfo("Mesh with name '" + mesh.name + "' had original vertex count " + std::to_string(mesh.vertexCount) + ", new vertex count " + std::to_string(vertices.size()));
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

        // Match texture coordinate quantization for textured emissives to match PackedEmissiveTriangle.
        // This is to avoid mismatch when sampling and evaluating emissive triangles.
        if (mesh.pMaterial->getEmissiveTexture() != nullptr)
        {
            float2 minTexCrd = float2(std::numeric_limits<float>::infinity());
            float2 maxTexCrd = float2(-std::numeric_limits<float>::infinity());
            float2 maxError = float2(0);

            for (auto& v : vertices)
            {
                float2 texCrd = v.first.texCrd;
                minTexCrd = min(minTexCrd, texCrd);
                maxTexCrd = max(maxTexCrd, texCrd);
                v.first.texCrd = f16tof32(f32tof16(texCrd));
                maxError = max(maxError, abs(v.first.texCrd - texCrd));
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
                uint2 maxTexDim = mesh.pMaterial->getMaxTextureDimensions();
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

        // Add the mesh to the scene.
        // If the non-indexed vertices build flag is set, we will de-index the data below.
        const bool isIndexed = !is_set(mFlags, Flags::NonIndexedVertices);
        const uint32_t outputVertexCount = isIndexed ? (uint32_t)vertices.size() : mesh.indexCount;

        mMeshes.push_back({});
        MeshSpec& spec = mMeshes.back();
        assert(mBuffersData.staticData.size() <= std::numeric_limits<uint32_t>::max() && mBuffersData.dynamicData.size() <= std::numeric_limits<uint32_t>::max() && mBuffersData.indices.size() <= std::numeric_limits<uint32_t>::max());
        spec.staticVertexOffset = (uint32_t)mBuffersData.staticData.size();
        spec.dynamicVertexOffset = (uint32_t)mBuffersData.dynamicData.size();

        if (isIndexed)
        {
            spec.indexOffset = (uint32_t)mBuffersData.indices.size();
            spec.indexCount = mesh.indexCount;
        }

        spec.vertexCount = outputVertexCount;
        spec.topology = mesh.topology;
        spec.materialId = addMaterial(mesh.pMaterial, is_set(mFlags, Flags::RemoveDuplicateMaterials));

        if (mesh.hasBones())
        {
            spec.hasDynamicData = true;
        }

        // Copy indices into global index array.
        if (isIndexed)
        {
            mBuffersData.indices.insert(mBuffersData.indices.end(), indices.begin(), indices.end());
        }

        // Copy vertices into global vertex arrays.
        for (uint32_t i = 0; i < outputVertexCount; i++)
        {
            uint32_t index = isIndexed ? i : indices[i];
            assert(index < vertices.size());
            const Mesh::Vertex& v = vertices[index].first;

            StaticVertexData s;
            s.position = v.position;
            s.normal = v.normal;
            s.texCrd = v.texCrd;
            s.tangent = v.tangent;
            mBuffersData.staticData.push_back(PackedStaticVertexData(s));

            if (mesh.hasBones())
            {
                DynamicVertexData d;
                d.boneWeight = v.boneWeights;
                d.boneID = v.boneIDs;
                d.staticIndex = (uint32_t)mBuffersData.staticData.size() - 1;
                d.globalMatrixID = 0; // This will be initialized in createMeshData()
                mBuffersData.dynamicData.push_back(d);
            }
        }

        mDirty = true;

        assert(mMeshes.size() <= std::numeric_limits<uint32_t>::max());
        return (uint32_t)mMeshes.size() - 1;
    }

    uint32_t SceneBuilder::addMaterial(const Material::SharedPtr& pMaterial, bool removeDuplicate)
    {
        assert(pMaterial);

        // Reuse previously added materials
        if (auto it = std::find(mMaterials.begin(), mMaterials.end(), pMaterial); it != mMaterials.end())
        {
            return (uint32_t)std::distance(mMaterials.begin(), it);
        }

        // Try to find previously added material with equal properties (duplicate)
        if (auto it = std::find_if(mMaterials.begin(), mMaterials.end(), [&pMaterial] (const auto& m) { return *m == *pMaterial; }); it != mMaterials.end())
        {
            const auto& equalMaterial = *it;

            // ASSIMP sometimes creates internal copies of a material: Always de-duplicate if name and properties are equal.
            if (removeDuplicate || pMaterial->getName() == equalMaterial->getName())
            {
                return (uint32_t)std::distance(mMaterials.begin(), it);
            }
            else
            {
                logInfo("Material '" + pMaterial->getName() + "' is a duplicate (has equal properties) of material '" + equalMaterial->getName() + "'.");
            }
        }

        mDirty = true;
        mMaterials.push_back(pMaterial);
        assert(mMaterials.size() <= std::numeric_limits<uint32_t>::max());
        return (uint32_t)mMaterials.size() - 1;
    }

    uint32_t SceneBuilder::addCamera(const Camera::SharedPtr& pCamera)
    {
        assert(pCamera);
        mCameras.push_back(pCamera);
        mDirty = true;
        assert(mCameras.size() <= std::numeric_limits<uint32_t>::max());
        return (uint32_t)mCameras.size() - 1;
    }

    uint32_t SceneBuilder::addLight(const Light::SharedPtr& pLight)
    {
        assert(pLight);
        mLights.push_back(pLight);
        mDirty = true;
        assert(mLights.size() <= std::numeric_limits<uint32_t>::max());
        return (uint32_t)mLights.size() - 1;
    }

    void SceneBuilder::setCamera(const std::string name)
    {
        for (uint i = 0; i < mCameras.size(); i++)
        {
            if (mCameras[i]->getName() == name)
            {
                mSelectedCamera = i;
                return;
            }
        }
    }

    Vao::SharedPtr SceneBuilder::createVao(uint16_t drawCount)
    {
        for (auto& mesh : mMeshes) assert(mesh.topology == mMeshes[0].topology);
        const size_t vertexCount = (uint32_t)mBuffersData.staticData.size();
        size_t ibSize = sizeof(uint32_t) * mBuffersData.indices.size();
        size_t staticVbSize = sizeof(PackedStaticVertexData) * vertexCount;
        size_t prevVbSize = sizeof(PrevVertexData) * vertexCount;
        assert(ibSize <= std::numeric_limits<uint32_t>::max() && staticVbSize <= std::numeric_limits<uint32_t>::max() && prevVbSize <= std::numeric_limits<uint32_t>::max());

        // Create the index buffer
        Buffer::SharedPtr pIB = nullptr;
        if (ibSize > 0)
        {
            ResourceBindFlags ibBindFlags = Resource::BindFlags::Index | ResourceBindFlags::ShaderResource;
            pIB = Buffer::create((uint32_t)ibSize, ibBindFlags, Buffer::CpuAccess::None, mBuffersData.indices.data());
        }

        // Create the vertex data as structured buffers
        ResourceBindFlags vbBindFlags = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Vertex;
        Buffer::SharedPtr pStaticBuffer = Buffer::createStructured(sizeof(PackedStaticVertexData), (uint32_t)vertexCount, vbBindFlags, Buffer::CpuAccess::None, nullptr, false);
        Buffer::SharedPtr pPrevBuffer = Buffer::createStructured(sizeof(PrevVertexData), (uint32_t)vertexCount, vbBindFlags, Buffer::CpuAccess::None, nullptr, false);

        Vao::BufferVec pVBs(Scene::kVertexBufferCount);
        pVBs[Scene::kStaticDataBufferIndex] = pStaticBuffer;
        pVBs[Scene::kPrevVertexBufferIndex] = pPrevBuffer;
        std::vector<uint16_t> drawIDs(drawCount);
        for (uint32_t i = 0; i < drawCount; i++) drawIDs[i] = i;
        pVBs[Scene::kDrawIdBufferIndex] = Buffer::create(drawCount * sizeof(uint16_t), ResourceBindFlags::Vertex, Buffer::CpuAccess::None, drawIDs.data());

        // The layout only initializes the vertex data and draw ID layout. The skinning data doesn't get passed into the vertex shader.
        VertexLayout::SharedPtr pLayout = VertexLayout::create();

        // Add the packed static vertex data layout
        VertexBufferLayout::SharedPtr pStaticLayout = VertexBufferLayout::create();
        pStaticLayout->addElement(VERTEX_POSITION_NAME, offsetof(PackedStaticVertexData, position), ResourceFormat::RGB32Float, 1, VERTEX_POSITION_LOC);
        pStaticLayout->addElement(VERTEX_PACKED_NORMAL_TANGENT_NAME, offsetof(PackedStaticVertexData, packedNormalTangent), ResourceFormat::RGB32Float, 1, VERTEX_PACKED_NORMAL_TANGENT_LOC);
        pStaticLayout->addElement(VERTEX_TEXCOORD_NAME, offsetof(PackedStaticVertexData, texCrd), ResourceFormat::RG32Float, 1, VERTEX_TEXCOORD_LOC);
        pLayout->addBufferLayout(Scene::kStaticDataBufferIndex, pStaticLayout);

        // Add the previous vertex data layout
        VertexBufferLayout::SharedPtr pPrevLayout = VertexBufferLayout::create();
        pPrevLayout->addElement(VERTEX_PREV_POSITION_NAME, offsetof(PrevVertexData, position), ResourceFormat::RGB32Float, 1, VERTEX_PREV_POSITION_LOC);
        pLayout->addBufferLayout(Scene::kPrevVertexBufferIndex, pPrevLayout);

        // Add the draw ID layout
        VertexBufferLayout::SharedPtr pInstLayout = VertexBufferLayout::create();
        pInstLayout->addElement(INSTANCE_DRAW_ID_NAME, 0, ResourceFormat::R16Uint, 1, INSTANCE_DRAW_ID_LOC);
        pInstLayout->setInputClass(VertexBufferLayout::InputClass::PerInstanceData, 1);
        pLayout->addBufferLayout(Scene::kDrawIdBufferIndex, pInstLayout);

        Vao::SharedPtr pVao = Vao::create(mMeshes[0].topology, pLayout, pVBs, pIB, ResourceFormat::R32Uint);
        return pVao;
    }

    void SceneBuilder::createGlobalMatricesBuffer(Scene* pScene)
    {
        pScene->mSceneGraph.resize(mSceneGraph.size());

        for (size_t i = 0; i < mSceneGraph.size(); i++)
        {
            assert(mSceneGraph[i].parent <= std::numeric_limits<uint32_t>::max());
            pScene->mSceneGraph[i] = Scene::Node(mSceneGraph[i].name, (uint32_t)mSceneGraph[i].parent, mSceneGraph[i].transform, mSceneGraph[i].localToBindPose);
        }
    }

    uint32_t SceneBuilder::createMeshData(Scene* pScene)
    {
        auto& meshData = pScene->mMeshDesc;
        auto& instanceData = pScene->mMeshInstanceData;
        meshData.resize(mMeshes.size());
        pScene->mMeshHasDynamicData.resize(mMeshes.size());

        size_t drawCount = 0;
        for (uint32_t meshID = 0; meshID < mMeshes.size(); meshID++)
        {
            // Mesh data
            const auto& mesh = mMeshes[meshID];
            meshData[meshID].materialID = mesh.materialId;
            meshData[meshID].vbOffset = mesh.staticVertexOffset;
            meshData[meshID].ibOffset = mesh.indexOffset;
            meshData[meshID].vertexCount = mesh.vertexCount;
            meshData[meshID].indexCount = mesh.indexCount;

            drawCount += mesh.instances.size();

            // Mesh instance data
            for (const auto& instance : mesh.instances)
            {
                instanceData.push_back({});
                auto& meshInstance = instanceData.back();
                meshInstance.globalMatrixID = instance;
                meshInstance.materialID = mesh.materialId;
                meshInstance.meshID = meshID;
                meshInstance.vbOffset = mesh.staticVertexOffset;
                meshInstance.ibOffset = mesh.indexOffset;
            }

            if (mesh.hasDynamicData)
            {
                assert(mesh.instances.size() == 1);
                pScene->mMeshHasDynamicData[meshID] = true;

                for (uint32_t i = 0; i < mesh.vertexCount; i++)
                {
                    mBuffersData.dynamicData[mesh.dynamicVertexOffset + i].globalMatrixID = (uint32_t)mesh.instances[0];
                }
            }
        }
        assert(drawCount <= std::numeric_limits<uint32_t>::max());
        return (uint32_t)drawCount;
    }

    Scene::SharedPtr SceneBuilder::getScene()
    {
        // We cache the scene because creating it is not cheap.
        // With the PythonImporter, the scene is fetched twice, once for running
        // the scene script and another time when the scene has finished loading.
        if (mpScene && !mDirty)
        {
            // PythonImporter sets the filename after loading the nested scene,
            // so we need to set it to the correct value here.
            mpScene->mFilename = mFilename;
            return mpScene;
        }

        if (mMeshes.size() == 0)
        {
            logError("Can't build scene. No meshes were loaded");
            return nullptr;
        }

        TimeReport timeReport;

        mpScene = Scene::create();
        mpScene->mCameras = mCameras;
        mpScene->mSelectedCamera = mSelectedCamera;
        mpScene->mCameraSpeed = mCameraSpeed;
        mpScene->mLights = mLights;
        mpScene->mMaterials = mMaterials;
        mpScene->mpLightProbe = mpLightProbe;
        mpScene->mpEnvMap = mpEnvMap;
        mpScene->mFilename = mFilename;

        createGlobalMatricesBuffer(mpScene.get());
        uint32_t drawCount = createMeshData(mpScene.get());
        assert(drawCount <= std::numeric_limits<uint16_t>::max());
        mpScene->mpVao = createVao(drawCount);
        calculateMeshBoundingBoxes(mpScene.get());
        createAnimationController(mpScene.get());
        mpScene->finalize();
        mDirty = false;

        timeReport.measure("Creating resources");
        timeReport.printToLog();

        return mpScene;
    }

    void SceneBuilder::calculateMeshBoundingBoxes(Scene* pScene)
    {
        // Calculate mesh bounding boxes
        pScene->mMeshBBs.resize(mMeshes.size());
        for (size_t i = 0; i < mMeshes.size(); i++)
        {
            const auto& mesh = mMeshes[i];
            float3 boxMin(FLT_MAX);
            float3 boxMax(-FLT_MAX);

            const auto* staticData = &mBuffersData.staticData[mesh.staticVertexOffset];
            for (uint32_t v = 0; v < mesh.vertexCount; v++)
            {
                boxMin = glm::min(boxMin, staticData[v].position);
                boxMax = glm::max(boxMax, staticData[v].position);
            }

            pScene->mMeshBBs[i] = BoundingBox::fromMinMax(boxMin, boxMax);
        }
    }

    void SceneBuilder::addAnimation(const Animation::SharedPtr& pAnimation)
    {
        mAnimations.push_back(pAnimation);
        mDirty = true;
    }

    void SceneBuilder::createAnimationController(Scene* pScene)
    {
        pScene->mpAnimationController = AnimationController::create(pScene, mBuffersData.staticData, mBuffersData.dynamicData);
        for (const auto& pAnim : mAnimations)
        {
            pScene->mpAnimationController->addAnimation(pAnim);
        }
    }

    SCRIPT_BINDING(SceneBuilder)
    {
        pybind11::enum_<SceneBuilder::Flags> flags(m, "SceneBuilderFlags");
        flags.value("Default", SceneBuilder::Flags::Default);
        flags.value("RemoveDuplicateMaterials", SceneBuilder::Flags::RemoveDuplicateMaterials);
        flags.value("UseOriginalTangentSpace", SceneBuilder::Flags::UseOriginalTangentSpace);
        flags.value("AssumeLinearSpaceTextures", SceneBuilder::Flags::AssumeLinearSpaceTextures);
        flags.value("DontMergeMeshes", SceneBuilder::Flags::DontMergeMeshes);
        flags.value("BuffersAsShaderResource", SceneBuilder::Flags::BuffersAsShaderResource);
        flags.value("UseSpecGlossMaterials", SceneBuilder::Flags::UseSpecGlossMaterials);
        flags.value("UseMetalRoughMaterials", SceneBuilder::Flags::UseMetalRoughMaterials);
        flags.value("NonIndexedVertices", SceneBuilder::Flags::NonIndexedVertices);
        ScriptBindings::addEnumBinaryOperators(flags);
    }
}
