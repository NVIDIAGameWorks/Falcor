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
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
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
#include "../Externals/mikktspace/mikktspace.h"
#include <filesystem>

namespace Falcor
{
    namespace
    {
        class MikkTSpaceWrapper
        {
        public:
            static std::vector<float3> generateBitangents(const float3* pPositions, const float3* pNormals, const float2* pTexCrd, const uint32_t* pIndices, size_t vertexCount, size_t indexCount)
            {
                if (!pNormals || !pPositions || !pTexCrd || !pIndices)
                {
                    logWarning("Can't generate tangent space. The mesh doesn't have positions/normals/texCrd/indices");
                    return std::vector<float3>(vertexCount, float3(0, 0, 0));
                }

                SMikkTSpaceInterface mikktspace = {};
                mikktspace.m_getNumFaces = [](const SMikkTSpaceContext* pContext) {return ((MikkTSpaceWrapper*)(pContext->m_pUserData))->getFaceCount(); };
                mikktspace.m_getNumVerticesOfFace = [](const SMikkTSpaceContext * pContext, int32_t face) {return 3; };
                mikktspace.m_getPosition = [](const SMikkTSpaceContext * pContext, float position[], int32_t face, int32_t vert) {((MikkTSpaceWrapper*)(pContext->m_pUserData))->getPosition(position, face, vert); };
                mikktspace.m_getNormal = [](const SMikkTSpaceContext * pContext, float normal[], int32_t face, int32_t vert) {((MikkTSpaceWrapper*)(pContext->m_pUserData))->getNormal(normal, face, vert); };
                mikktspace.m_getTexCoord = [](const SMikkTSpaceContext * pContext, float texCrd[], int32_t face, int32_t vert) {((MikkTSpaceWrapper*)(pContext->m_pUserData))->getTexCrd(texCrd, face, vert); };
                mikktspace.m_setTSpaceBasic = [](const SMikkTSpaceContext * pContext, const float tangent[], float sign, int32_t face, int32_t vert) {((MikkTSpaceWrapper*)(pContext->m_pUserData))->setTangent(tangent, sign, face, vert); };

                MikkTSpaceWrapper wrapper(pPositions, pNormals, pTexCrd, pIndices, vertexCount, indexCount);
                SMikkTSpaceContext context = {};
                context.m_pInterface = &mikktspace;
                context.m_pUserData = &wrapper;

                if (genTangSpaceDefault(&context) == false)
                {
                    logError("Failed to generate MikkTSpace tangents");
                    return std::vector<float3>(vertexCount, float3(0, 0, 0));
                }

                return wrapper.mBitangents;
            }

        private:
            MikkTSpaceWrapper(const float3* pPositions, const float3* pNormals, const float2* pTexCrd, const uint32_t* pIndices, size_t vertexCount, size_t indexCount) :
                mpPositions(pPositions), mpNormals(pNormals), mpTexCrd(pTexCrd), mpIndices(pIndices), mFaceCount(indexCount / 3), mBitangents(vertexCount) {}
            const float3* mpPositions;
            const float3* mpNormals;
            const float2* mpTexCrd;
            const uint32_t* mpIndices;
            size_t mFaceCount;
            std::vector<float3> mBitangents;
            int32_t getFaceCount() const { return (int32_t)mFaceCount; }
            int32_t getIndex(int32_t face, int32_t vert) { return mpIndices[face * 3 + vert]; }
            void getPosition(float position[], int32_t face, int32_t vert) { *(float3*)position = mpPositions[getIndex(face, vert)]; }
            void getNormal(float normal[], int32_t face, int32_t vert) { *(float3*)normal = mpNormals[getIndex(face, vert)]; }
            void getTexCrd(float texCrd[], int32_t face, int32_t vert) { *(float2*)texCrd = mpTexCrd[getIndex(face, vert)]; }

            void setTangent(const float tangent[], float sign, int32_t face, int32_t vert)
            {
                int32_t index = getIndex(face, vert);
                float3 T(*(float3*)tangent), N;
                getNormal(&N[0], face, vert);
                // bitangent = fSign * cross(vN, tangent);
                mBitangents[index] = cross(N, T); // Not using fSign because... I don't know why. It flips the tangent space. Need to go read the paper
            }
        };

        void validateTangentSpace(const float3 bitangents[], uint32_t vertexCount)
        {
            auto isValid = [](const float3& bitangent)
            {
                if (glm::any(glm::isinf(bitangent) || glm::isnan(bitangent))) return false;
                if (length(bitangent) < 1e-6f) return false;
                return true;
            };

            uint32_t numInvalid = 0;
            for (uint32_t i = 0; i < vertexCount; i++)
            {
                if (!isValid(bitangents[i])) numInvalid++;
            }

            if (numInvalid > 0)
            {
                logWarning("Loaded tangent space is invalid at " + std::to_string(numInvalid) + " vertices. Please fix the asset.");
            }
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

    bool SceneBuilder::import(const std::string& filename, const InstanceMatrices& instances)
    {
        bool success = false;
        if (std::filesystem::path(filename).extension() == ".py")
        {
            success = PythonImporter::import(filename, *this);
        }
        else if (std::filesystem::path(filename).extension() == ".fscene")
        {
            success = SceneImporter::import(filename, *this);
        }
        else
        {
            success = AssimpImporter::import(filename, *this, instances);
        }
        mFilename = filename;
        return success;
    }

    uint32_t SceneBuilder::addNode(const Node& node)
    {
        assert(node.parent == kInvalidNode || node.parent < mSceneGraph.size());

        assert(mSceneGraph.size() <= UINT32_MAX);
        uint32_t newNodeID = (uint32_t)mSceneGraph.size();
        mSceneGraph.push_back(InternalNode(node));
        if(node.parent != kInvalidNode) mSceneGraph[node.parent].children.push_back(newNodeID);
        mDirty = true;
        return newNodeID;
    }

    void SceneBuilder::addMeshInstance(uint32_t nodeID, uint32_t meshID)
    {
        assert(meshID < mMeshes.size());
        mSceneGraph.at(nodeID).meshes.push_back(meshID);
        mMeshes.at(meshID).instances.push_back(nodeID);
        mDirty = true;
    }

    uint32_t SceneBuilder::addMesh(const Mesh& mesh)
    {
        const auto& prevMesh = mMeshes.size() ? mMeshes.back() : MeshSpec();

        // Create the new mesh spec
        mMeshes.push_back({});
        MeshSpec& spec = mMeshes.back();
        assert(mBuffersData.staticData.size() <= UINT32_MAX && mBuffersData.dynamicData.size() <= UINT32_MAX && mBuffersData.indices.size() <= UINT32_MAX);
        spec.staticVertexOffset = (uint32_t)mBuffersData.staticData.size();
        spec.dynamicVertexOffset = (uint32_t)mBuffersData.dynamicData.size();
        spec.indexOffset = (uint32_t)mBuffersData.indices.size();
        spec.indexCount = mesh.indexCount;
        spec.vertexCount = mesh.vertexCount;
        spec.topology = mesh.topology;
        spec.materialId = addMaterial(mesh.pMaterial, is_set(mFlags, Flags::RemoveDuplicateMaterials));

        // Error checking
        auto throw_on_missing_element = [&](const std::string& element)
        {
            throw std::runtime_error("Error when adding the mesh " + mesh.name + " to the scene.\nThe mesh is missing " + element);
        };

        auto missing_element_warning = [&](const std::string& element)
        {
            logWarning("The mesh " + mesh.name + " is missing the element " + element + ". This is not an error, the element will be filled with zeros which may result in incorrect rendering");
        };

        // Initialize the static data
        if (mesh.indexCount == 0 || !mesh.pIndices) throw_on_missing_element("indices");
        mBuffersData.indices.insert(mBuffersData.indices.end(), mesh.pIndices, mesh.pIndices + mesh.indexCount);

        if (mesh.vertexCount == 0) throw_on_missing_element("vertices");
        if (mesh.pPositions == nullptr) throw_on_missing_element("positions");
        if (mesh.pNormals == nullptr) missing_element_warning("normals");
        if (mesh.pTexCrd == nullptr) missing_element_warning("texture coordinates");

        // Initialize the dynamic data
        if (mesh.pBoneWeights || mesh.pBoneIDs)
        {
            if (mesh.pBoneIDs == nullptr) throw_on_missing_element("bone IDs");
            if (mesh.pBoneWeights == nullptr) throw_on_missing_element("bone weights");
            spec.hasDynamicData = true;
        }

        // Generate tangent space if that's required
        std::vector<float3> bitangents;
        if (!is_set(mFlags, Flags::UseOriginalTangentSpace) || !mesh.pBitangents)
        {
            bitangents = MikkTSpaceWrapper::generateBitangents(mesh.pPositions, mesh.pNormals, mesh.pTexCrd, mesh.pIndices, mesh.vertexCount, mesh.indexCount);
        }
        else
        {
            validateTangentSpace(mesh.pBitangents, mesh.vertexCount);
        }

        for (uint32_t v = 0; v < mesh.vertexCount; v++)
        {
            StaticVertexData s;
            s.position = mesh.pPositions[v];
            s.normal = mesh.pNormals ? mesh.pNormals[v] : float3(0, 0, 0);
            s.texCrd = mesh.pTexCrd ? mesh.pTexCrd[v] : float2(0, 0);
            s.bitangent = bitangents.size() ? bitangents[v] : mesh.pBitangents[v];
            mBuffersData.staticData.push_back(PackedStaticVertexData(s));

            if (mesh.pBoneWeights)
            {
                DynamicVertexData d;
                d.boneWeight = mesh.pBoneWeights[v];
                d.boneID = mesh.pBoneIDs[v];
                d.staticIndex = (uint32_t)mBuffersData.staticData.size() - 1;
                mBuffersData.dynamicData.push_back(d);
            }
        }

        mDirty = true;

        assert(mMeshes.size() <= UINT32_MAX);
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
                logWarning("Material '" + pMaterial->getName() + "' is a duplicate (has equal properties) of material '" + equalMaterial->getName() + "'.");
            }
        }

        mDirty = true;
        mMaterials.push_back(pMaterial);
        assert(mMaterials.size() <= UINT32_MAX);
        return (uint32_t)mMaterials.size() - 1;
    }

    void SceneBuilder::setCamera(const Camera::SharedPtr& pCamera, uint32_t nodeID)
    {
        mCamera.nodeID = nodeID;
        mCamera.pObject = pCamera;
        mDirty = true;
    }

    uint32_t SceneBuilder::addLight(const Light::SharedPtr& pLight, uint32_t nodeID)
    {
        Scene::AnimatedObject<Light> light;
        light.pObject = pLight;
        light.nodeID = nodeID;
        mLights.push_back(light);
        mDirty = true;
        assert(mLights.size() <= UINT32_MAX);
        return (uint32_t)mLights.size() - 1;
    }

    Vao::SharedPtr SceneBuilder::createVao(uint16_t drawCount)
    {
        for (auto& mesh : mMeshes) assert(mesh.topology == mMeshes[0].topology);
        const size_t vertexCount = (uint32_t)mBuffersData.staticData.size();
        size_t ibSize = sizeof(uint32_t) * mBuffersData.indices.size();
        size_t staticVbSize = sizeof(PackedStaticVertexData) * vertexCount;
        size_t prevVbSize = sizeof(PrevVertexData) * vertexCount;
        assert(ibSize <= UINT32_MAX && staticVbSize <= UINT32_MAX && prevVbSize <= UINT32_MAX);

        // Create the index buffer
        ResourceBindFlags ibBindFlags = Resource::BindFlags::Index | ResourceBindFlags::ShaderResource;
        Buffer::SharedPtr pIB = Buffer::create((uint32_t)ibSize, ibBindFlags, Buffer::CpuAccess::None, mBuffersData.indices.data());

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
        pStaticLayout->addElement(VERTEX_PACKED_NORMAL_BITANGENT_NAME, offsetof(PackedStaticVertexData, packedNormalBitangent), ResourceFormat::RGB32Float, 1, VERTEX_PACKED_NORMAL_BITANGENT_LOC);
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
            assert(mSceneGraph[i].parent <= UINT32_MAX);
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
        assert(drawCount <= UINT32_MAX);
        return (uint32_t)drawCount;
    }

    Scene::SharedPtr SceneBuilder::getScene()
    {
        // We cache the scene because creating it is not cheap.
        // With the PythonImporter, the scene is fetched twice, once for running
        // the scene script and another time when the scene has finished loading.
        if (mpScene && !mDirty) return mpScene;

        if (mMeshes.size() == 0)
        {
            logError("Can't build scene. No meshes were loaded");
            return nullptr;
        }
        mpScene = Scene::create();
        if (mCamera.pObject == nullptr) mCamera.pObject = Camera::create();
        mpScene->mCamera = mCamera;
        mpScene->mCameraSpeed = mCameraSpeed;
        mpScene->mLights = mLights;
        mpScene->mMaterials = mMaterials;
        mpScene->mpLightProbe = mpLightProbe;
        mpScene->mpEnvMap = mpEnvMap;
        mpScene->mFilename = mFilename;

        createGlobalMatricesBuffer(mpScene.get());
        uint32_t drawCount = createMeshData(mpScene.get());
        assert(drawCount <= UINT16_MAX);
        mpScene->mpVao = createVao(drawCount);
        calculateMeshBoundingBoxes(mpScene.get());
        createAnimationController(mpScene.get());
        mpScene->finalize();
        mDirty = false;

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

    uint32_t SceneBuilder::addAnimation(uint32_t meshID, Animation::ConstSharedPtrRef pAnimation)
    {
        assert(meshID < mMeshes.size());
        mMeshes[meshID].animations.push_back(pAnimation);
        mDirty = true;
        assert(mMeshes[meshID].animations.size() <= UINT32_MAX);
        return (uint32_t)mMeshes[meshID].animations.size() - 1;
    }

    void SceneBuilder::createAnimationController(Scene* pScene)
    {
        pScene->mpAnimationController = AnimationController::create(pScene, mBuffersData.staticData, mBuffersData.dynamicData);
        for (uint32_t i = 0; i < mMeshes.size(); i++)
        {
            for (const auto& pAnim : mMeshes[i].animations)
            {
                pScene->mpAnimationController->addAnimation(i, pAnim);
            }
        }
    }

    SCRIPT_BINDING(SceneBuilder)
    {
        auto buildFlags = m.enum_<SceneBuilder::Flags>("SceneBuilderFlags");
        buildFlags.regEnumVal(SceneBuilder::Flags::None);
        buildFlags.regEnumVal(SceneBuilder::Flags::RemoveDuplicateMaterials);
        buildFlags.regEnumVal(SceneBuilder::Flags::UseOriginalTangentSpace);
        buildFlags.regEnumVal(SceneBuilder::Flags::AssumeLinearSpaceTextures);
        buildFlags.regEnumVal(SceneBuilder::Flags::DontMergeMeshes);
        buildFlags.regEnumVal(SceneBuilder::Flags::BuffersAsShaderResource);
        buildFlags.regEnumVal(SceneBuilder::Flags::UseSpecGlossMaterials);
        buildFlags.regEnumVal(SceneBuilder::Flags::UseMetalRoughMaterials);
        buildFlags.addBinaryOperators();
    }
}
