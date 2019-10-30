/***************************************************************************
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
***************************************************************************/
#include "stdafx.h"
#include "Scene.h"
#include "Raytracing/RtState.h"
#include "Raytracing/RtProgramVars.h"

namespace Falcor
{
    namespace
    {
        // Checks if the transform flips the coordinate system handedness (its determinant is negative).
        bool doesTransformFlip(const mat4& m)
        {
            return determinant((mat3)m) < 0.f;
        }

        const std::string kParameterBlockName = "gScene";
        const std::string kMeshBufferName = "meshes";
        const std::string kMeshInstanceBufferName = "meshInstances";
        const std::string kIndexBufferName = "indices";
        const std::string kVertexBufferName = "vertices";
        const std::string kLightsBufferName = "lights";
        const std::string kCameraVarName = "camera";
    }

    const FileDialogFilterVec Scene::kFileExtensionFilters =
    {
        {"fscene"},
        {"fbx"},
        {"gltf"},
        {"obj"},
        {"dae"},
        {"x"},
        {"md5mesh"},
        {"ply"},
        {"3ds"},
        {"blend"},
        {"ase"},
        {"ifc"},
        {"xgl"},
        {"zgl"},
        {"dxf"},
        {"lwo"},
        {"lws"},
        {"lxo"},
        {"stl"},
        {"x"},
        {"ac"},
        {"ms3d"},
        {"cob"},
        {"scn"},
        {"3d"},
        {"mdl"},
        {"mdl2"},
        {"pk3"},
        {"smd"},
        {"vta"},
        {"raw"},
        {"ter"},
        {"glb"}
    };

    Scene::Scene()
    {
        mpFrontClockwiseRS = RasterizerState::create(RasterizerState::Desc().setFrontCounterCW(false));
        mpNoCullRS = RasterizerState::create(RasterizerState::Desc().setCullMode(RasterizerState::CullMode::None));
    }

    Scene::SharedPtr Scene::create(const std::string& filename)
    {
        auto pBuilder = SceneBuilder::create(filename);
        return pBuilder ? pBuilder->getScene() : nullptr;
    }

    Scene::SharedPtr Scene::create()
    {
        return Scene::SharedPtr(new Scene());
    }

    Shader::DefineList Scene::getSceneDefines()
    {
        Shader::DefineList defines;
        defines.add("MATERIAL_COUNT", std::to_string(mMaterials.size()));
        return defines;
    }

    void Scene::render(RenderContext* pContext, GraphicsState* pState, GraphicsVars* pVars, RenderFlags flags)
    {
        PROFILE("renderScene");

        pState->setVao(mpVao);
        pVars->setParameterBlock("gScene", mpSceneBlock);

        bool overrideRS = !is_set(flags, RenderFlags::UserRasterizerState);
        auto pCurrentRS = pState->getRasterizerState();

        if (mDrawCounterClockwiseMeshes.count)
        {
            if (overrideRS) pState->setRasterizerState(nullptr);
            pContext->drawIndexedIndirect(pState, pVars, mDrawCounterClockwiseMeshes.count, mDrawCounterClockwiseMeshes.pBuffer.get(), 0, nullptr, 0);
        }

        if (mDrawClockwiseMeshes.count)
        {
            if (overrideRS) pState->setRasterizerState(mpFrontClockwiseRS);
            pContext->drawIndexedIndirect(pState, pVars, mDrawClockwiseMeshes.count, mDrawClockwiseMeshes.pBuffer.get(), 0, nullptr, 0);
        }

        if (mDrawAlphaTestedMeshes.count)
        {
            if (overrideRS) pState->setRasterizerState(mpNoCullRS);
            pContext->drawIndexedIndirect(pState, pVars, mDrawAlphaTestedMeshes.count, mDrawAlphaTestedMeshes.pBuffer.get(), 0, nullptr, 0);
        }

        if (overrideRS) pState->setRasterizerState(pCurrentRS);
    }

    void Scene::raytrace(RenderContext* pContext, const std::shared_ptr<RtState>& pState, const std::shared_ptr<RtProgramVars>& pRtVars, uvec3 dispatchDims)
    {
        // On first execution, create BLAS for each mesh
        if (mBlasData.empty())
        {
            initGeomDesc();
            buildBlas(pContext);
            updateAsToInstanceDataMapping();
        }

        // If not set yet, set geometry indices for this RtProgramVars
        if (mRtVarsWithGeometryIndex.count(pRtVars.get()) == 0)
        {
            setGeometryIndexIntoRtVars(pRtVars);
            mRtVarsWithGeometryIndex.insert(pRtVars.get());
        }

        // On first execution, when meshes have moved, when there's a new ray count, or when a BLAS has changed, create/update the TLAS
        auto tlasIt = mTlasCache.find(pRtVars->getHitProgramsCount());
        if (tlasIt == mTlasCache.end())
        {
            // We need a hit entry per mesh right now to pass GeometryIndex()
            assert(pRtVars->hasPerMeshHitEntry());
            buildTlas(pContext, pRtVars->getHitProgramsCount(), true);

            // If new TLAS was just created, get it so the iterator is valid
            if (tlasIt == mTlasCache.end()) tlasIt = mTlasCache.find(pRtVars->getHitProgramsCount());
        }

        {
            PROFILE("applyRtVars");
            // Bind Scene Param Block
            const GraphicsVars::SharedPtr& pGlobalVars = pRtVars->getGlobalVars();
            mCamera.pObject->setIntoConstantBuffer(mpSceneBlock->getDefaultConstantBuffer().get(), kCameraVarName);
            pGlobalVars->setParameterBlock("gScene", mpSceneBlock);

            // Bind TLAS
            ParameterBlockReflection::BindLocation loc = pGlobalVars->getReflection()->getDefaultParameterBlock()->getResourceBinding("gRtScene");
            if (loc.setIndex != ProgramReflection::kInvalidLocation)
            {
                pGlobalVars->getDefaultBlock()->setSrv(loc, 0, tlasIt->second.pSrv);
            }

            // Nothing to set? Materials, etc is global in the param block already
            // Set miss-shader data
            pGlobalVars["DxrPerFrame"]["hitProgramCount"] = pRtVars->getHitProgramsCount();
            pGlobalVars->setRawBuffer("gAsToInstance", mpAsToInstanceMapping);

            if (!pRtVars->apply(pContext, pState->getRtso().get()))
            {
                logError("Scene::raytrace() - applying RtProgramVars failed, most likely because we ran out of descriptors.");
                assert(false);

            }
        }

        PROFILE("raytrace");
        pContext->raytrace(pRtVars, pState, dispatchDims.x, dispatchDims.y, dispatchDims.z);
    }

    void Scene::initResources()
    {
        GraphicsProgram::SharedPtr pProgram = GraphicsProgram::createFromFile("Framework/Shaders/SceneBlock.slang", "", "main");
        pProgram->addDefines(getSceneDefines());
        ParameterBlockReflection::SharedConstPtr pReflection = pProgram->getReflector()->getParameterBlock(kParameterBlockName);
        assert(pReflection);

        mpSceneBlock = ParameterBlock::create(pReflection, true);

        ReflectionVar::SharedConstPtr pMeshRefl = pReflection->getResource(kMeshBufferName);
        mpMeshesBuffer = StructuredBuffer::create(pMeshRefl->getName(), std::dynamic_pointer_cast<const ReflectionResourceType>(pMeshRefl->getType()), mMeshDesc.size(), Resource::BindFlags::ShaderResource);

        ReflectionVar::SharedConstPtr pInstRefl = pReflection->getResource(kMeshInstanceBufferName);
        mpMeshInstancesBuffer = StructuredBuffer::create(pInstRefl->getName(), std::dynamic_pointer_cast<const ReflectionResourceType>(pInstRefl->getType()), mMeshInstanceData.size(), Resource::BindFlags::ShaderResource);

        if(mLights.size())
        {
            ReflectionVar::SharedConstPtr pLightsRefl = pReflection->getResource(kLightsBufferName);
            mpLightsBuffer = StructuredBuffer::create(pLightsRefl->getName(), std::dynamic_pointer_cast<const ReflectionResourceType>(pLightsRefl->getType()), mLights.size(), Resource::BindFlags::ShaderResource);
        }
    }

    void Scene::uploadResources()
    {
        // Upload geometry
        checkOffsets();
        mpMeshesBuffer->setBlob(mMeshDesc.data(), 0, sizeof(MeshDesc) * mMeshDesc.size());
        mpMeshInstancesBuffer->setBlob(mMeshInstanceData.data(), 0, sizeof(MeshInstanceData) * mMeshInstanceData.size());

        mpSceneBlock->setStructuredBuffer(kMeshInstanceBufferName, mpMeshInstancesBuffer);
        mpSceneBlock->setStructuredBuffer(kMeshBufferName, mpMeshesBuffer);
        mpSceneBlock->setStructuredBuffer(kLightsBufferName, mpLightsBuffer);
        mpSceneBlock->setRawBuffer(kIndexBufferName, mpVao->getIndexBuffer());
        mpSceneBlock->setStructuredBuffer(kVertexBufferName, mpVao->getVertexBuffer(Scene::kStaticDataBufferIndex)->asStructuredBuffer());

        // Set material data
        for (uint32_t i = 0; i < (uint32_t)mMaterials.size(); i++)
        {
            mMaterials[i]->setIntoParameterBlock(mpSceneBlock, "materials[" + std::to_string(i) + "]");
        }

        if (mpLightProbe)
        {
            LightProbe::setSharedIntoParameterBlock(mpSceneBlock, "probeShared");
            mpLightProbe->setIntoParameterBlock(mpSceneBlock, "lightProbe");
        }
    }

    void Scene::checkOffsets()
    {
        // Reflector, Struct type, Member
#define assert_offset(_r, _s, _m) assert(_r->getOffsetDesc(offsetof(_s, _m)).type != ReflectionBasicType::Type::Unknown)

        // MeshDesc
        auto pMeshReflector = mpMeshesBuffer->getBufferReflector();
        assert_offset(pMeshReflector, MeshDesc, vbOffset);
        assert_offset(pMeshReflector, MeshDesc, ibOffset);
        assert_offset(pMeshReflector, MeshDesc, vertexCount);
        assert_offset(pMeshReflector, MeshDesc, indexCount);
        assert_offset(pMeshReflector, MeshDesc, materialID);

        // MeshInstanceData
        auto pInstanceReflector = mpMeshInstancesBuffer->getBufferReflector();
        assert_offset(pInstanceReflector, MeshInstanceData, meshID);
        assert_offset(pInstanceReflector, MeshInstanceData, globalMatrixID);

#undef assert_offset
    }

    void Scene::updateBounds()
    {
        const auto& globalMatrices = mpAnimationController->getGlobalMatrices();
        std::vector<BoundingBox> instanceBBs;
        instanceBBs.reserve(mMeshInstanceData.size());

        for (const auto& inst : mMeshInstanceData)
        {
            const BoundingBox& meshBB = mMeshBBs[inst.meshID];
            const mat4& transform = globalMatrices[inst.globalMatrixID];
            instanceBBs.push_back(meshBB.transform(transform));
        }

        mSceneBB = instanceBBs.front();
        for (const BoundingBox& bb : instanceBBs)
        {
            mSceneBB = BoundingBox::fromUnion(mSceneBB, bb);
        }
    }

    void Scene::updateMeshInstanceFlags()
    {
        for (auto& inst : mMeshInstanceData)
        {
            inst.flags = MeshInstanceFlags::None;

            const mat4& transform = mpAnimationController->getGlobalMatrices()[inst.globalMatrixID];
            if (doesTransformFlip(transform)) inst.flags |= MeshInstanceFlags::Flipped;
        }
    }

    void Scene::finalize()
    {
        // Create mapping of meshes to their instances.
        mMeshIdToInstanceIds.clear();
        mMeshIdToInstanceIds.resize(mMeshDesc.size());
        for (uint32_t i = 0; i < (uint32_t)mMeshInstanceData.size(); i++)
        {
            mMeshIdToInstanceIds[mMeshInstanceData[i].meshID].push_back(i);
        }

        initResources();
        mpAnimationController->animate(gpDevice->getRenderContext(), 0); // Requires Scene block to exist
        updateMeshInstanceFlags();
        updateBounds();
        createDrawList();
        if (mCamera.pObject == nullptr)
        {
            mCamera.pObject = Camera::create();
            resetCamera();
        }
        setCameraController(mCamCtrlType);
        updateCamera(true);
        updateLights(true);
        uploadResources(); // Upload data after initialization is complete

        if (mpAnimationController->getMeshAnimationCount(0)) mpAnimationController->setActiveAnimation(0, 0);
    }

    template<>
    void Scene::AnimatedObject<Camera>::setIntoObject(const vec3& pos, const vec3& up, const vec3& lookAt)
    {
        pObject->setUpVector(up);
        pObject->setPosition(pos);
        pObject->setTarget(pos + lookAt);
    }

    template<>
    void Scene::AnimatedObject<Light>::setIntoObject(const vec3& pos, const vec3& up, const vec3& lookAt)
    {
        DirectionalLight* pDirLight = dynamic_cast<DirectionalLight*>(pObject.get());
        if (pDirLight)
        {
            pDirLight->setWorldDirection(lookAt);
            return;
        }
        PointLight* pPointLight = dynamic_cast<PointLight*>(pObject.get());
        if (pPointLight)
        {
            pPointLight->setWorldPosition(pos);
            pPointLight->setWorldDirection(lookAt);
        }
    }

    template<typename Object>
    bool Scene::AnimatedObject<Object>::enabled(bool force) const
    {
        return (animate || force) && (nodeID != kInvalidNode);
    }

    template<typename Object>
    bool Scene::AnimatedObject<Object>::update(const AnimationController* pAnimCtrl, bool force)
    {
        bool update = (force || animate) && nodeID != kInvalidNode;
        if (update)
        {
            if (!pAnimCtrl->didMatrixChanged(nodeID) && !force) return false;

            mat4 camMat = pAnimCtrl->getGlobalMatrices()[nodeID];
            vec3 pos = vec3(camMat[3]);
            vec3 up = vec3(camMat[1]);
            vec3 lookAt = vec3(camMat[2]);
            setIntoObject(pos, up, lookAt);
            return true;
        }
        return false;
    }

    Scene::UpdateFlags Scene::updateLights(bool forceUpdate)
    {
        UpdateFlags flags = UpdateFlags::None;

        for (size_t i = 0 ; i < mLights.size() ; i++)
        {
            auto& l = mLights[i];
            l.update(mpAnimationController.get(), forceUpdate);
            auto lightChanges = l.pObject->beginFrame();

            if (lightChanges != Light::Changes::None)
            {
                l.pObject->setIntoVariableBuffer(mpLightsBuffer.get(), sizeof(LightData)*i);
                if (is_set(lightChanges, Light::Changes::Intensity)) flags |= UpdateFlags::LightIntensityChanged;
                if (is_set(lightChanges, Light::Changes::Position)) flags |= UpdateFlags::LightsMoved;
                if (is_set(lightChanges, Light::Changes::Direction)) flags |= UpdateFlags::LightsMoved;
                const Light::Changes otherChanges = ~(Light::Changes::Intensity | Light::Changes::Position | Light::Changes::Direction);
                if ((lightChanges & otherChanges) != Light::Changes::None) flags |= UpdateFlags::LightPropertiesChanged;
            }
        }
        return flags;
    }

    Scene::UpdateFlags Scene::updateCamera(bool force)
    {
        UpdateFlags flags = UpdateFlags::None;
        mCamera.enabled(force) ? mCamera.update(mpAnimationController.get(), force) : mpCamCtrl->update();
        auto cameraChanges = mCamera.pObject->beginFrame();
        if (cameraChanges != Camera::Changes::None)
        {
            mCamera.pObject->setIntoConstantBuffer(mpSceneBlock->getDefaultConstantBuffer().get(), kCameraVarName);
            if (is_set(cameraChanges, Camera::Changes::Movement)) flags |= UpdateFlags::CameraMoved;
            if ((cameraChanges & (~Camera::Changes::Movement)) != Camera::Changes::None) flags |= UpdateFlags::CameraPropertiesChanged;
        }
        return flags;
    }

    Scene::UpdateFlags Scene::update(RenderContext* pContext, double currentTime)
    {
        mUpdates = UpdateFlags::None;
        if (mpAnimationController->animate(pContext, currentTime))
        {
            mUpdates |= UpdateFlags::SceneGraphChanged;
            for (const auto& inst : mMeshInstanceData)
            {
                if (mpAnimationController->didMatrixChanged(inst.globalMatrixID))
                {
                    mUpdates |= UpdateFlags::MeshesMoved;
                }
            }
        }

        mUpdates |= updateCamera(false);
        mUpdates |= updateLights(false);
        pContext->flush();
        if (is_set(mUpdates, UpdateFlags::MeshesMoved))
        {
            mTlasCache.clear();
            updateMeshInstanceFlags();
        }

        // If a transform in the scene changed, update BLASes with skinned meshes
        if (mBlasData.size() && mHasSkinnedMesh && is_set(mUpdates, UpdateFlags::SceneGraphChanged))
        {
            mTlasCache.clear();
            buildBlas(pContext);
        }

        return mUpdates;
    }

    void Scene::renderUI(Gui::Widgets& widget)
    {
        mpAnimationController->renderUI(widget);
        if(mCamera.hasGlobalTransform()) widget.checkbox("Animate Camera", mCamera.animate);

        auto cameraGroup = Gui::Group(widget, "Camera");
        if (cameraGroup.open())
        {            
            if (cameraGroup.var("Camera Speed", mCameraSpeed, 0.f, FLT_MAX, 0.01f))
            {
                mpCamCtrl->setCameraSpeed(mCameraSpeed);
            }
            mCamera.pObject->renderUI(cameraGroup.gui());

            cameraGroup.release();
        }

        auto lightsGroup = Gui::Group(widget, "Lights");
        if (lightsGroup.open())
        {
            if (mLights.size() && lightsGroup.open())
            {
                for (auto& light : mLights)
                {
                    auto name = light.pObject->getName();
                    auto g = Gui::Group(widget, name);
                    if (g.open())
                    {
                        if (light.hasGlobalTransform()) g.checkbox(("Animate##" + light.pObject->getName()).c_str(), light.animate);
                        light.pObject->renderUI(g.gui());
                        g.release();
                    }
                }
            }

            lightsGroup.release();
        }

        // Filtering mode
        // Camera controller
    }

    void Scene::resetCamera(bool resetDepthRange)
    {
        float radius = length(mSceneBB.extent);
        mCamera.pObject->setPosition(mSceneBB.center);
        mCamera.pObject->setTarget(mSceneBB.center + vec3(0, 0, -1));
        mCamera.pObject->setUpVector(glm::vec3(0, 1, 0));

        if(resetDepthRange)
        {
            float nearZ = std::max(0.1f, radius / 750.0f);
            float farZ = radius * 50;
            mCamera.pObject->setDepthRange(nearZ, farZ);
        }
    }

    void Scene::createDrawList()
    {
        std::vector<D3D12_DRAW_INDEXED_ARGUMENTS> drawClockwiseMeshes, drawCounterClockwiseMeshes, drawAlphaTestedMeshes;
        auto pMatricesBuffer = mpSceneBlock->getTypedBuffer("worldMatrices");
        const mat4* matrices = (mat4*)pMatricesBuffer->getData();

        for (const auto& instance : mMeshInstanceData)
        {
            const auto& mesh = mMeshDesc[instance.meshID];
            const auto& transform = matrices[instance.globalMatrixID];

            D3D12_DRAW_INDEXED_ARGUMENTS draw;
            draw.IndexCountPerInstance = mesh.indexCount;
            draw.InstanceCount = 1;
            draw.StartIndexLocation = mesh.ibOffset;
            draw.BaseVertexLocation = mesh.vbOffset;
            draw.StartInstanceLocation = (uint32_t)(drawClockwiseMeshes.size() + drawCounterClockwiseMeshes.size() + drawAlphaTestedMeshes.size());

            if (mMaterials[mesh.materialID]->getAlphaMode() == AlphaModeMask)
            {
                drawAlphaTestedMeshes.push_back(draw);
            }
            else
            {
                (doesTransformFlip(transform)) ? drawClockwiseMeshes.push_back(draw) : drawCounterClockwiseMeshes.push_back(draw);
            }
        }

        // Create the draw-indirect buffer
        if (drawCounterClockwiseMeshes.size())
        {
            mDrawCounterClockwiseMeshes.pBuffer = Buffer::create(sizeof(drawCounterClockwiseMeshes[0]) * drawCounterClockwiseMeshes.size(), Resource::BindFlags::IndirectArg, Buffer::CpuAccess::None, drawCounterClockwiseMeshes.data());
            mDrawCounterClockwiseMeshes.count = (uint32_t)drawCounterClockwiseMeshes.size();
        }

        if (drawClockwiseMeshes.size())
        {
            mDrawClockwiseMeshes.pBuffer = Buffer::create(sizeof(drawClockwiseMeshes[0]) * drawClockwiseMeshes.size(), Resource::BindFlags::IndirectArg, Buffer::CpuAccess::None, drawClockwiseMeshes.data());
            mDrawClockwiseMeshes.count = (uint32_t)drawClockwiseMeshes.size();
        }

        if (drawAlphaTestedMeshes.size())
        {
            mDrawAlphaTestedMeshes.pBuffer = Buffer::create(sizeof(drawAlphaTestedMeshes[0]) * drawAlphaTestedMeshes.size(), Resource::BindFlags::IndirectArg, Buffer::CpuAccess::None, drawAlphaTestedMeshes.data());
            mDrawAlphaTestedMeshes.count = (uint32_t)drawAlphaTestedMeshes.size();
        }

        size_t drawCount = drawClockwiseMeshes.size() + drawCounterClockwiseMeshes.size() + drawAlphaTestedMeshes.size();
        assert(drawCount <= UINT32_MAX);
    }

    void Scene::sortBlasMeshes()
    {
        // Of the non-instanced meshes, group based on what global matrix ID their transform is.
        std::unordered_map<uint32_t, std::vector<uint32_t>> nodeToMeshList;
        for (uint32_t meshId = 0; meshId < (uint32_t)mMeshIdToInstanceIds.size(); meshId++)
        {
            auto& instanceList = mMeshIdToInstanceIds[meshId];
            if (instanceList.size() > 1) continue; // Only processing non-instanced meshes here

            uint32_t globalMatrixId = mMeshInstanceData[instanceList[0]].globalMatrixID;
            nodeToMeshList[globalMatrixId].push_back(meshId);
        }

        // This should currently only be run on scene initialization
        assert(mBlasData.empty());

        // Build final result. Format is a list of Mesh ID's per BLAS

        // Non-instanced meshes were sorted above so just copy each list
        for (auto& it : nodeToMeshList) mBlasData.push_back(it.second);

        // Meshes that have multiple instances go in their own BLAS
        for (uint32_t meshId = 0; meshId < (uint32_t)mMeshIdToInstanceIds.size(); meshId++)
        {
            auto& instanceList = mMeshIdToInstanceIds[meshId];
            if (instanceList.size() == 1) continue; // Only processing instanced meshes here
            mBlasData.push_back(std::vector<uint32_t>({ meshId }));
        }
    }

    void Scene::initGeomDesc()
    {
        sortBlasMeshes();

        const VertexBufferLayout::SharedConstPtr& pVbLayout = mpVao->getVertexLayout()->getBufferLayout(kStaticDataBufferIndex);
        const Buffer::SharedPtr& pVb = mpVao->getVertexBuffer(kStaticDataBufferIndex);
        const Buffer::SharedPtr& pIb = mpVao->getIndexBuffer();

        for (uint32_t i = 0; i < (uint32_t)mBlasData.size(); i++)
        {
            auto& blas = mBlasData[i];
            auto& meshList = blas.meshList;
            auto& geomDescs = blas.geomDescs;
            geomDescs.resize(meshList.size());

            for (uint32_t j = 0; j < (uint32_t)meshList.size(); j++)
            {
                const MeshDesc& mesh = mMeshDesc[meshList[j]];
                blas.hasSkinnedMesh |= mMeshHasDynamicData[meshList[j]];

                D3D12_RAYTRACING_GEOMETRY_DESC& desc = geomDescs[j];
                desc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
                desc.Triangles.Transform3x4 = 0;
                // If this is an opaque mesh, set the opaque flag
                desc.Flags = (mMaterials[mesh.materialID]->getAlphaMode() == AlphaModeOpaque) ? D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE : D3D12_RAYTRACING_GEOMETRY_FLAG_NONE;

                // Set the position data
                desc.Triangles.VertexBuffer.StartAddress = pVb->getGpuAddress() + (mesh.vbOffset * pVbLayout->getStride());
                desc.Triangles.VertexBuffer.StrideInBytes = pVbLayout->getStride();
                desc.Triangles.VertexCount = mesh.vertexCount;
                desc.Triangles.VertexFormat = getDxgiFormat(pVbLayout->getElementFormat(0));

                // Set index data
                desc.Triangles.IndexBuffer = pIb->getGpuAddress() + (mesh.ibOffset * getFormatBytesPerBlock(mpVao->getIndexBufferFormat()));
                desc.Triangles.IndexCount = mesh.indexCount;
                desc.Triangles.IndexFormat = getDxgiFormat(mpVao->getIndexBufferFormat());
            }

            mHasSkinnedMesh |= blas.hasSkinnedMesh;
        }
    }

    void Scene::buildBlas(RenderContext* pContext)
    {
        PROFILE("buildBlas");

        // Get the VB and IB
        const VertexBufferLayout::SharedConstPtr& pVbLayout = mpVao->getVertexLayout()->getBufferLayout(kStaticDataBufferIndex);
        const Buffer::SharedPtr& pVb = mpVao->getVertexBuffer(kStaticDataBufferIndex);
        const Buffer::SharedPtr& pIb = mpVao->getIndexBuffer();
        pContext->resourceBarrier(pVb.get(), Resource::State::NonPixelShader);
        pContext->resourceBarrier(pIb.get(), Resource::State::NonPixelShader);

        // For each BLAS
        for (uint32_t i = 0; i < (uint32_t)mBlasData.size(); i++)
        {
            auto& blas = mBlasData[i];
            auto& meshList = blas.meshList;

            if (blas.pBlas != nullptr && !blas.hasSkinnedMesh) continue; // Skip updating BLASes not containing skinned meshes

            // Setup build parameters and get prebuild info
            D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs = {};
            inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
            inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
            inputs.NumDescs = (uint32_t)blas.geomDescs.size();
            inputs.pGeometryDescs = blas.geomDescs.data();
            inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_NONE;

            // Determine if this BLAS is, or will be refit, and add necessary flags
            if (blas.hasSkinnedMesh && mBlasUpdateMode == UpdateMode::Refit)
            {
                inputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE; // Subsequent updates need this flag too

                // Refit if BLAS exists, and it was previously created with ALLOW_UPDATE
                if (blas.pBlas != nullptr && blas.updateMode == UpdateMode::Refit) inputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
            }

            // Allocate scratch and BLAS buffers on the first build
            if (blas.pBlas == nullptr)
            {
                GET_COM_INTERFACE(gpDevice->getApiHandle(), ID3D12Device5, pDevice5);
                pDevice5->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &blas.prebuildInfo);

                // #SCENE This isn't guaranteed according to the spec, and the scratch buffer being stored should be sized differently depending on update mode
                assert(blas.prebuildInfo.UpdateScratchDataSizeInBytes <= blas.prebuildInfo.ScratchDataSizeInBytes);

                blas.pScratchBuffer = Buffer::create(blas.prebuildInfo.ScratchDataSizeInBytes, Buffer::BindFlags::UnorderedAccess, Buffer::CpuAccess::None);
                blas.pBlas = Buffer::create(blas.prebuildInfo.ResultDataMaxSizeInBytes, Buffer::BindFlags::AccelerationStructure, Buffer::CpuAccess::None);
            }
            // For any rebuild and refits, just add a barrier
            else
            {
                assert(blas.pScratchBuffer != nullptr);

                pContext->uavBarrier(blas.pBlas.get());
                pContext->uavBarrier(blas.pScratchBuffer.get());
            }

            // Build BLAS
            D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC asDesc = {};
            asDesc.Inputs = inputs;
            asDesc.ScratchAccelerationStructureData = blas.pScratchBuffer->getGpuAddress();
            asDesc.DestAccelerationStructureData = blas.pBlas->getGpuAddress();

            // Set buffer address to update in place if this is a refit
            if ((inputs.Flags & D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE) > 0) asDesc.SourceAccelerationStructureData = asDesc.DestAccelerationStructureData;

            GET_COM_INTERFACE(pContext->getLowLevelData()->getCommandList(), ID3D12GraphicsCommandList4, pList4);
            pList4->BuildRaytracingAccelerationStructure(&asDesc, 0, nullptr);

            // Insert a UAV barrier
            pContext->uavBarrier(blas.pBlas.get());

            if (!blas.hasSkinnedMesh) blas.pScratchBuffer.reset(); // Release 
            else blas.updateMode = mBlasUpdateMode;
        }
    }

    void Scene::fillInstanceDesc(std::vector<D3D12_RAYTRACING_INSTANCE_DESC>& instanceDescs, uint32_t rayCount, bool perMeshHitEntry)
    {
        instanceDescs.clear();
        uint32_t instanceContributionToHitGroupIndex = 0;
        uint32_t instanceId = 0;
        for (uint32_t i = 0; i < (uint32_t)mBlasData.size(); i++)
        {
            auto& meshList = mBlasData[i].meshList;
            D3D12_RAYTRACING_INSTANCE_DESC desc = {};
            desc.AccelerationStructure = mBlasData[i].pBlas->getGpuAddress();
            desc.InstanceMask = 0xFF;
            desc.InstanceContributionToHitGroupIndex = perMeshHitEntry ? instanceContributionToHitGroupIndex : 0;
            instanceContributionToHitGroupIndex += rayCount * (uint32_t)meshList.size();

            // If multiple meshes are in a BLAS:
            // - Their global matrix is the same.
            // - From sortBlasMeshes(), each mesh in the BLAS is guaranteed to be non-instanced, so only one INSTANCE_DESC is needed
            if (meshList.size() > 1)
            {
                desc.InstanceID = instanceId;
                instanceId += (uint32_t)meshList.size();

                // Any instances of the mesh will get you the correct matrix, so just pick the first mesh then the first instance.
                uint32_t firstInstanceId = mMeshIdToInstanceIds[meshList[0]][0];
                uint32_t matrixId = mMeshInstanceData[firstInstanceId].globalMatrixID;
                mat4 transform4x4 = transpose(mpAnimationController->getGlobalMatrices()[matrixId]);
                std::memcpy(desc.Transform, &transform4x4, sizeof(desc.Transform));
                instanceDescs.push_back(desc);
            }
            // If only one mesh is in the BLAS, there CAN be multiple instances of it. It is either:
            // - A non-instanced mesh that was unable to be merged with others
            // - A mesh with multiple instances
            else
            {
                // For every instance of the mesh, create an INSTANCE_DESC
                auto& instanceList = mMeshIdToInstanceIds[meshList[0]];
                for (uint32_t instId : instanceList)
                {
                    desc.InstanceID = instanceId++;
                    uint32_t matrixId = mMeshInstanceData[instId].globalMatrixID;
                    mat4 transform4x4 = transpose(mpAnimationController->getGlobalMatrices()[matrixId]);
                    std::memcpy(desc.Transform, &transform4x4, sizeof(desc.Transform));
                    instanceDescs.push_back(desc);
                }
            }
        }
    }

    void Scene::buildTlas(RenderContext* pContext, uint32_t rayCount, bool perMeshHitEntry)
    {
        PROFILE("buildTlas");

        TlasData tlas;
        auto it = mTlasCache.find(rayCount);
        if (it != mTlasCache.end()) tlas = it->second;

        fillInstanceDesc(mInstanceDescs, rayCount, perMeshHitEntry);

        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs = {};
        inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
        inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
        inputs.NumDescs = (uint32_t)mInstanceDescs.size();
        inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_NONE;

        // Add build flags for dynamic scenes if TLAS should be updating instead of rebuilt
        if (mpAnimationController->hasAnimations() && mTlasUpdateMode == UpdateMode::Refit)
        {
            inputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;

            // If TLAS has been built already and it was built with ALLOW_UPDATE
            if(tlas.pTlas != nullptr && tlas.updateMode == UpdateMode::Refit) inputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
        }

        tlas.updateMode = mTlasUpdateMode;

        // On first build for the scene, create scratch buffer and cache prebuild info. As long as INSTANCE_DESC count doesn't change, we can reuse these
        if (mpTlasScratch == nullptr)
        {
            // Prebuild
            GET_COM_INTERFACE(gpDevice->getApiHandle(), ID3D12Device5, pDevice5);
            pDevice5->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &mTlasPrebuildInfo);
            mpTlasScratch = Buffer::create(mTlasPrebuildInfo.ScratchDataSizeInBytes, Buffer::BindFlags::UnorderedAccess, Buffer::CpuAccess::None);

            // #SCENE This isn't guaranteed according to the spec, and the scratch buffer being stored should be sized differently depending on update mode
            assert(mTlasPrebuildInfo.UpdateScratchDataSizeInBytes <= mTlasPrebuildInfo.ScratchDataSizeInBytes);
        }

        // Setup GPU buffers
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC asDesc = {};
        asDesc.Inputs = inputs;

        // If first time building this TLAS
        if (tlas.pTlas == nullptr)
        {
            assert(tlas.pInstanceDescs == nullptr); // Instance desc should also be null if no TLAS
            tlas.pTlas = Buffer::create(mTlasPrebuildInfo.ResultDataMaxSizeInBytes, Buffer::BindFlags::AccelerationStructure, Buffer::CpuAccess::None);
            tlas.pInstanceDescs = Buffer::create((uint32_t)mInstanceDescs.size() * sizeof(D3D12_RAYTRACING_INSTANCE_DESC), Buffer::BindFlags::None, Buffer::CpuAccess::Write, mInstanceDescs.data());
        }
        // Else update instance descs and barrier TLAS buffers
        else
        {
            assert(mpAnimationController->hasAnimations());
            pContext->uavBarrier(tlas.pTlas.get());
            pContext->uavBarrier(mpTlasScratch.get());
            tlas.pInstanceDescs->setBlob(mInstanceDescs.data(), 0, inputs.NumDescs * sizeof(D3D12_RAYTRACING_INSTANCE_DESC));
            asDesc.SourceAccelerationStructureData = tlas.pTlas->getGpuAddress(); // Perform the update in-place
        }

        assert((inputs.NumDescs != 0) && tlas.pInstanceDescs->getApiHandle() && tlas.pTlas->getApiHandle() && mpTlasScratch->getApiHandle());

        asDesc.Inputs.InstanceDescs = tlas.pInstanceDescs->getGpuAddress();
        asDesc.ScratchAccelerationStructureData = mpTlasScratch->getGpuAddress();
        asDesc.DestAccelerationStructureData = tlas.pTlas->getGpuAddress();

        // Set the source buffer to update in place if this is an update
        if ((inputs.Flags & D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE) > 0) asDesc.SourceAccelerationStructureData = asDesc.DestAccelerationStructureData;

        // Create TLAS
        GET_COM_INTERFACE(pContext->getLowLevelData()->getCommandList(), ID3D12GraphicsCommandList4, pList4);
        pContext->resourceBarrier(tlas.pInstanceDescs.get(), Resource::State::NonPixelShader);
        pList4->BuildRaytracingAccelerationStructure(&asDesc, 0, nullptr);
        pContext->uavBarrier(tlas.pTlas.get());

        // Create TLAS SRV
        if (tlas.pSrv == nullptr)
        {
            D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
            srvDesc.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
            srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srvDesc.RaytracingAccelerationStructure.Location = tlas.pTlas->getGpuAddress();

            DescriptorSet::Layout layout;
            layout.addRange(DescriptorSet::Type::TextureSrv, 0, 1);
            DescriptorSet::SharedPtr pSet = DescriptorSet::create(gpDevice->getCpuDescriptorPool(), layout);
            assert(pSet);
            gpDevice->getApiHandle()->CreateShaderResourceView(nullptr, &srvDesc, pSet->getCpuHandle(0));

            ResourceWeakPtr pWeak = tlas.pTlas;
            tlas.pSrv = std::make_shared<ShaderResourceView>(pWeak, pSet, 0, 1, 0, 1);
        }

        mTlasCache[rayCount] = tlas;
    }

    void Scene::updateAsToInstanceDataMapping()
    {
        // Calculate acceleration structure indexing to mMeshInstanceData index
        // Essentially: mMeshInstanceData[ buffer[TLAS InstanceID() + GeometryIndex] ]
        // Here, just append mesh instance ID's in order they'd appear in the TLAS.
        std::vector<uint32_t> asToInstanceMapping;
        for (uint32_t blasIndex = 0; blasIndex < (uint32_t)mBlasData.size(); blasIndex++)
        {
            for (const uint32_t& meshId : mBlasData[blasIndex].meshList)
            {
                auto& instList = mMeshIdToInstanceIds[meshId];
                for (const uint32_t instId : instList)
                {
                    asToInstanceMapping.push_back(instId);
                }
            }
        }

        mpAsToInstanceMapping = Buffer::create(asToInstanceMapping.size() * sizeof(uint32_t), Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, asToInstanceMapping.data());
    }

    void Scene::setGeometryIndexIntoRtVars(const std::shared_ptr<RtProgramVars>& pRtVars)
    {
        // Set BLAS geometry index as constant buffer data
        for (uint32_t ray = 0; ray < pRtVars->getHitProgramsCount(); ray++)
        {
            RtProgramVars::VarsVector& rayVars = pRtVars->getHitVars(ray);

            uint32_t blasIndex = 0;
            uint32_t geometryIndex = 0;
            for (uint32_t i = 0; i < rayVars.size(); i++)
            {
                auto& pVar = rayVars[i];
                pVar["DxrPerGeometry"]["geometryIndex"] = geometryIndex++;

                // If at the end of this BLAS, reset counters and start checking next BLAS
                uint32_t geomCount = (uint32_t)mBlasData[blasIndex].meshList.size();
                if (geometryIndex == geomCount)
                {
                    geometryIndex = 0;
                    blasIndex++;
                }
            }
        }
    }

    void Scene::setEnvironmentMap(Texture::ConstSharedPtrRef pEnvMap)
    {
        if (mpEnvMap == pEnvMap) return;
        mpEnvMap = pEnvMap;
        mpSceneBlock["envMap"] = mpEnvMap;
    }

    void Scene::setCameraAspectRatio(float ratio)
    {
        mCamera.pObject->setAspectRatio(ratio);
    }

    void Scene::bindSamplerToMaterials(Sampler::ConstSharedPtrRef pSampler)
    {
        for (auto& pMaterial : mMaterials)
        {
            pMaterial->setSampler(pSampler);
        }
    }

    void Scene::setCameraController(CameraControllerType type)
    {
        if (mCamCtrlType == type && mpCamCtrl) return;

        switch (type)
        {
        case CameraControllerType::FirstPerson:
            mpCamCtrl = FirstPersonCameraController::create(mCamera.pObject);
            break;
        case CameraControllerType::Orbiter:
            mpCamCtrl = OrbiterCameraController::create(mCamera.pObject);
            ((OrbiterCameraController*)mpCamCtrl.get())->setModelParams(mSceneBB.center, length(mSceneBB.extent), 3.5f);
            break;
        case CameraControllerType::SixDOF:
            mpCamCtrl = SixDoFCameraController::create(mCamera.pObject);
            break;
        default:
            should_not_get_here();
        }
        mpCamCtrl->setCameraSpeed(mCameraSpeed);
    }

    bool Scene::onMouseEvent(const MouseEvent& mouseEvent)
    {
        return mpCamCtrl->onMouseEvent(mouseEvent);
    }

    bool Scene::onKeyEvent(const KeyboardEvent& keyEvent)
    {
        return mpCamCtrl->onKeyEvent(keyEvent);
    }
}
