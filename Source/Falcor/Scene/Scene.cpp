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
#include "Scene.h"
#include "Raytracing/RtProgram/RtProgram.h"
#include "Raytracing/RtProgramVars.h"
#include <sstream>

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
        const std::string kMaterialsBufferName = "materials";
        const std::string kLightsBufferName = "lights";
        const std::string kCameraVarName = "camera";

        const std::string kCamera = "c";
        const std::string kViewpoint = "viewpoint";
        const std::string kPosition = "position";
        const std::string kTarget = "target";
        const std::string kUp = "up";
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

    Shader::DefineList Scene::getSceneDefines() const
    {
        Shader::DefineList defines;
        defines.add("MATERIAL_COUNT", std::to_string(mMaterials.size()));
        return defines;
    }

    LightCollection::ConstSharedPtrRef Scene::getLightCollection(RenderContext* pContext)
    {
        if (!mpLightCollection)
        {
            mpLightCollection = LightCollection::create(pContext, shared_from_this());
            mpLightCollection->setShaderData(mpSceneBlock["lightCollection"]);
        }
        return mpLightCollection;
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

        if (overrideRS) pState->setRasterizerState(pCurrentRS);
    }

    void Scene::raytrace(RenderContext* pContext, RtProgram* pProgram, const std::shared_ptr<RtProgramVars>& pVars, uvec3 dispatchDims)
    {
        PROFILE("raytraceScene");

        auto rayTypeCount = pProgram->getHitProgramCount();
        setRaytracingShaderData(pContext, pVars->getRootVar(), rayTypeCount);

        // If not set yet, set geometry indices for this RtProgramVars.
        if (pVars->getSceneForGeometryIndices().get() != this)
        {
            setGeometryIndexIntoRtVars(pVars);
            pVars->setSceneForGeometryIndices(shared_from_this());
        }

        // Set ray type constant.
        pVars->getRootVar()["DxrPerFrame"]["hitProgramCount"] = rayTypeCount;

        pContext->raytrace(pProgram, pVars.get(), dispatchDims.x, dispatchDims.y, dispatchDims.z);
    }

    void Scene::initResources()
    {
        GraphicsProgram::SharedPtr pProgram = GraphicsProgram::createFromFile("Scene/SceneBlock.slang", "", "main");
        pProgram->addDefines(getSceneDefines());
        ParameterBlockReflection::SharedConstPtr pReflection = pProgram->getReflector()->getParameterBlock(kParameterBlockName);
        assert(pReflection);

        mpSceneBlock = ParameterBlock::create(pReflection);
        mpMeshesBuffer = Buffer::createStructured(mpSceneBlock[kMeshBufferName], (uint32_t)mMeshDesc.size(), Resource::BindFlags::ShaderResource);
        mpMeshInstancesBuffer = Buffer::createStructured(mpSceneBlock[kMeshInstanceBufferName], (uint32_t)mMeshInstanceData.size(), Resource::BindFlags::ShaderResource);

        mpMaterialsBuffer = Buffer::createStructured(mpSceneBlock[kMaterialsBufferName], (uint32_t)mMaterials.size(), Resource::BindFlags::ShaderResource);

        if (mLights.size())
        {
            mpLightsBuffer = Buffer::createStructured(mpSceneBlock[kLightsBufferName], (uint32_t)mLights.size(), Resource::BindFlags::ShaderResource);
        }
    }

    void Scene::uploadResources()
    {
        // Upload geometry
        checkOffsets();
        mpMeshesBuffer->setBlob(mMeshDesc.data(), 0, sizeof(MeshDesc) * mMeshDesc.size());
        mpMeshInstancesBuffer->setBlob(mMeshInstanceData.data(), 0, sizeof(MeshInstanceData) * mMeshInstanceData.size());

        mpSceneBlock->setBuffer(kMeshInstanceBufferName, mpMeshInstancesBuffer);
        mpSceneBlock->setBuffer(kMeshBufferName, mpMeshesBuffer);
        mpSceneBlock->setBuffer(kLightsBufferName, mpLightsBuffer);
        mpSceneBlock->setBuffer(kMaterialsBufferName, mpMaterialsBuffer);
        mpSceneBlock->setBuffer(kIndexBufferName, mpVao->getIndexBuffer());
        mpSceneBlock->setBuffer(kVertexBufferName, mpVao->getVertexBuffer(Scene::kStaticDataBufferIndex));

        if (mpLightProbe)
        {
            mpLightProbe->setShaderData(mpSceneBlock["lightProbe"]);
        }
    }

    // TODO: On initial upload of materials, we could improve this by not having separate calls to setElement()
    // but instead prepare a buffer containing all data.
    void Scene::uploadMaterial(uint32_t materialID)
    {
        assert(materialID < mMaterials.size());

        const auto &material = mMaterials[materialID];

        mpMaterialsBuffer->setElement(materialID, material->getData());

        const auto& resources = material->getResources();

        auto var = mpSceneBlock["materialResources"][materialID];

#define set_texture(texName) var[#texName] = resources.texName;
        set_texture(baseColor);
        set_texture(specular);
        set_texture(emissive);
        set_texture(normalMap);
        set_texture(occlusionMap);
#undef set_texture

        var["samplerState"] = resources.samplerState;
    }

    void Scene::checkOffsets()
    {
#if 0 // #SHADER_VAR fix this after we get rid of StructuredBuffer
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
#endif
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
        saveNewViewpoint();
        updateLights(true);
        updateMaterials(true);
        uploadResources(); // Upload data after initialization is complete
        updateGeometryStats();

        if (mpAnimationController->getMeshAnimationCount(0)) mpAnimationController->setActiveAnimation(0, 0);
    }

    void Scene::updateGeometryStats()
    {
        mGeometryStats = {};
        auto& s = mGeometryStats;

        for (uint32_t meshID = 0; meshID < getMeshCount(); meshID++)
        {
            const auto& mesh = getMesh(meshID);
            s.uniqueVertexCount += mesh.vertexCount;
            s.uniqueTriangleCount += mesh.indexCount / 3;
        }
        for (uint32_t instanceID = 0; instanceID < getMeshInstanceCount(); instanceID++)
        {
            const auto& instance = getMeshInstance(instanceID);
            const auto& mesh = getMesh(instance.meshID);
            s.instancedVertexCount += mesh.vertexCount;
            s.instancedTriangleCount += mesh.indexCount / 3;
        }
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

    Scene::UpdateFlags Scene::updateCamera(bool forceUpdate)
    {
        UpdateFlags flags = UpdateFlags::None;
        mCamera.enabled(forceUpdate) ? mCamera.update(mpAnimationController.get(), forceUpdate) : mpCamCtrl->update();
        auto cameraChanges = mCamera.pObject->beginFrame();
        if (cameraChanges != Camera::Changes::None)
        {
            mCamera.pObject->setShaderData(mpSceneBlock[kCameraVarName]);
            if (is_set(cameraChanges, Camera::Changes::Movement)) flags |= UpdateFlags::CameraMoved;
            if ((cameraChanges & (~Camera::Changes::Movement)) != Camera::Changes::None) flags |= UpdateFlags::CameraPropertiesChanged;
        }
        return flags;
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
                // TODO: This is slow since the buffer is not CPU writable. Copy into CPU buffer and upload once instead.
                mpLightsBuffer->setBlob(&l.pObject->getData(), sizeof(LightData) * i, sizeof(LightData));
                if (is_set(lightChanges, Light::Changes::Intensity)) flags |= UpdateFlags::LightIntensityChanged;
                if (is_set(lightChanges, Light::Changes::Position)) flags |= UpdateFlags::LightsMoved;
                if (is_set(lightChanges, Light::Changes::Direction)) flags |= UpdateFlags::LightsMoved;
                const Light::Changes otherChanges = ~(Light::Changes::Intensity | Light::Changes::Position | Light::Changes::Direction);
                if ((lightChanges & otherChanges) != Light::Changes::None) flags |= UpdateFlags::LightPropertiesChanged;
            }
        }
        return flags;
    }

    Scene::UpdateFlags Scene::updateMaterials(bool forceUpdate)
    {
        UpdateFlags flags = UpdateFlags::None;

        // Early out if no materials have changed
        if (!forceUpdate && Material::getGlobalUpdates() == Material::UpdateFlags::None) return flags;

        for (uint32_t i = 0; i < (uint32_t)mMaterials.size(); ++i)
        {
            auto& material = mMaterials[i];
            auto materialUpdates = material->getUpdates();
            if (forceUpdate || materialUpdates != Material::UpdateFlags::None)
            {
                material->clearUpdates();
                uploadMaterial(i);
                flags |= UpdateFlags::MaterialsChanged;
            }
        }

        Material::clearGlobalUpdates();

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
        mUpdates |= updateMaterials(false);
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

        // Update light collection
        if (mpLightCollection && mpLightCollection->update(pContext)) mUpdates |= UpdateFlags::LightCollectionChanged;

        return mUpdates;
    }

    void Scene::renderUI(Gui::Widgets& widget)
    {
        mpAnimationController->renderUI(widget);
        if(mCamera.hasGlobalTransform()) widget.checkbox("Animate Camera", mCamera.animate);

        auto cameraGroup = Gui::Group(widget, "Camera");
        if (cameraGroup.open())
        {
            if (cameraGroup.button("Save New Viewpoint")) saveNewViewpoint();
            Gui::DropdownList viewpoints;
            viewpoints.push_back({ 0, "Default Viewpoint" });
            for (uint32_t i = 1; i < mViewpoints.size(); i++)
            {
                viewpoints.push_back({ i, "Saved Viewpoint " + to_string(i) });
            }
            uint32_t index = mCurrentViewpoint;
            if (cameraGroup.dropdown("Saved Viewpoints", viewpoints, index)) gotoViewpoint(index);
            if (cameraGroup.button("Remove Current Viewpoint")) removeViewpoint();

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

        auto mtlGroup = Gui::Group(widget, "Materials");
        if (mtlGroup.open())
        {
            uint32_t materialID = 0;
            for (auto& material : mMaterials)
            {
                auto name = material->getName();
                auto g = Gui::Group(widget, std::to_string(materialID) + ": " + name);
                if (g.open())
                {
                    if (material->renderUI(g))
                    {
                        uploadMaterial(materialID);
                    }
                }
                materialID++;
            }

            mtlGroup.release();
        }

        auto statsGroup = Gui::Group(widget, "Statistics");
        if (statsGroup.open())
        {
            uint32_t lightProbeCount = getLightProbe() != nullptr ? 1 : 0;
            std::ostringstream oss;
            oss << "Mesh count: " << getMeshCount() << std::endl
                << "Mesh instance count: " << getMeshInstanceCount() << std::endl
                << "Unique triangle count: " << mGeometryStats.uniqueTriangleCount << std::endl
                << "Unique vertex count: " << mGeometryStats.uniqueVertexCount << std::endl
                << "Instanced triangle count: " << mGeometryStats.instancedTriangleCount << std::endl
                << "Instanced vertex count: " << mGeometryStats.instancedVertexCount << std::endl
                << "Material count: " << getMaterialCount() << std::endl
                << "Analytic light count: " << getLightCount() << std::endl
                << "Light probe count: " << lightProbeCount << std::endl;
            statsGroup.text(oss.str());

            if (mpLightCollection)
            {
                auto lightCollectionGroup = Gui::Group(widget, "Mesh lights", true);
                if (lightCollectionGroup.open()) mpLightCollection->renderUI(lightCollectionGroup);
                lightCollectionGroup.release();
            }
            else
            {
                statsGroup.text("Mesh light count: N/A");
            }

            statsGroup.release();
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

    void Scene::saveNewViewpoint()
    {
        auto camera = getCamera();
        auto position = camera->getPosition();
        auto target = camera->getTarget();
        auto up = camera->getUpVector();

        Viewpoint v = { position, target, up };
        mViewpoints.push_back(v);
    }

    void Scene::removeViewpoint()
    {
        if (mCurrentViewpoint == 0)
        {
            logWarning("Cannot remove default viewpoint");
            return;
        }
        mViewpoints.erase(mViewpoints.begin() + mCurrentViewpoint);
        mCurrentViewpoint = 0;
    }

    void Scene::gotoViewpoint(uint32_t index)
    {
        auto camera = getCamera();
        camera->setPosition(mViewpoints[index].position);
        camera->setTarget(mViewpoints[index].target);
        camera->setUpVector(mViewpoints[index].up);
        mCurrentViewpoint = index;
    }

    Material::SharedPtr Scene::getMaterialByName(const std::string &name) const
    {
        for (const auto& m : mMaterials)
        {
            if (m->getName() == name) return m;
        }

        return nullptr;
    }

    Light::SharedPtr Scene::getLightByName(const std::string &name) const
    {
        for (const auto& l : mLights)
        {
            if (l.pObject->getName() == name) return l.pObject;
        }

        return nullptr;
    }

    void Scene::toggleAnimations(bool animate)
    {
        for (int i = 0; i < mLights.size(); i++)
        {
            toggleLightAnimation(i, animate);
        }
        toggleCameraAnimation(animate);
        mpAnimationController->toggleAnimations(animate);
    }

    void Scene::createDrawList()
    {
        std::vector<D3D12_DRAW_INDEXED_ARGUMENTS> drawClockwiseMeshes, drawCounterClockwiseMeshes;
        auto pMatricesBuffer = mpSceneBlock->getBuffer("worldMatrices");
        const mat4* matrices = (mat4*)pMatricesBuffer->map(Buffer::MapType::Read); // #SCENEV2 This will cause the pipeline to flush and sync, but it's probably not too bad as this only happens once

        for (const auto& instance : mMeshInstanceData)
        {
            const auto& mesh = mMeshDesc[instance.meshID];
            const auto& transform = matrices[instance.globalMatrixID];

            D3D12_DRAW_INDEXED_ARGUMENTS draw;
            draw.IndexCountPerInstance = mesh.indexCount;
            draw.InstanceCount = 1;
            draw.StartIndexLocation = mesh.ibOffset;
            draw.BaseVertexLocation = mesh.vbOffset;
            draw.StartInstanceLocation = (uint32_t)(drawClockwiseMeshes.size() + drawCounterClockwiseMeshes.size());

            (doesTransformFlip(transform)) ? drawClockwiseMeshes.push_back(draw) : drawCounterClockwiseMeshes.push_back(draw);
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

        size_t drawCount = drawClockwiseMeshes.size() + drawCounterClockwiseMeshes.size();
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
                const auto& material = mMaterials[mesh.materialID];
                bool opaque = (material->getAlphaMode() == AlphaModeOpaque) && material->getSpecularTransmission() == 0.f;
                desc.Flags = opaque ? D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE : D3D12_RAYTRACING_GEOMETRY_FLAG_NONE;

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

        updateAsToInstanceDataMapping();
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
            for (const uint32_t meshId : mBlasData[blasIndex].meshList)
            {
                auto& instList = mMeshIdToInstanceIds[meshId];
                for (const uint32_t instId : instList)
                {
                    asToInstanceMapping.push_back(instId);
                }
            }
        }

        mpAsToInstanceMapping = Buffer::createTyped<uint32_t>((uint32_t)asToInstanceMapping.size(), Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, asToInstanceMapping.data());
    }

    void Scene::setGeometryIndexIntoRtVars(const std::shared_ptr<RtProgramVars>& pVars)
    {
        assert(!mBlasData.empty());
        auto meshCount = getMeshCount();
        uint32_t descHitCount = pVars->getDescHitGroupCount();

        uint32_t blasIndex = 0;
        uint32_t geometryIndex = 0;
        for (uint32_t i = 0; i < meshCount; i++)
        {
            for (uint32_t hit = 0; hit < descHitCount; hit++)
            {
                auto pHitVars = pVars->getHitVars(hit, i);
                auto var = pHitVars->findMember(0).findMember("geometryIndex");
                if( var.isValid() )
                {
                    var = geometryIndex;
                }
            }

            geometryIndex++;

            // If at the end of this BLAS, reset counters and start checking next BLAS
            uint32_t geomCount = (uint32_t)mBlasData[blasIndex].meshList.size();
            if (geometryIndex == geomCount)
            {
                geometryIndex = 0;
                blasIndex++;
            }
        }
    }

    void Scene::setRaytracingShaderData(RenderContext* pContext, const ShaderVar& var, uint32_t rayTypeCount)
    {
        // On first execution, create BLAS for each mesh.
        if (mBlasData.empty())
        {
            initGeomDesc();
            buildBlas(pContext);
            updateAsToInstanceDataMapping();
        }

        // On first execution, when meshes have moved, when there's a new ray count, or when a BLAS has changed, create/update the TLAS
        //
        // TODO: The notion of "ray count" is being treated as fundamental here, and intrinsicly
        // linked to the number of hit groups in the program, without checking if this matches
        // other things like the number of miss shaders. If/when we support meshes with custom
        // intersection shaders, then the assumption that number of ray types and number of
        // hit groups match will be incorrect.
        //
        // It really seems like a first-class notion of ray types (and the number thereof) is required.
        //
        auto tlasIt = mTlasCache.find(rayTypeCount);
        if (tlasIt == mTlasCache.end())
        {
            // We need a hit entry per mesh right now to pass GeometryIndex()
            buildTlas(pContext, rayTypeCount, true);

            // If new TLAS was just created, get it so the iterator is valid
            if (tlasIt == mTlasCache.end()) tlasIt = mTlasCache.find(rayTypeCount);
        }

        // Bind Scene parameter block.
        mCamera.pObject->setShaderData(mpSceneBlock[kCameraVarName]);
        var["gScene"] = mpSceneBlock;

        // Bind TLAS.
        var["gRtScene"].setSrv(tlasIt->second.pSrv);

        // Bind lookup table for mesh instance ID.
        var["gAsToInstance"] = mpAsToInstanceMapping;
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
        if (keyEvent.type == KeyboardEvent::Type::KeyPressed)
        {
            if (!(keyEvent.mods.isAltDown || keyEvent.mods.isCtrlDown || keyEvent.mods.isShiftDown))
            {
                if (keyEvent.key == KeyboardEvent::Key::F3)
                {
                    saveNewViewpoint();
                    return true;
                }
            }
        }
        return mpCamCtrl->onKeyEvent(keyEvent);
    }

    std::string Scene::getConfig()
    {
        std::string c;
        c += std::string(kCamera) + " = " + Scripting::makeMemberFunc(kScene, kCameraVarName);
        for (int i = 0; i < mViewpoints.size(); i++)
        {
            if (i == 0) continue;
            auto v = mViewpoints[i];
            c += std::string(kCamera) + "." + kPosition + "=" + to_string(v.position) + "\n";
            c += std::string(kCamera) + "." + kTarget + "=" + to_string(v.target) + "\n";
            c += std::string(kCamera) + "." + kUp + "=" + to_string(v.up) + "\n";
            c += Scripting::makeMemberFunc(kScene, kViewpoint);
        }
        c += Scripting::makeMemberFunc(kScene, kViewpoint, 0);
        return c;
    }

    SCRIPT_BINDING(Scene)
    {
        auto s = m.regClass(Scene);
        s.func_("animate", &Scene::toggleAnimations); // toggle animations on or off
        s.func_("light", &Scene::getLight); // get specific light
        s.func_("light", &Scene::getLightByName); // get specific light
        s.func_("animateLight", &Scene::toggleLightAnimation); // toggle animation for a specific light on or off
        s.func_("material", &Scene::getMaterial); // get specific material
        s.func_("material", &Scene::getMaterialByName); // get specific material
        s.func_("camera", &Scene::getCamera); // get camera
        s.func_("animateCamera", &Scene::toggleCameraAnimation); // toggle camera animation on or off
        s.func_("viewpoint", &Scene::saveNewViewpoint); // save the current camera position etc.
        s.func_("viewpoint", &Scene::gotoViewpoint); // select a previously saved camera viewpoint
        s.func_("removeViewpoint", &Scene::removeViewpoint); // remove the current camera viewpoint
    }
}
