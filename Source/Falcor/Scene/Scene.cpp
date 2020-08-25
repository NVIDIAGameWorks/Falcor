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
#include "Scene.h"
#include "HitInfo.h"
#include "Raytracing/RtProgram/RtProgram.h"
#include "Raytracing/RtProgramVars.h"
#include <sstream>

namespace Falcor
{
    static_assert(sizeof(PackedStaticVertexData) % 16 == 0, "PackedStaticVertexData size should be a multiple of 16");
    static_assert(sizeof(PackedMeshInstanceData) % 16 == 0, "PackedMeshInstanceData size should be a multiple of 16");
    static_assert(PackedMeshInstanceData::kMatrixBits + PackedMeshInstanceData::kMeshBits + PackedMeshInstanceData::kFlagsBits <= 32);

    namespace
    {
        // Checks if the transform flips the coordinate system handedness (its determinant is negative).
        bool doesTransformFlip(const glm::mat4& m)
        {
            return glm::determinant((glm::mat3)m) < 0.f;
        }

        const std::string kParameterBlockName = "gScene";
        const std::string kMeshBufferName = "meshes";
        const std::string kMeshInstanceBufferName = "meshInstances";
        const std::string kIndexBufferName = "indices";
        const std::string kVertexBufferName = "vertices";
        const std::string kPrevVertexBufferName = "prevVertices";
        const std::string kMaterialsBufferName = "materials";
        const std::string kLightsBufferName = "lights";

        const std::string kCamera = "camera";
        const std::string kCameras = "cameras";
        const std::string kCameraSpeed = "cameraSpeed";
        const std::string kAnimated = "animated";
        const std::string kRenderSettings = "renderSettings";
        const std::string kEnvMap = "envMap";
        const std::string kMaterials = "materials";
        const std::string kGetLight = "getLight";
        const std::string kGetMaterial = "getMaterial";
        const std::string kSetEnvMap = "setEnvMap";
        const std::string kAddViewpoint = "addViewpoint";
        const std::string kRemoveViewpoint = "kRemoveViewpoint";
        const std::string kSelectViewpoint = "selectViewpoint";
    }

    const FileDialogFilterVec& Scene::getFileExtensionFilters()
    {
        return Importer::getFileExtensionFilters();
    }

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
        defines.add("INDEXED_VERTICES", hasIndexBuffer() ? "1" : "0");
        defines.add(HitInfo::getDefines(this));
        return defines;
    }

    const LightCollection::SharedPtr& Scene::getLightCollection(RenderContext* pContext)
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
        bool isIndexed = hasIndexBuffer();

        if (mDrawCounterClockwiseMeshes.count)
        {
            if (overrideRS) pState->setRasterizerState(nullptr);
            if (isIndexed) pContext->drawIndexedIndirect(pState, pVars, mDrawCounterClockwiseMeshes.count, mDrawCounterClockwiseMeshes.pBuffer.get(), 0, nullptr, 0);
            else pContext->drawIndirect(pState, pVars, mDrawCounterClockwiseMeshes.count, mDrawCounterClockwiseMeshes.pBuffer.get(), 0, nullptr, 0);
        }

        if (mDrawClockwiseMeshes.count)
        {
            if (overrideRS) pState->setRasterizerState(mpFrontClockwiseRS);
            if (isIndexed) pContext->drawIndexedIndirect(pState, pVars, mDrawClockwiseMeshes.count, mDrawClockwiseMeshes.pBuffer.get(), 0, nullptr, 0);
            else pContext->drawIndirect(pState, pVars, mDrawClockwiseMeshes.count, mDrawClockwiseMeshes.pBuffer.get(), 0, nullptr, 0);
        }

        if (overrideRS) pState->setRasterizerState(pCurrentRS);
    }

    void Scene::raytrace(RenderContext* pContext, RtProgram* pProgram, const std::shared_ptr<RtProgramVars>& pVars, uint3 dispatchDims)
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
        GraphicsProgram::SharedPtr pProgram = GraphicsProgram::createFromFile("Scene/SceneBlock.slang", "", "main", getSceneDefines());
        ParameterBlockReflection::SharedConstPtr pReflection = pProgram->getReflector()->getParameterBlock(kParameterBlockName);
        assert(pReflection);

        mpSceneBlock = ParameterBlock::create(pReflection);
        mpMeshesBuffer = Buffer::createStructured(mpSceneBlock[kMeshBufferName], (uint32_t)mMeshDesc.size(), Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false);
        mpMeshesBuffer->setName("Scene::mpMeshesBuffer");
        mpMeshInstancesBuffer = Buffer::createStructured(mpSceneBlock[kMeshInstanceBufferName], (uint32_t)mMeshInstanceData.size(), Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false);
        mpMeshInstancesBuffer->setName("Scene::mpMeshInstancesBuffer");

        mpMaterialsBuffer = Buffer::createStructured(mpSceneBlock[kMaterialsBufferName], (uint32_t)mMaterials.size(), Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false);
        mpMaterialsBuffer->setName("Scene::mpMaterialsBuffer");

        if (mLights.size())
        {
            mpLightsBuffer = Buffer::createStructured(mpSceneBlock[kLightsBufferName], (uint32_t)mLights.size(), Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false);
            mpLightsBuffer->setName("Scene::mpLightsBuffer");
        }
    }

    void Scene::uploadResources()
    {
        // Upload geometry
        mpMeshesBuffer->setBlob(mMeshDesc.data(), 0, sizeof(MeshDesc) * mMeshDesc.size());

        mpSceneBlock->setBuffer(kMeshInstanceBufferName, mpMeshInstancesBuffer);
        mpSceneBlock->setBuffer(kMeshBufferName, mpMeshesBuffer);
        mpSceneBlock->setBuffer(kLightsBufferName, mpLightsBuffer);
        mpSceneBlock->setBuffer(kMaterialsBufferName, mpMaterialsBuffer);
        if (hasIndexBuffer()) mpSceneBlock->setBuffer(kIndexBufferName, mpVao->getIndexBuffer());
        mpSceneBlock->setBuffer(kVertexBufferName, mpVao->getVertexBuffer(Scene::kStaticDataBufferIndex));
        mpSceneBlock->setBuffer(kPrevVertexBufferName, mpVao->getVertexBuffer(Scene::kPrevVertexBufferIndex));

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

        const auto& material = mMaterials[materialID];

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

    void Scene::uploadSelectedCamera()
    {
        getCamera()->setShaderData(mpSceneBlock[kCamera]);
    }

    void Scene::updateBounds()
    {
        const auto& globalMatrices = mpAnimationController->getGlobalMatrices();
        std::vector<BoundingBox> instanceBBs;
        instanceBBs.reserve(mMeshInstanceData.size());

        for (const auto& inst : mMeshInstanceData)
        {
            const BoundingBox& meshBB = mMeshBBs[inst.meshID];
            const glm::mat4& transform = globalMatrices[inst.globalMatrixID];
            instanceBBs.push_back(meshBB.transform(transform));
        }

        mSceneBB = instanceBBs.front();
        for (const BoundingBox& bb : instanceBBs)
        {
            mSceneBB = BoundingBox::fromUnion(mSceneBB, bb);
        }
    }

    void Scene::updateMeshInstances(bool forceUpdate)
    {
        bool dataChanged = false;
        const auto& globalMatrices = mpAnimationController->getGlobalMatrices();

        for (auto& inst : mMeshInstanceData)
        {
            uint32_t prevFlags = inst.flags;
            inst.flags = (uint32_t)MeshInstanceFlags::None;

            const glm::mat4& transform = globalMatrices[inst.globalMatrixID];
            if (doesTransformFlip(transform)) inst.flags |= (uint32_t)MeshInstanceFlags::Flipped;

            dataChanged |= (inst.flags != prevFlags);
        }

        if (forceUpdate || dataChanged)
        {
            // Make sure the scene data fits in the packed format.
            // TODO: If we run into the limits, use bits from the materialID field.
            if (globalMatrices.size() >= (1 << PackedMeshInstanceData::kMatrixBits)) throw std::exception("Number of transform matrices exceed the maximum");
            if (getMeshCount() >= (1 << PackedMeshInstanceData::kMeshBits)) throw std::exception("Number of meshes exceed the maximum");

            // Prepare packed mesh instance data.
            assert(mMeshInstanceData.size() > 0);
            mPackedMeshInstanceData.resize(mMeshInstanceData.size());

            for (size_t i = 0; i < mMeshInstanceData.size(); i++)
            {
                mPackedMeshInstanceData[i].pack(mMeshInstanceData[i]);
            }

            size_t byteSize = sizeof(PackedMeshInstanceData) * mPackedMeshInstanceData.size();
            assert(mpMeshInstancesBuffer && mpMeshInstancesBuffer->getSize() == byteSize);
            mpMeshInstancesBuffer->setBlob(mPackedMeshInstanceData.data(), 0, byteSize);
        }
    }

    void Scene::finalize()
    {
        sortMeshes();
        initResources();
        mpAnimationController->animate(gpDevice->getRenderContext(), 0); // Requires Scene block to exist
        updateMeshInstances(true);
        updateBounds();
        createDrawList();
        if (mCameras.size() == 0)
        {
            // Create a new camera to use in the event of a scene with no cameras
            mCameras.push_back(Camera::create());
            resetCamera();
        }
        setCameraController(mCamCtrlType);
        initializeCameras();
        uploadSelectedCamera();
        addViewpoint();
        updateLights(true);
        updateEnvMap(true);
        updateMaterials(true);
        uploadResources(); // Upload data after initialization is complete
        updateGeometryStats();
        updateLightStats();
        prepareUI();
    }

    void Scene::initializeCameras()
    {
        for (auto& camera : mCameras)
        {
            updateAnimatable(*camera, *mpAnimationController, true);
            camera->beginFrame();
        }
    }

    void Scene::prepareUI()
    {
        for (uint32_t camId = 0; camId < (uint32_t)mCameras.size(); camId++)
        {
            mCameraList.push_back({ camId, mCameras[camId]->getName() });
        }
    }

    void Scene::updateGeometryStats()
    {
        auto& s = mSceneStats;

        s.uniqueVertexCount = 0;
        s.uniqueTriangleCount = 0;
        s.instancedVertexCount = 0;
        s.instancedTriangleCount = 0;

        for (uint32_t meshID = 0; meshID < getMeshCount(); meshID++)
        {
            const auto& mesh = getMesh(meshID);
            s.uniqueVertexCount += mesh.vertexCount;
            s.uniqueTriangleCount += mesh.getTriangleCount();
        }
        for (uint32_t instanceID = 0; instanceID < getMeshInstanceCount(); instanceID++)
        {
            const auto& instance = getMeshInstance(instanceID);
            const auto& mesh = getMesh(instance.meshID);
            s.instancedVertexCount += mesh.vertexCount;
            s.instancedTriangleCount += mesh.getTriangleCount();
        }
    }

    void Scene::updateRaytracingStats()
    {
        auto& s = mSceneStats;

        s.blasCount = mBlasData.size();
        s.blasCompactedCount = 0;
        s.blasMemoryInBytes = 0;

        for (const auto& blas : mBlasData)
        {
            if (blas.useCompaction) s.blasCompactedCount++;
            s.blasMemoryInBytes += blas.blasByteSize;
        }
    }

    void Scene::updateLightStats()
    {
        auto& s = mSceneStats;

        s.activeLightCount = 0;
        s.totalLightCount = 0;
        s.pointLightCount = 0;
        s.directionalLightCount = 0;
        s.rectLightCount = 0;
        s.sphereLightCount = 0;
        s.distantLightCount = 0;

        for (const auto& light : mLights)
        {
            if (light->isActive()) s.activeLightCount++;
            s.totalLightCount++;

            switch (light->getType())
            {
            case LightType::Point:
                s.pointLightCount++;
                break;
            case LightType::Directional:
                s.directionalLightCount++;
                break;
            case LightType::Rect:
                s.rectLightCount++;
                break;
            case LightType::Sphere:
                s.sphereLightCount++;
                break;
            case LightType::Distant:
                s.distantLightCount++;
                break;
            }
        }
    }

    bool Scene::updateAnimatable(Animatable& animatable, const AnimationController& controller, bool force)
    {
        uint32_t nodeID = animatable.getNodeID();

        // It is possible for this to be called on an object with no associated node in the scene graph (kInvalidNode),
        // e.g. non-animated lights. This check ensures that we return immediately instead of trying to check
        // matrices for a non-existent node.
        if (nodeID == kInvalidNode) return false;

        if (force || (animatable.hasAnimation() && animatable.isAnimated()))
        {
            if (!controller.didMatrixChanged(nodeID) && !force) return false;

            glm::mat4 transform = controller.getGlobalMatrices()[nodeID];
            animatable.updateFromAnimation(transform);
            return true;
        }
        return false;
    }

    Scene::UpdateFlags Scene::updateSelectedCamera(bool forceUpdate)
    {
        auto camera = mCameras[mSelectedCamera];

        if (forceUpdate || (camera->hasAnimation() && camera->isAnimated()))
        {
            updateAnimatable(*camera, *mpAnimationController, forceUpdate);
        }
        else
        {
            mpCamCtrl->update();
        }

        UpdateFlags flags = UpdateFlags::None;
        auto cameraChanges = camera->beginFrame();
        if (mCameraSwitched || cameraChanges != Camera::Changes::None)
        {
            uploadSelectedCamera();
            if (is_set(cameraChanges, Camera::Changes::Movement)) flags |= UpdateFlags::CameraMoved;
            if ((cameraChanges & (~Camera::Changes::Movement)) != Camera::Changes::None) flags |= UpdateFlags::CameraPropertiesChanged;
            if (mCameraSwitched) flags |= UpdateFlags::CameraSwitched;
        }
        mCameraSwitched = false;
        return flags;
    }

    Scene::UpdateFlags Scene::updateLights(bool forceUpdate)
    {
        Light::Changes combinedChanges = Light::Changes::None;

        // Animate lights and get list of changes.
        for (const auto& light : mLights)
        {
            updateAnimatable(*light, *mpAnimationController, forceUpdate);
            auto changes = light->beginFrame();
            combinedChanges |= changes;
        }

        // Update changed lights.
        uint32_t lightCount = 0;

        for (const auto& light : mLights)
        {
            if (!light->isActive()) continue;
            auto changes = light->getChanges();

            if (changes != Light::Changes::None || is_set(combinedChanges, Light::Changes::Active) || forceUpdate)
            {
                // TODO: This is slow since the buffer is not CPU writable. Copy into CPU buffer and upload once instead.
                mpLightsBuffer->setElement(lightCount, light->getData());
            }

            lightCount++;
        }

        if (combinedChanges != Light::Changes::None || forceUpdate)
        {
            mpSceneBlock["lightCount"] = lightCount;
            updateLightStats();
        }

        // Compute update flags.
        UpdateFlags flags = UpdateFlags::None;
        if (is_set(combinedChanges, Light::Changes::Intensity)) flags |= UpdateFlags::LightIntensityChanged;
        if (is_set(combinedChanges, Light::Changes::Position)) flags |= UpdateFlags::LightsMoved;
        if (is_set(combinedChanges, Light::Changes::Direction)) flags |= UpdateFlags::LightsMoved;
        if (is_set(combinedChanges, Light::Changes::Active)) flags |= UpdateFlags::LightCountChanged;
        const Light::Changes otherChanges = ~(Light::Changes::Intensity | Light::Changes::Position | Light::Changes::Direction | Light::Changes::Active);
        if ((combinedChanges & otherChanges) != Light::Changes::None) flags |= UpdateFlags::LightPropertiesChanged;

        return flags;
    }

    Scene::UpdateFlags Scene::updateEnvMap(bool forceUpdate)
    {
        UpdateFlags flags = UpdateFlags::None;

        if (mpEnvMap)
        {
            auto envMapChanges = mpEnvMap->beginFrame();
            if (envMapChanges != EnvMap::Changes::None || forceUpdate)
            {
                if (envMapChanges != EnvMap::Changes::None) flags |= UpdateFlags::EnvMapChanged;
                mpEnvMap->setShaderData(mpSceneBlock[kEnvMap]);
            }
        }

        return flags;
    }

    Scene::UpdateFlags Scene::updateMaterials(bool forceUpdate)
    {
        UpdateFlags flags = UpdateFlags::None;

        // Early out if no materials have changed
        if (!forceUpdate && Material::getGlobalUpdates() == Material::UpdateFlags::None) return flags;

        for (uint32_t materialId = 0; materialId < (uint32_t)mMaterials.size(); ++materialId)
        {
            auto& material = mMaterials[materialId];
            auto materialUpdates = material->getUpdates();
            if (forceUpdate || materialUpdates != Material::UpdateFlags::None)
            {
                material->clearUpdates();
                uploadMaterial(materialId);
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

        mUpdates |= updateSelectedCamera(false);
        mUpdates |= updateLights(false);
        mUpdates |= updateEnvMap(false);
        mUpdates |= updateMaterials(false);
        pContext->flush();
        if (is_set(mUpdates, UpdateFlags::MeshesMoved))
        {
            mTlasCache.clear();
            updateMeshInstances(false);
        }

        // If a transform in the scene changed, update BLASes with skinned meshes
        if (mBlasData.size() && mHasSkinnedMesh && is_set(mUpdates, UpdateFlags::SceneGraphChanged))
        {
            mTlasCache.clear();
            buildBlas(pContext);
        }

        // Update light collection
        if (mpLightCollection && mpLightCollection->update(pContext)) mUpdates |= UpdateFlags::LightCollectionChanged;

        if (mRenderSettings != mPrevRenderSettings)
        {
            mUpdates |= UpdateFlags::RenderSettingsChanged;
            mPrevRenderSettings = mRenderSettings;
        }

        return mUpdates;
    }

    void Scene::renderUI(Gui::Widgets& widget)
    {
        if (mpAnimationController->hasAnimations())
        {
            bool isEnabled = mpAnimationController->isEnabled();
            if (widget.checkbox("Animate Scene", isEnabled)) mpAnimationController->setEnabled(isEnabled);
        }

        auto camera = mCameras[mSelectedCamera];
        if (camera->hasAnimation())
        {
            bool isAnimated = camera->isAnimated();
            if (widget.checkbox("Animate Camera", isAnimated)) camera->setIsAnimated(isAnimated);
        }

        if (widget.var("Camera Speed", mCameraSpeed, 0.f, std::numeric_limits<float>::max(), 0.01f))
        {
            mpCamCtrl->setCameraSpeed(mCameraSpeed);
        }

        if (mCameraList.size() > 1)
        {
            uint32_t camIndex = mSelectedCamera;
            if (widget.dropdown("Selected Camera", mCameraList, camIndex)) selectCamera(camIndex);
        }

        if (widget.button("Add Viewpoint")) addViewpoint();

        if (mViewpoints.size() > 1)
        {
            if (widget.button("Remove Viewpoint", true)) removeViewpoint();

            Gui::DropdownList viewpoints;
            viewpoints.push_back({ 0, "Default Viewpoint" });
            for (uint32_t viewId = 1; viewId < (uint32_t)mViewpoints.size(); viewId++)
            {
                viewpoints.push_back({ viewId, "Viewpoint " + std::to_string(viewId) });
            }
            uint32_t viewIndex = mCurrentViewpoint;
            if (widget.dropdown("Viewpoints", viewpoints, viewIndex)) selectViewpoint(viewIndex);
        }

        if (auto cameraGroup = widget.group("Camera"))
        {
            camera->renderUI(cameraGroup);
        }

        if (auto lightingGroup = widget.group("Render Settings"))
        {
            lightingGroup.checkbox("Use environment light", mRenderSettings.useEnvLight);
            lightingGroup.tooltip("This enables using the environment map as a distant light source.", true);

            lightingGroup.checkbox("Use analytic lights", mRenderSettings.useAnalyticLights);
            lightingGroup.tooltip("This enables using analytic lights.", true);

            lightingGroup.checkbox("Emissive", mRenderSettings.useEmissiveLights);
            lightingGroup.tooltip("This enables using emissive triangles as lights.", true);
        }

        if (mpEnvMap)
        {
            if (auto envMapGroup = widget.group("EnvMap"))
            {
                mpEnvMap->renderUI(envMapGroup);
            }
        }

        if (auto lightsGroup = widget.group("Lights"))
        {
            uint32_t lightID = 0;
            for (auto& light : mLights)
            {
                auto name = std::to_string(lightID) + ": " + light->getName();
                if (auto lightGroup = lightsGroup.group(name))
                {
                    light->renderUI(lightGroup);
                }
                lightID++;
            }
        }

        if (auto materialsGroup = widget.group("Materials"))
        {
            uint32_t materialID = 0;
            for (auto& material : mMaterials)
            {
                auto name = std::to_string(materialID) + ": " + material->getName();
                if (auto materialGroup = materialsGroup.group(name))
                {
                    if (material->renderUI(materialGroup)) uploadMaterial(materialID);
                }
                materialID++;
            }
        }

        if (auto statsGroup = widget.group("Statistics"))
        {
            std::ostringstream oss;

            // Geometry stats.
            oss << "Geometry stats:" << std::endl
                << "  Mesh count: " << getMeshCount() << std::endl
                << "  Mesh instance count: " << getMeshInstanceCount() << std::endl
                << "  Transform matrix count: " << getAnimationController()->getGlobalMatrices().size() << std::endl
                << "  Unique triangle count: " << mSceneStats.uniqueTriangleCount << std::endl
                << "  Unique vertex count: " << mSceneStats.uniqueVertexCount << std::endl
                << "  Instanced triangle count: " << mSceneStats.instancedTriangleCount << std::endl
                << "  Instanced vertex count: " << mSceneStats.instancedVertexCount << std::endl
                << std::endl;

            // Raytracing stats.
            oss << "Raytracing stats:" << std::endl
                << "  BLAS count (total): " << mSceneStats.blasCount << std::endl
                << "  BLAS count (compacted): " << mSceneStats.blasCompactedCount << std::endl
                << "  BLAS memory (bytes): " << mSceneStats.blasMemoryInBytes << std::endl
                << std::endl;

            // Material stats.
            oss << "Materials stats:" << std::endl
                << "  Material count: " << getMaterialCount() << std::endl
                << std::endl;

            // Analytic light stats.
            oss << "Analytic light stats:" << std::endl
                << "  Active light count: " << mSceneStats.activeLightCount << std::endl
                << "  Total light count: " << mSceneStats.totalLightCount << std::endl
                << "  Point light count: " << mSceneStats.pointLightCount << std::endl
                << "  Directional light count: " << mSceneStats.directionalLightCount << std::endl
                << "  Rect light count: " << mSceneStats.rectLightCount << std::endl
                << "  Sphere light count: " << mSceneStats.sphereLightCount << std::endl
                << "  Distant light count: " << mSceneStats.distantLightCount << std::endl
                << std::endl;

            // Emissive light stats.
            oss << "Emissive light stats:" << std::endl;
            if (mpLightCollection)
            {
                auto stats = mpLightCollection->getStats();
                oss << "  Active triangle count: " << stats.trianglesActive << std::endl
                    << "  Active uniform triangle count: " << stats.trianglesActiveUniform << std::endl
                    << "  Active textured triangle count: " << stats.trianglesActiveTextured << std::endl
                    << "  Details:" << std::endl
                    << "    Total mesh count: " << stats.meshLightCount << std::endl
                    << "    Textured mesh count: " << stats.meshesTextured << std::endl
                    << "    Total triangle count: " << stats.triangleCount << std::endl
                    << "    Texture triangle count: " << stats.trianglesTextured << std::endl
                    << "    Culled triangle count: " << stats.trianglesCulled << std::endl;
            }
            else
            {
                oss << "  N/A" << std::endl;
            }
            oss << std::endl;

            // Environment map stats.
            oss << "Environment map:" << std::endl;
            if (mpEnvMap)
            {
                oss << "  Filename: " << mpEnvMap->getFilename() << std::endl;
                oss << "  Resolution: " << mpEnvMap->getEnvMap()->getWidth() << "x" << mpEnvMap->getEnvMap()->getHeight() << std::endl;
            }
            else
            {
                oss << "  N/A" << std::endl;
            }
            oss << std::endl;

            statsGroup.text(oss.str());
        }

        // Filtering mode
        // Camera controller
    }

    bool Scene::useEnvBackground() const
    {
        return mpEnvMap != nullptr;
    }

    bool Scene::useEnvLight() const
    {
        return mRenderSettings.useEnvLight && mpEnvMap != nullptr;
    }

    bool Scene::useAnalyticLights() const
    {
        return mRenderSettings.useAnalyticLights && mLights.empty() == false;
    }

    bool Scene::useEmissiveLights() const
    {
        return mRenderSettings.useEmissiveLights && mpLightCollection != nullptr && mpLightCollection->getActiveLightCount() > 0;
    }

    void Scene::setCamera(const Camera::SharedPtr& pCamera)
    {
        auto name = pCamera->getName();
        for (uint index = 0; index < mCameras.size(); index++)
        {
            if (mCameras[index]->getName() == name)
            {
                selectCamera(index);
                return;
            }
        }
        logWarning("Selected camera " + name + " does not exist.");
        pybind11::print("Selected camera", name, "does not exist.");
    }

    void Scene::selectCamera(uint32_t index)
    {
        if (index == mSelectedCamera) return;
        if (index >= mCameras.size())
        {
            logWarning("Selected camera index " + std::to_string(index) + " is invalid.");
            pybind11::print("Selected camera index", index, "is invalid.");
            return;
        }

        mSelectedCamera = index;
        mCameraSwitched = true;
        setCameraController(mCamCtrlType);
        updateSelectedCamera(false);
    }

    void Scene::resetCamera(bool resetDepthRange)
    {
        auto camera = getCamera();
        float radius = length(mSceneBB.extent);
        camera->setPosition(mSceneBB.center);
        camera->setTarget(mSceneBB.center + float3(0, 0, -1));
        camera->setUpVector(float3(0, 1, 0));

        if (resetDepthRange)
        {
            float nearZ = std::max(0.1f, radius / 750.0f);
            float farZ = radius * 50;
            camera->setDepthRange(nearZ, farZ);
        }
    }

    void Scene::setCameraSpeed(float speed)
    {
        mCameraSpeed = clamp(speed, 0.f, std::numeric_limits<float>::max());
        mpCamCtrl->setCameraSpeed(speed);
    }

    void Scene::addViewpoint()
    {
        auto camera = getCamera();
        addViewpoint(camera->getPosition(), camera->getTarget(), camera->getUpVector(), mSelectedCamera);
    }

    void Scene::addViewpoint(const float3& position, const float3& target, const float3& up, uint32_t cameraIndex)
    {
        Viewpoint viewpoint = { cameraIndex, position, target, up };
        mViewpoints.push_back(viewpoint);
        mCurrentViewpoint = (uint32_t)mViewpoints.size() - 1;
    }

    void Scene::removeViewpoint()
    {
        if (mCurrentViewpoint == 0)
        {
            logWarning("Cannot remove default viewpoint");
            return;
        }
        mViewpoints.erase(mViewpoints.begin() + mCurrentViewpoint);
        mCurrentViewpoint = std::min(mCurrentViewpoint, (uint32_t)mViewpoints.size() - 1);
    }

    void Scene::selectViewpoint(uint32_t index)
    {
        if (index >= mViewpoints.size())
        {
            logWarning("Viewpoint does not exist");
            return;
        }

        auto& viewpoint = mViewpoints[index];
        selectCamera(viewpoint.index);
        auto camera = getCamera();
        camera->setPosition(viewpoint.position);
        camera->setTarget(viewpoint.target);
        camera->setUpVector(viewpoint.up);
        mCurrentViewpoint = index;
    }

    Material::SharedPtr Scene::getMaterialByName(const std::string& name) const
    {
        for (const auto& m : mMaterials)
        {
            if (m->getName() == name) return m;
        }

        return nullptr;
    }

    Light::SharedPtr Scene::getLightByName(const std::string& name) const
    {
        for (const auto& l : mLights)
        {
            if (l->getName() == name) return l;
        }

        return nullptr;
    }

    void Scene::toggleAnimations(bool animate)
    {
        for (auto& light : mLights) light->setIsAnimated(animate);
        for (auto& camera : mCameras) camera->setIsAnimated(animate);
        mpAnimationController->setEnabled(animate);
    }

    void Scene::setBlasUpdateMode(UpdateMode mode)
    {
        if (mode != mBlasUpdateMode) mRebuildBlas = true;
        mBlasUpdateMode = mode;
    }

    void Scene::createDrawList()
    {
        auto pMatricesBuffer = mpSceneBlock->getBuffer("worldMatrices");
        const glm::mat4* matrices = (glm::mat4*)pMatricesBuffer->map(Buffer::MapType::Read); // #SCENEV2 This will cause the pipeline to flush and sync, but it's probably not too bad as this only happens once

        auto createBuffers = [&](const auto& drawClockwiseMeshes, const auto& drawCounterClockwiseMeshes)
        {
            // Create the draw-indirect buffer
            if (drawCounterClockwiseMeshes.size())
            {
                mDrawCounterClockwiseMeshes.pBuffer = Buffer::create(sizeof(drawCounterClockwiseMeshes[0]) * drawCounterClockwiseMeshes.size(), Resource::BindFlags::IndirectArg, Buffer::CpuAccess::None, drawCounterClockwiseMeshes.data());
                mDrawCounterClockwiseMeshes.pBuffer->setName("Scene::mDrawCounterClockwiseMeshes::pBuffer");
                mDrawCounterClockwiseMeshes.count = (uint32_t)drawCounterClockwiseMeshes.size();
            }

            if (drawClockwiseMeshes.size())
            {
                mDrawClockwiseMeshes.pBuffer = Buffer::create(sizeof(drawClockwiseMeshes[0]) * drawClockwiseMeshes.size(), Resource::BindFlags::IndirectArg, Buffer::CpuAccess::None, drawClockwiseMeshes.data());
                mDrawClockwiseMeshes.pBuffer->setName("Scene::mDrawClockwiseMeshes::pBuffer");
                mDrawClockwiseMeshes.count = (uint32_t)drawClockwiseMeshes.size();
            }

            size_t drawCount = drawClockwiseMeshes.size() + drawCounterClockwiseMeshes.size();
            assert(drawCount <= std::numeric_limits<uint32_t>::max());
        };

        if (hasIndexBuffer())
        {
            std::vector<D3D12_DRAW_INDEXED_ARGUMENTS> drawClockwiseMeshes, drawCounterClockwiseMeshes;

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
            createBuffers(drawClockwiseMeshes, drawCounterClockwiseMeshes);
        }
        else
        {
            std::vector<D3D12_DRAW_ARGUMENTS> drawClockwiseMeshes, drawCounterClockwiseMeshes;

            for (const auto& instance : mMeshInstanceData)
            {
                const auto& mesh = mMeshDesc[instance.meshID];
                const auto& transform = matrices[instance.globalMatrixID];
                assert(mesh.indexCount == 0);

                D3D12_DRAW_ARGUMENTS draw;
                draw.VertexCountPerInstance = mesh.vertexCount;
                draw.InstanceCount = 1;
                draw.StartVertexLocation = mesh.vbOffset;
                draw.StartInstanceLocation = (uint32_t)(drawClockwiseMeshes.size() + drawCounterClockwiseMeshes.size());

                (doesTransformFlip(transform)) ? drawClockwiseMeshes.push_back(draw) : drawCounterClockwiseMeshes.push_back(draw);
            }
            createBuffers(drawClockwiseMeshes, drawCounterClockwiseMeshes);
        }
    }

    void Scene::sortMeshes()
    {
        // We first sort meshes into groups with the same transform.
        // The mesh instances list is then reordered to match this order.
        //
        // For ray tracing, we create one BLAS per mesh group and the mesh instances
        // can therefore be directly indexed by [InstanceID() + GeometryIndex()].
        // This avoids the need to have a lookup table from hit IDs to mesh instance.

        // Build a list of mesh instance indices per mesh.
        std::vector<std::vector<size_t>> instanceLists(mMeshDesc.size());
        for (size_t i = 0; i < mMeshInstanceData.size(); i++)
        {
            assert(mMeshInstanceData[i].meshID < instanceLists.size());
            instanceLists[mMeshInstanceData[i].meshID].push_back(i);
        }

        // The non-instanced meshes are grouped based on what global matrix ID their transform is.
        std::unordered_map<uint32_t, std::vector<uint32_t>> nodeToMeshList;
        for (uint32_t meshId = 0; meshId < (uint32_t)instanceLists.size(); meshId++)
        {
            const auto& instanceList = instanceLists[meshId];
            if (instanceList.size() > 1) continue; // Only processing non-instanced meshes here

            assert(instanceList.size() == 1);
            uint32_t globalMatrixId = mMeshInstanceData[instanceList[0]].globalMatrixID;
            nodeToMeshList[globalMatrixId].push_back(meshId);
        }

        // Build final result. Format is a list of Mesh ID's per mesh group.

        // This should currently only be run on scene initialization.
        assert(mMeshGroups.empty());

        // Non-instanced meshes were sorted above so just copy each list.
        for (const auto& it : nodeToMeshList) mMeshGroups.push_back({ it.second });

        // Meshes that have multiple instances go in their own groups.
        for (uint32_t meshId = 0; meshId < (uint32_t)instanceLists.size(); meshId++)
        {
            const auto& instanceList = instanceLists[meshId];
            if (instanceList.size() == 1) continue; // Only processing instanced meshes here
            mMeshGroups.push_back({ std::vector<uint32_t>({ meshId }) });
        }

        // Calculate mapping from new mesh instance ID to existing instance index.
        // Here, just append existing instance ID's in order they appear in the mesh groups.
        std::vector<size_t> instanceMapping;
        for (const auto& meshGroup : mMeshGroups)
        {
            for (const uint32_t meshId : meshGroup.meshList)
            {
                const auto& instanceList = instanceLists[meshId];
                for (size_t idx : instanceList)
                {
                    instanceMapping.push_back(idx);
                }
            }
        }
        assert(instanceMapping.size() == mMeshInstanceData.size());
        {
            // Check that all indices exist
            std::set<size_t> instanceIndices(instanceMapping.begin(), instanceMapping.end());
            assert(instanceIndices.size() == mMeshInstanceData.size());
        }

        // Now reorder mMeshInstanceData based on the new mapping.
        // We'll make a copy of the existing data first, and the populate the array.
        std::vector<MeshInstanceData> prevInstanceData = mMeshInstanceData;
        for (size_t i = 0; i < mMeshInstanceData.size(); i++)
        {
            assert(instanceMapping[i] < prevInstanceData.size());
            mMeshInstanceData[i] = prevInstanceData[instanceMapping[i]];
        }

        // Create mapping of meshes to their instances.
        mMeshIdToInstanceIds.clear();
        mMeshIdToInstanceIds.resize(mMeshDesc.size());
        for (uint32_t instId = 0; instId < (uint32_t)mMeshInstanceData.size(); instId++)
        {
            mMeshIdToInstanceIds[mMeshInstanceData[instId].meshID].push_back(instId);
        }
    }

    void Scene::initGeomDesc()
    {
        assert(mBlasData.empty());

        const VertexBufferLayout::SharedConstPtr& pVbLayout = mpVao->getVertexLayout()->getBufferLayout(kStaticDataBufferIndex);
        const Buffer::SharedPtr& pVb = mpVao->getVertexBuffer(kStaticDataBufferIndex);
        const Buffer::SharedPtr& pIb = mpVao->getIndexBuffer();

        assert(mMeshGroups.size() > 0);
        mBlasData.resize(mMeshGroups.size());
        mRebuildBlas = true;
        mHasSkinnedMesh = false;

        for (size_t i = 0; i < mBlasData.size(); i++)
        {
            const auto& meshList = mMeshGroups[i].meshList;
            auto& blas = mBlasData[i];
            auto& geomDescs = blas.geomDescs;
            geomDescs.resize(meshList.size());

            for (size_t j = 0; j < meshList.size(); j++)
            {
                const MeshDesc& mesh = mMeshDesc[meshList[j]];
                blas.hasSkinnedMesh |= mMeshHasDynamicData[meshList[j]];

                D3D12_RAYTRACING_GEOMETRY_DESC& desc = geomDescs[j];
                desc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
                desc.Triangles.Transform3x4 = 0;

                // If this is an opaque mesh, set the opaque flag
                const auto& material = mMaterials[mesh.materialID];
                bool opaque = material->getAlphaMode() == AlphaModeOpaque;
                desc.Flags = opaque ? D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE : D3D12_RAYTRACING_GEOMETRY_FLAG_NONE;

                // Set the position data
                desc.Triangles.VertexBuffer.StartAddress = pVb->getGpuAddress() + (mesh.vbOffset * pVbLayout->getStride());
                desc.Triangles.VertexBuffer.StrideInBytes = pVbLayout->getStride();
                desc.Triangles.VertexCount = mesh.vertexCount;
                desc.Triangles.VertexFormat = getDxgiFormat(pVbLayout->getElementFormat(0));

                // Set index data
                if (pIb)
                {
                    desc.Triangles.IndexBuffer = pIb->getGpuAddress() + (mesh.ibOffset * getFormatBytesPerBlock(mpVao->getIndexBufferFormat()));
                    desc.Triangles.IndexCount = mesh.indexCount;
                    desc.Triangles.IndexFormat = getDxgiFormat(mpVao->getIndexBufferFormat());
                }
                else
                {
                    assert(mesh.indexCount == 0);
                    desc.Triangles.IndexBuffer = NULL;
                    desc.Triangles.IndexCount = 0;
                    desc.Triangles.IndexFormat = DXGI_FORMAT_UNKNOWN;
                }
            }

            mHasSkinnedMesh |= blas.hasSkinnedMesh;
        }
    }

    void Scene::buildBlas(RenderContext* pContext)
    {
        PROFILE("buildBlas");

        // Add barriers for the VB and IB which will be accessed by the build.
        const Buffer::SharedPtr& pVb = mpVao->getVertexBuffer(kStaticDataBufferIndex);
        const Buffer::SharedPtr& pIb = mpVao->getIndexBuffer();
        pContext->resourceBarrier(pVb.get(), Resource::State::NonPixelShader);
        if (pIb) pContext->resourceBarrier(pIb.get(), Resource::State::NonPixelShader);

        // On the first time, or if a full rebuild is necessary we will:
        // - Update all build inputs and prebuild info
        // - Calculate total intermediate buffer sizes
        // - Build all BLASes into an intermediate buffer
        // - Calculate total compacted buffer size
        // - Compact/clone all BLASes to their final location
        if (mRebuildBlas)
        {
            uint64_t totalMaxBlasSize = 0;
            uint64_t totalScratchSize = 0;

            for (auto& blas : mBlasData)
            {
                // Determine how BLAS build/update should be done.
                // The default choice is to compact all static BLASes and those that don't need to be rebuilt every frame. For those compaction just adds overhead.
                // TODO: Add compaction on/off switch for profiling.
                // TODO: Disable compaction for skinned meshes if update performance becomes a problem.
                blas.updateMode = mBlasUpdateMode;
                blas.useCompaction = !blas.hasSkinnedMesh || blas.updateMode != UpdateMode::Rebuild;

                // Setup build parameters.
                D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS& inputs = blas.buildInputs;
                inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
                inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
                inputs.NumDescs = (uint32_t)blas.geomDescs.size();
                inputs.pGeometryDescs = blas.geomDescs.data();
                inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_NONE;

                // Add necessary flags depending on settings.
                if (blas.useCompaction)
                {
                    inputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION;
                }
                if (blas.hasSkinnedMesh && blas.updateMode == UpdateMode::Refit)
                {
                    inputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;
                }

                // Set optional performance hints.
                // TODO: Set FAST_BUILD for skinned meshes if update/rebuild performance becomes a problem.
                // TODO: Add FAST_TRACE on/off switch for profiling. It is disabled by default as it is scene-dependent.
                //if (!blas.hasSkinnedMesh)
                //{
                //    inputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
                //}

                // Get prebuild info.
                GET_COM_INTERFACE(gpDevice->getApiHandle(), ID3D12Device5, pDevice5);
                pDevice5->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &blas.prebuildInfo);

                // Figure out the padded allocation sizes to have proper alignement.
                uint64_t paddedMaxBlasSize = align_to(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT, blas.prebuildInfo.ResultDataMaxSizeInBytes);
                blas.blasByteOffset = totalMaxBlasSize;
                totalMaxBlasSize += paddedMaxBlasSize;

                uint64_t scratchSize = std::max(blas.prebuildInfo.ScratchDataSizeInBytes, blas.prebuildInfo.UpdateScratchDataSizeInBytes);
                uint64_t paddedScratchSize = align_to(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT, scratchSize);
                blas.scratchByteOffset = totalScratchSize;
                totalScratchSize += paddedScratchSize;
            }

            // Allocate intermediate buffers and scratch buffer.
            // The scratch buffer we'll retain because it's needed for subsequent rebuilds and updates.
            // TODO: Save memory by reducing the scratch buffer to the minimum required for the dynamic objects.
            if (mpBlasScratch == nullptr || mpBlasScratch->getSize() < totalScratchSize)
            {
                mpBlasScratch = Buffer::create(totalScratchSize, Buffer::BindFlags::UnorderedAccess, Buffer::CpuAccess::None);
                mpBlasScratch->setName("Scene::mpBlasScratch");
            }
            else
            {
                // If we didn't need to reallocate, just insert a barrier so it's safe to use.
                pContext->uavBarrier(mpBlasScratch.get());
            }

            Buffer::SharedPtr pDestBuffer = Buffer::create(totalMaxBlasSize, Buffer::BindFlags::AccelerationStructure, Buffer::CpuAccess::None);

            const size_t postBuildInfoSize = sizeof(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE_DESC);
            static_assert(postBuildInfoSize == sizeof(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_CURRENT_SIZE_DESC));
            Buffer::SharedPtr pPostbuildInfoBuffer = Buffer::create(mBlasData.size() * postBuildInfoSize, Buffer::BindFlags::None, Buffer::CpuAccess::Read);

            // Build the BLASes into the intermediate destination buffer.
            // We output postbuild info to a separate buffer to find out the final size requirements.
            assert(pDestBuffer && pPostbuildInfoBuffer && mpBlasScratch);
            uint64_t postBuildInfoOffset = 0;

            for (const auto& blas : mBlasData)
            {
                D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC asDesc = {};
                asDesc.Inputs = blas.buildInputs;
                asDesc.ScratchAccelerationStructureData = mpBlasScratch->getGpuAddress() + blas.scratchByteOffset;
                asDesc.DestAccelerationStructureData = pDestBuffer->getGpuAddress() + blas.blasByteOffset;

                // Need to find out the the postbuild compacted BLAS size to know the final allocation size.
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC postbuildInfoDesc = {};
                postbuildInfoDesc.InfoType = blas.useCompaction ? D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE : D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_CURRENT_SIZE;
                postbuildInfoDesc.DestBuffer = pPostbuildInfoBuffer->getGpuAddress() + postBuildInfoOffset;
                postBuildInfoOffset += postBuildInfoSize;

                GET_COM_INTERFACE(pContext->getLowLevelData()->getCommandList(), ID3D12GraphicsCommandList4, pList4);
                pList4->BuildRaytracingAccelerationStructure(&asDesc, 1, &postbuildInfoDesc);
            }

            // Release scratch buffer if there is no animated content. We will not need it.
            if (!mHasSkinnedMesh) mpBlasScratch.reset();

            // Read back the calculated final size requirements for each BLAS.
            // For this purpose we have to flush and map the postbuild info buffer for readback.
            // TODO: We could copy to a staging buffer first and wait on a GPU fence for when it's ready.
            // But there is no other work to do inbetween so it probably wouldn't help. This is only done once at startup anyway.
            pContext->flush(true);
            const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE_DESC* postBuildInfo =
                (const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE_DESC*) pPostbuildInfoBuffer->map(Buffer::MapType::Read);

            uint64_t totalBlasSize = 0;
            for (size_t i = 0; i < mBlasData.size(); i++)
            {
                auto& blas = mBlasData[i];
                blas.blasByteSize = postBuildInfo[i].CompactedSizeInBytes;
                assert(blas.blasByteSize <= blas.prebuildInfo.ResultDataMaxSizeInBytes);
                uint64_t paddedBlasSize = align_to(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT, blas.blasByteSize);
                totalBlasSize += paddedBlasSize;
            }
            pPostbuildInfoBuffer->unmap();

            // Allocate final BLAS buffer.
            if (mpBlas == nullptr || mpBlas->getSize() < totalBlasSize)
            {
                mpBlas = Buffer::create(totalBlasSize, Buffer::BindFlags::AccelerationStructure, Buffer::CpuAccess::None);
                mpBlas->setName("Scene::mpBlas");
            }
            else
            {
                // If we didn't need to reallocate, just insert a barrier so it's safe to use.
                pContext->uavBarrier(mpBlas.get());
            }

            // Insert barriers for the intermediate buffer. This is probably not necessary since we flushed above, but it's not going to hurt.
            pContext->uavBarrier(pDestBuffer.get());

            // Compact/clone all BLASes to their final location.
            uint64_t blasOffset = 0;
            for (auto& blas : mBlasData)
            {
                GET_COM_INTERFACE(pContext->getLowLevelData()->getCommandList(), ID3D12GraphicsCommandList4, pList4);
                pList4->CopyRaytracingAccelerationStructure(
                    mpBlas->getGpuAddress() + blasOffset,
                    pDestBuffer->getGpuAddress() + blas.blasByteOffset,
                    blas.useCompaction ? D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_COMPACT : D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_CLONE);

                uint64_t paddedBlasSize = align_to(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT, blas.blasByteSize);
                blas.blasByteOffset = blasOffset;
                blasOffset += paddedBlasSize;
            }
            assert(blasOffset == totalBlasSize);

            // Insert barrier. The BLAS buffer is now ready for use.
            pContext->uavBarrier(mpBlas.get());

            updateRaytracingStats();
            mRebuildBlas = false;

            return;
        }

        // If we get here, all BLASes have previously been built and compacted. We will:
        // - Early out if there are no animated meshes.
        // - Update or rebuild in-place the ones that are animated.
        assert(!mRebuildBlas);
        if (mHasSkinnedMesh == false) return;

        // Insert barriers. The buffers are now ready to be written to.
        assert(mpBlas && mpBlasScratch);
        pContext->uavBarrier(mpBlas.get());
        pContext->uavBarrier(mpBlasScratch.get());

        for (const auto& blas : mBlasData)
        {
            // Skip updating BLASes not containing skinned meshes.
            if (!blas.hasSkinnedMesh) continue;

            // Build/update BLAS.
            D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC asDesc = {};
            asDesc.Inputs = blas.buildInputs;
            asDesc.ScratchAccelerationStructureData = mpBlasScratch->getGpuAddress() + blas.scratchByteOffset;
            asDesc.DestAccelerationStructureData = mpBlas->getGpuAddress() + blas.blasByteOffset;

            if (blas.updateMode == UpdateMode::Refit)
            {
                // Set source address to destination address to update in place.
                asDesc.SourceAccelerationStructureData = asDesc.DestAccelerationStructureData;
                asDesc.Inputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
            }
            else
            {
                // We'll rebuild in place. The BLAS should not be compacted, check that size matches prebuild info.
                assert(blas.blasByteSize == blas.prebuildInfo.ResultDataMaxSizeInBytes);
            }

            GET_COM_INTERFACE(pContext->getLowLevelData()->getCommandList(), ID3D12GraphicsCommandList4, pList4);
            pList4->BuildRaytracingAccelerationStructure(&asDesc, 0, nullptr);
        }

        // Insert barrier. The BLAS buffer is now ready for use.
        pContext->uavBarrier(mpBlas.get());
    }

    void Scene::fillInstanceDesc(std::vector<D3D12_RAYTRACING_INSTANCE_DESC>& instanceDescs, uint32_t rayCount, bool perMeshHitEntry) const
    {
        assert(mpBlas);
        instanceDescs.clear();
        uint32_t instanceContributionToHitGroupIndex = 0;
        uint32_t instanceId = 0;

        for (size_t i = 0; i < mBlasData.size(); i++)
        {
            const auto& meshList = mMeshGroups[i].meshList;

            D3D12_RAYTRACING_INSTANCE_DESC desc = {};
            desc.AccelerationStructure = mpBlas->getGpuAddress() + mBlasData[i].blasByteOffset;
            desc.InstanceMask = 0xFF;
            desc.InstanceContributionToHitGroupIndex = perMeshHitEntry ? instanceContributionToHitGroupIndex : 0;
            instanceContributionToHitGroupIndex += rayCount * (uint32_t)meshList.size();

            // If multiple meshes are in a BLAS:
            // - Their global matrix is the same.
            // - From sortMeshes(), each mesh in the BLAS is guaranteed to be non-instanced, so only one INSTANCE_DESC is needed
            if (meshList.size() > 1)
            {
                assert(mMeshIdToInstanceIds[meshList[0]].size() == 1);
                assert(mMeshIdToInstanceIds[meshList[0]][0] == instanceId); // Mesh instances are sorted by instanceId
                desc.InstanceID = instanceId;
                instanceId += (uint32_t)meshList.size();

                // Any instances of the mesh will get you the correct matrix, so just pick the first mesh then the first instance.
                uint32_t matrixId = mMeshInstanceData[desc.InstanceID].globalMatrixID;
                glm::mat4 transform4x4 = transpose(mpAnimationController->getGlobalMatrices()[matrixId]);
                std::memcpy(desc.Transform, &transform4x4, sizeof(desc.Transform));
                instanceDescs.push_back(desc);
            }
            // If only one mesh is in the BLAS, there CAN be multiple instances of it. It is either:
            // - A non-instanced mesh that was unable to be merged with others
            // - A mesh with multiple instances
            else
            {
                assert(meshList.size() == 1);
                const auto& instanceList = mMeshIdToInstanceIds[meshList[0]];

                // For every instance of the mesh, create an INSTANCE_DESC
                for (uint32_t instId : instanceList)
                {
                    assert(instId == instanceId); // Mesh instances are sorted by instanceId
                    desc.InstanceID = instanceId++;
                    uint32_t matrixId = mMeshInstanceData[desc.InstanceID].globalMatrixID;
                    glm::mat4 transform4x4 = transpose(mpAnimationController->getGlobalMatrices()[matrixId]);
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
            if (tlas.pTlas != nullptr && tlas.updateMode == UpdateMode::Refit) inputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
        }

        tlas.updateMode = mTlasUpdateMode;

        // On first build for the scene, create scratch buffer and cache prebuild info. As long as INSTANCE_DESC count doesn't change, we can reuse these
        if (mpTlasScratch == nullptr)
        {
            // Prebuild
            GET_COM_INTERFACE(gpDevice->getApiHandle(), ID3D12Device5, pDevice5);
            pDevice5->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &mTlasPrebuildInfo);
            mpTlasScratch = Buffer::create(mTlasPrebuildInfo.ScratchDataSizeInBytes, Buffer::BindFlags::UnorderedAccess, Buffer::CpuAccess::None);
            mpTlasScratch->setName("Scene::mpTlasScratch");

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

    void Scene::setGeometryIndexIntoRtVars(const std::shared_ptr<RtProgramVars>& pVars)
    {
        // Sets the 'geometryIndex' hit shader variable for each mesh.
        // This is the local index of which mesh in the BLAS was hit.
        // In DXR 1.0 we have to pass it via a constant buffer to the shader,
        // in DXR 1.1 it is available through the GeometryIndex() system value.
        //
        assert(!mBlasData.empty());
        uint32_t meshCount = getMeshCount();
        uint32_t descHitCount = pVars->getDescHitGroupCount();

        uint32_t blasIndex = 0;
        uint32_t geometryIndex = 0;
        for (uint32_t meshId = 0; meshId < meshCount; meshId++)
        {
            for (uint32_t hit = 0; hit < descHitCount; hit++)
            {
                auto pHitVars = pVars->getHitVars(hit, meshId);
                auto var = pHitVars->findMember(0).findMember("geometryIndex");
                if (var.isValid())
                {
                    var = geometryIndex;
                }
            }

            geometryIndex++;

            // If at the end of this BLAS, reset counters and start checking next BLAS
            uint32_t geomCount = (uint32_t)mMeshGroups[blasIndex].meshList.size();
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
        }

        // On first execution, when meshes have moved, when there's a new ray count, or when a BLAS has changed, create/update the TLAS
        //
        // TODO: The notion of "ray count" is being treated as fundamental here, and intrinsically
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

        assert(mpSceneBlock);
        assert(tlasIt->second.pSrv);

        // Bind Scene parameter block.
        getCamera()->setShaderData(mpSceneBlock[kCamera]);
        var["gScene"] = mpSceneBlock;

        // Bind TLAS.
        var["gRtScene"].setSrv(tlasIt->second.pSrv);
    }

    void Scene::setEnvMap(EnvMap::SharedPtr pEnvMap)
    {
        if (mpEnvMap == pEnvMap) return;
        mpEnvMap = pEnvMap;
        if (mpEnvMap) mpEnvMap->setShaderData(mpSceneBlock[kEnvMap]);
    }

    void Scene::loadEnvMap(const std::string& filename)
    {
        EnvMap::SharedPtr pEnvMap = EnvMap::create(filename);
        setEnvMap(pEnvMap);
    }

    void Scene::setCameraAspectRatio(float ratio)
    {
        getCamera()->setAspectRatio(ratio);
    }

    void Scene::bindSamplerToMaterials(const Sampler::SharedPtr& pSampler)
    {
        for (auto& pMaterial : mMaterials)
        {
            pMaterial->setSampler(pSampler);
        }
    }

    void Scene::setCameraController(CameraControllerType type)
    {
        if (!mCameraSwitched && mCamCtrlType == type && mpCamCtrl) return;

        auto camera = getCamera();
        switch (type)
        {
        case CameraControllerType::FirstPerson:
            mpCamCtrl = FirstPersonCameraController::create(camera);
            break;
        case CameraControllerType::Orbiter:
            mpCamCtrl = OrbiterCameraController::create(camera);
            ((OrbiterCameraController*)mpCamCtrl.get())->setModelParams(mSceneBB.center, length(mSceneBB.extent), 3.5f);
            break;
        case CameraControllerType::SixDOF:
            mpCamCtrl = SixDoFCameraController::create(camera);
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
                    addViewpoint();
                    return true;
                }
            }
        }
        return mpCamCtrl->onKeyEvent(keyEvent);
    }

    std::string Scene::getScript(const std::string& sceneVar)
    {
        std::string c;

        // Render settings.
        c += Scripting::makeSetProperty(sceneVar, kRenderSettings, mRenderSettings);

        // Animations.
        if (hasAnimation() && !isAnimated())
        {
            c += Scripting::makeSetProperty(sceneVar, kAnimated, false);
        }
        for (size_t i = 0; i < mLights.size(); ++i)
        {
            const auto& light = mLights[i];
            if (light->hasAnimation() && !light->isAnimated())
            {
                c += Scripting::makeSetProperty(sceneVar + "." + kGetLight + "(" + std::to_string(i) + ").", kAnimated, false);
            }
        }

        // Camera.
        if (mSelectedCamera != 0)
        {
            c += sceneVar + "." + kCamera + " = " + sceneVar + "." + kCameras + "[" + std::to_string(mSelectedCamera) + "]\n";
        }
        c += getCamera()->getScript(sceneVar + "." + kCamera);

        // Camera speed.
        c += Scripting::makeSetProperty(sceneVar, kCameraSpeed, mCameraSpeed);

        // Viewpoints.
        if (hasSavedViewpoints())
        {
            for (size_t i = 1; i < mViewpoints.size(); i++)
            {
                auto v = mViewpoints[i];
                c += Scripting::makeMemberFunc(sceneVar, kAddViewpoint, v.position, v.target, v.up, v.index);
            }
        }

        return c;
    }

    SCRIPT_BINDING(Scene)
    {
        pybind11::class_<Scene, Scene::SharedPtr> scene(m, "Scene");
        scene.def_property(kCamera.c_str(), &Scene::getCamera, &Scene::setCamera);
        scene.def_property_readonly(kCameras.c_str(), &Scene::getCameras);
        scene.def_property_readonly(kEnvMap.c_str(), &Scene::getEnvMap);
        scene.def_property_readonly(kMaterials.c_str(), &Scene::getMaterials);
        scene.def_property(kCameraSpeed.c_str(), &Scene::getCameraSpeed, &Scene::setCameraSpeed);
        scene.def_property(kAnimated.c_str(), &Scene::isAnimated, &Scene::setIsAnimated);
        scene.def_property(kRenderSettings.c_str(), pybind11::overload_cast<void>(&Scene::getRenderSettings, pybind11::const_), &Scene::setRenderSettings);

        scene.def("animate", &Scene::toggleAnimations, "animate"_a); // PYTHONDEPRECATED
        auto animateCamera = [](Scene* pScene, bool animate) { pScene->getCamera()->setIsAnimated(animate); };
        scene.def("animateCamera", animateCamera, "animate"_a); // PYTHONDEPRECATED
        auto animateLight = [](Scene* pScene, uint32_t index, bool animate) { pScene->getLight(index)->setIsAnimated(animate); };
        scene.def("animateLight", animateLight, "index"_a, "animate"_a); // PYTHONDEPRECATED

        scene.def(kSetEnvMap.c_str(), &Scene::loadEnvMap, "filename"_a);
        scene.def(kGetLight.c_str(), &Scene::getLight, "index"_a);
        scene.def(kGetLight.c_str(), &Scene::getLightByName, "name"_a);
        scene.def("light", &Scene::getLight); // PYTHONDEPRECATED
        scene.def("light", &Scene::getLightByName); // PYTHONDEPRECATED
        scene.def(kGetMaterial.c_str(), &Scene::getMaterial, "index"_a);
        scene.def(kGetMaterial.c_str(), &Scene::getMaterialByName, "name"_a);
        scene.def("material", &Scene::getMaterial); // PYTHONDEPRECATED
        scene.def("material", &Scene::getMaterialByName); // PYTHONDEPRECATED

        // Viewpoints
        scene.def(kAddViewpoint.c_str(), pybind11::overload_cast<>(&Scene::addViewpoint)); // add current camera as viewpoint
        scene.def(kAddViewpoint.c_str(), pybind11::overload_cast<const float3&, const float3&, const float3&, uint32_t>(&Scene::addViewpoint), "position"_a, "target"_a, "up"_a, "cameraIndex"_a=0); // add specified viewpoint
        scene.def(kRemoveViewpoint.c_str(), &Scene::removeViewpoint); // remove the selected viewpoint
        scene.def(kSelectViewpoint.c_str(), &Scene::selectViewpoint, "index"_a); // select a viewpoint by index

        scene.def("viewpoint", pybind11::overload_cast<>(&Scene::addViewpoint)); // PYTHONDEPRECATED save the current camera position etc.
        scene.def("viewpoint", pybind11::overload_cast<uint32_t>(&Scene::selectViewpoint)); // PYTHONDEPRECATED select a previously saved camera viewpoint

        // RenderSettings
        ScriptBindings::SerializableStruct<Scene::RenderSettings> renderSettings(m, "SceneRenderSettings");
#define field(f_) field(#f_, &Scene::RenderSettings::f_)
        renderSettings.field(useEnvLight);
        renderSettings.field(useAnalyticLights);
        renderSettings.field(useEmissiveLights);
#undef field
    }
}
