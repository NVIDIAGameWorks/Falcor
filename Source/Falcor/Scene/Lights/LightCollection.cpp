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
#include "LightCollection.h"
#include "LightCollectionShared.slang"
#include "Core/API/Device.h"
#include "Scene/Scene.h"
#include "Scene/Material/BasicMaterial.h"
#include "Utils/Logger.h"
#include "Utils/Timing/TimeReport.h"
#include "Utils/Timing/Profiler.h"
#include <sstream>

namespace Falcor
{
    static_assert(sizeof(MeshLightData) % 16 == 0, "MeshLightData size should be a multiple of 16");
    static_assert(sizeof(PackedEmissiveTriangle) % 16 == 0, "PackedEmissiveTriangle size should be a multiple of 16");
    static_assert(sizeof(EmissiveFlux) % 16 == 0, "EmissiveFlux size should be a multiple of 16");

    namespace
    {
        const char kEmissiveIntegratorFile[] = "Scene/Lights/EmissiveIntegrator.3d.slang";
        const char kBuildTriangleListFile[] = "Scene/Lights/BuildTriangleList.cs.slang";
        const char kUpdateTriangleVerticesFile[] = "Scene/Lights/UpdateTriangleVertices.cs.slang";
        const char kFinalizeIntegrationFile[] = "Scene/Lights/FinalizeIntegration.cs.slang";
    }

    LightCollection::SharedPtr LightCollection::create(RenderContext* pRenderContext, const std::shared_ptr<Scene>& pScene)
    {
        return SharedPtr(new LightCollection(pRenderContext, pScene));
    }

    LightCollection::LightCollection(RenderContext* pRenderContext, const std::shared_ptr<Scene>& pScene)
    {
        FALCOR_ASSERT(pScene);
        mpScene = pScene;

        // Setup the lights.
        setupMeshLights(*pScene);

        // Create program for integrating emissive textures.
        // This should be done after lights are setup, so that we know which sampler state etc. to use.
        initIntegrator(*pScene);

        // Create programs for building/updating the mesh lights.
        Shader::DefineList defines = pScene->getSceneDefines();
        mpTriangleListBuilder = ComputePass::create(kBuildTriangleListFile, "buildTriangleList", defines);
        mpTrianglePositionUpdater = ComputePass::create(kUpdateTriangleVerticesFile, "updateTriangleVertices", defines);
        mpFinalizeIntegration = ComputePass::create(kFinalizeIntegrationFile, "finalizeIntegration", defines);

        mpStagingFence = GpuFence::create();

        // Now build the mesh light data.
        build(pRenderContext, *pScene);
    }

    bool LightCollection::update(RenderContext* pRenderContext, UpdateStatus* pUpdateStatus)
    {
        FALCOR_PROFILE("LightCollection::update()");

        auto pScene = mpScene.lock();
        if (!pScene) return false;

        if (pUpdateStatus)
        {
            pUpdateStatus->lightsUpdateInfo.clear();
            pUpdateStatus->lightsUpdateInfo.reserve(mMeshLights.size());
        }

        // Update transform matrices and check for updates.
        // TODO: Move per-mesh instance update flags into Scene. Return just a list of mesh lights that have changed.
        std::vector<uint32_t> updatedLights;
        updatedLights.reserve(mMeshLights.size());

        for (uint32_t lightIdx = 0; lightIdx < mMeshLights.size(); ++lightIdx)
        {
            const GeometryInstanceData& instanceData = pScene->getGeometryInstance(mMeshLights[lightIdx].instanceID);
            UpdateFlags updateFlags = UpdateFlags::None;

            // Check if instance transform changed.
            if (pScene->getAnimationController()->isMatrixChanged(NodeID{ instanceData.globalMatrixID })) updateFlags |= UpdateFlags::MatrixChanged;

            // Store update status.
            if (updateFlags != UpdateFlags::None) updatedLights.push_back(lightIdx);
            if (pUpdateStatus) pUpdateStatus->lightsUpdateInfo.push_back(updateFlags);
        }

        // Update light data if needed.
        if (!updatedLights.empty())
        {
            updateTrianglePositions(pRenderContext, *pScene, updatedLights);
            return true;
        }

        return false;
    }

    void LightCollection::initIntegrator(const Scene& scene)
    {
        // The current algorithm rasterizes emissive triangles in texture space,
        // and uses atomic operations to sum up the contribution from all covered texels.
        // We do this in a raster pass, so we get one thread per texel/triangle.

        // Check for required features.
        if (!gpDevice->isFeatureSupported(Device::SupportedFeatures::ConservativeRasterizationTier3))
        {
            throw RuntimeError("LightCollection requires conservative rasterization tier 3 support.");
        }
        if (!gpDevice->isShaderModelSupported(Device::ShaderModel::SM6_6))
        {
            throw RuntimeError("LightCollection requires Shader Model 6.6 support.");
        }

        // Create program.
        auto defines = scene.getSceneDefines();
        defines.add("INTEGRATOR_PASS", "1");

        Program::Desc desc;
        desc.addShaderLibrary(kEmissiveIntegratorFile).vsEntry("vsMain").gsEntry("gsMain").psEntry("psMain");
        desc.setShaderModel("6_6");
        mIntegrator.pProgram = GraphicsProgram::create(desc, defines);

        // Create graphics state.
        mIntegrator.pState = GraphicsState::create();

        // Set state.
        mIntegrator.pState->setProgram(mIntegrator.pProgram);
        mIntegrator.pState->setVao(Vao::create(Vao::Topology::TriangleList));

        // Set viewport. Note we don't bind any render targets so the size just determines the dispatch limits.
        const uint32_t vpDim = 16384;       // 16K x 16K
        mIntegrator.pState->setViewport(0, GraphicsState::Viewport(0.f, 0.f, (float)vpDim, (float)vpDim, 0.f, 1.f));
        mIntegrator.pProgram->addDefine("_VIEWPORT_DIM", std::to_string(vpDim));    // Pass size to shader

        // Set raster state to disable culling. We don't care about winding in texture space when integrating.
        RasterizerState::Desc rsDesc;
        rsDesc.setCullMode(RasterizerState::CullMode::None);
        rsDesc.setFillMode(RasterizerState::FillMode::Solid);
        rsDesc.setConservativeRasterization(true);
        mIntegrator.pState->setRasterizerState(RasterizerState::create(rsDesc));

        // Set depth-stencil state to disable depth test/writes.
        DepthStencilState::Desc dsDesc;
        dsDesc.setDepthEnabled(false);
        dsDesc.setDepthWriteMask(false);
        mIntegrator.pState->setDepthStencilState(DepthStencilState::create(dsDesc));

        // Create sampler for texel fetch. This is identical to material sampler but uses point sampling.
        Sampler::Desc samplerDesc = mpSamplerState ? mpSamplerState->getDesc() : Sampler::Desc();
        samplerDesc.setFilterMode(Sampler::Filter::Point, Sampler::Filter::Point, Sampler::Filter::Point);
        mIntegrator.pPointSampler = Sampler::create(samplerDesc);
    }

    void LightCollection::setupMeshLights(const Scene& scene)
    {
        mMeshLights.clear();
        mpSamplerState = nullptr;
        mTriangleCount = 0;

        // Create mesh lights for all emissive mesh instances.
        for (uint32_t instanceID = 0; instanceID < scene.getGeometryInstanceCount(); instanceID++)
        {
            const GeometryInstanceData& instanceData = scene.getGeometryInstance(instanceID);

            // We only support triangle meshes.
            if (instanceData.getType() != GeometryType::TriangleMesh) continue;

            const MeshDesc& meshData = scene.getMesh(MeshID::fromSlang( instanceData.geometryID ));

            // Only mesh lights with basic materials are supported.
            auto pMaterial = scene.getMaterial(MaterialID::fromSlang( instanceData.materialID ))->toBasicMaterial();

            if (pMaterial && pMaterial->isEmissive())
            {
                // We've found a mesh instance with an emissive material => Setup mesh light data.
                MeshLightData meshLight;
                meshLight.instanceID = instanceID;
                meshLight.triangleCount = meshData.getTriangleCount();
                meshLight.triangleOffset = mTriangleCount;
                meshLight.materialID = instanceData.materialID;

                mMeshLights.push_back(meshLight);
                mTriangleCount += meshLight.triangleCount;

                // Store ptr to texture sampler. We currently assume all the mesh lights' materials have the same sampler, which is true in current Falcor.
                // If this changes in the future, we'll have to support multiple samplers.
                if (pMaterial->getEmissiveTexture())
                {
                    if (!mpSamplerState)
                    {
                        mpSamplerState = pMaterial->getDefaultTextureSampler();
                    }
                    else if (mpSamplerState != pMaterial->getDefaultTextureSampler())
                    {
                        throw RuntimeError("Material '{}' is using a different sampler.", pMaterial->getName());
                    }
                }
            }
        }
    }

    void LightCollection::build(RenderContext* pRenderContext, const Scene& scene)
    {
        prepareMeshData(scene);

        if (mTriangleCount == 0)
        {
            // If there are no emissive triangle, clear everything and mark the CPU data/stats as valid.
            mMeshLightTriangles.clear();
            mMeshLightStats = MeshLightStats();

            mCPUInvalidData = CPUOutOfDateFlags::None;
            mStagingBufferValid = true;
            mStatsValid = true;
        }
        else
        {
            TimeReport timeReport;

            // Prepare GPU buffers.
            prepareTriangleData(pRenderContext, scene);
            timeReport.measure("LightCollection::build preparation");

            // Pre-integrate emissive triangles.
            // TODO: We might want to redo this in update() for animated meshes or after scale changes as that affects the flux.
            integrateEmissive(pRenderContext, scene);

            timeReport.measure("LightCollection::build integrate emissive");

            // Build list of active triangles.
            mCPUInvalidData = CPUOutOfDateFlags::All;
            mStagingBufferValid = false;
            mStatsValid = false;

            prepareSyncCPUData(pRenderContext);
            updateActiveTriangleList();

            timeReport.measure("LightCollection::build finalize");
            timeReport.printToLog();
        }
    }

    void LightCollection::prepareTriangleData(RenderContext* pRenderContext, const Scene& scene)
    {
        FALCOR_ASSERT(mTriangleCount > 0);

        // Create GPU buffers.
        mpTriangleData = Buffer::createStructured(mpTriangleListBuilder["gTriangleData"], mTriangleCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, Buffer::CpuAccess::None, nullptr, false);
        mpTriangleData->setName("LightCollection::mpTriangleData");
        if (mpTriangleData->getStructSize() != sizeof(PackedEmissiveTriangle)) throw RuntimeError("Struct PackedEmissiveTriangle size mismatch between CPU/GPU");

        mpFluxData = Buffer::createStructured(mpFinalizeIntegration["gFluxData"], mTriangleCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, Buffer::CpuAccess::None, nullptr, false);
        mpFluxData->setName("LightCollection::mpFluxData");
        if (mpFluxData->getStructSize() != sizeof(EmissiveFlux)) throw RuntimeError("Struct EmissiveFlux size mismatch between CPU/GPU");

        // Compute triangle data (vertices, uv-coordinates, materialID) for all mesh lights.
        buildTriangleList(pRenderContext, scene);
    }

    void LightCollection::prepareMeshData(const Scene& scene)
    {
        // Create buffer for the mesh data if needed.
        if (!mMeshLights.empty())
        {
            mpMeshData = Buffer::createStructured(
                mpTrianglePositionUpdater["gMeshData"],
                uint32_t(mMeshLights.size()),
                ResourceBindFlags::ShaderResource,
                Buffer::CpuAccess::None, nullptr, false);
            mpMeshData->setName("LightCollection::mpMeshData");
            if (mpMeshData->getStructSize() != sizeof(MeshLightData))
            {
                throw RuntimeError("Size mismatch for structured buffer of MeshLightData");
            }
            size_t meshDataSize = mMeshLights.size() * sizeof(mMeshLights[0]);
            FALCOR_ASSERT(mpMeshData->getSize() == meshDataSize);
            mpMeshData->setBlob(mMeshLights.data(), 0, meshDataSize);
        }

        // Build a lookup table from instance ID to emissive triangle offset.
        // This is useful in ray tracing for locating the emissive triangle that was hit for MIS computation etc.
        uint32_t instanceCount = scene.getGeometryInstanceCount();
        if (instanceCount > 0)
        {
            std::vector<uint32_t> triangleOffsets(instanceCount, MeshLightData::kInvalidIndex);
            for (const auto& it : mMeshLights)
            {
                FALCOR_ASSERT(it.instanceID < instanceCount);
                triangleOffsets[it.instanceID] = it.triangleOffset;
            }

            mpPerMeshInstanceOffset = Buffer::createStructured(sizeof(uint32_t), (uint32_t)triangleOffsets.size(), Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, triangleOffsets.data(), false);
            mpPerMeshInstanceOffset->setName("LightCollection::mpPerMeshInstanceOffset");
        }
    }

    void LightCollection::integrateEmissive(RenderContext* pRenderContext, const Scene& scene)
    {
        FALCOR_ASSERT(mTriangleCount > 0);
        FALCOR_ASSERT(mMeshLights.size() > 0);

        // Prepare program vars.
        mIntegrator.pVars = GraphicsVars::create(mIntegrator.pProgram.get());
        mIntegrator.pVars["gScene"] = scene.getParameterBlock();
        mIntegrator.pVars["gPointSampler"] = mIntegrator.pPointSampler;
        setShaderData(mIntegrator.pVars["gLightCollection"]);

        // 1st pass: Rasterize emissive triangles in texture space to find maximum texel value.
        // The maximum is needed to rescale the texels to fixed-point format in the accumulation pass.
        Buffer::SharedPtr pTexelMax;
        {
            // Allocate intermediate buffer.
            // Move into a member variable if we're integrating multiple times to avoid re-allocation.
            pTexelMax = Buffer::create(mTriangleCount * sizeof(uint32_t), Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess, Buffer::CpuAccess::None);
            pTexelMax->setName("LightCollection: pTexelMax");

            pRenderContext->clearUAV(pTexelMax->getUAV().get(), uint4(0));

            // Bind our resources.
            mIntegrator.pVars["gTexelMax"] = pTexelMax;
            mIntegrator.pVars["gTexelSum"].setUav(UnorderedAccessView::getNullView(ReflectionResourceType::Dimensions::Buffer));

            // Execute.
            mIntegrator.pProgram->addDefine("INTEGRATOR_PASS", "1");
            pRenderContext->draw(mIntegrator.pState.get(), mIntegrator.pVars.get(), mTriangleCount * 3, 0);
        }

        // 2nd pass: Rasterize emissive triangles in texture space to sum up their texels.
        // The summation is done in fixed-point format to guarantee deterministic results independent
        // of rasterization order. We use 64-bit atomics to avoid large accumulated errors.
        {
            // Re-allocate result buffer if needed.
            const uint32_t bufSize = mTriangleCount * 4 * sizeof(uint64_t);
            if (!mIntegrator.pResultBuffer || mIntegrator.pResultBuffer->getSize() < bufSize)
            {
                mIntegrator.pResultBuffer = Buffer::create(bufSize, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess, Buffer::CpuAccess::None);
                mIntegrator.pResultBuffer->setName("LightCollection::mIntegrator::pResultBuffer");
            }

            pRenderContext->clearUAV(mIntegrator.pResultBuffer->getUAV().get(), uint4(0));

            // Bind our resources.
            mIntegrator.pVars["gTexelSum"] = mIntegrator.pResultBuffer;

            // Execute.
            mIntegrator.pProgram->addDefine("INTEGRATOR_PASS", "2");
            pRenderContext->draw(mIntegrator.pState.get(), mIntegrator.pVars.get(), mTriangleCount * 3, 0);
        }

        // 3rd pass: Finalize the per-triangle flux values.
        {
            // Bind scene.
            mpFinalizeIntegration["gScene"] = scene.getParameterBlock();

            mpFinalizeIntegration["gPointSampler"] = mIntegrator.pPointSampler;
            mpFinalizeIntegration["gTexelMax"] = pTexelMax;
            mpFinalizeIntegration["gTexelSum"] = mIntegrator.pResultBuffer;
            mpFinalizeIntegration["gTriangleData"] = mpTriangleData;
            mpFinalizeIntegration["gFluxData"] = mpFluxData;

            mpFinalizeIntegration["CB"]["gTriangleCount"] = mTriangleCount;

            // Execute.
            FALCOR_ASSERT(mpFinalizeIntegration->getThreadGroupSize().y == 1);
            uint32_t rows = div_round_up(mTriangleCount, mpFinalizeIntegration->getThreadGroupSize().x);
            mpFinalizeIntegration->execute(pRenderContext, mpFinalizeIntegration->getThreadGroupSize().x, rows);
        }
#if 0
        // Output a list of per-triangle results to file for debugging purposes.
        std::ofstream ofs("flux.txt");
        const float* pMax = (const float*)pTexelMax->map(Buffer::MapType::Read);
        const float4* flux = (const float4*)mpFluxData->map(Buffer::MapType::Read);
        const uint64_t* result = (const uint64_t*)mIntegrator.pResultBuffer->map(Buffer::MapType::Read);
        for (uint32_t i = 0; i < mpFluxData->getElementCount(); i++)
        {
            uint64_t w = result[4 * i + 3];
            float weight = w / (float)(1ull << 35);
            ofs << std::setw(6) << i << " : max " << std::setw(12) << pMax[i] << "  weight " << std::setw(12) << weight << "  value " << to_string(flux[i]) << std::endl;
        }
        mpFluxData->unmap();
        pTexelMax->unmap();
        mIntegrator.pResultBuffer->unmap();
#endif
    }

    void LightCollection::computeStats() const
    {
        if (mStatsValid) return;

        auto pScene = mpScene.lock();
        FALCOR_ASSERT(pScene);

        // Read back the current data. This is potentially expensive.
        syncCPUData();

        // Stats on input data.
        MeshLightStats stats;
        stats.meshLightCount = (uint32_t)mMeshLights.size();
        stats.triangleCount = (uint32_t)mMeshLightTriangles.size();

        uint32_t trianglesTotal = 0;
        for (const auto& meshLight : mMeshLights)
        {
            auto pMaterial = pScene->getMaterial(MaterialID::fromSlang(meshLight.materialID))->toBasicMaterial();
            FALCOR_ASSERT(pMaterial);
            bool isTextured = pMaterial->getEmissiveTexture() != nullptr;

            if (isTextured)
            {
                stats.meshesTextured++;
                stats.trianglesTextured += meshLight.triangleCount;
            }
            trianglesTotal += meshLight.triangleCount;
        }
        FALCOR_ASSERT(trianglesTotal == stats.triangleCount);

        // Stats on pre-processed data.
        for (const auto& tri : mMeshLightTriangles)
        {
            FALCOR_ASSERT(tri.flux >= 0.f);
            if (tri.flux == 0.f)
            {
                stats.trianglesCulled++;
            }
            else
            {
                // TODO: Currently we don't detect uniform radiance for textured lights, so just look at whether the mesh light is textured or not.
                // This code will change when we tag individual triangles as textured vs non-textured.
                auto pMaterial = pScene->getMaterial(MaterialID::fromSlang(mMeshLights[tri.lightIdx].materialID))->toBasicMaterial();
                FALCOR_ASSERT(pMaterial);
                bool isTextured = pMaterial->getEmissiveTexture() != nullptr;

                if (isTextured) stats.trianglesActiveTextured++;
                else stats.trianglesActiveUniform++;
            }
        }
        stats.trianglesActive = stats.trianglesActiveUniform + stats.trianglesActiveTextured;

        mMeshLightStats = stats;
        mStatsValid = true;
    }

    void LightCollection::buildTriangleList(RenderContext* pRenderContext, const Scene& scene)
    {
        FALCOR_ASSERT(mMeshLights.size() > 0);

        // Bind scene.
        mpTriangleListBuilder["gScene"] = scene.getParameterBlock();

        // Bind our output buffer.
        mpTriangleListBuilder["gTriangleData"] = mpTriangleData;

        // TODO: Single dispatch over all emissive triangles instead of per-mesh dispatches.
        // This code is not performance critical though, as it's currently only run once at init time.
        for (uint32_t lightIdx = 0; lightIdx < mMeshLights.size(); ++lightIdx)
        {
            const MeshLightData& meshLight = mMeshLights[lightIdx];

            mpTriangleListBuilder["CB"]["gLightIdx"] = lightIdx;
            mpTriangleListBuilder["CB"]["gMaterialID"] = meshLight.materialID;
            mpTriangleListBuilder["CB"]["gInstanceID"] = meshLight.instanceID;
            mpTriangleListBuilder["CB"]["gTriangleCount"] = meshLight.triangleCount;
            mpTriangleListBuilder["CB"]["gTriangleOffset"] = meshLight.triangleOffset;

            // TODO: Disable automatic UAV barriers.
            // Each kernel writes to non-overlapping parts of the output buffers, but currently Falcor inserts barriers between each dispatch.
            mpTriangleListBuilder->execute(pRenderContext, meshLight.triangleCount, 1u, 1u);
        }
    }

    void LightCollection::updateActiveTriangleList()
    {
        // This function updates the list of active (non-culled) triangles based on the pre-integrated flux.
        // We currently run this as part of initialization. To support animated emissive textures and/or
        // dynamically changing emissive intensities, it should be run as part of update().
        // In that case, we may want to move it to the GPU to avoid syncing the data to the CPU first.

        // Read back the current data. This is potentially expensive.
        syncCPUData();

        const uint32_t triCount = (uint32_t)mMeshLightTriangles.size();
        const uint32_t kInvalidActiveIndex = ~0u;

        mTriToActiveList.clear();
        mTriToActiveList.resize(triCount, kInvalidActiveIndex);
        mActiveTriangleList.clear();
        mActiveTriangleList.reserve(triCount);

        // Iterate over the emissive triangles.
        for (uint32_t triIdx = 0; triIdx < triCount; triIdx++)
        {
            if (mMeshLightTriangles[triIdx].flux > 0.f)
            {
                mTriToActiveList[triIdx] = (uint32_t)mActiveTriangleList.size();
                mActiveTriangleList.push_back(triIdx);
            }
        }

        FALCOR_ASSERT(mActiveTriangleList.size() <= std::numeric_limits<uint32_t>::max());
        const uint32_t activeCount = (uint32_t)mActiveTriangleList.size();

        // Update GPU buffer.
        if (activeCount > 0)
        {
            if (!mpActiveTriangleList || mpActiveTriangleList->getElementCount() < activeCount)
            {
                mpActiveTriangleList = Buffer::createStructured(sizeof(uint32_t), activeCount, Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, mActiveTriangleList.data(), false);
                mpActiveTriangleList->setName("LightCollection::mpActiveTriangleList");
            }
            else
            {
                mpActiveTriangleList->setBlob(mActiveTriangleList.data(), 0, activeCount * sizeof(mActiveTriangleList[0]));
            }
        }

        if (!mpTriToActiveList || mpTriToActiveList->getElementCount() < triCount)
        {
            mpTriToActiveList = Buffer::createStructured(sizeof(uint32_t), triCount, Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, mTriToActiveList.data(), false);
            mpTriToActiveList->setName("LightCollection::mpTriToActiveList");
        }
        else
        {
            mpTriToActiveList->setBlob(mTriToActiveList.data(), 0, triCount * sizeof(mTriToActiveList[0]));
        }
    }

    void LightCollection::updateTrianglePositions(RenderContext* pRenderContext, const Scene& scene, const std::vector<uint32_t>& updatedLights)
    {
        // This pass pre-transforms all emissive triangles into world space and updates their area and face normals.
        // It is executed if any geometry in the scene has moved, which is wasteful since it will update also things
        // that didn't move. However, due to the CPU overhead of doing per-mesh dispatches as before, it's still faster.
        // TODO: Test using ExecuteIndirect to do the dispatches on the GPU for only the updated meshes.
        // Alternatively, upload the list of updated meshes and early out unnecessary threads at runtime.
        FALCOR_ASSERT(!updatedLights.empty());

        // Bind scene.
        mpTrianglePositionUpdater["gScene"] = scene.getParameterBlock();

        // Bind our resources.
        mpTrianglePositionUpdater["gTriangleData"] = mpTriangleData;
        mpTrianglePositionUpdater["gMeshData"] = mpMeshData;

        mpTrianglePositionUpdater["CB"]["gTriangleCount"] = mTriangleCount;

        // Run compute pass to update all triangles.
        mpTrianglePositionUpdater->execute(pRenderContext, mTriangleCount, 1u, 1u);

        mCPUInvalidData |= CPUOutOfDateFlags::TriangleData;
        mStagingBufferValid = false;
    }

    void LightCollection::setShaderData(const ShaderVar& var) const
    {
        FALCOR_ASSERT(var.isValid());

        // Set variables.
        var["triangleCount"] = mTriangleCount;
        var["activeTriangleCount"] = (uint32_t)mActiveTriangleList.size();
        var["meshCount"] = (uint32_t)mMeshLights.size();

        // Bind buffers.
        var["perMeshInstanceOffset"] = mpPerMeshInstanceOffset; // Can be nullptr

        if (mTriangleCount > 0)
        {
            // These buffers must exist if triangle count is > 0.
            FALCOR_ASSERT(mpTriangleData && mpFluxData && mpMeshData);
            var["triangleData"] = mpTriangleData;
            var["fluxData"] = mpFluxData;
            var["meshData"] = mpMeshData;

            if (!mActiveTriangleList.empty())
            {
                FALCOR_ASSERT(mpActiveTriangleList);
                var["activeTriangles"] = mpActiveTriangleList;
            }

            var["triToActiveMapping"] = mpTriToActiveList;
        }
        else
        {
            FALCOR_ASSERT(mMeshLights.empty());
        }
    }

    void LightCollection::copyDataToStagingBuffer(RenderContext* pRenderContext) const
    {
        if (mStagingBufferValid) return;

        // Allocate staging buffer for readback. The data from our different GPU buffers is stored consecutively.
        const size_t stagingSize = mpTriangleData->getSize() + mpFluxData->getSize();
        if (!mpStagingBuffer || mpStagingBuffer->getSize() < stagingSize)
        {
            mpStagingBuffer = Buffer::create(stagingSize, Resource::BindFlags::None, Buffer::CpuAccess::Read);
            mpStagingBuffer->setName("LightCollection::mpStagingBuffer");
            mCPUInvalidData = CPUOutOfDateFlags::All;
        }

        // Schedule the copy operations for data that is invalid.
        // Note that the staging buffer is allocated for the worst-case encountered so far.
        // If the number of triangles ever decreases, we'll be copying unnecessary data. This currently doesn't happen as geometry is not added/removed from the scene.
        // TODO: Update this code if we start removing geometry dynamically.
        FALCOR_ASSERT(mCPUInvalidData != CPUOutOfDateFlags::None); // We shouldn't get here unless at least some data is out of date.
        bool copyTriangleData = is_set(mCPUInvalidData, CPUOutOfDateFlags::TriangleData);
        bool copyFluxData = is_set(mCPUInvalidData, CPUOutOfDateFlags::FluxData);

        uint64_t offset = 0;
        if (copyTriangleData) pRenderContext->copyBufferRegion(mpStagingBuffer.get(), offset, mpTriangleData.get(), 0, mpTriangleData->getSize());
        offset += mpTriangleData->getSize();
        if (copyFluxData) pRenderContext->copyBufferRegion(mpStagingBuffer.get(), offset, mpFluxData.get(), 0, mpFluxData->getSize());
        offset += mpFluxData->getSize();
        FALCOR_ASSERT(offset == stagingSize);

        // Submit command list and insert signal.
        pRenderContext->flush(false);
        mpStagingFence->gpuSignal(pRenderContext->getLowLevelData()->getCommandQueue());

        // Resize the CPU-side triangle list (array-of-structs) buffer and mark the data as invalid.
        mMeshLightTriangles.resize(mTriangleCount);

        mStagingBufferValid = true;
    }

    void LightCollection::syncCPUData() const
    {
        if (mCPUInvalidData == CPUOutOfDateFlags::None) return;

        // If the data has not yet been copied to the staging buffer, we have to do that first.
        // This should normally have done by calling prepareSyncCPUData().
        if (!mStagingBufferValid)
        {
            logWarning("LightCollection::syncCPUData() performance warning - Call LightCollection::prepareSyncCPUData() ahead of time if possible");
            prepareSyncCPUData(gpDevice->getRenderContext());
        }

        // Wait for signal.
        mpStagingFence->syncCpu();

        FALCOR_ASSERT(mStagingBufferValid);
        FALCOR_ASSERT(mpTriangleData && mpFluxData);
        const void* mappedData = mpStagingBuffer->map(Buffer::MapType::Read);

        uint64_t offset = 0;
        const PackedEmissiveTriangle* triangleData = reinterpret_cast<const PackedEmissiveTriangle*>(reinterpret_cast<uintptr_t>(mappedData) + offset);
        offset += mpTriangleData->getSize();
        const EmissiveFlux* fluxData = reinterpret_cast<const EmissiveFlux*>(reinterpret_cast<uintptr_t>(mappedData) + offset);
        offset += mpFluxData->getSize();
        FALCOR_ASSERT(offset <= mpStagingBuffer->getSize());

        bool updateTriangleData = is_set(mCPUInvalidData, CPUOutOfDateFlags::TriangleData);
        bool updateFluxData = is_set(mCPUInvalidData, CPUOutOfDateFlags::FluxData);

        FALCOR_ASSERT(mTriangleCount > 0);
        FALCOR_ASSERT(mMeshLightTriangles.size() == (size_t)mTriangleCount);
        for (uint32_t triIdx = 0; triIdx < mTriangleCount; triIdx++)
        {
            const auto tri = triangleData[triIdx].unpack();
            auto& meshLightTri = mMeshLightTriangles[triIdx];

            if (updateTriangleData)
            {
                meshLightTri.lightIdx = tri.lightIdx;
                meshLightTri.normal = tri.normal;
                meshLightTri.area = tri.area;

                for (uint32_t j = 0; j < 3; j++)
                {
                    meshLightTri.vtx[j].pos = tri.posW[j];
                    meshLightTri.vtx[j].uv = tri.texCoords[j];
                }
            }

            if (updateFluxData)
            {
                meshLightTri.flux = fluxData[triIdx].flux;
                meshLightTri.averageRadiance = fluxData[triIdx].averageRadiance;
            }
        }

        mpStagingBuffer->unmap();
        mCPUInvalidData = CPUOutOfDateFlags::None;
    }

    uint64_t LightCollection::getMemoryUsageInBytes() const
    {
        uint64_t m = 0;
        if (mpTriangleData) m += mpTriangleData->getSize();
        if (mpActiveTriangleList) m += mpActiveTriangleList->getSize();
        if (mpTriToActiveList) m += mpTriToActiveList->getSize();
        if (mpFluxData) m += mpFluxData->getSize();
        if (mpMeshData) m += mpMeshData->getSize();
        if (mpPerMeshInstanceOffset) m += mpPerMeshInstanceOffset->getSize();
        if (mpStagingBuffer) m += mpStagingBuffer->getSize();
        if (mIntegrator.pResultBuffer) m += mIntegrator.pResultBuffer->getSize();
        return m;
    }
}
