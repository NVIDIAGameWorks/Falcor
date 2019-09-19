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
#include "LightCollection.h"
#include "LightCollectionShared.h"
#include <sstream>

namespace Falcor
{
    namespace
    {
        const char kFileEmissiveIntegrator[] = "Experimental/Scene/Lights/EmissiveIntegrator.ps.slang";
        const char kFileSetup[] = "Experimental/Scene/Lights/LightCollection.cs.slang";
    }

    LightCollection::SharedPtr LightCollection::create(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
    {
        SharedPtr ptr = SharedPtr(new LightCollection());
        return ptr->init(pRenderContext, pScene) ? ptr : nullptr;
    }

    bool LightCollection::update(RenderContext* pRenderContext, UpdateStatus* pUpdateStatus)
    {
        PROFILE("LightCollection::update()");

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
            const MeshInstanceData& instanceData = mpScene->getMeshInstance(mMeshLights[lightIdx].meshInstanceID);
            UpdateFlags updateFlags = UpdateFlags::None;

            // Check if mesh has an active animation.
            // This check is over-conservative. We should inspect all matrices referenced by the bones to detect skinning changes.
            // TODO: Implement this check inside AnimationController and store extra flag to differentiate between rigid body and skinning changes.
            if (mpScene->getAnimationController()->getMeshAnimationCount(instanceData.meshID) > 0)
            {
                bool hasAnimation = mpScene->getAnimationController()->getActiveAnimation(instanceData.meshID) != AnimationController::kBindPoseAnimationId;
                bool isPaused = gpFramework->getGlobalClock().isPaused();
                if (hasAnimation && !isPaused) updateFlags |= UpdateFlags::AnimationChanged;
            }

            // Check if instance transform changed.
            if (mpScene->getAnimationController()->didMatrixChanged(instanceData.globalMatrixID)) updateFlags |= UpdateFlags::MatrixChanged;

            // Store update status.
            if (updateFlags != UpdateFlags::None) updatedLights.push_back(lightIdx);
            if (pUpdateStatus) pUpdateStatus->lightsUpdateInfo.push_back(updateFlags);
        }

        // Update light data if needed.
        if (!updatedLights.empty())
        {
            updateTrianglePositions(pRenderContext, updatedLights);
            return true;
        }

        return false;
    }

    bool LightCollection::prepareProgram(ProgramBase* pProgram) const
    {
        return pProgram->addDefine("_NUM_MESH_LIGHTS", std::to_string(mMeshLights.size()));
    }

    bool LightCollection::renderUI(Gui::Widgets& widget)
    {
        // Prints stats about the number of lights etc.
        const MeshLightStats& stats = getStats();
        std::ostringstream oss;
        oss << "Mesh lights (input)" << std::endl
            << " Meshes (total)         : " << stats.meshLightCount << std::endl
            << " Meshes (textured)      : " << stats.meshesTextured << std::endl
            << " Triangles (total)      : " << stats.triangleCount << std::endl
            << " Triangles (textured)   : " << stats.trianglesTextured << std::endl
            << std::endl
            << "Mesh lights (pre-processed)" << std::endl
            << " Triangles (culled)     : " << stats.trianglesCulled << std::endl
            << " Triangles (active)     : " << stats.trianglesActive << std::endl
            << " -> uniform emissive    : " << stats.trianglesActiveUniform << std::endl
            << " -> textured emissive   : " << stats.trianglesActiveTextured << std::endl;

        widget.text(oss.str().c_str());

        return false;
    }

    bool LightCollection::init(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
    {
        assert(pScene);
        mpScene = pScene;

        // Setup the lights.
        if (!setupMeshLights()) return false;

        // Create program for integrating emissive textures.
        // This should be done after lights are setup, so that we know which sampler state etc. to use.
        if (!initIntegrator()) return false;

        // Create programs for building/updating the mesh lights.
        Shader::DefineList defines = mpScene->getSceneDefines();
        mpTriangleListBuilder = ComputePass::create(kFileSetup, "buildTriangleList", defines);
        if (!mpTriangleListBuilder) return false;
        mpTrianglePositionUpdater = ComputePass::create(kFileSetup, "updateTriangleVertices", defines);
        if (!mpTrianglePositionUpdater) return false;
        mpFinalizeIntegration = ComputePass::create(kFileSetup, "finalizeIntegration", Program::DefineList(), false);
        if (!mpFinalizeIntegration) return false;

        mpStagingFence = GpuFence::create();
        if (!mpStagingFence) return false;

        // Now build the mesh light data.
        build(pRenderContext);

        return true;
    }

    bool LightCollection::initIntegrator()
    {
        // The current algorithm rasterizes emissive triangles in texture space,
        // and uses atomic operations to sum up the contribution from all covered texels.
        // We do this in a raster pass, so we get one thread per texel/triangle.
        // TODO: Make this deterministic with regards to floating-point errors in the integration.

        std::string s;
        if (findFileInDataDirectories("NVAPI/nvHLSLExtns.h", s) == false)
        {
            logError("LightCollection relies on NVAPI, which appears to be missing. Please make sure you have NVAPI installed (instructions are in the readme file)");
            return false;
        }

        // Create program.
        Program::Desc desc;
        desc.addShaderLibrary(kFileEmissiveIntegrator).vsEntry("vsMain").psEntry("psMain");
        mIntegrator.pProgram = GraphicsProgram::create(desc);
        if (!mIntegrator.pProgram) return false;

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

        return true;
    }

    bool LightCollection::setupMeshLights()
    {
        mMeshLights.clear();
        mpSamplerState = nullptr;
        mTriangleCount = 0;

        // Create mesh lights for all emissive mesh instances.
        for (uint32_t meshInstanceID = 0; meshInstanceID < mpScene->getMeshInstanceCount(); meshInstanceID++)
        {
            const MeshInstanceData& instanceData = mpScene->getMeshInstance(meshInstanceID);
            const MeshDesc& meshData = mpScene->getMesh(instanceData.meshID);
            const Material::SharedPtr& pMaterial = mpScene->getMaterial(instanceData.materialID);
            assert(pMaterial);

            if (pMaterial->isEmissive())
            {
                // We've found a mesh instance with an emissive material => Setup mesh light data.
                MeshLightData meshLight;
                meshLight.meshInstanceID = meshInstanceID;
                meshLight.triangleCount = meshData.indexCount / 3;
                meshLight.triangleOffset = mTriangleCount;
                meshLight.flags = PACK_EMISSIVE_TYPE(meshLight.flags, pMaterial->getEmissiveTexture() != nullptr ? ChannelTypeTexture : ChannelTypeConst);
                meshLight.emissiveColor = pMaterial->getEmissiveColor();
                meshLight.emissiveFactor = pMaterial->getEmissiveFactor();

                mMeshLights.push_back(meshLight);
                mTriangleCount += meshLight.triangleCount;

                // Store ptr to texture sampler. We currently assume all the mesh lights' materials have the same sampler, which is true in current Falcor.
                // If this changes in the future, we'll have to support multiple samplers.
                if (pMaterial->getEmissiveTexture())
                {
                    if (!mpSamplerState)
                    {
                        mpSamplerState = pMaterial->getSampler();
                    }
                    else if (mpSamplerState != pMaterial->getSampler())
                    {
                        logError("Material '" + pMaterial->getName() + "' is using a different sampler.");
                        return false;
                    }
                }
            }
        }

        return true;
    }

    void LightCollection::build(RenderContext* pRenderContext)
    {
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
            // Prepare GPU buffers.
            prepareTriangleData(pRenderContext);
            prepareMeshData();

            // Pre-integrate emissive triangles.
            // TODO: We might want to redo this in update() for animated meshes or after scale changes as that affects the flux.
            integrateEmissive(pRenderContext);

            mCPUInvalidData = CPUOutOfDateFlags::All;
            mStagingBufferValid = false;
            mStatsValid = false;

            prepareSyncCPUData(pRenderContext);
        }
    }

    void LightCollection::prepareTriangleData(RenderContext* pRenderContext)
    {
        // Create GPU buffers.
        assert(mTriangleCount > 0);
        const size_t bufSize = mTriangleCount * 3 * sizeof(glm::vec3);
        const size_t uvBufSize = mTriangleCount * 3 * sizeof(glm::vec2);

        mpMeshLightsVertexPos = Buffer::create(bufSize, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess, Buffer::CpuAccess::None);
        mpMeshLightsVertexPos->setName("LightCollection_MeshLightsVertexPos");
        mpMeshLightsTexCoords = Buffer::create(uvBufSize, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess, Buffer::CpuAccess::None);
        mpMeshLightsTexCoords->setName("LightCollection_MeshLightsTexCoords");
        mpTriangleData = StructuredBuffer::create(mpTriangleListBuilder->getProgram().get(), "gTriangleData", mTriangleCount);
        mpTriangleData->setName("LightCollection_TriangleData");

        // Compute triangle data (vertices, uv-coordinates, materialID) for all mesh lights.
        buildTriangleList(pRenderContext);
    }

    void LightCollection::prepareMeshData()
    {
        assert(mMeshLights.size() > 0);

        // Create buffer for the mesh data.
        mpMeshData = StructuredBuffer::create(mpTriangleListBuilder->getProgram().get(), "gLights.meshData", mMeshLights.size(), ResourceBindFlags::ShaderResource);
        if (!mpMeshData || mpMeshData->getElementSize() != sizeof(MeshLightData))
        {
            throw std::exception("Failed to create structured buffer of MeshLightData");
        }
        size_t meshDataSize = mMeshLights.size() * sizeof(mMeshLights[0]);
        assert(mpMeshData && mpMeshData->getSize() == meshDataSize);
        mpMeshData->setBlob(mMeshLights.data(), 0, meshDataSize);

        // Build a lookup table from mesh instance ID to emissive triangle offset.
        // This is useful in ray tracing for locating the emissive triangle that was hit for MIS computation etc.
        uint32_t instanceCount = mpScene->getMeshInstanceCount();
        assert(instanceCount > 0);
        std::vector<uint32_t> triangleOffsets(instanceCount, kInvalidIndex);
        for (const auto& it : mMeshLights)
        {
            assert(it.meshInstanceID < instanceCount);
            triangleOffsets[it.meshInstanceID] = it.triangleOffset;
        }

        // Create the GPU buffer.
        mpPerMeshInstanceOffset = TypedBuffer<uint32_t>::create(instanceCount, Resource::BindFlags::ShaderResource);
        mpPerMeshInstanceOffset->setName("LightCollection_PerMeshInstanceOffset");

        const size_t sizeInBytes = triangleOffsets.size() * sizeof(triangleOffsets[0]);
        assert(mpPerMeshInstanceOffset->getSize() == sizeInBytes);
        mpPerMeshInstanceOffset->setBlob(triangleOffsets.data(), 0, sizeInBytes);
    }

    void LightCollection::integrateEmissive(RenderContext* pRenderContext)
    {
        assert(mTriangleCount > 0);
        assert(mMeshLights.size() > 0);

        // 1st pass: Rasterize emissive triangles in texture space to sum up their texels.
        {
            // Re-allocate result buffer if needed.
            const uint32_t bufSize = mTriangleCount * sizeof(glm::vec4);
            if (!mIntegrator.pResultBuffer || mIntegrator.pResultBuffer->getSize() < bufSize)
            {
                mIntegrator.pResultBuffer = Buffer::create(bufSize, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess, Buffer::CpuAccess::None);
                mIntegrator.pResultBuffer->setName("LightCollection_IntegratorResults");
                assert(mIntegrator.pResultBuffer);
            }

            // Clear to zero before we start.
            pRenderContext->clearUAV(mIntegrator.pResultBuffer->getUAV().get(), glm::vec4(0.f));

            // Specialize the program and re-create the vars.
            prepareProgram(mIntegrator.pProgram.get());
            mIntegrator.pVars = GraphicsVars::create(mIntegrator.pProgram.get());
            if (!mIntegrator.pVars) throw std::exception("Failed to create program vars");

            // Bind our resources.
            bool success = setIntoProgramVars(mIntegrator.pVars.get(), mIntegrator.pVars->getConstantBuffer("CB"), "gLights");
            assert(success);

            mIntegrator.pVars["gTexelSum"] = mIntegrator.pResultBuffer;
            mIntegrator.pVars["gPointSampler"] = mIntegrator.pPointSampler;

            // Execute.
            pRenderContext->draw(mIntegrator.pState.get(), mIntegrator.pVars.get(), mTriangleCount * 3, 0);
        }

        // 2nd pass: Finalize the per-triangle flux values.
        {
            // Specialize the program and re-create the vars.
            prepareProgram(mpFinalizeIntegration->getProgram().get());
            mpFinalizeIntegration->setVars(nullptr); // Re-create vars.

            // Bind our resources.
            bool success = setIntoProgramVars(mpFinalizeIntegration->getVars().get(), mpFinalizeIntegration->getVars()->getConstantBuffer("FinalCB"), "gLights");
            assert(success);

            mpFinalizeIntegration["gTexelSum"] = mIntegrator.pResultBuffer;
            mpFinalizeIntegration["gTriangleData"] = mpTriangleData; // TODO: This buffer is already bound by setIntoProgramVars, is it dangerous to alias it like this?

            // Execute.
            assert(mpFinalizeIntegration->getThreadGroupSize().y == 1);
            uint32_t rows = div_round_up(mTriangleCount, mpFinalizeIntegration->getThreadGroupSize().x);
            mpFinalizeIntegration->execute(pRenderContext, mpFinalizeIntegration->getThreadGroupSize().x, rows);
        }
    }

    void LightCollection::computeStats() const
    {
        if (mStatsValid) return;

        // Read back the current data. This is potentially expensive.
        syncCPUData();

        // Stats on input data.
        MeshLightStats stats;
        stats.meshLightCount = (uint32_t)mMeshLights.size();
        stats.triangleCount = (uint32)mMeshLightTriangles.size();

        uint32_t trianglesTotal = 0;
        for (const auto& meshLight : mMeshLights)
        {
            if (meshLight.isTextured())
            {
                stats.meshesTextured++;
                stats.trianglesTextured += meshLight.triangleCount;
            }
            trianglesTotal += meshLight.triangleCount;
        }
        assert(trianglesTotal == stats.triangleCount);

        // Stats on pre-processed data.
        for (const auto& tri : mMeshLightTriangles)
        {
            assert(tri.luminousFlux >= 0.f);
            if (tri.luminousFlux == 0.f)
            {
                stats.trianglesCulled++;
            }
            else
            {
                // TODO: Currently we don't detect uniform radiance for textured lights, so just look at whether the mesh light is textured or not.
                // This code will change when we tag individual triangles as textured vs non-textured.
                if (mMeshLights[tri.lightIdx].isTextured()) stats.trianglesActiveTextured++;
                else stats.trianglesActiveUniform++;
            }
        }
        stats.trianglesActive = stats.trianglesActiveUniform + stats.trianglesActiveTextured;

        mMeshLightStats = stats;
        mStatsValid = true;
    }

    void LightCollection::buildTriangleList(RenderContext* pRenderContext)
    {
        assert(mMeshLights.size() > 0);

        // Bind scene.
        mpTriangleListBuilder["gScene"] = mpScene->getParameterBlock();

        // Bind our output buffers.
        mpTriangleListBuilder["gVertexPosOutput"] = mpMeshLightsVertexPos;
        mpTriangleListBuilder["gTexCoordsOutput"] = mpMeshLightsTexCoords;
        mpTriangleListBuilder["gTriangleData"] = mpTriangleData;

        // TODO: Single dispatch over all emissive triangles instad of per-mesh dispatches.
        // This code is not performance critical though, as it's currently only run once at init time.
        for (uint32_t lightIdx = 0; lightIdx < mMeshLights.size(); ++lightIdx)
        {
            const MeshLightData& meshLight = mMeshLights[lightIdx];

            mpTriangleListBuilder["PerMeshCB"]["gLightIdx"] = lightIdx;
            mpTriangleListBuilder["PerMeshCB"]["gMeshInstanceID"] = meshLight.meshInstanceID;
            mpTriangleListBuilder["PerMeshCB"]["gTriangleCount"] = meshLight.triangleCount;
            mpTriangleListBuilder["PerMeshCB"]["gTriangleOffset"] = meshLight.triangleOffset;

            // as each kernel will write to non-overlapping parts of the output buffers.
            mpTriangleListBuilder->execute(pRenderContext, meshLight.triangleCount, 1u, 1u);
        }
    }

    void LightCollection::updateTrianglePositions(RenderContext* pRenderContext, const std::vector<uint32_t>& updatedLights)
    {
        // This pass pre-transforms all emissive triangles into world space and updates their area and face normals.
        // It is executed if any geometry in the scene has moved, which is wasteful since it will update also things
        // that didn't move. However, due to the CPU overhead of doing per-mesh dispatches as before, it's still faster.
        // TODO: Test using ExecuteIndirect to do the dispatches on the GPU for only the updated meshes.
        // Alternatively, upload the list of updated meshes and early out unnecessary threads at runtime.
        assert(!updatedLights.empty());

        // Bind scene.
        mpTrianglePositionUpdater["gScene"] = mpScene->getParameterBlock();

        // Bind our resources.
        mpTrianglePositionUpdater["gVertexPosOutput"] = mpMeshLightsVertexPos;
        mpTrianglePositionUpdater["gTriangleData"] = mpTriangleData;
        mpTrianglePositionUpdater["gMeshData"] = mpMeshData;
        mpTrianglePositionUpdater["PerMeshCB"]["gTriangleCount"] = mTriangleCount;

        // Run compute pass to update all triangles.
        mpTrianglePositionUpdater->execute(pRenderContext, mTriangleCount, 1u, 1u);

        mCPUInvalidData |= (CPUOutOfDateFlags::Positions | CPUOutOfDateFlags::TriangleData);
        mStagingBufferValid = false;
    }

    bool LightCollection::setIntoBlockCommon(const ParameterBlock::SharedPtr& pBlock, const ConstantBuffer::SharedPtr& pCB, const std::string& varName) const
    {
        assert(pBlock);
        assert(pCB);

        // Check that the struct exists.
        if (pCB->getVariableOffset(varName) == ConstantBuffer::kInvalidOffset)
        {
            logError("LightCollection::setIntoBlockCommon() - Variable " + varName + " does not exist");
            return false;
        }
        std::string prefix = varName + ".";

        // Ok. The struct exists.
        // In the following we validate it has the correct fields and set the data.

        // Set variables.
        pCB[prefix + "triangleCount"] = mTriangleCount;
        pCB[prefix + "meshCount"] = (uint32_t)mMeshLights.size();

        // Bind mesh light triangles.
        if (mTriangleCount > 0)
        {
            // Bind per-triangle data (these buffers must exist if triangle count is > 0).
            pBlock[prefix + "meshLightsVertexPos"] = mpMeshLightsVertexPos;
            pBlock[prefix + "meshLightsTexCoords"] = mpMeshLightsTexCoords;
            pBlock[prefix + "triangleData"] = mpTriangleData;
        }

        // Bind per-mesh instance triangle offsets (if buffer exists, reset binding otherwise).
        // This buffer can exist even if mTriangleCount == 0. It would just be an array of kInvalidIndex then.
        pBlock[prefix + "perMeshInstanceOffset"] = mpPerMeshInstanceOffset->asTypedBufferBase();

        // Bind mesh lights array.
        if (!mMeshLights.empty())
        {
            static_assert(sizeof(MeshLightData) % (sizeof(float) * 4) == 0, "MeshLightData size should be a multiple of 16");

            // Bind mesh data.
            pBlock[prefix + "meshData"] = mpMeshData;

            // Validate that the emissive textures array exists and is of correct size.
            auto pTexVar = pBlock->getReflection()->getResource(prefix + "emissiveTextures");
            assert(pTexVar && pTexVar->getType()->asArrayType()->getArraySize() == mMeshLights.size());

            // Bind array of emissive textures. One texture per mesh light (can be the same for multiple lights).
            // TODO: Index into a compact array of emissive textures instead.
            for (size_t lightIdx = 0; lightIdx < mMeshLights.size(); lightIdx++)
            {
                const MeshInstanceData& instanceData = mpScene->getMeshInstance(mMeshLights[lightIdx].meshInstanceID);
                const Material::SharedPtr& pMaterial = mpScene->getMaterial(instanceData.materialID);
                assert(pMaterial);

                const std::string texVar = prefix + "emissiveTextures[" + std::to_string(lightIdx) + "]";
                pBlock[texVar] = pMaterial->getEmissiveTexture();   // May be nullptr
            }
        }

        // Set sampler state. We only have a single sampler that is used for all emissive textures.
        if (mpSamplerState)
        {
            bool success = pBlock->setSampler(prefix + "samplerState", mpSamplerState);
            assert(success);
        }

        return true;
    }

    void LightCollection::copyDataToStagingBuffer(RenderContext* pRenderContext) const
    {
        if (mStagingBufferValid) return;

        // Allocate staging buffer for readback. The data from our different GPU buffers is stored consecutively.
        const size_t stagingSize = mpMeshLightsVertexPos->getSize() + mpMeshLightsTexCoords->getSize() + mpTriangleData->getSize();
        if (!mpStagingBuffer || mpStagingBuffer->getSize() < stagingSize)
        {
            mpStagingBuffer = Buffer::create(stagingSize, Resource::BindFlags::None, Buffer::CpuAccess::Read);
            mpStagingBuffer->setName("LightCollection_StagingBuffer");
            mCPUInvalidData = CPUOutOfDateFlags::All;
        }

        // Schedule the copy operations for data that is invalid.
        // Note that the staging buffer is allocated for the worst-case encountered so far.
        // If the number of triangles ever decreases, we'll be copying unnecessary data. This currently doesn't happen as geometry is not added/removed from the scene.
        // TODO: Update this code if we start removing geometry dynamically.
        assert(mCPUInvalidData != CPUOutOfDateFlags::None); // We shouldn't get here unless at least some data is out of date.
        bool copyPositions = (mCPUInvalidData & CPUOutOfDateFlags::Positions) == CPUOutOfDateFlags::Positions;
        bool copyTexCoords = (mCPUInvalidData & CPUOutOfDateFlags::TexCoords) == CPUOutOfDateFlags::TexCoords;
        bool copyTriangleData = (mCPUInvalidData & CPUOutOfDateFlags::TriangleData) == CPUOutOfDateFlags::TriangleData;

        if (copyPositions) pRenderContext->copyBufferRegion(mpStagingBuffer.get(), 0, mpMeshLightsVertexPos.get(), 0, mpMeshLightsVertexPos->getSize());
        if (copyTexCoords) pRenderContext->copyBufferRegion(mpStagingBuffer.get(), mpMeshLightsVertexPos->getSize(), mpMeshLightsTexCoords.get(), 0, mpMeshLightsTexCoords->getSize());
        if (copyTriangleData) pRenderContext->copyBufferRegion(mpStagingBuffer.get(), mpMeshLightsVertexPos->getSize() + mpMeshLightsTexCoords->getSize(), mpTriangleData.get(), 0, mpTriangleData->getSize());

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

        assert(mStagingBufferValid);
        const void* mappedData = mpStagingBuffer->map(Buffer::MapType::Read);
        const glm::vec3* vertexPos = reinterpret_cast<const glm::vec3*>(mappedData);
        const glm::vec2* vertexTexCrd = reinterpret_cast<const glm::vec2*>(reinterpret_cast<uintptr_t>(mappedData) + mpMeshLightsVertexPos->getSize());
        assert(mpTriangleData);
        if (mpTriangleData->getElementSize() != sizeof(EmissiveTriangle)) throw std::exception("Struct EmissiveTriangle size mismatch between CPU/GPU");
        const EmissiveTriangle* triangleData = reinterpret_cast<const EmissiveTriangle*>(reinterpret_cast<uintptr_t>(mappedData) + mpMeshLightsVertexPos->getSize() + mpMeshLightsTexCoords->getSize());

        assert(mTriangleCount > 0);
        const bool updatePositions = (mCPUInvalidData & CPUOutOfDateFlags::Positions) == CPUOutOfDateFlags::Positions;
        const bool updateTexCoords = (mCPUInvalidData & CPUOutOfDateFlags::TexCoords) == CPUOutOfDateFlags::TexCoords;
        const bool updateTriangleData = (mCPUInvalidData & CPUOutOfDateFlags::TriangleData) == CPUOutOfDateFlags::TriangleData;

        assert(mMeshLightTriangles.size() == (size_t)mTriangleCount);
        for (uint32_t triIdx = 0; triIdx < mTriangleCount; triIdx++)
        {
            auto& tri = mMeshLightTriangles[triIdx];

            // Store triangle data.
            if (updateTriangleData)
            {
                tri.lightIdx = triangleData[triIdx].lightIdx;
                tri.normal = triangleData[triIdx].normal;
                tri.area = triangleData[triIdx].area;
                tri.averageRadiance = triangleData[triIdx].averageRadiance;
                tri.luminousFlux = triangleData[triIdx].flux;
            }

            // Store texcoords.
            if (updateTexCoords)
            {
                for (uint32_t j = 0; j < 3; j++) tri.vtx[j].uv = vertexTexCrd[triIdx * 3 + j];
            }

            // Store positions.
            if (updatePositions)
            {
                for (uint32_t j = 0; j < 3; j++) tri.vtx[j].pos = vertexPos[triIdx * 3 + j];
            }
        }

        mpStagingBuffer->unmap();

        mCPUInvalidData = CPUOutOfDateFlags::None;
    }
}
