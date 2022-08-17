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
#include "AnimationController.h"
#include "Core/API/RenderContext.h"
#include "Utils/Timing/Profiler.h"
#include "Scene/Scene.h"
#include <fstream>

namespace Falcor
{
    namespace
    {
        const std::string kWorldMatrices = "worldMatrices";
        const std::string kInverseTransposeWorldMatrices = "inverseTransposeWorldMatrices";
        const std::string kPrevWorldMatrices = "prevWorldMatrices";
        const std::string kPrevInverseTransposeWorldMatrices = "prevInverseTransposeWorldMatrices";
    }

    AnimationController::AnimationController(Scene* pScene, const StaticVertexVector& staticVertexData, const SkinningVertexVector& skinningVertexData, uint32_t prevVertexCount, const std::vector<Animation::SharedPtr>& animations)
        : mpScene(pScene)
        , mAnimations(animations)
        , mNodesEdited(pScene->mSceneGraph.size())
        , mLocalMatrices(pScene->mSceneGraph.size())
        , mGlobalMatrices(pScene->mSceneGraph.size())
        , mInvTransposeGlobalMatrices(pScene->mSceneGraph.size())
        , mMatricesChanged(pScene->mSceneGraph.size())
    {
        // Create GPU resources.
        FALCOR_ASSERT(mLocalMatrices.size() <= std::numeric_limits<uint32_t>::max());

        if (!mLocalMatrices.empty())
        {
            mpWorldMatricesBuffer = Buffer::createStructured(sizeof(float4x4), (uint32_t)mLocalMatrices.size(), Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false);
            mpWorldMatricesBuffer->setName("AnimationController::mpWorldMatricesBuffer");
            mpPrevWorldMatricesBuffer = Buffer::createStructured(sizeof(float4x4), (uint32_t)mLocalMatrices.size(), Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false);
            mpPrevWorldMatricesBuffer->setName("AnimationController::mpPrevWorldMatricesBuffer");
            mpInvTransposeWorldMatricesBuffer = Buffer::createStructured(sizeof(float4x4), (uint32_t)mLocalMatrices.size(), Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false);
            mpInvTransposeWorldMatricesBuffer->setName("AnimationController::mpInvTransposeWorldMatricesBuffer");
            mpPrevInvTransposeWorldMatricesBuffer = Buffer::createStructured(sizeof(float4x4), (uint32_t)mLocalMatrices.size(), Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false);
            mpPrevInvTransposeWorldMatricesBuffer->setName("AnimationController::mpPrevInvTransposeWorldMatricesBuffer");
        }

        // An extra buffer is required to store the previous frame vertex data for skinned and vertex-animated meshes.
        // The buffer contains data for skinned meshes first, followed by vertex-animated meshes.
        //
        // Initialize the previous positions for skinned vertices. AnimatedVertexCache will initialize the remaining data if necessary
        // This ensures we have valid data in the buffer before the skinning pass runs for the first time.
        if (prevVertexCount > 0)
        {
            std::vector<PrevVertexData> prevVertexData(prevVertexCount);
            for (size_t i = 0; i < skinningVertexData.size(); i++)
            {
                uint32_t staticIndex = skinningVertexData[i].staticIndex;
                prevVertexData[i].position = staticVertexData[staticIndex].position;
            }
            mpPrevVertexData = Buffer::createStructured(sizeof(PrevVertexData), prevVertexCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, Buffer::CpuAccess::None, prevVertexData.data(), false);
            mpPrevVertexData->setName("AnimationController::mpPrevVertexData");
        }

        createSkinningPass(staticVertexData, skinningVertexData);

        // Determine length of global animation loop.
        for (const auto& pAnimation : mAnimations)
        {
            mGlobalAnimationLength = std::max(mGlobalAnimationLength, pAnimation->getDuration());
        }
    }

    AnimationController::UniquePtr AnimationController::create(Scene* pScene, const StaticVertexVector& staticVertexData, const SkinningVertexVector& skinningVertexData, uint32_t prevVertexCount, const std::vector<Animation::SharedPtr>& animations)
    {
        return UniquePtr(new AnimationController(pScene, staticVertexData, skinningVertexData, prevVertexCount, animations));
    }

    void AnimationController::addAnimatedVertexCaches(std::vector<CachedCurve>&& cachedCurves, std::vector<CachedMesh>&& cachedMeshes, const StaticVertexVector& staticVertexData)
    {
        size_t totalAnimatedMeshVertexCount = 0;

        for (auto& cache : cachedMeshes)
        {
            totalAnimatedMeshVertexCount += mpScene->getMesh(cache.meshID).vertexCount;
        }
        for (auto& cache : cachedCurves)
        {
            if (cache.tessellationMode != CurveTessellationMode::LinearSweptSphere)
            {
                totalAnimatedMeshVertexCount += mpScene->getMesh(MeshID{ cache.geometryID }).vertexCount;
            }
        }

        if (totalAnimatedMeshVertexCount > 0)
        {
            // Initialize remaining previous position data
            std::vector<PrevVertexData> prevVertexData;
            prevVertexData.reserve(totalAnimatedMeshVertexCount);

            for (auto& cache : cachedMeshes)
            {
                uint32_t offset = mpScene->getMesh(cache.meshID).vbOffset;
                for (size_t i = 0; i < cache.vertexData.front().size(); i++)
                {
                    prevVertexData.push_back({ staticVertexData[offset + i].position });
                }
            }

            for (auto& cache : cachedCurves)
            {
                if (cache.tessellationMode != CurveTessellationMode::LinearSweptSphere)
                {
                    uint32_t offset = mpScene->getMesh(MeshID{ cache.geometryID }).vbOffset;
                    uint32_t vertexCount = mpScene->getMesh(MeshID{ cache.geometryID }).vertexCount;
                    for (size_t i = 0; i < vertexCount; i++)
                    {
                        prevVertexData.push_back({ staticVertexData[offset + i].position });
                    }
                }
            }

            uint32_t byteOffset = 0;
            if (!cachedMeshes.empty())
            {
                byteOffset = mpScene->getMesh(cachedMeshes.front().meshID).prevVbOffset * sizeof(PrevVertexData);
            }
            else // !cachedCurves.empty()
            {
                for (auto& cache : cachedCurves)
                {
                    if (cache.tessellationMode != CurveTessellationMode::LinearSweptSphere)
                    {
                        byteOffset = mpScene->getMesh(MeshID{ cache.geometryID }).prevVbOffset * sizeof(PrevVertexData);
                        break;
                    }
                }
            }

            mpPrevVertexData->setBlob(prevVertexData.data(), byteOffset, prevVertexData.size() * sizeof(PrevVertexData));
        }

        mpVertexCache = AnimatedVertexCache::create(mpScene, mpPrevVertexData, std::move(cachedCurves), std::move(cachedMeshes));

        // Note: It is a workaround to have two pre-infinity behaviors for the cached animation.
        // We need `Cycle` behavior when the length of cached animation is smaller than the length of mesh animation (e.g., tiger forest).
        // We need `Constant` behavior when both animation lengths are equal (e.g., a standalone tiger).
        if (mpVertexCache->getGlobalAnimationLength() < mGlobalAnimationLength)
        {
            mpVertexCache->setPreInfinityBehavior(Animation::Behavior::Cycle);
        }
    }

    void AnimationController::setEnabled(bool enabled)
    {
        mEnabled = enabled;
    }

    void AnimationController::setIsLooped(bool looped)
    {
        mLoopAnimations = looped;

        if (mpVertexCache)
        {
            mpVertexCache->setIsLooped(looped);
        }
    }

    void AnimationController::initLocalMatrices()
    {
        for (size_t i = 0; i < mLocalMatrices.size(); i++)
        {
            mLocalMatrices[i] = mpScene->mSceneGraph[i].transform;
        }
    }

    bool AnimationController::animate(RenderContext* pContext, double currentTime)
    {
        FALCOR_PROFILE("animate");

        std::fill(mMatricesChanged.begin(), mMatricesChanged.end(), false);

        // Check for edited scene nodes and update local matrices.
        const auto& sceneGraph = mpScene->mSceneGraph;
        bool edited = false;
        for (size_t i = 0; i < sceneGraph.size(); ++i)
        {
            if (mNodesEdited[i])
            {
                mLocalMatrices[i] = sceneGraph[i].transform;
                mNodesEdited[i] = false;
                mMatricesChanged[i] = true;
                edited = true;
            }
        }

        bool changed = false;
        double time = mLoopAnimations ? std::fmod(currentTime, mGlobalAnimationLength) : currentTime;

        // Check if animation controller was enabled/disabled since last call.
        // When enabling/disabling, all data for the current and previous frame is initialized,
        // including transformation matrices, dynamic vertex data etc.
        if (mFirstUpdate || mEnabled != mPrevEnabled)
        {
            initLocalMatrices();
            if (mEnabled)
            {
                updateLocalMatrices(time);
                mTime = mPrevTime = time;
            }
            updateWorldMatrices(true);
            uploadWorldMatrices(true);

            if (!sceneGraph.empty())
            {
                FALCOR_ASSERT(mpWorldMatricesBuffer && mpPrevWorldMatricesBuffer);
                FALCOR_ASSERT(mpInvTransposeWorldMatricesBuffer && mpPrevInvTransposeWorldMatricesBuffer);
                pContext->copyResource(mpPrevWorldMatricesBuffer.get(), mpWorldMatricesBuffer.get());
                pContext->copyResource(mpPrevInvTransposeWorldMatricesBuffer.get(), mpInvTransposeWorldMatricesBuffer.get());
                bindBuffers();
                executeSkinningPass(pContext, true);
            }

            if (mpVertexCache)
            {
                if (mEnabled && mpVertexCache->hasAnimations())
                {
                    // Recompute time based on the cycle length of vertex caches.
                    double vertexCacheTime = (mGlobalAnimationLength == 0) ? currentTime : time;
                    mpVertexCache->animate(pContext, vertexCacheTime);
                }
                mpVertexCache->copyToPrevVertices(pContext);
            }

            mFirstUpdate = false;
            mPrevEnabled = mEnabled;
            changed = true;
        }

        // Perform incremental update.
        // This updates all animated matrices and dynamic vertex data.
        if (edited || mEnabled && (time != mTime || mTime != mPrevTime))
        {
            if (edited || hasAnimations())
            {
                FALCOR_ASSERT(mpWorldMatricesBuffer && mpPrevWorldMatricesBuffer);
                FALCOR_ASSERT(mpInvTransposeWorldMatricesBuffer && mpPrevInvTransposeWorldMatricesBuffer);
                swap(mpPrevWorldMatricesBuffer, mpWorldMatricesBuffer);
                swap(mpPrevInvTransposeWorldMatricesBuffer, mpInvTransposeWorldMatricesBuffer);
                updateLocalMatrices(time);
                updateWorldMatrices();
                uploadWorldMatrices();
                bindBuffers();
                executeSkinningPass(pContext);
                changed = true;
            }

            if (mpVertexCache && mpVertexCache->hasAnimations())
            {
                // Recompute time based on the cycle length of vertex caches.
                double vertexCacheTime = (mGlobalAnimationLength == 0) ? currentTime : time;
                mpVertexCache->animate(pContext, vertexCacheTime);
                changed = true;
            }

            mPrevTime = mTime;
            mTime = time;
        }

        return changed;
    }

    void AnimationController::updateLocalMatrices(double time)
    {
        for (auto& pAnimation : mAnimations)
        {
            NodeID nodeID = pAnimation->getNodeID();
            FALCOR_ASSERT(nodeID.get() < mLocalMatrices.size());
            mLocalMatrices[nodeID.get()] = pAnimation->animate(time);
            mMatricesChanged[nodeID.get()] = true;
        }
    }

    void AnimationController::updateWorldMatrices(bool updateAll)
    {
        const auto& sceneGraph = mpScene->mSceneGraph;

        for (size_t i = 0; i < mGlobalMatrices.size(); i++)
        {
            // Propagate matrix change flag to children.
            if (sceneGraph[i].parent != NodeID::Invalid())
            {
                mMatricesChanged[i] = mMatricesChanged[i] || mMatricesChanged[sceneGraph[i].parent.get()];
            }

            if (!mMatricesChanged[i] && !updateAll) continue;

            mGlobalMatrices[i] = mLocalMatrices[i];

            if (mpScene->mSceneGraph[i].parent != NodeID::Invalid())
            {
                mGlobalMatrices[i] = mGlobalMatrices[sceneGraph[i].parent.get()] * mGlobalMatrices[i];
            }

            mInvTransposeGlobalMatrices[i] = transpose(inverse(mGlobalMatrices[i]));

            if (mpSkinningPass)
            {
                mSkinningMatrices[i] = mGlobalMatrices[i] * sceneGraph[i].localToBindSpace;
                mInvTransposeSkinningMatrices[i] = transpose(inverse(mSkinningMatrices[i]));
            }
        }
    }

    void AnimationController::uploadWorldMatrices(bool uploadAll)
    {
        if (mGlobalMatrices.empty()) return;

        FALCOR_ASSERT(mGlobalMatrices.size() == mInvTransposeGlobalMatrices.size());
        FALCOR_ASSERT(mpWorldMatricesBuffer && mpInvTransposeWorldMatricesBuffer);

        if (uploadAll)
        {
            // Upload all matrices.
            mpWorldMatricesBuffer->setBlob(mGlobalMatrices.data(), 0, mpWorldMatricesBuffer->getSize());
            mpInvTransposeWorldMatricesBuffer->setBlob(mInvTransposeGlobalMatrices.data(), 0, mpInvTransposeWorldMatricesBuffer->getSize());
        }
        else
        {
            // Upload changed matrices only.
            for (size_t i = 0; i < mGlobalMatrices.size();)
            {
                // Detect ranges of consecutive matrices that have all changed or not.
                size_t offset = i;
                bool changed = mMatricesChanged[i];
                while (i < mGlobalMatrices.size() && mMatricesChanged[i] == changed) ++i;

                // Upload range of changed matrices.
                if (changed)
                {
                    size_t count = i - offset;
                    mpWorldMatricesBuffer->setBlob(&mGlobalMatrices[offset], offset * sizeof(float4x4), count * sizeof(float4x4));
                    mpInvTransposeWorldMatricesBuffer->setBlob(&mInvTransposeGlobalMatrices[offset], offset * sizeof(float4x4), count * sizeof(float4x4));
                }
            }
        }
    }

    void AnimationController::bindBuffers()
    {
        ParameterBlock* pBlock = mpScene->mpSceneBlock.get();
        pBlock->setBuffer(kWorldMatrices, mpWorldMatricesBuffer);
        pBlock->setBuffer(kInverseTransposeWorldMatrices, mpInvTransposeWorldMatricesBuffer);
        bool usePrev = mEnabled && hasAnimations();
        pBlock->setBuffer(kPrevWorldMatrices, usePrev ? mpPrevWorldMatricesBuffer : mpWorldMatricesBuffer);
        pBlock->setBuffer(kPrevInverseTransposeWorldMatrices, usePrev ? mpPrevInvTransposeWorldMatricesBuffer : mpInvTransposeWorldMatricesBuffer);
    }

    uint64_t AnimationController::getMemoryUsageInBytes() const
    {
        uint64_t m = 0;
        m += mpWorldMatricesBuffer ? mpWorldMatricesBuffer->getSize() : 0;
        m += mpPrevWorldMatricesBuffer ? mpPrevWorldMatricesBuffer->getSize() : 0;
        m += mpInvTransposeWorldMatricesBuffer ? mpInvTransposeWorldMatricesBuffer->getSize() : 0;
        m += mpPrevInvTransposeWorldMatricesBuffer ? mpPrevInvTransposeWorldMatricesBuffer->getSize() : 0;
        m += mpSkinningMatricesBuffer ? mpSkinningMatricesBuffer->getSize() : 0;
        m += mpInvTransposeSkinningMatricesBuffer ? mpInvTransposeSkinningMatricesBuffer->getSize() : 0;
        m += mpMeshBindMatricesBuffer ? mpMeshBindMatricesBuffer->getSize() : 0;
        m += mpStaticVertexData ? mpStaticVertexData->getSize() : 0;
        m += mpSkinningVertexData ? mpSkinningVertexData->getSize() : 0;
        m += mpPrevVertexData ? mpPrevVertexData->getSize() : 0;
        m += mpVertexCache ? mpVertexCache->getMemoryUsageInBytes() : 0;
        return m;
    }

    void AnimationController::createSkinningPass(const std::vector<PackedStaticVertexData>& staticVertexData, const std::vector<SkinningVertexData>& skinningVertexData)
    {
        if (staticVertexData.empty()) return;

        // We always copy the static data, to initialize the non-skinned vertices.
        FALCOR_ASSERT(mpScene->getMeshVao());
        const Buffer::SharedPtr& pVB = mpScene->getMeshVao()->getVertexBuffer(Scene::kStaticDataBufferIndex);
        FALCOR_ASSERT(pVB->getSize() == staticVertexData.size() * sizeof(staticVertexData[0]));
        pVB->setBlob(staticVertexData.data(), 0, pVB->getSize());

        if (!skinningVertexData.empty())
        {
            mSkinningMatrices.resize(mpScene->mSceneGraph.size());
            mInvTransposeSkinningMatrices.resize(mSkinningMatrices.size());
            mMeshBindMatrices.resize(mpScene->mSceneGraph.size());

            mpSkinningPass = ComputePass::create("Scene/Animation/Skinning.slang");
            auto block = mpSkinningPass->getVars()["gData"];

            // Initialize mesh bind transforms
            std::vector<float4x4> meshInvBindMatrices(mMeshBindMatrices.size());
            for (size_t i = 0; i < mpScene->mSceneGraph.size(); i++)
            {
                mMeshBindMatrices[i] = mpScene->mSceneGraph[i].meshBind;
                meshInvBindMatrices[i] = rmcv::inverse(mMeshBindMatrices[i]);
            }

            // Bind vertex data.
            FALCOR_ASSERT(staticVertexData.size() <= std::numeric_limits<uint32_t>::max());
            FALCOR_ASSERT(skinningVertexData.size() <= std::numeric_limits<uint32_t>::max());
            mpStaticVertexData = Buffer::createStructured(block["staticData"], (uint32_t)staticVertexData.size(), ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, staticVertexData.data(), false);
            mpStaticVertexData->setName("AnimationController::mpStaticVertexData");
            mpSkinningVertexData = Buffer::createStructured(block["skinningData"], (uint32_t)skinningVertexData.size(), ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, skinningVertexData.data(), false);
            mpSkinningVertexData->setName("AnimationController::mpSkinningVertexData");

            block["staticData"] = mpStaticVertexData;
            block["skinningData"] = mpSkinningVertexData;
            block["skinnedVertices"] = pVB;
            block["prevSkinnedVertices"] = mpPrevVertexData;

            // Bind transforms.
            FALCOR_ASSERT(mSkinningMatrices.size() < std::numeric_limits<uint32_t>::max());
            mpMeshBindMatricesBuffer = Buffer::createStructured(sizeof(float4x4), (uint32_t)mSkinningMatrices.size(), ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, mMeshBindMatrices.data(), false);
            mpMeshBindMatricesBuffer->setName("AnimationController::mpMeshBindMatricesBuffer");
            mpMeshInvBindMatricesBuffer = Buffer::createStructured(sizeof(float4x4), (uint32_t)mSkinningMatrices.size(), ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, meshInvBindMatrices.data(), false);
            mpMeshInvBindMatricesBuffer->setName("AnimationController::mpMeshInvBindMatricesBuffer");
            mpSkinningMatricesBuffer = Buffer::createStructured(sizeof(float4x4), (uint32_t)mSkinningMatrices.size(), ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false);
            mpSkinningMatricesBuffer->setName("AnimationController::mpSkinningMatricesBuffer");
            mpInvTransposeSkinningMatricesBuffer = Buffer::createStructured(sizeof(float4x4), (uint32_t)mSkinningMatrices.size(), ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false);
            mpInvTransposeSkinningMatricesBuffer->setName("AnimationController::mpInvTransposeSkinningMatricesBuffer");

            block["boneMatrices"].setBuffer(mpSkinningMatricesBuffer);
            block["inverseTransposeBoneMatrices"].setBuffer(mpInvTransposeSkinningMatricesBuffer);
            block["meshBindMatrices"].setBuffer(mpMeshBindMatricesBuffer);
            block["meshInvBindMatrices"].setBuffer(mpMeshInvBindMatricesBuffer);

            mSkinningDispatchSize = (uint32_t)skinningVertexData.size();
        }
    }

    void AnimationController::executeSkinningPass(RenderContext* pContext, bool initPrev)
    {
        if (!mpSkinningPass) return;

        // Update matrices.
        FALCOR_ASSERT(mpSkinningMatricesBuffer && mpInvTransposeSkinningMatricesBuffer);
        mpSkinningMatricesBuffer->setBlob(mSkinningMatrices.data(), 0, mpSkinningMatricesBuffer->getSize());
        mpInvTransposeSkinningMatricesBuffer->setBlob(mInvTransposeSkinningMatrices.data(), 0, mpInvTransposeSkinningMatricesBuffer->getSize());

        // Execute skinning pass.
        auto vars = mpSkinningPass->getVars()["gData"];
        vars["inverseTransposeWorldMatrices"].setBuffer(mpInvTransposeWorldMatricesBuffer);
        vars["worldMatrices"].setBuffer(mpWorldMatricesBuffer);
        vars["initPrev"] = initPrev;
        mpSkinningPass->execute(pContext, mSkinningDispatchSize, 1, 1);
    }

    void AnimationController::renderUI(Gui::Widgets& widget)
    {
        if (widget.checkbox("Loop Animations", mLoopAnimations))
        {
            if (mpVertexCache)
            {
                mpVertexCache->setIsLooped(mLoopAnimations);
            }
        }
        widget.tooltip("Enable/disable global animation looping.");

        for (auto& animation : mAnimations)
        {
            if (auto animGroup = widget.group(animation->getName()))
            {
                animation->renderUI(animGroup);
            }
        }
    }
}
