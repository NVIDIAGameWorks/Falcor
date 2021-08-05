/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
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
#include "AnimationController.h"
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

    AnimationController::AnimationController(Scene* pScene, const StaticVertexVector& staticVertexData, const DynamicVertexVector& dynamicVertexData, const std::vector<Animation::SharedPtr>& animations)
        : mpScene(pScene)
        , mLocalMatrices(pScene->mSceneGraph.size())
        , mGlobalMatrices(pScene->mSceneGraph.size())
        , mInvTransposeGlobalMatrices(pScene->mSceneGraph.size())
        , mMatricesAnimated(pScene->mSceneGraph.size())
        , mMatricesChanged(pScene->mSceneGraph.size())
        , mAnimations(animations)
    {
        initFlags();

        // Create GPU resources.
        assert(mLocalMatrices.size() * 4 <= std::numeric_limits<uint32_t>::max());
        uint32_t float4Count = (uint32_t)mLocalMatrices.size() * 4;

        mpWorldMatricesBuffer = Buffer::createStructured(sizeof(float4), float4Count, Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false);
        mpWorldMatricesBuffer->setName("AnimationController::mpWorldMatricesBuffer");
        mpPrevWorldMatricesBuffer = Buffer::createStructured(sizeof(float4), float4Count, Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false);
        mpPrevWorldMatricesBuffer->setName("AnimationController::mpPrevWorldMatricesBuffer");
        mpInvTransposeWorldMatricesBuffer = Buffer::createStructured(sizeof(float4), float4Count, Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false);
        mpInvTransposeWorldMatricesBuffer->setName("AnimationController::mpInvTransposeWorldMatricesBuffer");
        mpPrevInvTransposeWorldMatricesBuffer = Buffer::createStructured(sizeof(float4), float4Count, Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false);
        mpPrevInvTransposeWorldMatricesBuffer->setName("AnimationController::mpPrevInvTransposeWorldMatricesBuffer");

        createSkinningPass(staticVertexData, dynamicVertexData);

        // Determine length of global animation loop.
        for (const auto& pAnimation : mAnimations)
        {
            mGlobalAnimationLength = std::max(mGlobalAnimationLength, pAnimation->getDuration());
        }
    }

    AnimationController::UniquePtr AnimationController::create(Scene* pScene, const StaticVertexVector& staticVertexData, const DynamicVertexVector& dynamicVertexData, const std::vector<Animation::SharedPtr>& animations)
    {
        return UniquePtr(new AnimationController(pScene, staticVertexData, dynamicVertexData, animations));
    }

    void AnimationController::addAnimatedVertexCaches(std::vector<CachedCurve>& cachedCurves, std::vector<CachedMesh>& cachedMeshes)
    {
        mpVertexCache = AnimatedVertexCache::create(mpScene, cachedCurves, cachedMeshes);
    }

    void AnimationController::setEnabled(bool enabled)
    {
        mEnabled = enabled;
    }

    void AnimationController::initFlags()
    {
        std::fill(mMatricesAnimated.begin(), mMatricesAnimated.end(), false);

        // Tag all matrices affected by an animation.
        for (const auto& pAnimation : mAnimations)
        {
            mMatricesAnimated[pAnimation->getNodeID()] = true;
        }

        // Traverse the scene graph hierarchy to propagate the flags.
        assert(mpScene->mSceneGraph.size() == mMatricesAnimated.size());
        for (size_t i = 0; i < mMatricesAnimated.size(); i++)
        {
            if (uint32_t parent = mpScene->mSceneGraph[i].parent; parent != SceneBuilder::kInvalidNode)
            {
                assert(parent < i);
                mMatricesAnimated[i] = mMatricesAnimated[i] || mMatricesAnimated[parent];
            }
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
        PROFILE("animate");

        bool changed = false;
        double time = mLoopAnimations ? std::fmod(currentTime, mGlobalAnimationLength) : currentTime;
        std::fill(mMatricesChanged.begin(), mMatricesChanged.end(), false);

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
            pContext->copyResource(mpPrevWorldMatricesBuffer.get(), mpWorldMatricesBuffer.get());
            pContext->copyResource(mpPrevInvTransposeWorldMatricesBuffer.get(), mpInvTransposeWorldMatricesBuffer.get());
            bindBuffers();
            executeSkinningPass(pContext, true);

            if (mpVertexCache) mpVertexCache->copyToPrevVertices(pContext);

            mFirstUpdate = false;
            mPrevEnabled = mEnabled;
            changed = true;
        }

        // Perform incremental update.
        // This updates all animated matrices and dynamic vertex data.
        if (mEnabled && (time != mTime || mTime != mPrevTime))
        {
            if (hasAnimations())
            {
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
                double time = mLoopAnimations ? std::fmod(currentTime, mpVertexCache->getGlobalAnimationLength()) : currentTime;
                mpVertexCache->animate(pContext, time);
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
            uint32_t nodeID = pAnimation->getNodeID();
            mLocalMatrices[nodeID] = pAnimation->animate(time);
            mMatricesChanged[nodeID] = true;
        }
    }

    void AnimationController::updateWorldMatrices(bool updateAll)
    {
        for (size_t i = 0; i < mGlobalMatrices.size(); i++)
        {
            if (!mMatricesAnimated[i] && !updateAll) continue;

            mGlobalMatrices[i] = mLocalMatrices[i];

            if (mpScene->mSceneGraph[i].parent != SceneBuilder::kInvalidNode)
            {
                mGlobalMatrices[i] = mGlobalMatrices[mpScene->mSceneGraph[i].parent] * mGlobalMatrices[i];
                mMatricesChanged[i] = mMatricesChanged[i] || mMatricesChanged[mpScene->mSceneGraph[i].parent];
                assert(!mMatricesChanged[i] || mMatricesAnimated[i]);
            }

            mInvTransposeGlobalMatrices[i] = transpose(inverse(mGlobalMatrices[i]));

            if (mpSkinningPass)
            {
                mSkinningMatrices[i] = mGlobalMatrices[i] * mpScene->mSceneGraph[i].localToBindSpace;
                mInvTransposeSkinningMatrices[i] = transpose(inverse(mSkinningMatrices[i]));
            }
        }
    }

    void AnimationController::uploadWorldMatrices(bool uploadAll)
    {
        if (uploadAll)
        {
            // Upload all matrices.
            mpWorldMatricesBuffer->setBlob(mGlobalMatrices.data(), 0, mpWorldMatricesBuffer->getSize());
            mpInvTransposeWorldMatricesBuffer->setBlob(mInvTransposeGlobalMatrices.data(), 0, mpInvTransposeWorldMatricesBuffer->getSize());
        }
        else
        {
            // Upload animated matrices only.
            for (size_t i = 0; i < mGlobalMatrices.size();)
            {
                // Detect ranges of consecutive matrices that are all animated or not.
                size_t offset = i;
                bool animated = mMatricesAnimated[i];
                while (i < mGlobalMatrices.size() && mMatricesAnimated[i] == animated) ++i;

                // Upload range of animated matrices.
                if (animated)
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
        m += mpSkinningStaticVertexData ? mpSkinningStaticVertexData->getSize() : 0;
        m += mpSkinningDynamicVertexData ? mpSkinningDynamicVertexData->getSize() : 0;
        m += mpPrevVertexData ? mpPrevVertexData->getSize() : 0;
        m += mpVertexCache ? mpVertexCache->getMemoryUsageInBytes() : 0;
        return m;
    }

    void AnimationController::createSkinningPass(const std::vector<PackedStaticVertexData>& staticVertexData, const std::vector<DynamicVertexData>& dynamicVertexData)
    {
        // We always copy the static data, to initialize the non-skinned vertices.
        const Buffer::SharedPtr& pVB = mpScene->mpVao->getVertexBuffer(Scene::kStaticDataBufferIndex);
        assert(pVB->getSize() == staticVertexData.size() * sizeof(staticVertexData[0]));
        pVB->setBlob(staticVertexData.data(), 0, pVB->getSize());

        if (!dynamicVertexData.empty())
        {
            mSkinningMatrices.resize(mpScene->mSceneGraph.size());
            mInvTransposeSkinningMatrices.resize(mSkinningMatrices.size());
            mMeshBindMatrices.resize(mpScene->mSceneGraph.size());

            mpSkinningPass = ComputePass::create("Scene/Animation/Skinning.slang");
            auto block = mpSkinningPass->getVars()["gData"];

            // Initialize the previous positions for skinned vertices.
            // This ensures we have valid data in the buffer before the skinning pass runs for the first time.
            std::vector<PrevVertexData> prevVertexData(dynamicVertexData.size());
            for (size_t i = 0; i < dynamicVertexData.size(); i++)
            {
                uint32_t staticIndex = dynamicVertexData[i].staticIndex;
                prevVertexData[i].position = staticVertexData[staticIndex].position;
            }

            // Initialize mesh bind transforms
            std::vector<float4x4> meshInvBindMatrices(mMeshBindMatrices.size());
            for (size_t i = 0; i < mpScene->mSceneGraph.size(); i++)
            {
                mMeshBindMatrices[i] = mpScene->mSceneGraph[i].meshBind;
                meshInvBindMatrices[i] = glm::inverse(mMeshBindMatrices[i]);
            }

            // Bind vertex data.
            assert(staticVertexData.size() <= std::numeric_limits<uint32_t>::max());
            assert(dynamicVertexData.size() <= std::numeric_limits<uint32_t>::max());
            mpSkinningStaticVertexData = Buffer::createStructured(block["staticData"], (uint32_t)staticVertexData.size(), ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, staticVertexData.data(), false);
            mpSkinningStaticVertexData->setName("AnimationController::mpSkinningStaticVertexData");
            mpSkinningDynamicVertexData = Buffer::createStructured(block["dynamicData"], (uint32_t)dynamicVertexData.size(), ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, dynamicVertexData.data(), false);
            mpSkinningDynamicVertexData->setName("AnimationController::mpSkinningDynamicVertexData");
            mpPrevVertexData = Buffer::createStructured(block["prevSkinnedVertices"], (uint32_t)dynamicVertexData.size(), ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, Buffer::CpuAccess::None, prevVertexData.data(), false);
            mpPrevVertexData->setName("AnimationController::mpPrevVertexData");

            block["staticData"] = mpSkinningStaticVertexData;
            block["dynamicData"] = mpSkinningDynamicVertexData;
            block["skinnedVertices"] = pVB;
            block["prevSkinnedVertices"] = mpPrevVertexData;

            // Bind transforms.
            assert(mSkinningMatrices.size() * 4 < std::numeric_limits<uint32_t>::max());
            uint32_t float4Count = (uint32_t)mSkinningMatrices.size() * 4;
            mpMeshBindMatricesBuffer = Buffer::createStructured(sizeof(float4), float4Count, ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, mMeshBindMatrices.data(), false);
            mpMeshBindMatricesBuffer->setName("AnimationController::mpMeshBindMatricesBuffer");
            mpMeshInvBindMatricesBuffer = Buffer::createStructured(sizeof(float4), float4Count, ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, meshInvBindMatrices.data(), false);
            mpMeshInvBindMatricesBuffer->setName("AnimationController::mpMeshInvBindMatricesBuffer");
            mpSkinningMatricesBuffer = Buffer::createStructured(sizeof(float4), float4Count, ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false);
            mpSkinningMatricesBuffer->setName("AnimationController::mpSkinningMatricesBuffer");
            mpInvTransposeSkinningMatricesBuffer = Buffer::createStructured(sizeof(float4), float4Count, ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false);
            mpInvTransposeSkinningMatricesBuffer->setName("AnimationController::mpInvTransposeSkinningMatricesBuffer");

            block["boneMatrices"].setBuffer(mpSkinningMatricesBuffer);
            block["inverseTransposeBoneMatrices"].setBuffer(mpInvTransposeSkinningMatricesBuffer);
            block["meshBindMatrices"].setBuffer(mpMeshBindMatricesBuffer);
            block["meshInvBindMatrices"].setBuffer(mpMeshInvBindMatricesBuffer);

            mSkinningDispatchSize = (uint32_t)dynamicVertexData.size();
        }
    }

    void AnimationController::executeSkinningPass(RenderContext* pContext, bool initPrev)
    {
        if (!mpSkinningPass) return;
        mpSkinningMatricesBuffer->setBlob(mSkinningMatrices.data(), 0, mpSkinningMatricesBuffer->getSize());
        mpInvTransposeSkinningMatricesBuffer->setBlob(mInvTransposeSkinningMatrices.data(), 0, mpInvTransposeSkinningMatricesBuffer->getSize());
        auto vars = mpSkinningPass->getVars()["gData"];
        vars["inverseTransposeWorldMatrices"].setBuffer(mpInvTransposeWorldMatricesBuffer);
        vars["worldMatrices"].setBuffer(mpWorldMatricesBuffer);
        vars["initPrev"] = initPrev;
        mpSkinningPass->execute(pContext, mSkinningDispatchSize, 1, 1);
    }

    void AnimationController::renderUI(Gui::Widgets& widget)
    {
        widget.checkbox("Loop Animations", mLoopAnimations);
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
