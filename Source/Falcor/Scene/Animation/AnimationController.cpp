/***************************************************************************
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
#include "AnimationController.h"
#include <fstream>

namespace Falcor
{
    namespace
    {
        const static std::string kWorldMatricesBufferName = "worldMatrices";
        const static std::string kInverseTransposeWorldMatrices = "inverseTransposeWorldMatrices";
        const static std::string kPreviousWorldMatrices = "previousFrameWorldMatrices";
    }

    AnimationController::AnimationController(Scene* pScene, const StaticVertexVector& staticVertexData, const DynamicVertexVector& dynamicVertexData) :
        mpScene(pScene), mLocalMatrices(pScene->mSceneGraph.size()), mInvTransposeGlobalMatrices(pScene->mSceneGraph.size()), mMatricesChanged(pScene->mSceneGraph.size())
    {
        size_t l2wBufSize = mLocalMatrices.size() * 4;
        assert(l2wBufSize <= UINT32_MAX);
        mpWorldMatricesBuffer = TypedBuffer<float4>::create((uint32_t)l2wBufSize);
        mpPrevWorldMatricesBuffer = mpWorldMatricesBuffer;
        mpInvTransposeWorldMatricesBuffer = TypedBuffer<float4>::create((uint32_t)l2wBufSize);
        createSkinningPass(staticVertexData, dynamicVertexData);
    }

    AnimationController::UniquePtr AnimationController::create(Scene* pScene, const StaticVertexVector& staticVertexData, const DynamicVertexVector& dynamicVertexData)
    {
        return UniquePtr(new AnimationController(pScene, staticVertexData, dynamicVertexData));
    }

    void AnimationController::addAnimation(uint32_t meshID, Animation::ConstSharedPtrRef pAnimation)
    {
        mMeshes[meshID].pAnimations.push_back(pAnimation);
        mHasAnimations = true;
    }

    void AnimationController::initLocalMatrices()
    {
        for (size_t i = 0; i < mLocalMatrices.size(); i++) mLocalMatrices[i] = mpScene->mSceneGraph[i].transform;
    }

    bool AnimationController::animate(RenderContext* pContext, double currentTime)
    {
        PROFILE("animate");

        mMatricesChanged.assign(mMatricesChanged.size(), false);

        if (mAnimationChanged == false)
        {
            if (mActiveAnimationCount == 0) return false;
            if (mLastAnimationTime == currentTime)
            {
                // Copy the current matrices to the previous matrices. We can do that only once, but not sure if it we'll help perf (it only occures when the animation is paused)
                pContext->copyResource(mpPrevWorldMatricesBuffer.get(), mpWorldMatricesBuffer.get());
                return false;
            }
        }
        else initLocalMatrices();

        mAnimationChanged = false;
        mLastAnimationTime = currentTime;

        for (const auto& a : mMeshes)
        {
            const auto& mesh = a.second;
            if (mesh.activeAnimation == kBindPoseAnimationId) continue; // Bind pose was pre-computed
            auto& pAnimation = mesh.pAnimations[mesh.activeAnimation];
            pAnimation->animate(currentTime, mLocalMatrices);
            for (size_t i = 0; i < pAnimation->getChannelCount(); i++)
            {
                mMatricesChanged[pAnimation->getChannelMatrixID(i)] = true;
            }
        }

        swap(mpPrevWorldMatricesBuffer, mpWorldMatricesBuffer);
        updateMatrices();
        bindBuffers();
        executeSkinningPass(pContext);

        return true;
    }

    bool AnimationController::validateIndices(uint32_t meshID, uint32_t animID, const std::string& warningPrefix) const
    {
        const auto& m = mMeshes.find(meshID);
        if (m == mMeshes.end())
        {
            logWarning(warningPrefix + " - the mesh doesn't have animations");
            return false;
        }

        if (animID >= m->second.pAnimations.size())
        {
            logWarning(warningPrefix + " - the animation ID doesn't exist");
            return false;
        }
        return true;
    }

    uint32_t AnimationController::getMeshAnimationCount(uint32_t meshID) const
    {
        const auto& m = mMeshes.find(meshID);
        return (m == mMeshes.end()) ? 0 : (uint32_t)m->second.pAnimations.size();
    }

    const std::string& AnimationController::getAnimationName(uint32_t meshID, uint32_t animID) const
    {
        static std::string s;
        if (validateIndices(meshID, animID, "AnimationController::getAnimationName") == false) return s;
        return mMeshes.at(meshID).pAnimations[animID]->getName();
    }

    bool AnimationController::setActiveAnimation(uint32_t meshID, uint32_t animID)
    {
        if (animID != kBindPoseAnimationId && validateIndices(meshID, animID, "AnimationController::setActiveAnimation") == false) return false;

        if (mMeshes[meshID].activeAnimation != animID)
        {
            mAnimationChanged = true;
            mMeshes[meshID].activeAnimation = animID;
            (animID != kBindPoseAnimationId) ? mActiveAnimationCount++ : mActiveAnimationCount--;
            assert(mActiveAnimationCount < UINT32_MAX);
            allocatePrevWorldMatrixBuffer();
        }
        return true;
    }

    uint32_t AnimationController::getActiveAnimation(uint32_t meshID) const
    {
        if (validateIndices(meshID, 0, "AnimationController::getActiveAnimation") == false) return kBindPoseAnimationId;
        return mMeshes.at(meshID).activeAnimation;
    }

    void AnimationController::renderUI(Gui::Widgets& widget)
    {
        if (mMeshes[0].pAnimations.size())
        {
            bool active = mMeshes[0].activeAnimation != kBindPoseAnimationId;
            if (widget.checkbox("Animate Scene", active))
            {
                setActiveAnimation(0, active ? 0 : kBindPoseAnimationId);
            }
        }
    }

    void AnimationController::updateMatrices()
    {
        // We can optimize this
        mGlobalMatrices = mLocalMatrices;

        for (size_t i = 0; i < mGlobalMatrices.size(); i++)
        {
            if (mpScene->mSceneGraph[i].parent != SceneBuilder::kInvalidNode)
            {
                mGlobalMatrices[i] = mGlobalMatrices[mpScene->mSceneGraph[i].parent] * mGlobalMatrices[i];
                mMatricesChanged[i] = mMatricesChanged[i] || mMatricesChanged[mpScene->mSceneGraph[i].parent];
            }

            mInvTransposeGlobalMatrices[i] = transpose(inverse(mGlobalMatrices[i]));

            if(mpSkinningPass)
            {
                mSkinningMatrices[i] = mGlobalMatrices[i] * mpScene->mSceneGraph[i].localToBindSpace;
                mInvTransposeSkinningMatrices[i] = transpose(inverse(mSkinningMatrices[i]));
            }
        }
        mpWorldMatricesBuffer->setBlob(mGlobalMatrices.data(), 0, mpWorldMatricesBuffer->getSize());
        mpInvTransposeWorldMatricesBuffer->setBlob(mInvTransposeGlobalMatrices.data(), 0, mpInvTransposeWorldMatricesBuffer->getSize());
    }

    void AnimationController::bindBuffers()
    {
        ParameterBlock* pBlock = mpScene->mpSceneBlock.get();
        pBlock->setTypedBuffer(kWorldMatricesBufferName, mpWorldMatricesBuffer);
        pBlock->setTypedBuffer(kPreviousWorldMatrices, mpPrevWorldMatricesBuffer);
        pBlock->setTypedBuffer(kInverseTransposeWorldMatrices, mpInvTransposeWorldMatricesBuffer);
    }

    void AnimationController::allocatePrevWorldMatrixBuffer()
    {
        if (mActiveAnimationCount)
        {
            if(mpWorldMatricesBuffer == mpPrevWorldMatricesBuffer)
            {
                mpPrevWorldMatricesBuffer = TypedBuffer<float4>::create(mpWorldMatricesBuffer->getElementCount());
            }
        }
        else mpPrevWorldMatricesBuffer = mpWorldMatricesBuffer;
    }

    void AnimationController::createSkinningPass(const std::vector<StaticVertexData>& staticVertexData, const std::vector<DynamicVertexData>& dynamicVertexData)
    {
        StructuredBuffer::SharedPtr pVB = mpScene->mpVao->getVertexBuffer(Scene::kStaticDataBufferIndex)->asStructuredBuffer();
        assert(pVB->getSize() == staticVertexData.size() * sizeof(staticVertexData[0]));
        // We always copy the static data, to initialize the non-skinned vertices
        pVB->setBlob(staticVertexData.data(), 0, pVB->getSize());
        pVB->uploadToGPU();

        if (dynamicVertexData.size())
        {
            mSkinningMatrices.resize(mpScene->mSceneGraph.size());
            mInvTransposeSkinningMatrices.resize(mSkinningMatrices.size());

            mpSkinningPass = ComputePass::create("Skinning.slang");
            auto pBlock = mpSkinningPass->getVars()->getParameterBlock("gData");
            pBlock->setStructuredBuffer("skinnedVertices", pVB);

            auto createBuffer = [&](const std::string& name, const auto& initData)
            {
                ReflectionResourceType::SharedConstPtr pReflector = pBlock->getReflection()->getResource(name)->getType()->asResourceType()->shared_from_this();
                auto pBuffer = StructuredBuffer::create(name, pReflector, (uint32_t)initData.size(), ResourceBindFlags::ShaderResource);
                pBuffer->setBlob(initData.data(), 0, pBuffer->getSize());
                pBlock->setStructuredBuffer(name, pBuffer);
            };

            createBuffer("staticData", staticVertexData);
            createBuffer("dynamicData", dynamicVertexData);

            mpSkinningMatricesBuffer = TypedBuffer<float4>::create((uint32_t)mSkinningMatrices.size() * 4, ResourceBindFlags::ShaderResource);
            mpInvTransposeSkinningMatricesBuffer = TypedBuffer<float4>::create((uint32_t)mSkinningMatrices.size() * 4, ResourceBindFlags::ShaderResource);
            pBlock->setTypedBuffer("boneMatrices", mpSkinningMatricesBuffer);
            pBlock->setTypedBuffer("inverseTransposeBoneMatrices", mpInvTransposeSkinningMatricesBuffer);
            pBlock->setTypedBuffer("inverseTransposeWorldMatrices", mpInvTransposeWorldMatricesBuffer);
            pBlock->setTypedBuffer("worldMatrices", mpWorldMatricesBuffer);
            
            mSkinningDispatchSize = (uint32_t)dynamicVertexData.size();
        }
    }

    void AnimationController::executeSkinningPass(RenderContext* pContext)
    {
        if (!mpSkinningPass) return;
        mpSkinningMatricesBuffer->setBlob(mSkinningMatrices.data(), 0, mpSkinningMatricesBuffer->getSize());
        mpInvTransposeSkinningMatricesBuffer->setBlob(mInvTransposeSkinningMatrices.data(), 0, mpInvTransposeSkinningMatricesBuffer->getSize());
        mpSkinningPass->execute(pContext, mSkinningDispatchSize, 1, 1);
    }
}
