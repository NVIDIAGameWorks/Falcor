/***************************************************************************
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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
#include "Framework.h"
#include "AnimationController.h"
#include "Model.h"
#include <fstream>
#include "Animation.h"
#include <algorithm>

namespace Falcor
{
    void dumpBonesHeirarchy(const std::string& filename, Bone* pBone, uint32_t count)
    {
        std::ofstream dotfile;
        dotfile.open(filename.c_str());

        // Header
        dotfile << "digraph BonesGraph {" << std::endl;

        for(uint32_t i = 0; i < count; i++)
        {
            const Bone& bone = pBone[i];
            if(bone.parentID != AnimationController::kInvalidBoneID)
            {
                std::string parent = pBone[bone.parentID].name;
                std::string me = bone.name;
                std::replace(parent.begin(), parent.end(), '.', '_');
                std::replace(me.begin(), me.end(), '.', '_');

                dotfile << parent << " -> " << me << std::endl;
            }
        }

        // Close the file
        dotfile << "}" << std::endl; // closing graph scope
        dotfile.close();
    }

    AnimationController::UniquePtr AnimationController::create(const std::vector<Bone>& Bones)
    {
        return UniquePtr(new AnimationController(Bones));
    }

    AnimationController::UniquePtr AnimationController::create(const AnimationController& other)
    {
        return UniquePtr(new AnimationController(other.mBones));
    }

    AnimationController::AnimationController(const std::vector<Bone>& Bones)
    {
        mBones = Bones;
        mBoneTransforms.resize(mBones.size());
        mBoneInvTransposeTransforms.resize(mBones.size());
        setActiveAnimation(kBindPoseAnimationId);
    }

    void AnimationController::addAnimation(Animation::UniquePtr pAnimation)
    {
        mAnimations.push_back(std::move(pAnimation));
    }

    AnimationController::~AnimationController() = default;

    void AnimationController::setBoneLocalTransform(uint32_t boneID, const glm::mat4& transform)
    {
        assert(boneID < mBones.size());
        mBones[boneID].localTransform = transform;
    }

    void AnimationController::animate(double currentTime)
    {
        if(mActiveAnimation != kBindPoseAnimationId)
        {
            mAnimations[mActiveAnimation]->animate(currentTime, this);
        }

        for(uint32_t i = 0; i < mBones.size(); i++)
        {
            mBones[i].globalTransform = mBones[i].localTransform;
            if(mBones[i].parentID != kInvalidBoneID)
            {
                mBones[i].globalTransform = mBones[mBones[i].parentID].globalTransform * mBones[i].localTransform;
            }
            mBoneTransforms[i] = mBones[i].globalTransform * mBones[i].offset;
            mBoneInvTransposeTransforms[i] = transpose(inverse(mBoneTransforms[i]));
        }
    }

    void AnimationController::setActiveAnimation(uint32_t id)
    {
        assert(id == kBindPoseAnimationId || id < mAnimations.size());
        mActiveAnimation = id;
        if(id == kBindPoseAnimationId)
        {
            for(auto& bone : mBones)
            {
                bone.localTransform = bone.originalLocalTransform;
            }
        }
        animate(0);
    }

    const std::string& AnimationController::getAnimationName(uint32_t ID) const
    { 
        return mAnimations[ID]->getName(); 
    }
}