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
#pragma once
#include <map>
#include <vector>
#include "glm/mat4x4.hpp"
#include "Animation.h"

namespace Falcor
{
    struct Bone
    {
        uint32_t parentID;
        uint32_t boneID;
        std::string name;
        glm::mat4 offset;
        glm::mat4 localTransform;
        glm::mat4 originalLocalTransform;
        glm::mat4 globalTransform;
    };

    class Model;
    class AssimpModelImporter;

    class AnimationController
    {
    public:
        using UniquePtr = std::unique_ptr<AnimationController>;
        using UniqueConstPtr = std::unique_ptr<const AnimationController>;
        static const uint32_t kInvalidBoneID = -1;
        static const uint32_t kBindPoseAnimationId = -1;

        static UniquePtr create(const std::vector<Bone>& bones);
        static UniquePtr create(const AnimationController& other);
        ~AnimationController();

        void addAnimation(Animation::UniquePtr pAnimation);
        void animate(double currentTime);

        uint32_t getAnimationCount() const { return uint32_t(mAnimations.size()); }
        const std::string& getAnimationName(uint32_t ID) const;
        void setActiveAnimation(uint32_t id);
        uint32_t getActiveAnimation() const {return mActiveAnimation;}

        const std::vector<mat4>& getBoneMatrices() const { return mBoneTransforms; }
        const std::vector<mat4>& getBoneInvTransposeMatrices() const { return mBoneInvTransposeTransforms; }
        uint32_t getBoneCount() const { return uint32_t(mBones.size()); }

        uint32_t getBoneIdFromName(const std::string& name) const;
        void setBoneLocalTransform(uint32_t boneID, const glm::mat4& transform);

    private:
        AnimationController(const std::vector<Bone>& bones);

        std::vector<Bone> mBones;
        std::vector<glm::mat4> mBoneTransforms;
        std::vector<glm::mat4> mBoneInvTransposeTransforms;
        std::vector<Animation::UniquePtr> mAnimations;

        uint32_t mActiveAnimation = kBindPoseAnimationId;

        void calculateBoneTransforms();
    };
}