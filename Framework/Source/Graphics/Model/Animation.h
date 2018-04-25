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
#include <vector>
#include "glm/vec3.hpp"
#include "glm/gtc/quaternion.hpp"

namespace Falcor
{
    class AnimationController;

    class Animation
    {
    public:
        using UniquePtr = std::unique_ptr<Animation>;
        using UniqueConstPtr = std::unique_ptr<const Animation>;

        template<typename T>
        struct AnimationKey
        {
            T value;
            float time;
        };

        template<typename T>
        struct AnimationChannel
        {
            std::vector<AnimationKey<T>> keys;
            uint32_t lastKeyUsed = 0;
        };

        struct AnimationSet
        {
            uint32_t boneID;
            AnimationChannel<glm::vec3> translation;
            AnimationChannel<glm::vec3> scaling;
            AnimationChannel<glm::quat> rotation;
            float lastUpdateTime = 0;
        };

        static UniquePtr create(const std::string& name, const std::vector<AnimationSet>& animationSets, float duration, float ticksPerSecond);
        static UniquePtr create(const Animation& other);
        ~Animation();
        void animate(double totalTime, AnimationController* pAnimationController);
        const std::string& getName() const { return mName; }

    private:
        Animation(const std::string& name, const std::vector<AnimationSet>& animationSets, float duration, float ticksPerSecond);
        Animation(const Animation& other);
        
        const std::string mName;
        float mDuration;
        float mTicksPerSecond;

        std::vector<AnimationSet> mAnimationSets;

        template<typename _KeyType>
        _KeyType calcCurrentKey(AnimationChannel<_KeyType>& channel, float ticks, float lastUpdateTime);
    };
}