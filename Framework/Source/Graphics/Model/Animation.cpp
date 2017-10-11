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
#include "Animation.h"
#include "AnimationController.h"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtx/transform.hpp"

namespace Falcor
{
    Animation::UniquePtr Animation::create(const std::string& name, const std::vector<AnimationSet>& animationSets, float duration, float ticksPerSecond)
    {
        return UniquePtr(new Animation(name, animationSets, duration, ticksPerSecond));
    }

    Animation::Animation(const std::string& name, const std::vector<AnimationSet>& animationSets, float duration, float ticksPerSecond) : mName(name), mAnimationSets(animationSets), mDuration(duration), mTicksPerSecond(ticksPerSecond)
    {

    }

    Animation::~Animation() = default;

    template<typename T>
    uint32_t findCurrentFrame(T channel, float ticks)
    {
        uint32_t curKeyID = channel.lastKeyUsed;
        while(curKeyID < channel.keys.size() - 1)
        {
            if(channel.keys[curKeyID + 1].time > ticks)
            {
                break;
            }
            curKeyID++;
        }
        return curKeyID;
    }

    glm::vec3 interpolate(const glm::vec3& start, const glm::vec3& end, float ratio)
    {
        return start + ((end - start) * ratio);
    }

    glm::quat interpolate(const glm::quat& start, const glm::quat& end, float ratio)
    {
        return glm::slerp(start, end, ratio);
    }

    template<typename KeyType>
    KeyType Animation::calcCurrentKey(AnimationChannel<KeyType>& channel, float ticks, float lastUpdateTime)
    {
        KeyType curValue;
        if(channel.keys.size() > 0)
        {
            if(ticks < lastUpdateTime)
            {
                channel.lastKeyUsed = 0;
            }

            // search for the next keyframe
            uint32_t curKeyIndex = findCurrentFrame(channel, ticks);
            uint32_t nextKeyIndex = (curKeyIndex + 1) % channel.keys.size();
            const AnimationKey<KeyType>& curKey = channel.keys[curKeyIndex];
            const AnimationKey<KeyType>& nextKey = channel.keys[nextKeyIndex];

            assert(ticks >= curKey.time);
            // Interpolate between them
            float diff = nextKey.time - curKey.time;
            if(diff < 0)
            {
                diff += mDuration;
            }
            else if(diff == 0)
            {
                curValue = curKey.value;
            }
            else
            {
                float ratio = (ticks - curKey.time) / diff;
                curValue = interpolate(curKey.value, nextKey.value, ratio);
            }
            channel.lastKeyUsed = curKeyIndex;
        }
        return curValue;
    }

    void Animation::animate(double totalTime, AnimationController* pAnimationController)
    {
        // Calculate the relative time
        float ticks = (float)fmod(totalTime * mTicksPerSecond, mDuration);

        for(auto& Key : mAnimationSets)
        {
            glm::mat4 translation;
            translation[3] = glm::vec4(calcCurrentKey(Key.translation, ticks, Key.lastUpdateTime), 1);

            glm::mat4 scaling;
            if (Key.scaling.keys.size() > 0)
            {
                glm::scale(calcCurrentKey(Key.scaling, ticks, Key.lastUpdateTime));
            }

            glm::quat q = calcCurrentKey(Key.rotation, ticks, Key.lastUpdateTime);
            glm::mat4 rotation = glm::mat4_cast(q);

            Key.lastUpdateTime = ticks;

            glm::mat4 T = translation * rotation * scaling;
            pAnimationController->setBoneLocalTransform(Key.boneID, T);
        }
    }
}
