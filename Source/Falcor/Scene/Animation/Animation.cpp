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
#include "Animation.h"
#include "glm/gtc/quaternion.hpp"
#include "glm/gtx/transform.hpp"
#include "AnimationController.h"

namespace Falcor
{
    Animation::SharedPtr Animation::create(const std::string& name, double durationInSeconds)
    {
        return SharedPtr(new Animation(name, durationInSeconds));
    }

    Animation::Animation(const std::string& name, double durationInSeconds) : mName(name), mDurationInSeconds(durationInSeconds) {}

    size_t Animation::findChannelFrame(const Channel& c, double time) const
    {
        size_t frameID = (time < c.lastUpdateTime) ? 0 : c.lastKeyframeUsed;
        while (frameID < c.keyframes.size() - 1)
        {
            if (c.keyframes[frameID + 1].time > time) break;
            frameID++;
        }
        return frameID;
    }

    mat4 Animation::interpolate(const Keyframe& start, const Keyframe& end, double curTime) const
    {
        double localTime = curTime - start.time;
        double keyframeDuration = end.time - start.time;
        if (keyframeDuration < 0) keyframeDuration += mDurationInSeconds;
        float factor = keyframeDuration != 0 ? (float)(localTime / keyframeDuration) : 1;

        vec3 translation = lerp(start.translation, end.translation, factor);
        vec3 scaling = lerp(start.scaling, end.scaling, factor);
        quat rotation = slerp(start.rotation, end.rotation, factor);

        mat4 T;
        T[3] = vec4(translation, 1);
        mat4 R = mat4_cast(rotation);
        mat4 S = scale(scaling);
        mat4 transform = T * R * S;
        return transform;
    }

    mat4 Animation::animateChannel(Channel& c, double time)
    {
        size_t curKeyIndex = findChannelFrame(c, time);
        size_t nextKeyIndex = curKeyIndex + 1;
        if (nextKeyIndex == c.keyframes.size()) nextKeyIndex = 0;

        c.lastUpdateTime = time;
        c.lastKeyframeUsed = curKeyIndex;

        return interpolate(c.keyframes[curKeyIndex], c.keyframes[nextKeyIndex], time);
    }

    void Animation::animate(double totalTime, std::vector<mat4>& matrices)
    {
        // Calculate the relative time
        double modTime = fmod(totalTime, mDurationInSeconds);
        for (auto& c : mChannels)
        {
            matrices[c.matrixID] = animateChannel(c, modTime);
        }
    }

    size_t Animation::addChannel(size_t matrixID)
    {
        mChannels.push_back(Channel(matrixID));
        return mChannels.size() - 1;
    }

    void Animation::addKeyframe(size_t channelID, const Keyframe& keyframe)
    {
        assert(channelID < mChannels.size());
        assert(keyframe.time <= mDurationInSeconds);

        mChannels[channelID].lastKeyframeUsed = 0;
        auto& channelFrames = mChannels[channelID].keyframes;

        if (channelFrames.size() == 0 || channelFrames[0].time > keyframe.time)
        {
            channelFrames.insert(channelFrames.begin(), keyframe);
            return;
        }
        else
        {
            for (size_t i = 0; i < channelFrames.size(); i++)
            {
                auto& current = channelFrames[i];
                // If we already have a key-frame at the same time, replace it
                if (current.time == keyframe.time)
                {
                    current = keyframe;
                    return;
                }

                // If this is not the last frame, Check if we are in between frames
                if (i < channelFrames.size() - 1)
                {
                    auto& Next = channelFrames[i + 1];
                    if (current.time < keyframe.time && Next.time > keyframe.time)
                    {
                        channelFrames.insert(channelFrames.begin() + i + 1, keyframe);
                        return;
                    }
                }
            }

            // If we got here, need to push it to the end of the list
            channelFrames.push_back(keyframe);
        }
    }

    const Animation::Keyframe& Animation::getKeyframe(size_t channelID, double time) const
    {
        assert(channelID < mChannels.size());
        for (const auto& k : mChannels[channelID].keyframes)
        {
            if (k.time == time) return k;
        }
        throw std::runtime_error(("Animation::getKeyframe() - can't find a keyframe at time " + to_string(time)).c_str());
    }

    bool Animation::doesKeyframeExists(size_t channelID, double time) const
    {
        assert(channelID < mChannels.size());
        for (const auto& k : mChannels[channelID].keyframes)
        {
            if (k.time == time) return true;
        }
        return false;
    }
}
