/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include "Animation.h"
#include "glm/gtc/quaternion.hpp"
#include "glm/gtx/transform.hpp"
#include "AnimationController.h"

namespace Falcor
{
    // Bezier form hermite spline
    static float3 interpolateHermite(const float3& p0, const float3& p1, const float3& p2, const float3& p3, float t)
    {
        float3 v1 = (p2 - p0) * 0.5f;
        float3 v2 = (p3 - p1) * 0.5f;

        float3 b0 = p1;
        float3 b1 = p1 + (p2 - p0) * 0.5f / 3.f;
        float3 b2 = p2 - (p3 - p1) * 0.5f / 3.f;
        float3 b3 = p2;

        float3 q0 = lerp(b0, b1, t);
        float3 q1 = lerp(b1, b2, t);
        float3 q2 = lerp(b2, b3, t);

        float3 qq0 = lerp(q0, q1, t);
        float3 qq1 = lerp(q1, q2, t);

        return lerp(qq0, qq1, t);
    }

    // Bezier hermite slerp
    static glm::quat interpolateHermite(const glm::quat& r0, const glm::quat& r1, const glm::quat& r2, const glm::quat& r3, float t)
    {
        glm::quat b0 = r1;
        glm::quat b1 = r1 + (r2 - r0) * 0.5f / 3.0f;
        glm::quat b2 = r2 - (r3 - r1) * 0.5f / 3.0f;
        glm::quat b3 = r2;

        glm::quat q0 = slerp(b0, b1, t);
        glm::quat q1 = slerp(b1, b2, t);
        glm::quat q2 = slerp(b2, b3, t);

        glm::quat qq0 = slerp(q0, q1, t);
        glm::quat qq1 = slerp(q1, q2, t);

        return slerp(qq0, qq1, t);
    }

    static Animation::Keyframe interpolateLinear(const Animation::Keyframe& k0, const Animation::Keyframe& k1, float t)
    {
        assert(t >= 0.f && t <= 1.f);
        Animation::Keyframe result;
        result.translation = lerp(k0.translation, k1.translation, t);
        result.scaling = lerp(k0.scaling, k1.scaling, t);
        result.rotation = slerp(k0.rotation, k1.rotation, t);
        return result;
    }

    static Animation::Keyframe interpolateHermite(const Animation::Keyframe& k0, const Animation::Keyframe& k1, const Animation::Keyframe& k2, const Animation::Keyframe& k3, float t)
    {
        assert(t >= 0.f && t <= 1.f);
        Animation::Keyframe result;
        result.translation = interpolateHermite(k0.translation, k1.translation, k2.translation, k3.translation, t);
        result.scaling = lerp(k1.scaling, k2.scaling, t);
        result.rotation = interpolateHermite(k0.rotation, k1.rotation, k2.rotation, k3.rotation, t);
        return result;
    }

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

        // Cache last used key frame.
        c.lastUpdateTime = time;
        c.lastKeyframeUsed = frameID;

        return frameID;
    }

    glm::mat4 Animation::animateChannel(const Channel& c, double time) const
    {
        auto mode = c.interpolationMode;

        // Use linear interpolation if there are less than 4 keyframes.
        if (c.keyframes.size() < 4) mode = InterpolationMode::Linear;

        Keyframe interpolated;

        // Compute index of adjacent frame including optional warping.
        auto adjacentFrame = [] (const Channel& c, size_t frame, int32_t offset = 1)
        {
            size_t count = c.keyframes.size();
            if ((int64_t)frame + offset < 0) frame += count;
            return c.enableWarping ? (frame + offset) % count : std::min(frame + offset, count - 1);
        };

        if (mode == InterpolationMode::Linear)
        {
            size_t i0 = findChannelFrame(c, time);
            size_t i1 = adjacentFrame(c, i0);

            const Keyframe& k0 = c.keyframes[i0];
            const Keyframe& k1 = c.keyframes[i1];

            double segmentDuration = k1.time - k0.time;
            if (c.enableWarping && segmentDuration < 0.0) segmentDuration += mDurationInSeconds;
            float t = (float)clamp(segmentDuration > 0.0 ? (time - k0.time) / segmentDuration : 1.0, 0.0, 1.0);

            interpolated = interpolateLinear(k0, k1, t);
        }
        else if (mode == InterpolationMode::Hermite)
        {
            size_t i1 = findChannelFrame(c, time);
            size_t i0 = adjacentFrame(c, i1, -1);
            size_t i2 = adjacentFrame(c, i1, 1);
            size_t i3 = adjacentFrame(c, i1, 2);

            const Keyframe& k0 = c.keyframes[i0];
            const Keyframe& k1 = c.keyframes[i1];
            const Keyframe& k2 = c.keyframes[i2];
            const Keyframe& k3 = c.keyframes[i3];

            double segmentDuration = k2.time - k1.time;
            if (c.enableWarping && segmentDuration < 0.0) segmentDuration += mDurationInSeconds;
            float t = (float)clamp(segmentDuration > 0.0 ? (time - k1.time) / segmentDuration : 1.0, 0.0, 1.0);

            interpolated = interpolateHermite(k0, k1, k2, k3, t);
        }

        glm::mat4 T = translate(interpolated.translation);
        glm::mat4 R = mat4_cast(interpolated.rotation);
        glm::mat4 S = scale(interpolated.scaling);
        glm::mat4 transform = T * R * S;

        return transform;
    }

    void Animation::animate(double totalTime, std::vector<glm::mat4>& matrices)
    {
        // Calculate the relative time
        double modTime = std::fmod(totalTime, mDurationInSeconds);
        for (auto& c : mChannels)
        {
            matrices[c.matrixID] = animateChannel(c, modTime);
        }
    }

    uint32_t Animation::addChannel(uint32_t matrixID)
    {
        mChannels.push_back(Channel(matrixID));
        return (uint32_t)(mChannels.size() - 1);
    }

    uint32_t Animation::getChannel(uint32_t matrixID) const
    {
        for (uint32_t i = 0; i < mChannels.size(); ++i)
        {
            if (mChannels[i].matrixID == matrixID) return i;
        }
        return kInvalidChannel;
    }

    void Animation::addKeyframe(uint32_t channelID, const Keyframe& keyframe)
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

    const Animation::Keyframe& Animation::getKeyframe(uint32_t channelID, double time) const
    {
        assert(channelID < mChannels.size());
        for (const auto& k : mChannels[channelID].keyframes)
        {
            if (k.time == time) return k;
        }
        throw std::runtime_error(("Animation::getKeyframe() - can't find a keyframe at time " + std::to_string(time)).c_str());
    }

    bool Animation::doesKeyframeExists(uint32_t channelID, double time) const
    {
        assert(channelID < mChannels.size());
        for (const auto& k : mChannels[channelID].keyframes)
        {
            if (k.time == time) return true;
        }
        return false;
    }

    void Animation::setInterpolationMode(uint32_t channelID, InterpolationMode mode, bool enableWarping)
    {
        assert(channelID < mChannels.size());
        mChannels[channelID].interpolationMode = mode;
        mChannels[channelID].enableWarping = enableWarping;
    }

}
