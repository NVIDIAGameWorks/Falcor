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
#include "Animation.h"
#include "AnimationController.h"
#include "Utils/Math/Common.h"
#include "Utils/Scripting/ScriptBindings.h"
#include "Scene/Transform.h"
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/transform.hpp>

namespace Falcor
{
    namespace
    {
        const double kEpsilonTime = 1e-5f;

        const Gui::DropdownList kChannelLoopModeDropdown =
        {
            { (uint32_t)Animation::Behavior::Constant, "Constant" },
            { (uint32_t)Animation::Behavior::Linear, "Linear" },
            { (uint32_t)Animation::Behavior::Cycle, "Cycle" },
            { (uint32_t)Animation::Behavior::Oscillate, "Oscillate" },
        };

        // Bezier form hermite spline
        float3 interpolateHermite(const float3& p0, const float3& p1, const float3& p2, const float3& p3, float t)
        {
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
        glm::quat interpolateHermite(const glm::quat& r0, const glm::quat& r1, const glm::quat& r2, const glm::quat& r3, float t)
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

        // This function performs linear extrapolation when either t < 0 or t > 1
        Animation::Keyframe interpolateLinear(const Animation::Keyframe& k0, const Animation::Keyframe& k1, float t)
        {
            Animation::Keyframe result;
            result.translation = lerp(k0.translation, k1.translation, t);
            result.scaling = lerp(k0.scaling, k1.scaling, t);
            result.rotation = slerp(k0.rotation, k1.rotation, t);
            result.time = glm::lerp(k0.time, k1.time, (double)t);
            return result;
        }

        Animation::Keyframe interpolateHermite(const Animation::Keyframe& k0, const Animation::Keyframe& k1, const Animation::Keyframe& k2, const Animation::Keyframe& k3, float t)
        {
            FALCOR_ASSERT(t >= 0.f && t <= 1.f);
            Animation::Keyframe result;
            result.translation = interpolateHermite(k0.translation, k1.translation, k2.translation, k3.translation, t);
            result.scaling = lerp(k1.scaling, k2.scaling, t);
            result.rotation = interpolateHermite(k0.rotation, k1.rotation, k2.rotation, k3.rotation, t);
            result.time = glm::lerp(k1.time, k2.time, (double)t);
            return result;
        }
    }

    Animation::SharedPtr Animation::create(const std::string& name, NodeID nodeID, double duration)
    {
        return SharedPtr(new Animation(name, nodeID, duration));
    }

    Animation::Animation(const std::string& name, NodeID nodeID, double duration)
        : mName(name)
        , mNodeID(nodeID)
        , mDuration(duration)
    {}

    rmcv::mat4 Animation::animate(double currentTime)
    {
        // Calculate the sample time.
        double time = currentTime;
        if (time < mKeyframes.front().time || time > mKeyframes.back().time)
        {
            time = calcSampleTime(currentTime);
        }

        // Determine if the animation behaves linearly outside of defined keyframes.
        bool isLinearPostInfinity = time > mKeyframes.back().time && this->getPostInfinityBehavior() == Behavior::Linear;
        bool isLinearPreInfinity = time < mKeyframes.front().time && this->getPreInfinityBehavior() == Behavior::Linear;

        Keyframe interpolated;

        if (isLinearPreInfinity && mKeyframes.size() > 1)
        {
            const auto& k0 = mKeyframes.front();
            auto k1 = interpolate(mInterpolationMode, k0.time + kEpsilonTime);
            double segmentDuration = k1.time - k0.time;
            float t = (float)((time - k0.time) / segmentDuration);
            interpolated = interpolateLinear(k0, k1, t);
        }
        else if (isLinearPostInfinity && mKeyframes.size() > 1)
        {
            const auto& k1 = mKeyframes.back();
            auto k0 = interpolate(mInterpolationMode, k1.time - kEpsilonTime);
            double segmentDuration = k1.time - k0.time;
            float t = (float)((time - k0.time) / segmentDuration);
            interpolated = interpolateLinear(k0, k1, t);
        }
        else
        {
            interpolated = interpolate(mInterpolationMode, time);
        }

        rmcv::mat4 T = rmcv::translate(interpolated.translation);
        rmcv::mat4 R = rmcv::mat4_cast(interpolated.rotation);
        rmcv::mat4 S = rmcv::scale(interpolated.scaling);
        rmcv::mat4 transform = T * R * S;

        return transform;
    }

    Animation::Keyframe Animation::interpolate(InterpolationMode mode, double time) const
    {
        FALCOR_ASSERT(!mKeyframes.empty());

        // Validate cached frame index.
        size_t frameIndex = clamp(mCachedFrameIndex, (size_t)0, mKeyframes.size() - 1);
        if (time < mKeyframes[frameIndex].time) frameIndex = 0;

        // Find frame index.
        while (frameIndex < mKeyframes.size() - 1)
        {
            if (mKeyframes[frameIndex + 1].time > time) break;
            frameIndex++;
        }

        // Cache frame index;
        mCachedFrameIndex = frameIndex;

        // Compute index of adjacent frame including optional warping.
        auto adjacentFrame = [this] (size_t frame, int32_t offset = 1)
        {
            size_t count = mKeyframes.size();
            return mEnableWarping ? (frame + count + offset) % count : clamp(frame + offset, (size_t)0, count - 1);
        };

        if (mode == InterpolationMode::Linear || mKeyframes.size() < 4)
        {
            size_t i0 = frameIndex;
            size_t i1 = adjacentFrame(i0);

            const Keyframe& k0 = mKeyframes[i0];
            const Keyframe& k1 = mKeyframes[i1];

            double segmentDuration = k1.time - k0.time;
            if (mEnableWarping && segmentDuration < 0.0) segmentDuration += mDuration;
            float t = (float)clamp((segmentDuration > 0.0 ? (time - k0.time) / segmentDuration : 1.0), 0.0, 1.0);

            return interpolateLinear(k0, k1, t);
        }
        else if (mode == InterpolationMode::Hermite)
        {
            size_t i1 = frameIndex;
            size_t i0 = adjacentFrame(i1, -1);
            size_t i2 = adjacentFrame(i1, 1);
            size_t i3 = adjacentFrame(i1, 2);

            const Keyframe& k0 = mKeyframes[i0];
            const Keyframe& k1 = mKeyframes[i1];
            const Keyframe& k2 = mKeyframes[i2];
            const Keyframe& k3 = mKeyframes[i3];

            double segmentDuration = k2.time - k1.time;
            if (mEnableWarping && segmentDuration < 0.0) segmentDuration += mDuration;
            float t = (float)clamp(segmentDuration > 0.0 ? (time - k1.time) / segmentDuration : 1.0, 0.0, 1.0);

            return interpolateHermite(k0, k1, k2, k3, t);
        }
        else
        {
            throw ArgumentError("'mode' is unknown interpolation mode");
        }
    }

    // Calculates the sample time within the keyframe range if the current time lies outside and
    // the animation does not behave linearly. If the animation behaves linearly, then the
    // current time is returned. This function should not be used if the current time lies
    // within the range of defined keyframe times.
    double Animation::calcSampleTime(double currentTime)
    {
        double modifiedTime = currentTime;
        double firstKeyframeTime = mKeyframes.front().time;
        double lastKeyframeTime = mKeyframes.back().time;
        double duration = lastKeyframeTime - firstKeyframeTime;

        FALCOR_ASSERT(currentTime < firstKeyframeTime || currentTime > lastKeyframeTime);

        Behavior behavior = (currentTime < firstKeyframeTime) ? mPreInfinityBehavior : mPostInfinityBehavior;
        switch (behavior)
        {
        case Behavior::Constant:
            modifiedTime = clamp(currentTime, firstKeyframeTime, lastKeyframeTime);
            break;
        case Behavior::Cycle:
            // Calculate the relative time
            modifiedTime = firstKeyframeTime + std::fmod(currentTime - firstKeyframeTime, duration);
            if (modifiedTime < firstKeyframeTime) modifiedTime += duration;
            break;
        case Behavior::Oscillate:
            // Calculate the relative time
            double offset = std::fmod(currentTime - firstKeyframeTime, 2 * duration);
            if (offset < 0) offset += 2 * duration;
            if (offset > duration) offset = 2 * duration - offset;
            modifiedTime = firstKeyframeTime + offset;
        }

        return modifiedTime;
    }

    void Animation::addKeyframe(const Keyframe& keyframe)
    {
        FALCOR_ASSERT(keyframe.time <= mDuration);

        if (mKeyframes.size() == 0 || mKeyframes[0].time > keyframe.time)
        {
            mKeyframes.insert(mKeyframes.begin(), keyframe);
        }
        else if (mKeyframes.back().time < keyframe.time)
        {
            mKeyframes.push_back(keyframe);
        }
        else
        {
            for (size_t i = 0; i < mKeyframes.size(); i++)
            {
                auto& current = mKeyframes[i];
                // If we already have a key-frame at the same time, replace it
                if (current.time == keyframe.time)
                {
                    current = keyframe;
                    return;
                }

                // If this is not the last frame, Check if we are in between frames
                if (i < mKeyframes.size() - 1)
                {
                    auto& Next = mKeyframes[i + 1];
                    if (current.time < keyframe.time && Next.time > keyframe.time)
                    {
                        mKeyframes.insert(mKeyframes.begin() + i + 1, keyframe);
                        return;
                    }
                }
            }

            // If we got here, need to push it to the end of the list
            mKeyframes.push_back(keyframe);
        }
    }

    const Animation::Keyframe& Animation::getKeyframe(double time) const
    {
        for (const auto& k : mKeyframes)
        {
            if (k.time == time) return k;
        }
        throw ArgumentError("'time' ({}) does not refer to an existing keyframe", time);
    }

    bool Animation::doesKeyframeExists(double time) const
    {
        for (const auto& k : mKeyframes)
        {
            if (k.time == time) return true;
        }
        return false;
    }

    void Animation::renderUI(Gui::Widgets& widget)
    {
        widget.dropdown("Pre-Infinity Behavior", kChannelLoopModeDropdown, reinterpret_cast<uint32_t&>(mPreInfinityBehavior));
        widget.dropdown("Post-Infinity Behavior", kChannelLoopModeDropdown, reinterpret_cast<uint32_t&>(mPostInfinityBehavior));
    }

    FALCOR_SCRIPT_BINDING(Animation)
    {
        using namespace pybind11::literals;

        FALCOR_SCRIPT_BINDING_DEPENDENCY(Transform)

        pybind11::class_<Animation, Animation::SharedPtr> animation(m, "Animation");
        animation.def_property_readonly("name", &Animation::getName);
        animation.def_property_readonly("nodeID", &Animation::getNodeID);
        animation.def_property_readonly("duration", &Animation::getDuration);
        animation.def_property("preInfinityBehavior", &Animation::getPreInfinityBehavior, &Animation::setPreInfinityBehavior);
        animation.def_property("postInfinityBehavior", &Animation::getPostInfinityBehavior, &Animation::setPostInfinityBehavior);
        animation.def_property("interpolationMode", &Animation::getInterpolationMode, &Animation::setInterpolationMode);
        animation.def_property("enableWarping", &Animation::isWarpingEnabled, &Animation::setEnableWarping);
        animation.def(pybind11::init(&Animation::create), "name"_a, "nodeID"_a, "duration"_a);
        animation.def("addKeyframe", [] (Animation* pAnimation, double time, const Transform& transform) {
            Animation::Keyframe keyframe{ time, transform.getTranslation(), transform.getScaling(), transform.getRotation() };
            pAnimation->addKeyframe(keyframe);
        });

        pybind11::enum_<Animation::InterpolationMode> interpolationMode(animation, "InterpolationMode");
        interpolationMode.value("Linear", Animation::InterpolationMode::Linear);
        interpolationMode.value("Hermite", Animation::InterpolationMode::Hermite);

        pybind11::enum_<Animation::Behavior> behavior(animation, "Behavior");
        behavior.value("Constant", Animation::Behavior::Constant);
        behavior.value("Linear", Animation::Behavior::Linear);
        behavior.value("Cycle", Animation::Behavior::Cycle);
        behavior.value("Oscillate", Animation::Behavior::Oscillate);
    }
}
