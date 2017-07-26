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
#include "glm/vec3.hpp"
#include <vector>
#include "Graphics/Paths/MovableObject.h"
#include "Utils/Math/CubicSpline.h"

namespace Falcor
{
    using Vec3CubicSpline = CubicSpline <glm::vec3>;

    class ObjectPath : public std::enable_shared_from_this<ObjectPath>
    {
    public:
        using SharedPtr = std::shared_ptr<ObjectPath>;
        using SharedConstPtr = std::shared_ptr<const ObjectPath>;

        static ObjectPath::SharedPtr create();
        ~ObjectPath();

        enum class Interpolation
        {
            Linear,
            CubicSpline
        };

        void setInterpolationMode(Interpolation mode) { mMode = mode; }
        uint32_t addKeyFrame(float time, const glm::vec3& position, const glm::vec3& target, const glm::vec3& up);
        void removeKeyFrame(uint32_t frameID);

        bool animate(double currentTime);

        void attachObject(const IMovableObject::SharedPtr& pObject);
        void detachObject(const IMovableObject::SharedPtr& pObject);
        void detachAllObjects() { mpObjects.clear(); }

        const IMovableObject::SharedPtr& getAttachedObject(uint32_t i) const { return mpObjects[i]; }
        uint32_t getAttachedObjectCount() const { return (uint32_t)mpObjects.size(); }

        void setAnimationRepeat(bool repeatAnimation) { mRepeatAnimation = repeatAnimation; }

        const glm::vec3& getCurrentPosition() const { return mCurrentFrame.position; }
        const glm::vec3& getCurrentLookAtVector() const { return mCurrentFrame.target; }
        const glm::vec3& getCurrentUpVector() const { return mCurrentFrame.up; }
        bool isRepeatOn() const {return mRepeatAnimation;}

        const std::string& getName() const { return mName; }
        void setName(const std::string& name) { mName = name; }

        struct Frame
        {
            glm::vec3 position;
            glm::vec3 target;
            glm::vec3 up;
            float time = 0;
        };

        uint32_t getKeyFrameCount() const {return (uint32_t)mKeyFrames.size();}
        const Frame& getKeyFrame(uint32_t frameID) const { return mKeyFrames[frameID]; }

        void setFramePosition(uint32_t frameID, const glm::vec3& pos) { mDirty = true; mKeyFrames[frameID].position = pos; }
        void setFrameTarget(uint32_t frameID, const glm::vec3& target) { mDirty = true; mKeyFrames[frameID].target = target; }
        void setFrameUp(uint32_t frameID, const glm::vec3& up) { mDirty = true; mKeyFrames[frameID].up = up; }
        uint32_t setFrameTime(uint32_t frameID, float time);

        void getFrameAt(uint32_t frameID, float t, Frame& frameOut);

    private:
        ObjectPath() = default;

        float getInterpolationFactor(uint32_t frameID, double currentTime) const;

        Frame linearInterpolation(uint32_t currentFrame, float t) const;
        Frame cubicSplineInterpolation(uint32_t currentFrame, float t);

        std::vector<Frame> mKeyFrames;
        std::vector<IMovableObject::SharedPtr> mpObjects;
        std::string mName;
        bool mRepeatAnimation = false;

        Frame mCurrentFrame;
        Interpolation mMode = Interpolation::CubicSpline;
        bool mDirty = false;

        std::unique_ptr<Vec3CubicSpline> mpPositionSpline;
        std::unique_ptr<Vec3CubicSpline> mpTargetSpline;
        std::unique_ptr<Vec3CubicSpline> mpUpSpline;
    };
}