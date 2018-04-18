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

    /** Describes and manages a path consisting of some number of key frames. Objects in a scene, such as models, lights, and cameras, 
        can be attached to a path and multiple objects can be attached to each path. Updating the path through animate() will update the 
        positions of all attached objects.
    */
    class ObjectPath : public std::enable_shared_from_this<ObjectPath>
    {
    public:
        using SharedPtr = std::shared_ptr<ObjectPath>;
        using SharedConstPtr = std::shared_ptr<const ObjectPath>;

        /** Create a path.
        */
        static ObjectPath::SharedPtr create();
        ~ObjectPath();

        /** Ways to interpolate between key frames
        */
        enum class Interpolation
        {
            Linear,         ///< Linear interpolation. Requires at least 2 key frames
            CubicSpline     ///< Cubic spline interpolation.  Requires at least 3 key frames. Path updates will fall-back to linear interpolation when there are less than 3 key frames
        };

        /**  Set the interpolation mode.
        */
        void setInterpolationMode(Interpolation mode) { mMode = mode; }

        /** Insert a key frame. Key frame will be inserted/sorted into the path based on time.
            \param[in] time Time in seconds
            \param[in] position World-space position
            \param[in] target Look-at target position
            \param[in] up Up vector
        */
        uint32_t addKeyFrame(float time, const glm::vec3& position, const glm::vec3& target, const glm::vec3& up);

        /** Remove a key frame. Does not check bounds.
        */
        void removeKeyFrame(uint32_t frameID);

        /** Update the current frame and updates the transforms on all attached objects.
            \param[in] currentTime Elapsed time in seconds
            \return Whether update was successful.
        */
        bool animate(double currentTime);

        /** Attach a movable object to the path, such as models, cameras, and lights.
        */
        void attachObject(const IMovableObject::SharedPtr& pObject);

        /** Detach a movable object from the path.
        */
        void detachObject(const IMovableObject::SharedPtr& pObject);

        /** Detach all objects from the path.
        */
        void detachAllObjects();

        /** Get an object attached to the path.
        */
        const IMovableObject::SharedPtr& getAttachedObject(uint32_t i) const { return mpObjects[i]; }

        /** Get the number of attached objects.
        */
        uint32_t getAttachedObjectCount() const { return (uint32_t)mpObjects.size(); }

        /** Set whether the animation should loop upon reaching the last key frame.
        */
        void setAnimationRepeat(bool repeatAnimation) { mRepeatAnimation = repeatAnimation; }

        /** Get the current interpolated position along the path.
        */
        const glm::vec3& getCurrentPosition() const { return mCurrentFrame.position; }

        /** Get the current interpolated look-at target position (NOT vector) along the path.
        */
        const glm::vec3& getCurrentLookAtVector() const { return mCurrentFrame.target; }

        /** Get the current interpolated up vector along the path.
        */
        const glm::vec3& getCurrentUpVector() const { return mCurrentFrame.up; }

        /** Get whether the animation will loop.
        */
        bool isRepeatOn() const {return mRepeatAnimation;}

        /** Get the path name.
        */
        const std::string& getName() const { return mName; }

        /** Set the path name.
        */
        void setName(const std::string& name) { mName = name; }

        /** Frame data
        */
        struct Frame
        {
            glm::vec3 position;
            glm::vec3 target;
            glm::vec3 up;
            float time = 0;
        };

        /** Get the number of key frames in the path.
        */
        uint32_t getKeyFrameCount() const {return (uint32_t)mKeyFrames.size();}

        /** Get a particular keyframe.
        */
        const Frame& getKeyFrame(uint32_t frameID) const { return mKeyFrames[frameID]; }

        /** Set a key frame's position.
            \param[in] frameID Key frame index
            \param[in] pos Position
        */
        void setFramePosition(uint32_t frameID, const glm::vec3& pos) { mDirty = true; mKeyFrames[frameID].position = pos; }

        /** Set a key frame's look-at target.
            \param[in] frameID Key frame index
            \param[in] target Target position
        */
        void setFrameTarget(uint32_t frameID, const glm::vec3& target) { mDirty = true; mKeyFrames[frameID].target = target; }

        /** Set a key frame's up vector.
            \param[in] frameID Key frame index
            \param[in] up Up vector
        */
        void setFrameUp(uint32_t frameID, const glm::vec3& up) { mDirty = true; mKeyFrames[frameID].up = up; }

        /** Set a key frame's time. This will re-sort the key frame in the path.
            \param[in] frameID Key frame index
            \param[in] time Time in seconds
        */
        uint32_t setFrameTime(uint32_t frameID, float time);

        /** Get interpolated frame data without modifying the path's current state.
            \param[in] frameID Beginning frame ID
            \param[in] t Interpolation factor between 0 and 1 (between frameID, and the next frame). Respects the path's interpolation mode.
            \param[out] frameOut Frame data struct to store output
        */
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