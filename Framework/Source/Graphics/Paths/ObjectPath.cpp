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
#include "ObjectPath.h"
#include "MovableObject.h"
#include "glm/common.hpp"
#include <algorithm>

namespace Falcor
{
    ObjectPath::SharedPtr ObjectPath::create()
    {
        return SharedPtr(new ObjectPath);
    }

    ObjectPath::~ObjectPath() = default;

    uint32_t ObjectPath::addKeyFrame(float time, const glm::vec3& position, const glm::vec3& target, const glm::vec3& up)
    {
        Frame keyFrame;
        keyFrame.time = time;
        keyFrame.target = target;
        keyFrame.position = position;
        keyFrame.up = up;
        mDirty = true;

        if(mKeyFrames.size() == 0 || mKeyFrames[0].time > time)
        {
            mKeyFrames.insert(mKeyFrames.begin(), keyFrame);
            return 0;
        }
        else
        {
            for(size_t i = 0 ; i < mKeyFrames.size() ; i++)
            {
                auto& current = mKeyFrames[i];
                // If we already have a key-frame at the same time, replace it
                if(current.time == time)
                {
                    current = keyFrame;
                    return (uint32_t)i;
                }

                // If this is not the last frame, Check if we are in between frames
                if(i < mKeyFrames.size() - 1)
                {
                    auto& Next = mKeyFrames[i + 1];
                    if(current.time < time && Next.time > time)
                    {
                        mKeyFrames.insert(mKeyFrames.begin() + i + 1, keyFrame);
                        return (uint32_t)i + 1;
                    }
                }
            }

            // If we got here, need to push it to the end of the list
            mKeyFrames.push_back(keyFrame);
            return (uint32_t)mKeyFrames.size() - 1;
        }
    }

    bool ObjectPath::animate(double currentTime)
    {
        if(mKeyFrames.size() == 0 || mpObjects.size() == 0)
        {
            return false;
        }

        double animTime = currentTime;
        const auto& firstFrame = mKeyFrames[0];
        const auto& lastFrame = mKeyFrames[mKeyFrames.size() - 1];
        if(mRepeatAnimation)
        {
            float delta = lastFrame.time - firstFrame.time;
            if(delta)
            {
                animTime = float(fmod(currentTime, delta));
                animTime += firstFrame.time;
            }
            else
                animTime = lastFrame.time;
        }

        if(animTime >= lastFrame.time)
        {
            mCurrentFrame = lastFrame;
        }
        else if(animTime <= firstFrame.time)
        {
            mCurrentFrame = firstFrame;
        }
        else
        {
            // Find out where we are
            bool foundFrame = false;
            for(uint32_t i = 0 ; i < (uint32_t)mKeyFrames.size() - 1; i++)
            {
                const auto& curKey = mKeyFrames[i];
                const auto& nextKey = mKeyFrames[i + 1];

                if(animTime >= curKey.time && animTime < nextKey.time)
                {
                    // Found the animation keys. Interpolate
                    float t = getInterpolationFactor(i, animTime);
                    getFrameAt(i, t, mCurrentFrame);
                    foundFrame = true;
                    break;
                }
            }
            assert(foundFrame);
        }

        for(auto& pObj : mpObjects)
        {
            pObj->move(mCurrentFrame.position, mCurrentFrame.target, mCurrentFrame.up);
        }

        return true;
    }

    void ObjectPath::getFrameAt(uint32_t frameID, float t, Frame& frameOut)
    {
        if (getKeyFrameCount() == 1)
        {
            frameOut = mKeyFrames[0];
            return;
        }

        if (mMode == Interpolation::Linear || getKeyFrameCount() < 3)
        {
            frameOut = linearInterpolation(frameID, t);
            return;
        }
        else if(mMode == Interpolation::CubicSpline)
        {
            frameOut = cubicSplineInterpolation(frameID, t);
            return;
        }

        should_not_get_here();
        return;
    }

    float ObjectPath::getInterpolationFactor(uint32_t frameID, double currentTime) const
    {
        const Frame& current = mKeyFrames[frameID];
        const Frame& next = mKeyFrames[frameID + 1];
        double delta = next.time - current.time;
        double curTime = currentTime - current.time;
        return float(curTime / delta);
    }

    ObjectPath::Frame ObjectPath::linearInterpolation(uint32_t currentFrame, float t) const
    {
        const uint32_t nextFrame = std::min(currentFrame + 1, getKeyFrameCount() - 1);

        const Frame& current = mKeyFrames[currentFrame];
        const Frame& next = mKeyFrames[nextFrame];

        Frame result;
        result.position = glm::mix(current.position, next.position, t);
        result.target = glm::mix(current.target, next.target, t);
        result.up = glm::mix(current.up, next.up, t);
        result.time = glm::mix(current.time, next.time, t);

        return result;
    }

    ObjectPath::Frame ObjectPath::cubicSplineInterpolation(uint32_t currentFrame, float t)
    {
        if (mDirty)
        {
            mDirty = false;
            std::vector<glm::vec3> positions, targets, ups;
            for (auto& a : mKeyFrames)
            {
                positions.push_back(a.position);
                targets.push_back(a.target);
                ups.push_back(a.up);
            }

            mpPositionSpline = std::make_unique<Vec3CubicSpline>(positions.data(), uint32_t(mKeyFrames.size()));
            mpTargetSpline = std::make_unique<Vec3CubicSpline>(targets.data(), uint32_t(mKeyFrames.size()));
            mpUpSpline = std::make_unique<Vec3CubicSpline>(ups.data(), uint32_t(mKeyFrames.size()));
        }

        const Frame& current = mKeyFrames[currentFrame];
        const Frame& next = mKeyFrames[currentFrame + 1];

        Frame result;
        result.position = mpPositionSpline->interpolate(currentFrame, t);
        result.target = mpTargetSpline->interpolate(currentFrame, t);
        result.up = mpUpSpline->interpolate(currentFrame, t);
        result.time = glm::mix(current.time, next.time, t);

        return result;
    }

    void ObjectPath::attachObject(const IMovableObject::SharedPtr& pObject)
    {
        // Only attach the object if its not already found
        if(std::find(mpObjects.begin(), mpObjects.end(), pObject) == mpObjects.end())
        {
            mpObjects.push_back(pObject);
        }
    }

    void ObjectPath::detachObject(const IMovableObject::SharedPtr& pObject)
    {
        auto it = std::find(mpObjects.begin(), mpObjects.end(), pObject);
        if(it != mpObjects.end())
        {
            mpObjects.erase(it);
        }
    }

    void ObjectPath::removeKeyFrame(uint32_t frameID)
    {
        mKeyFrames.erase(mKeyFrames.begin() + frameID);
        mDirty = true;
    }

    uint32_t ObjectPath::setFrameTime(uint32_t frameID, float time)
    {
        const auto Frame = mKeyFrames[frameID];
        removeKeyFrame(frameID);
        return addKeyFrame(time, Frame.position, Frame.target, Frame.up);
    }

}