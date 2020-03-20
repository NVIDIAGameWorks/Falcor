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
 **************************************************************************/
#include "stdafx.h"
#include "Camera.h"
#include "Utils/Math/AABB.h"
#include "Utils/Math/FalcorMath.h"
#include "Utils/UI/Gui.h"
#include "glm/gtc/type_ptr.hpp"

namespace Falcor
{
    static_assert(sizeof(CameraData) % (sizeof(vec4)) == 0, "CameraData size should be a multiple of 16");

    // Default dimensions of full frame cameras and 35mm film
    const float Camera::kDefaultFrameHeight = 24.0f;

    Camera::Camera()
    {
    }

    Camera::~Camera() = default;

    Camera::SharedPtr Camera::create()
    {
        Camera* pCamera = new Camera;
        return SharedPtr(pCamera);
    }

    Camera::Changes Camera::beginFrame(bool firstFrame)
    {
        if (mJitterPattern.pGenerator)
        {
            vec2 jitter = mJitterPattern.pGenerator->next();
            jitter *= mJitterPattern.scale;
            setJitterInternal(jitter.x, jitter.y);
        }

        calculateCameraParameters();

        if (firstFrame) mPrevData = mData;

        // Keep copies of the transforms used for the previous frame. We need these for computing motion vectors etc.
        mData.prevViewProjMatNoJitter = mPrevData.viewProjMatNoJitter;

        mChanges = is_set(mChanges, Changes::Movement | Changes::Frustum) ? Changes::History : Changes::None;

        if (mPrevData.posW != mData.posW) mChanges |= Changes::Movement;
        if (mPrevData.up != mData.up) mChanges |= Changes::Movement;
        if (mPrevData.target != mData.target) mChanges |= Changes::Movement;

        if (mPrevData.focalDistance != mData.focalDistance) mChanges    |= Changes::FocalDistance;
        if (mPrevData.apertureRadius != mData.apertureRadius) mChanges  |= Changes::Aperture | Changes::Exposure;
        if (mPrevData.shutterSpeed != mData.shutterSpeed) mChanges      |= Changes::Exposure;
        if (mPrevData.ISOSpeed != mData.ISOSpeed) mChanges              |= Changes::Exposure;

        if (mPrevData.focalLength != mData.focalLength) mChanges |= Changes::Frustum;
        if (mPrevData.aspectRatio != mData.aspectRatio) mChanges |= Changes::Frustum;
        if (mPrevData.nearZ != mData.nearZ)             mChanges |= Changes::Frustum;
        if (mPrevData.farZ != mData.farZ)               mChanges |= Changes::Frustum;
        if (mPrevData.frameHeight != mData.frameHeight) mChanges |= Changes::Frustum;

        // Jitter
        if (mPrevData.jitterX != mData.jitterX) mChanges |= Changes::Jitter;
        if (mPrevData.jitterY != mData.jitterY) mChanges |= Changes::Jitter;

        mPrevData = mData;

        return getChanges();
    }

    void Camera::calculateCameraParameters() const
    {
        if (mDirty)
        {
            // Interpret focal length of 0 as 0 FOV. Technically 0 FOV should be focal length of infinity.
            const float fovY = mData.focalLength == 0.0f ? 0.0f : focalLengthToFovY(mData.focalLength, mData.frameHeight);

            if (mEnablePersistentViewMat)
            {
                mData.viewMat = mPersistentViewMat;
            }
            else
            {
                mData.viewMat = glm::lookAt(mData.posW, mData.target, mData.up);
            }

            // if camera projection is set to be persistent, don't override it.
            if (mEnablePersistentProjMat)
            {
                mData.projMat = mPersistentProjMat;
            }
            else
            {
                if (fovY != 0.f)
                {
                    mData.projMat = glm::perspective(fovY, mData.aspectRatio, mData.nearZ, mData.farZ);
                }
                else
                {
                    // Take the length of look-at vector as half a viewport size
                    const float halfLookAtLength = length(mData.posW - mData.target) * 0.5f;
                    mData.projMat = glm::ortho(-halfLookAtLength, halfLookAtLength, -halfLookAtLength, halfLookAtLength, mData.nearZ, mData.farZ);
                }
            }

            // Build jitter matrix
            // (jitterX and jitterY are expressed as subpixel quantities divided by the screen resolution
            //  for instance to apply an offset of half pixel along the X axis we set jitterX = 0.5f / Width)
            glm::mat4 jitterMat(1.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 1.0f, 0.0f,
                2.0f * mData.jitterX, 2.0f * mData.jitterY, 0.0f, 1.0f);
            // Apply jitter matrix to the projection matrix
            mData.viewProjMatNoJitter = mData.projMat * mData.viewMat;
            mData.projMat = jitterMat * mData.projMat;

            mData.viewProjMat = mData.projMat * mData.viewMat;
            mData.invViewProj = glm::inverse(mData.viewProjMat);

            // Extract camera space frustum planes from the VP matrix
            // See: https://fgiesen.wordpress.com/2012/08/31/frustum-planes-from-the-projection-matrix/
            glm::mat4 tempMat = glm::transpose(mData.viewProjMat);
            for (int i = 0; i < 6; i++)
            {
                glm::vec4 plane = (i & 1) ? tempMat[i >> 1] : -tempMat[i >> 1];
                if(i != 5) // Z range is [0, w]. For the 0 <= z plane we don't need to add w
                {
                    plane += tempMat[3];
                }

                mFrustumPlanes[i].xyz = glm::vec3(plane);
                mFrustumPlanes[i].sign = glm::sign(mFrustumPlanes[i].xyz);
                mFrustumPlanes[i].negW = -plane.w;
            }

            // Ray tracing related vectors
            mData.cameraW = glm::normalize(mData.target - mData.posW) * mData.focalDistance;
            mData.cameraU = glm::normalize(glm::cross(mData.cameraW, mData.up));
            mData.cameraV = glm::normalize(glm::cross(mData.cameraU, mData.cameraW));
            const float ulen = mData.focalDistance * tanf(fovY * 0.5f) * mData.aspectRatio;
            mData.cameraU *= ulen;
            const float vlen = mData.focalDistance * tanf(fovY * 0.5f);
            mData.cameraV *= vlen;

            mDirty = false;
        }
    }

    const glm::mat4& Camera::getViewMatrix() const
    {
        calculateCameraParameters();
        return mData.viewMat;
    }

    const glm::mat4& Camera::getProjMatrix() const
    {
        calculateCameraParameters();
        return mData.projMat;
    }

    const glm::mat4& Camera::getViewProjMatrix() const
    {
        calculateCameraParameters();
        return mData.viewProjMat;
    }

    const glm::mat4& Camera::getInvViewProjMatrix() const
    {
        calculateCameraParameters();
        return mData.invViewProj;
    }

    void Camera::setProjectionMatrix(const glm::mat4& proj)
    {
        mDirty = true;
        mPersistentProjMat = proj;
        togglePersistentProjectionMatrix(true);
    }

    void Camera::setViewMatrix(const glm::mat4& view)
    {
        mDirty = true;
        mPersistentViewMat = view;
        togglePersistentViewMatrix(true);
    }

    void Camera::togglePersistentProjectionMatrix(bool persistent)
    {
        mEnablePersistentProjMat = persistent;
    }

    void Camera::togglePersistentViewMatrix(bool persistent)
    {
        mEnablePersistentViewMat = persistent;
    }

    bool Camera::isObjectCulled(const BoundingBox& box) const
    {
        calculateCameraParameters();

        bool isInside = true;
        // AABB vs. frustum test
        // See method 4b: https://fgiesen.wordpress.com/2010/10/17/view-frustum-culling/
        for (int plane = 0; plane < 6; plane++)
        {
            glm::vec3 signedExtent = box.extent * mFrustumPlanes[plane].sign;
            float dr = glm::dot(box.center + signedExtent, mFrustumPlanes[plane].xyz);
            isInside = isInside && (dr > mFrustumPlanes[plane].negW);
        }

        return !isInside;
    }

    void Camera::setShaderData(const ShaderVar& var) const
    {
        calculateCameraParameters();
        var["data"].setBlob(mData);
    }

    void Camera::setPatternGenerator(const CPUSampleGenerator::SharedPtr& pGenerator, const vec2& scale)
    {
        mJitterPattern.pGenerator = pGenerator;
        mJitterPattern.scale = scale;
        if (!pGenerator)
        {
            setJitterInternal(0, 0);
        }
    }

    void Camera::setJitter(float jitterX, float jitterY)
    {
        if (mJitterPattern.pGenerator)
        {
            logWarning("Camera::setJitter() called when a pattern-generator object was attached to the camera. Detaching the pattern-generator");
            mJitterPattern.pGenerator = nullptr;
        }
        setJitterInternal(jitterX, jitterY);
    }

    void Camera::setJitterInternal(float jitterX, float jitterY)
    {
        mData.jitterX = jitterX;
        mData.jitterY = jitterY;
        mDirty = true;
    }

    void Camera::renderUI(Gui* pGui, const char* uiGroup)
    {
        if (!uiGroup) uiGroup = "Camera Settings";

        auto g = Gui::Group(pGui, uiGroup);
        if (g.open())
        {
            float focalLength = getFocalLength();
            if (g.var("Focal Length", focalLength, 0.0f, FLT_MAX, 0.25f)) setFocalLength(focalLength);

            float aspectRatio = getAspectRatio();
            if (g.var("Aspect Ratio", aspectRatio, 0.f, FLT_MAX, 0.001f)) setAspectRatio(aspectRatio);

            float focalDistance = getFocalDistance();
            if (g.var("Focal Distance", focalDistance, 0.f, FLT_MAX, 0.05f)) setFocalDistance(focalDistance);

            float apertureRadius = getApertureRadius();
            if (g.var("Aperture Radius", apertureRadius, 0.f, FLT_MAX, 0.001f)) setApertureRadius(apertureRadius);

            float shutterSpeed = getShutterSpeed();
            if (g.var("Shutter Speed", shutterSpeed, 0.f, FLT_MAX, 0.001f)) setShutterSpeed(shutterSpeed);

            float ISOSpeed = getISOSpeed();
            if (g.var("ISO Speed", ISOSpeed, 0.8f, FLT_MAX, 0.25f)) setISOSpeed(ISOSpeed);

            float2 depth = glm::vec2(mData.nearZ, mData.farZ);
            if (g.var("Depth Range", depth, 0.f, FLT_MAX, 0.1f)) setDepthRange(depth.x, depth.y);

            float3 pos = getPosition();
            if (g.var("Position", pos, -FLT_MAX, FLT_MAX, 0.001f)) setPosition(pos);

            float3 target = getTarget();
            if (g.var("Target", target, -FLT_MAX, FLT_MAX, 0.001f)) setTarget(target);

            float3 up = getUpVector();
            if (g.var("Up", up, -FLT_MAX, FLT_MAX, 0.001f)) setUpVector(up);

            g.release();
        }
    }

    SCRIPT_BINDING(Camera)
    {
        auto camera = m.regClass(Camera);
        camera.roProperty("name", &Camera::getName);
        camera.property("aspectRatio", &Camera::getAspectRatio, &Camera::setAspectRatio);
        camera.property("focalLength", &Camera::getFocalLength, &Camera::setFocalLength);
        camera.property("frameHeight", &Camera::getFrameHeight, &Camera::setFrameHeight);
        camera.property("focalDistance", &Camera::getFocalDistance, &Camera::setFocalDistance);
        camera.property("apertureRadius", &Camera::getApertureRadius, &Camera::setApertureRadius);
        camera.property("shutterSpeed", &Camera::getShutterSpeed, &Camera::setShutterSpeed);
        camera.property("ISOSpeed", &Camera::getISOSpeed, &Camera::setISOSpeed);
        camera.property("nearPlane", &Camera::getNearPlane, &Camera::setNearPlane);
        camera.property("farPlane", &Camera::getFarPlane, &Camera::setFarPlane);
        camera.property("position", &Camera::getPosition, &Camera::setPosition);
        camera.property("target", &Camera::getTarget, &Camera::setTarget);
        camera.property("up", &Camera::getUpVector, &Camera::setUpVector);
    }
}
