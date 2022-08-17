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
#include "Camera.h"
#include "Core/Program/ShaderVar.h"
#include "Utils/Logger.h"
#include "Utils/Math/AABB.h"
#include "Utils/Math/FalcorMath.h"
#include "Utils/UI/Gui.h"
#include "Utils/Scripting/ScriptBindings.h"
#include "Utils/Scripting/ScriptWriter.h"
#include <glm/gtc/type_ptr.hpp>

namespace Falcor
{
    namespace
    {
        const std::string kAnimated = "animated";
        const std::string kPosition = "position";
        const std::string kTarget = "target";
        const std::string kUp = "up";
    }

    static_assert(sizeof(CameraData) % (sizeof(float4)) == 0, "CameraData size should be a multiple of 16");

    // Default dimensions of full frame cameras and 35mm film
    const float Camera::kDefaultFrameHeight = 24.0f;

    Camera::Camera(const std::string& name)
        : mName(name)
    {
    }

    Camera::SharedPtr Camera::create(const std::string& name)
    {
        return SharedPtr(new Camera(name));
    }

    Camera::Changes Camera::beginFrame(bool firstFrame)
    {
        if (mJitterPattern.pGenerator)
        {
            float2 jitter = mJitterPattern.pGenerator->next();
            jitter *= mJitterPattern.scale;
            setJitterInternal(jitter.x, jitter.y);
        }

        calculateCameraParameters();

        if (firstFrame) mPrevData = mData;

        // Keep copies of the transforms used for the previous frame. We need these for computing motion vectors etc.
        mData.prevViewMat = mPrevData.viewMat;
        mData.prevViewProjMatNoJitter = mPrevData.viewProjMatNoJitter;
        mData.prevPosW = mPrevData.posW;

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
        if (mPrevData.frameWidth != mData.frameWidth)   mChanges |= Changes::Frustum;

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
            if (mPreserveHeight)
            {
                // Set frame width based on height and aspect ratio
                mData.frameWidth = mData.frameHeight * mData.aspectRatio;
            }
            else
            {
                // Set frame height based on width and aspect ratio
                mData.frameHeight = mData.frameWidth / mData.aspectRatio;
            }

            // Interpret focal length of 0 as 0 FOV. Technically 0 FOV should be focal length of infinity.
            const float fovY = mData.focalLength == 0.0f ? 0.0f : focalLengthToFovY(mData.focalLength, mData.frameHeight);

            if (mEnablePersistentViewMat)
            {
                mData.viewMat = mPersistentViewMat;
            }
            else
            {
                mData.viewMat = rmcv::lookAt(mData.posW, mData.target, mData.up);
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
                    mData.projMat = rmcv::perspective(fovY, mData.aspectRatio, mData.nearZ, mData.farZ);
                }
                else
                {
                    // Take the length of look-at vector as half a viewport size
                    const float halfLookAtLength = length(mData.posW - mData.target) * 0.5f;
                    mData.projMat = rmcv::ortho(-halfLookAtLength, halfLookAtLength, -halfLookAtLength, halfLookAtLength, mData.nearZ, mData.farZ);
                }
            }

            // Build jitter matrix
            // (jitterX and jitterY are expressed as subpixel quantities divided by the screen resolution
            //  for instance to apply an offset of half pixel along the X axis we set jitterX = 0.5f / Width)
            rmcv::mat4 jitterMat = rmcv::translate(float3(2.0f * mData.jitterX, 2.0f * mData.jitterY, 0.0f));

            // Apply jitter matrix to the projection matrix
            mData.viewProjMatNoJitter = mData.projMat * mData.viewMat;
            mData.projMatNoJitter = mData.projMat;
            mData.projMat = jitterMat * mData.projMat;

            mData.viewProjMat = mData.projMat * mData.viewMat;
            mData.invViewProj = rmcv::inverse(mData.viewProjMat);

            // Extract camera space frustum planes from the VP matrix
            // See: https://fgiesen.wordpress.com/2012/08/31/frustum-planes-from-the-projection-matrix/
            rmcv::mat4 tempMat = rmcv::transpose(mData.viewProjMat);
            for (int i = 0; i < 6; i++)
            {
                float4 plane = (i & 1) ? tempMat.getCol(i >> 1) : -tempMat.getCol(i >> 1);
                if(i != 5) // Z range is [0, w]. For the 0 <= z plane we don't need to add w
                {
                    plane += tempMat.getCol(3);
                }

                mFrustumPlanes[i].xyz = float3(plane);
                mFrustumPlanes[i].sign = glm::sign(mFrustumPlanes[i].xyz);
                mFrustumPlanes[i].negW = -plane.w;
            }

            // Ray tracing related vectors
            mData.cameraW = glm::normalize(mData.target - mData.posW) * mData.focalDistance;
            mData.cameraU = glm::normalize(glm::cross(mData.cameraW, mData.up));
            mData.cameraV = glm::normalize(glm::cross(mData.cameraU, mData.cameraW));
            const float ulen = mData.focalDistance * std::tan(fovY * 0.5f) * mData.aspectRatio;
            mData.cameraU *= ulen;
            const float vlen = mData.focalDistance * std::tan(fovY * 0.5f);
            mData.cameraV *= vlen;

            mDirty = false;
        }
    }

    const rmcv::mat4 Camera::getViewMatrix() const
    {
        calculateCameraParameters();
        return mData.viewMat;
    }

    const rmcv::mat4 Camera::getPrevViewMatrix() const
    {
        return mData.prevViewMat;
    }

    const rmcv::mat4 Camera::getProjMatrix() const
    {
        calculateCameraParameters();
        return mData.projMat;
    }

    const rmcv::mat4 Camera::getViewProjMatrix() const
    {
        calculateCameraParameters();
        return mData.viewProjMat;
    }

    const rmcv::mat4 Camera::getViewProjMatrixNoJitter() const
    {
        calculateCameraParameters();
        return mData.viewProjMatNoJitter;
    }

    const rmcv::mat4 Camera::getInvViewProjMatrix() const
    {
        calculateCameraParameters();
        return mData.invViewProj;
    }

    void Camera::setProjectionMatrix(const rmcv::mat4& proj)
    {
        mDirty = true;
        mPersistentProjMat = proj;
        togglePersistentProjectionMatrix(true);
    }

    void Camera::setViewMatrix(const rmcv::mat4& view)
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

    bool Camera::isObjectCulled(const AABB& box) const
    {
        calculateCameraParameters();

        bool isInside = true;
        // AABB vs. frustum test
        // See method 4b: https://fgiesen.wordpress.com/2010/10/17/view-frustum-culling/
        for (int plane = 0; plane < 6; plane++)
        {
            float3 signedHalfExtent = 0.5f * box.extent() * mFrustumPlanes[plane].sign;
            float dr = glm::dot(box.center() + signedHalfExtent, mFrustumPlanes[plane].xyz);
            isInside = isInside && (dr > mFrustumPlanes[plane].negW);
        }

        return !isInside;
    }

    void Camera::setShaderData(const ShaderVar& var) const
    {
        calculateCameraParameters();
        var["data"].setBlob(mData);
    }

    void Camera::setPatternGenerator(const CPUSampleGenerator::SharedPtr& pGenerator, const float2& scale)
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

    float Camera::computeScreenSpacePixelSpreadAngle(const uint32_t winHeightPixels) const
    {
        const float FOVrad = focalLengthToFovY(getFocalLength(), Camera::kDefaultFrameHeight);
        const float angle = std::atan(2.0f * std::tan(FOVrad * 0.5f) / winHeightPixels);
        return angle;
    }

    Ray Camera::computeRayPinhole(uint2 pixel, uint2 frameDim, bool applyJitter) const
    {
        Ray ray;
        ray.origin = mData.posW;

        // Compute the normalized ray direction assuming a pinhole camera.
        // Compute sample position in screen space in [0,1] with origin at the top-left corner.
        // The camera jitter offsets the sample by +-0.5 pixels from the pixel center.
        float2 p = (float2(pixel) + float2(0.5f, 0.5f)) / float2(frameDim);
        if (applyJitter) p += float2(-mData.jitterX, mData.jitterY);

        float2 ndc = float2(2.0f, -2.0f) * p + float2(-1.0f, 1.0f);

        // Compute the normalized ray direction assuming a pinhole camera.
        ray.dir = glm::normalize(ndc.x * mData.cameraU + ndc.y * mData.cameraV + mData.cameraW);

        float invCos = 1.f / glm::dot(glm::normalize(mData.cameraW), ray.dir);
        ray.tMin = mData.nearZ * invCos;
        ray.tMax = mData.farZ * invCos;

        return ray;
    }

    void Camera::updateFromAnimation(const rmcv::mat4& transform)
    {
        float3 up = float3(transform.getCol(1));
        float3 fwd = float3(-transform.getCol(2));
        float3 pos = float3(transform.getCol(3));
        setUpVector(up);
        setPosition(pos);
        setTarget(pos + fwd);
    }

    void Camera::renderUI(Gui::Widgets& widget)
    {
        if (mHasAnimation) widget.checkbox("Animated", mIsAnimated);

        float focalLength = getFocalLength();
        if (widget.var("Focal Length", focalLength, 0.0f, FLT_MAX, 0.25f)) setFocalLength(focalLength);

        float aspectRatio = getAspectRatio();
        if (widget.var("Aspect Ratio", aspectRatio, 0.f, FLT_MAX, 0.001f)) setAspectRatio(aspectRatio);

        float focalDistance = getFocalDistance();
        if (widget.var("Focal Distance", focalDistance, 0.f, FLT_MAX, 0.05f)) setFocalDistance(focalDistance);

        float apertureRadius = getApertureRadius();
        if (widget.var("Aperture Radius", apertureRadius, 0.f, FLT_MAX, 0.001f)) setApertureRadius(apertureRadius);

        float shutterSpeed = getShutterSpeed();
        if (widget.var("Shutter Speed", shutterSpeed, 0.f, FLT_MAX, 0.001f)) setShutterSpeed(shutterSpeed);

        float ISOSpeed = getISOSpeed();
        if (widget.var("ISO Speed", ISOSpeed, 0.8f, FLT_MAX, 0.25f)) setISOSpeed(ISOSpeed);

        float2 depth = float2(mData.nearZ, mData.farZ);
        if (widget.var("Depth Range", depth, 0.f, FLT_MAX, 0.1f)) setDepthRange(depth.x, depth.y);

        float3 pos = getPosition();
        if (widget.var("Position", pos, -FLT_MAX, FLT_MAX, 0.001f)) setPosition(pos);

        float3 target = getTarget();
        if (widget.var("Target", target, -FLT_MAX, FLT_MAX, 0.001f)) setTarget(target);

        float3 up = getUpVector();
        if (widget.var("Up", up, -FLT_MAX, FLT_MAX, 0.001f)) setUpVector(up);

        if (widget.button("Dump")) dumpProperties();
    }

    std::string Camera::getScript(const std::string& cameraVar)
    {
        std::string c;

        if (hasAnimation() && !isAnimated())
        {
            c += ScriptWriter::makeSetProperty(cameraVar, kAnimated, false);
        }

        if (!hasAnimation() || !isAnimated())
        {
            c += ScriptWriter::makeSetProperty(cameraVar, kPosition, getPosition());
            c += ScriptWriter::makeSetProperty(cameraVar, kTarget, getTarget());
            c += ScriptWriter::makeSetProperty(cameraVar, kUp, getUpVector());
        }

        return c;
    }

    void Camera::dumpProperties()
    {
        pybind11::dict d;
        d["position"] = getPosition();
        d["target"] = getTarget();
        d["up"] = getUpVector();
        d["focalLength"] = getFocalLength();
        d["focalDistance"] = getFocalDistance();
        d["apertureRadius"] = getApertureRadius();
        std::cout << pybind11::str(d) << std::endl;
    }

    FALCOR_SCRIPT_BINDING(Camera)
    {
        using namespace pybind11::literals;

        FALCOR_SCRIPT_BINDING_DEPENDENCY(Animatable)

        pybind11::class_<Camera, Animatable, Camera::SharedPtr> camera(m, "Camera");
        camera.def_property("name", &Camera::getName, &Camera::setName);
        camera.def_property("aspectRatio", &Camera::getAspectRatio, &Camera::setAspectRatio);
        camera.def_property("focalLength", &Camera::getFocalLength, &Camera::setFocalLength);
        camera.def_property("frameHeight", &Camera::getFrameHeight, &Camera::setFrameHeight);
        camera.def_property("frameWidth", &Camera::getFrameWidth, &Camera::setFrameWidth);
        camera.def_property("focalDistance", &Camera::getFocalDistance, &Camera::setFocalDistance);
        camera.def_property("apertureRadius", &Camera::getApertureRadius, &Camera::setApertureRadius);
        camera.def_property("shutterSpeed", &Camera::getShutterSpeed, &Camera::setShutterSpeed);
        camera.def_property("ISOSpeed", &Camera::getISOSpeed, &Camera::setISOSpeed);
        camera.def_property("nearPlane", &Camera::getNearPlane, &Camera::setNearPlane);
        camera.def_property("farPlane", &Camera::getFarPlane, &Camera::setFarPlane);
        camera.def_property(kPosition.c_str(), &Camera::getPosition, &Camera::setPosition);
        camera.def_property(kTarget.c_str(), &Camera::getTarget, &Camera::setTarget);
        camera.def_property(kUp.c_str(), &Camera::getUpVector, &Camera::setUpVector);
        camera.def(pybind11::init(&Camera::create), "name"_a = "");
    }
}
