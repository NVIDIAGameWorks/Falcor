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
#include "Camera.h"
#include "glm/gtx/quaternion.hpp"
#include "utils/AABB.h"
#include "Utils/math/FalcorMath.h"
#include "API/ConstantBuffer.h"

namespace Falcor
{
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

    void Camera::calculateCameraParameters() const
    {
        if (mDirty)
        {
            // Interpret focal length of 0 as 0 FOV. Technically 0 FOV should be focal length of infinity.
            const float fovY = mData.focalLength == 0.0f ? 0.0f : focalLengthToFovY(mData.focalLength, kDefaultFrameHeight);

            mData.prevViewProjMat = viewProjMatNoJitter;
            mData.rightEyePrevViewProjMat = mData.rightEyeViewProjMat;

            if (mEnablePersistentViewMat)
            {
                mData.viewMat = mPersistentViewMat;
            }
            else
            {
                mData.viewMat = glm::lookAt(mData.position, mData.target, mData.up);
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
                    const float halfLookAtLength = length(mData.position - mData.target) * 0.5f;
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
            viewProjMatNoJitter = mData.projMat * mData.viewMat;
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
            mData.cameraW = mData.target - mData.position;
            const float lookdir_len = length(mData.cameraW);
            mData.cameraU = glm::normalize(glm::cross(mData.cameraW, mData.up));
            mData.cameraV = glm::normalize(glm::cross(mData.cameraU, mData.cameraW));
            const float ulen = lookdir_len * tanf(fovY * 0.5f) * mData.aspectRatio;
            mData.cameraU *= ulen;
            const float vlen = lookdir_len * tanf(fovY * 0.5f);
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

    void Camera::setRightEyeMatrices(const glm::mat4& view, const glm::mat4& proj)
    {
        mData.rightEyeViewMat = view;
        mData.rightEyeProjMat = proj;
        mData.rightEyeViewProjMat = proj * view;
    }

    void Camera::setIntoConstantBuffer(ConstantBuffer* pCB, const std::string& varName) const
    {
        size_t offset = pCB->getVariableOffset(varName + ".viewMat");

        if (offset == ConstantBuffer::kInvalidOffset)
        {
            logWarning("Camera::setIntoConstantBuffer() - variable \"" + varName + "\"not found in constant buffer\n");
            return;
        }

        setIntoConstantBuffer(pCB, offset);
    }

    void Camera::setIntoConstantBuffer(ConstantBuffer* pBuffer, const std::size_t& offset) const
    {
        calculateCameraParameters();
        assert(offset + getShaderDataSize() <= pBuffer->getSize());

        pBuffer->setBlob(&mData, offset, getShaderDataSize());
    }

    void Camera::move(const glm::vec3& position, const glm::vec3& target, const glm::vec3& up)
    {
        setPosition(position);
        setTarget(target);
        setUpVector(up);
    }
}
