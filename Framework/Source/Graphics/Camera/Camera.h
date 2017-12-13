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
#include <glm/common.hpp>
#include "glm/geometric.hpp"
#include "glm/mat4x4.hpp"
#include "Data/HostDeviceData.h"
#include <vector>
#include "Graphics/Paths/MovableObject.h"

namespace Falcor
{
    struct BoundingBox;
    class ConstantBuffer;

    /** Camera class. Default transform matrices are interpreted as left eye transform during stereo rendering.
    */
    class Camera : public IMovableObject, public std::enable_shared_from_this<Camera>
    {
    public:
        using SharedPtr = std::shared_ptr<Camera>;
        using SharedConstPtr = std::shared_ptr<const Camera>;

        // Default dimensions of full frame cameras and 35mm film
        static const float kDefaultFrameHeight;

        /** Create a new camera object.
        */
        static SharedPtr create();
        ~Camera();

        /** Name the camera.
        */
        void setName(const std::string& name) { mName = name; }

        /** Get the camera's name.
        */
        const std::string& getName() const { return mName; }

        /** Set the camera's aspect ratio (width/height).
        */
        void setAspectRatio(float aspectRatio) { mData.aspectRatio = aspectRatio; mDirty = true; }

        /** Get the camera's aspect ratio.
        */
        float getAspectRatio() const { return mData.aspectRatio; }

        /** Set camera focal length in mm. See FalcorMath.h for helper functions to convert between fovY angles.
        */
        void setFocalLength(float length) { mData.focalLength = length; mDirty = true; }

        /** Get the camera's focal length. See FalcorMath.h for helper functions to convert between fovY angles.
        */
        float getFocalLength() const { return mData.focalLength; }

        /** Get the camera's world space position.
        */
        const glm::vec3& getPosition() const { return mData.position; }

        /** Get the camera's world space up vector.
        */
        const glm::vec3& getUpVector() const {return mData.up;}

        /** Get the camera's world space target position.
        */
        const glm::vec3& getTarget() const { return mData.target; }

        /** Set the camera's world space position.
        */
        void setPosition(const glm::vec3& pos) { mData.position = pos; mDirty = true; }

        /** Set the camera's world space up vector.
        */
        void setUpVector(const glm::vec3& up) { mData.up = up; mDirty = true; }

        /** Set the camera's world space target position.
        */
        void setTarget(const glm::vec3& target) { mData.target = target; mDirty = true; }

        /** IMovable object interface.
        */
        void move(const glm::vec3& position, const glm::vec3& target, const glm::vec3& up) override;

        /** Set the camera's depth range.
        */
        void setDepthRange(float nearZ, float farZ) { mData.farZ = farZ; mData.nearZ = nearZ; mDirty = true; }

        /** Get the near plane depth.
        */
        float getNearPlane() const { return mData.nearZ; }

        /** Get the far plane depth.
        */
        float getFarPlane() const { return mData.farZ; }

        /** Set the camera's jitter.
            \param[in] jitterX Subpixel offset along X axis divided by screen width
            \param[in] jitterY Subpixel offset along Y axis divided by screen height
        */
        void setJitter(float jitterX, float jitterY) { mData.jitterX = jitterX; mData.jitterY = jitterY; mDirty = true; }
        float getJitterX() const { return mData.jitterX; }
        float getJitterY() const { return mData.jitterY; }

        /** Get the view matrix.
        */
        const glm::mat4& getViewMatrix() const;

        /** Get the projection matrix.
        */
        const glm::mat4& getProjMatrix() const;

        /** Get the view-projection matrix.
        */
        const glm::mat4& getViewProjMatrix() const;

        /** Get the inverse of the view-projection matrix.
        */
        const glm::mat4& getInvViewProjMatrix() const;

        /** Set the persistent projection matrix and sets camera to use the persistent matrix instead of calculating the matrix from its other settings.
        */
        void setProjectionMatrix(const glm::mat4& proj);

        /** Set the persistent view matrix and sets camera to use the persistent matrix instead of calculating the matrix from its other settings.
        */
        void setViewMatrix(const glm::mat4& view);

        /** Enable or disable usage of persistent projection matrix
            \param[in] persistent whether to set it persistent
        */
        void togglePersistentProjectionMatrix(bool persistent);
        void togglePersistentViewMatrix(bool persistent);

        /** Check if an object should be culled
            \param[in] box Bounding box of the object to check
        */
        bool isObjectCulled(const BoundingBox& box) const;

        /** Set camera data into a program's constant buffer.
            \param[in] pBuffer The constant buffer to set the parameters into.
            \param[in] varName The name of the light variable in the program.
        */
        void setIntoConstantBuffer(ConstantBuffer* pBuffer, const std::string& varName) const;

        /** Set camera data into a program's constant buffer.
            \param[in] pBuffer The constant buffer to set the parameters into.
            \param[in] offset Byte offset into the constant buffer to set data to.
        */
        void setIntoConstantBuffer(ConstantBuffer* pBuffer, const std::size_t& offset) const;

        /** Returns the raw camera data
        */
        const CameraData& getData() const { calculateCameraParameters(); return  mData; }

        /** Set transform matrices for the right eye
            \param[in] view Right eye view matrix
            \param[in] proj Right eye projection matrix
        */
        void setRightEyeMatrices(const glm::mat4& view, const glm::mat4& proj);

        /** Get the right eye view matrix.
        */
        const glm::mat4& getRightEyeViewMatrix() const { return mData.rightEyeViewMat; }

        /** Get the right eye projection matrix.
        */
        const glm::mat4& getRightEyeProjMatrix() const { return mData.rightEyeProjMat; }

        /** get the right eye view-projection matrix.
        */
        const glm::mat4& getRightEyeViewProjMatrix() const { return mData.rightEyeViewProjMat; }

        /** Get the size of the CameraData struct in bytes.
        */
        static uint32_t getShaderDataSize() 
        {
            static const size_t dataSize = sizeof(CameraData);
            static_assert(dataSize % sizeof(float) * 4 == 0, "Camera::CameraData size should be a multiple of 16");
            return dataSize;
        }

    private:
        Camera();

        mutable bool mDirty = true;
        mutable bool mEnablePersistentProjMat = false;
        mutable bool mEnablePersistentViewMat = false;
        mutable glm::mat4 mPersistentProjMat;
        mutable glm::mat4 mPersistentViewMat;

        std::string mName;

        void calculateCameraParameters() const;
        mutable CameraData mData;
        mutable glm::mat4 viewProjMatNoJitter;

        struct 
        {
            glm::vec3   xyz;    ///< Camera frustum plane position
            float       negW;   ///< Camera frustum plane, sign of the coordinates
            glm::vec3   sign;   ///< Camera frustum plane position
        } mutable mFrustumPlanes[6];
    };
}