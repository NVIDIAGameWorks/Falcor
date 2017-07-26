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
#include "glm/mat3x3.hpp"
#include "Utils/UserInput.h"
#include "Utils/CpuTimer.h"
#include "Graphics/Camera/Camera.h"

namespace Falcor
{
    /** Camera controller interface. Camera controllers should inherit from this object.
    */
    class CameraController
    {
    public:
        using SharedPtr = std::shared_ptr<CameraController>;
        virtual ~CameraController() = default;

        /** Attach a camera to this controller
        */
        virtual void attachCamera(const Camera::SharedPtr& pCamera) { mpCamera = pCamera; };
        /** Handle mouse events
        */
        virtual bool onMouseEvent(const MouseEvent& mouseEvent) {return false;}
        /* Handle keyboard events
        */
        virtual bool onKeyEvent(const KeyboardEvent& keyboardEvent) { return false; }
        /** Update the camera position and orientation
            Returns whether the camera was updated/changed
        */
        virtual bool update() = 0;        

        /** Set the camera's speed
        \param[in] Speed Camera speed. Measured in WorldUnits per second.
        */
        void setCameraSpeed(float speed) { mSpeed = speed; }

    protected:
        CameraController() = default;
        Camera::SharedPtr mpCamera = nullptr;
        float mSpeed = 1;
    };

    /** Model-view camera controller. Orbits around a given point.
        To controll the camera:
        * Left mouse click+movement will orbit around the model.
        * Mouse wheel zooms in/out.
    */
    class ModelViewCameraController : public CameraController
    {
    public:
        /** Handle mouse events
        */
        bool onMouseEvent(const MouseEvent& mouseEvent) override;
        /** Set the model parameters
            \param[in] Center The model's center. This is the position in which the camera will orbit around.
            \param[in] Radius The model's radius. Used to determin the speed of movement when zooming in/out.
            \param[in] InitialDistanceInRadius The initial distance of the camera from the model, measured in the model's radius.
        */
        void setModelParams(const glm::vec3& center, float radius, float initialDistanceInRadius);
        /** Update the camera position and orientation
            Returns whether the camera was updated/changed
        */
        bool update() override;

    private:
        glm::vec3 mModelCenter;
        float mModelRadius;
        float mCameraDistance;
        bool mbDirty;

        glm::mat3x3 mRotation;
        glm::vec3 mLastVector;
        bool mIsLeftButtonDown = false;
        bool mShouldRotate = false;
    };

    /** First person camera controller.
        if b6DoF is false, camera will behave like a regular FPS camera (up vector doesn't change). If b6DoF is true, camera can rotate in all direction
        Controls:
        - W/S/A/D to move forward/backward/stride left/stride right.
        - Q/E to move down/up.
        - Left mouse button+mouse movement to rotate camera.
        - Right mouse button+mouse movement to roll camera (for 6DoF camera controller only).
        - Ctrl for slower movement.
        - Shift for faster movement.
    */
    template<bool b6DoF>
    class FirstPersonCameraControllerCommon : public CameraController
    {
    public:
        FirstPersonCameraControllerCommon();

        /** Handle mouse events
        */
        bool onMouseEvent(const MouseEvent& mouseEvent) override;
        /** Handle keyboard events
        */
        bool onKeyEvent(const KeyboardEvent& keyboardEvent) override;
        /** Update the camera position and orientation
            Returns whether the camera was updated/changed
        */
        bool update() override;

    private:
        bool mIsLeftButtonDown = false;
        bool mIsRightButtonDown = false;
        bool mShouldRotate = false;

        glm::vec2 mLastMousePos;
        glm::vec2 mMouseDelta;

        CpuTimer mTimer;
        union
        {
            struct
            {
                bool forward  : 1;
                bool backward : 1;
                bool right    : 1;
                bool left     : 1;
                bool up       : 1;
                bool down     : 1;
            };
            bool b = false;
        } mMovement;
        float mSpeedModifier = 1.0f;
    };

    using FirstPersonCameraController = FirstPersonCameraControllerCommon<false>;
    using SixDoFCameraController = FirstPersonCameraControllerCommon<true>;

    class HmdCameraController : public SixDoFCameraController
    {
    public:
        virtual void attachCamera(const Camera::SharedPtr& pCamera) override;
        ~HmdCameraController();

        /** Update the camera position and orientation
        Returns whether the camera was updated/changed
        */
        bool update() override;

    private:
        void detachCamera();
        float mOrigFocalLength;
        float mOrigAspectRatio;
        glm::mat4 mInvPrevHmdViewMat;
    };
}