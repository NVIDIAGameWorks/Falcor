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
#include "CameraController.h"
#include "Camera.h"
#include "Utils/Math/FalcorMath.h"
#include "glm/mat3x3.hpp"
#include "glm/gtx/euler_angles.hpp"

#include "VR/OpenVR/VRSystem.h"

namespace Falcor
{
    glm::vec2 convertCamPosRange(const glm::vec2 pos)
    {
        // Convert [0,1] range to [-1, 1], and inverse the Y (screen-space y==0 is top)
        const glm::vec2 scale(2, -2);
        const glm::vec2 offset(-1, 1);
        glm::vec2 res = (pos * scale) + offset;
        return res;
    }

    void ModelViewCameraController::setModelParams(const glm::vec3& center, float radius, float distanceInRadius)
    {
        mModelCenter = center;
        mModelRadius = radius;
        mCameraDistance = distanceInRadius;
        mRotation = glm::mat3();
        mbDirty = true;
    }

    bool ModelViewCameraController::onMouseEvent(const MouseEvent& mouseEvent)
    {
        bool handled = false;
        switch(mouseEvent.type)
        {
        case MouseEvent::Type::Wheel:
            mCameraDistance -= (mouseEvent.wheelDelta.y * 0.2f);
            mbDirty = true;
            handled = true;
            break;
        case MouseEvent::Type::LeftButtonDown:
            mLastVector = project2DCrdToUnitSphere(convertCamPosRange(mouseEvent.pos));
            mIsLeftButtonDown = true;
            handled = true;
            break;
        case MouseEvent::Type::LeftButtonUp:
            handled = mIsLeftButtonDown;
            mIsLeftButtonDown = false;
            break;
        case MouseEvent::Type::Move:
            if(mIsLeftButtonDown)
            {
                glm::vec3 curVec = project2DCrdToUnitSphere(convertCamPosRange(mouseEvent.pos));
                glm::quat q = createQuaternionFromVectors(mLastVector, curVec);
                glm::mat3x3 rot = (glm::mat3x3)q;
                mRotation = rot * mRotation;
                mbDirty = true;
                mLastVector = curVec;
                handled = true;
                mShouldRotate = true;
            }
            break;
        default:
            break;
        }

        return handled;
    }

    bool ModelViewCameraController::update()
    {
        if(mpCamera && mbDirty)
        {
            mbDirty = false;
            mShouldRotate = false;
            mpCamera->setTarget(mModelCenter);

            glm::vec3 camPos = mModelCenter;
			camPos += (glm::vec3(0,0,1) * mRotation) * mModelRadius * mCameraDistance;
            mpCamera->setPosition(camPos);

            glm::vec3 up(0, 1, 0);
            up = up * mRotation;
            mpCamera->setUpVector(up);
            return true;
        }
        return false;
    }

    template<bool b6DoF>
    FirstPersonCameraControllerCommon<b6DoF>::FirstPersonCameraControllerCommon()
    {
        mTimer.update();
    }

    template<bool b6DoF>
    bool FirstPersonCameraControllerCommon<b6DoF>::onKeyEvent(const KeyboardEvent& event)
    {
        bool handled = false;
        bool keyPressed = (event.type == KeyboardEvent::Type::KeyPressed);

        switch(event.key)
        {
        case KeyboardEvent::Key::W:
            mMovement.forward = keyPressed;
            handled = true;
            break;
        case KeyboardEvent::Key::S:
            mMovement.backward = keyPressed;
            handled = true;
            break;
        case KeyboardEvent::Key::A:
            mMovement.right = keyPressed;
            handled = true;
            break;
        case KeyboardEvent::Key::D:
            mMovement.left = keyPressed;
            handled = true;
            break;
        case KeyboardEvent::Key::Q:
            mMovement.down = keyPressed;
            handled = true;
            break;
        case KeyboardEvent::Key::E:
            mMovement.up = keyPressed;
            handled = true;
            break;
        default:
            break;
        }

        mSpeedModifier = 1.0f;
        if (event.mods.isCtrlDown) mSpeedModifier = 0.25f;
        else if (event.mods.isShiftDown) mSpeedModifier = 10.0f;

        return handled;
    }

    template<bool b6DoF>
    bool FirstPersonCameraControllerCommon<b6DoF>::update()
    {
        mTimer.update();

        bool dirty = false;
        if(mpCamera)
        {
            if(mShouldRotate)
            {
                glm::vec3 camPos = mpCamera->getPosition();
                glm::vec3 camTarget = mpCamera->getTarget();
                glm::vec3 camUp = b6DoF ? mpCamera->getUpVector() : glm::vec3(0, 1, 0);;

                glm::vec3 viewDir = glm::normalize(camTarget - camPos);
                if(mIsLeftButtonDown)
                {
                    glm::vec3 sideway = glm::cross(viewDir, normalize(camUp));

                    // Rotate around x-axis
                    glm::quat qy = glm::angleAxis(mMouseDelta.y * mSpeedModifier, sideway);
                    glm::mat3 rotY(qy);
                    viewDir = viewDir * rotY;
                    camUp = camUp * rotY;

                    // Rotate around y-axis
                    glm::quat qx = glm::angleAxis(mMouseDelta.x * mSpeedModifier, camUp);
                    glm::mat3 rotX(qx);
                    viewDir = viewDir * rotX;

                    mpCamera->setTarget(camPos + viewDir);
                    mpCamera->setUpVector(camUp);
                    dirty = true;
                }

                if(b6DoF && mIsRightButtonDown)
                {
                    // Rotate around x-axis
                    glm::quat q = glm::angleAxis(mMouseDelta.x * mSpeedModifier, viewDir);
                    glm::mat3 rot(q);
                    camUp = camUp * rot;
                    mpCamera->setUpVector(camUp);
                    dirty = true;
                }

                mShouldRotate = false;
            }

            if(mMovement.b)
            {
                glm::vec3 movement(0, 0, 0);
                movement.z += mMovement.forward ? 1 : 0;
                movement.z += mMovement.backward ? -1 : 0;
                movement.x += mMovement.left ? 1 : 0;
                movement.x += mMovement.right ? -1 : 0;
                movement.y += mMovement.up ? 1 : 0;
                movement.y += mMovement.down ? -1 : 0;

                glm::vec3 camPos = mpCamera->getPosition();
                glm::vec3 camTarget = mpCamera->getTarget();
                glm::vec3 camUp = mpCamera->getUpVector();

                glm::vec3 viewDir = normalize(camTarget - camPos);
                glm::vec3 sideway = glm::cross(viewDir, normalize(camUp));

                float elapsedTime = mTimer.getElapsedTime();

                float curMove = mSpeedModifier * mSpeed * elapsedTime;
                camPos += movement.z * curMove * viewDir;
                camPos += movement.x * curMove * sideway;
                camPos += movement.y * curMove * camUp;

                camTarget = camPos + viewDir;

                mpCamera->setPosition(camPos);
                mpCamera->setTarget(camTarget);
                dirty = true;
            }
        }

        return dirty;
    }

    template<bool b6DoF>
    bool FirstPersonCameraControllerCommon<b6DoF>::onMouseEvent(const MouseEvent& event)
    {
        bool handled = false;
        switch(event.type)
        {
        case MouseEvent::Type::LeftButtonDown:
            mLastMousePos = event.pos;
            mIsLeftButtonDown = true;
            handled = true;
            break;
        case MouseEvent::Type::LeftButtonUp:
            handled = mIsLeftButtonDown;
            mIsLeftButtonDown = false;
            break;
        case MouseEvent::Type::RightButtonDown:
            mLastMousePos = event.pos;
            mIsRightButtonDown = true;
            handled = true;
            break;
        case MouseEvent::Type::RightButtonUp:
            handled = mIsRightButtonDown;
            mIsRightButtonDown = false;
            break;
        case MouseEvent::Type::Move:
            if(mIsLeftButtonDown || mIsRightButtonDown)
            {
                mMouseDelta = event.pos - mLastMousePos;
                mLastMousePos = event.pos;
                mShouldRotate = true;
                handled = true;
            }
            break;
        default:
            break;
        }

        return handled;
    }

    template class FirstPersonCameraControllerCommon < true > ;
    template class FirstPersonCameraControllerCommon < false > ;

    void HmdCameraController::attachCamera(const Camera::SharedPtr& pCamera)
    {
        if(VRSystem::instance() == nullptr)
        {
            logError("Can't attach camera to HmdCameraController. VRSystem not initialized", true);
            return;
        }

        if(mpCamera == pCamera)
        {
            return;
        }
        detachCamera();
        
        if (pCamera)
        {
            CameraController::attachCamera(pCamera);

            // Store the original camera parameters
            mOrigFocalLength = pCamera->getFocalLength();
            mOrigAspectRatio = pCamera->getAspectRatio();

            // Initialize the parameters from the HMD
            VRDisplay* pDisplay = VRSystem::instance()->getHMD().get();

            pCamera->setFocalLength(fovYToFocalLength(pDisplay->getFovY(), Camera::kDefaultFrameHeight));
            pCamera->setAspectRatio(pDisplay->getAspectRatio());

            mInvPrevHmdViewMat = glm::mat4();
        }
    }

    void setCameraParamsFromViewMat(Camera* pCamera, const glm::mat4& viewMat)
    {
        const glm::mat4 invViewMat = glm::inverse(viewMat);
        glm::vec4 hmdPos = invViewMat * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        glm::vec3 hmdTarget = glm::mat3(invViewMat) * glm::vec3(0.0f, 0.0f, -1.0f);
        glm::vec3 hmdUp = glm::mat3(invViewMat) * glm::vec3(0.0f, 1.0f, 0.0f);
        pCamera->setPosition(glm::vec3(hmdPos));
        pCamera->setTarget(hmdTarget + glm::vec3(hmdPos));
        pCamera->setUpVector(hmdUp);
    }

    bool HmdCameraController::update()
    {
        if(mpCamera)
        {
            VRDisplay* pDisplay = VRSystem::instance()->getHMD().get();

            // Set the near/far planes
            pDisplay->setDepthRange(mpCamera->getNearPlane(), mpCamera->getFarPlane());

            // Calculate the HMD world space position
            const glm::mat4 hmdWorldMat  = pDisplay->getWorldMatrix();
            glm::mat4 viewMat = hmdWorldMat * mInvPrevHmdViewMat * mpCamera->getViewMatrix();

            // Calculate the view params based on the center matrix
            setCameraParamsFromViewMat(mpCamera.get(), viewMat);

            // Update based on the mouse/keyboard movement
            if(SixDoFCameraController::update())
            {
                mpCamera->togglePersistentViewMatrix(false);
                viewMat = mpCamera->getViewMatrix();
            }

            // Get the right eye matrix
            glm::mat4 rightEyeView = pDisplay->getOffsetMatrix(VRDisplay::Eye::Right) * viewMat;
            glm::mat4 rightEyeProj = pDisplay->getProjectionMatrix(VRDisplay::Eye::Right);
            mpCamera->setRightEyeMatrices(rightEyeView, rightEyeProj);

            // Get the left eye matrix
            mpCamera->setProjectionMatrix(pDisplay->getProjectionMatrix(VRDisplay::Eye::Left));
            glm::mat4 leftEyeOffset = pDisplay->getOffsetMatrix(VRDisplay::Eye::Left);
            mpCamera->setViewMatrix(leftEyeOffset * viewMat);
            mInvPrevHmdViewMat = glm::inverse(leftEyeOffset * hmdWorldMat);
        }
        return true;
    }

    void HmdCameraController::detachCamera()
    {
        if(mpCamera)
        {
            // Calculate the view params based on the center matrix
            setCameraParamsFromViewMat(mpCamera.get(), mInvPrevHmdViewMat * mpCamera->getViewMatrix());

            // Restore the original parameters
            mpCamera->setFocalLength(mOrigFocalLength);
            mpCamera->setAspectRatio(mOrigAspectRatio);
            mpCamera->togglePersistentProjectionMatrix(false);
            mpCamera->togglePersistentViewMatrix(false);

            mpCamera = nullptr;
        }
    }

    HmdCameraController::~HmdCameraController()
    {
        detachCamera();
    }
}