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
#include "CameraController.h"
#include "Camera.h"
#include "Utils/UI/InputTypes.h"
#include "Utils/Math/FalcorMath.h"

namespace Falcor
{
    namespace
    {
        const float kGamepadDeadZone = 0.1f;        ///< Gamepad dead zone.
        const float kGamepadPowerCurve = 1.2f;      ///< Gamepad power curve exponent.
        const float kGamepadRotationSpeed = 2.5f;   ///< Gamepad camera rotation speed.

        float2 convertCamPosRange(const float2 pos)
        {
            // Convert [0,1] range to [-1, 1], and inverse the Y (screen-space y==0 is top)
            const float2 scale(2, -2);
            const float2 offset(-1, 1);
            float2 res = (pos * scale) + offset;
            return res;
        }
    }

    float3 CameraController::getUpVector() const
    {
        uint32_t index = (uint32_t)mUpDirection;
        FALCOR_ASSERT(index < 6);
        float3 up{0.f};
        up[index / 2] = (index % 2 == 0) ? 1.f : -1.f;
        return up;
    }

    void OrbiterCameraController::setModelParams(const float3& center, float radius, float distanceInRadius)
    {
        mModelCenter = center;
        mModelRadius = radius;
        mCameraDistance = distanceInRadius;
        mRotation = rmcv::mat3(1.f);
        mbDirty = true;
    }

    bool OrbiterCameraController::onMouseEvent(const MouseEvent& mouseEvent)
    {
        bool handled = false;
        switch(mouseEvent.type)
        {
        case MouseEvent::Type::Wheel:
            mCameraDistance -= (mouseEvent.wheelDelta.y * 0.2f);
            mbDirty = true;
            handled = true;
            break;
        case MouseEvent::Type::ButtonDown:
            if (mouseEvent.button == Input::MouseButton::Left)
            {
                mLastVector = project2DCrdToUnitSphere(convertCamPosRange(mouseEvent.pos));
                mIsLeftButtonDown = true;
                handled = true;
            }
            break;
        case MouseEvent::Type::ButtonUp:
            if (mouseEvent.button == Input::MouseButton::Left)
            {
                handled = mIsLeftButtonDown;
                mIsLeftButtonDown = false;
            }
            break;
        case MouseEvent::Type::Move:
            if(mIsLeftButtonDown)
            {
                float3 curVec = project2DCrdToUnitSphere(convertCamPosRange(mouseEvent.pos));
                glm::quat q = createQuaternionFromVectors(mLastVector, curVec);
                rmcv::mat3 rot = rmcv::mat3_cast(q);
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

    bool OrbiterCameraController::update()
    {
        if(mpCamera && mbDirty)
        {
            mbDirty = false;
            mShouldRotate = false;
            mpCamera->setTarget(mModelCenter);

            float3 camPos = mModelCenter;
            // tdavidovic: Why do we multiply the rotation matrix from the left (i.e., as if we multiplied by a transpose?)
            camPos += (float3(0,0,1) * mRotation) * mModelRadius * mCameraDistance;
            mpCamera->setPosition(camPos);

            float3 up(0, 1, 0);
            up = up * mRotation;
            mpCamera->setUpVector(up);
            return true;
        }
        return false;
    }

    void OrbiterCameraController::resetInputState()
    {
        mIsLeftButtonDown   = false;
        mShouldRotate       = false;
        mbDirty             = false;
    }

    template<bool b6DoF>
    FirstPersonCameraControllerCommon<b6DoF>::FirstPersonCameraControllerCommon(const Camera::SharedPtr& pCamera) : CameraController(pCamera)
    {
        mTimer.update();
    }

    template<bool b6DoF>
    bool FirstPersonCameraControllerCommon<b6DoF>::onKeyEvent(const KeyboardEvent& event)
    {
        bool handled = false;

        if (event.type == KeyboardEvent::Type::KeyPressed || event.type == KeyboardEvent::Type::KeyReleased)
        {
            bool keyPressed = (event.type == KeyboardEvent::Type::KeyPressed);

            switch(event.key)
            {
            case Input::Key::W:
                mMovement[Direction::Forward] = keyPressed;
                handled = true;
                break;
            case Input::Key::S:
                mMovement[Direction::Backward] = keyPressed;
                handled = true;
                break;
            case Input::Key::A:
                mMovement[Direction::Right] = keyPressed;
                handled = true;
                break;
            case Input::Key::D:
                mMovement[Direction::Left] = keyPressed;
                handled = true;
                break;
            case Input::Key::Q:
                mMovement[Direction::Down] = keyPressed;
                handled = true;
                break;
            case Input::Key::E:
                mMovement[Direction::Up] = keyPressed;
                handled = true;
                break;
            default:
                break;
            }

            mSpeedModifier = 1.0f;
            if (event.hasModifier(Input::Modifier::Ctrl)) mSpeedModifier = 0.25f;
            else if (event.hasModifier(Input::Modifier::Shift)) mSpeedModifier = 10.0f;
        }

        return handled;
    }

    inline float applyDeadZone(const float v, const float deadZone)
    {
        return v * std::max(v - deadZone, 0.f) / (1.f - deadZone);
    }
    inline float2 applyDeadZone(const float2 v, const float deadZone)
    {
        return v * std::max(length(v) - deadZone, 0.f) / (1.f - deadZone);
    }
    inline float applyPowerCurve(const float v, const float power)
    {
        return std::pow(std::fabs(v), power) * (v < 0.f ? -1.f : 1.f);
    }
    inline float2 applyPowerCurve(const float2 v, const float power)
    {
        return float2(applyPowerCurve(v.x, power), applyPowerCurve(v.y, power));
    };

    template<bool b6DoF>
    bool FirstPersonCameraControllerCommon<b6DoF>::onGamepadState(const GamepadState& gamepadState)
    {
        mGamepadPresent = true;

        mGamepadLeftStick = float2(gamepadState.leftX, gamepadState.leftY);
        mGamepadRightStick = float2(gamepadState.rightX, gamepadState.rightY);
        mGamepadLeftTrigger = gamepadState.leftTrigger;
        mGamepadRightTrigger = gamepadState.rightTrigger;

        // Apply dead zone.
        mGamepadLeftStick = applyDeadZone(mGamepadLeftStick, kGamepadDeadZone);
        mGamepadRightStick = applyDeadZone(mGamepadRightStick, kGamepadDeadZone);
        mGamepadLeftTrigger = applyDeadZone(mGamepadLeftTrigger, kGamepadDeadZone);
        mGamepadRightTrigger = applyDeadZone(mGamepadRightTrigger, kGamepadDeadZone);

        // Apply power curve.
        mGamepadLeftStick = applyPowerCurve(mGamepadLeftStick, kGamepadPowerCurve);
        mGamepadRightStick = applyPowerCurve(mGamepadRightStick, kGamepadPowerCurve);
        mGamepadLeftTrigger = applyPowerCurve(mGamepadLeftTrigger, kGamepadPowerCurve);
        mGamepadRightTrigger = applyPowerCurve(mGamepadRightTrigger, kGamepadPowerCurve);

        return (length(mGamepadLeftStick) > 0.f || length(mGamepadRightStick) > 0.f || mGamepadLeftTrigger > 0.f || mGamepadRightTrigger > 0.f);
    }

    template<bool b6DoF>
    bool FirstPersonCameraControllerCommon<b6DoF>::update()
    {
        mTimer.update();

        // Clamp elapsed time to avoid huge jumps at long frame times (e.g. loading).
        float elapsedTime = std::min(0.1f, (float)mTimer.delta());

        bool dirty = false;
        if (mpCamera)
        {
            bool anyGamepadMovement = mGamepadPresent && (length(mGamepadLeftStick) > 0.f || mGamepadLeftTrigger > 0.f || mGamepadRightTrigger > 0.f);
            bool anyGamepadRotation = mGamepadPresent && (length(mGamepadRightStick) > 0.f);

            if (mShouldRotate || anyGamepadRotation)
            {
                float3 camPos = mpCamera->getPosition();
                float3 camTarget = mpCamera->getTarget();
                float3 camUp = b6DoF ? mpCamera->getUpVector() : getUpVector();

                float3 viewDir = glm::normalize(camTarget - camPos);

                if (mIsLeftButtonDown || anyGamepadRotation)
                {
                    float3 sideway = glm::cross(viewDir, normalize(camUp));

                    float2 mouseRotation = mIsLeftButtonDown ? mMouseDelta * mSpeedModifier : float2(0.f);
                    float2 gamepadRotation = anyGamepadRotation ? mGamepadRightStick * kGamepadRotationSpeed * elapsedTime : float2(0.f);
                    float2 rotation = mouseRotation + gamepadRotation;

                    // Rotate around x-axis
                    glm::quat qy = glm::angleAxis(rotation.y, sideway);
                    rmcv::mat3 rotY(rmcv::mat3_cast(qy));
                    viewDir = viewDir * rotY;
                    camUp = camUp * rotY;

                    // Rotate around y-axis
                    glm::quat qx = glm::angleAxis(rotation.x, camUp);
                    rmcv::mat3 rotX(rmcv::mat3_cast(qx));
                    viewDir = viewDir * rotX;

                    mpCamera->setTarget(camPos + viewDir);
                    mpCamera->setUpVector(camUp);
                    dirty = true;
                }

                if (b6DoF && mIsRightButtonDown)
                {
                    // Rotate around x-axis
                    glm::quat q = glm::angleAxis(mMouseDelta.x * mSpeedModifier, viewDir);
                    rmcv::mat3 rot(rmcv::mat3_cast(q));
                    camUp = camUp * rot;
                    mpCamera->setUpVector(camUp);
                    dirty = true;
                }

                mShouldRotate = false;
            }

            if (mMovement.any() || anyGamepadMovement)
            {
                float3 movement(0, 0, 0);
                movement.z += mMovement.test(Direction::Forward) ? 1 : 0;
                movement.z += mMovement.test(Direction::Backward) ? -1 : 0;
                movement.x += mMovement.test(Direction::Left) ? 1 : 0;
                movement.x += mMovement.test(Direction::Right) ? -1 : 0;
                movement.y += mMovement.test(Direction::Up) ? 1 : 0;
                movement.y += mMovement.test(Direction::Down) ? -1 : 0;

                if (anyGamepadMovement)
                {
                    movement.x += mGamepadLeftStick.x;
                    movement.z -= mGamepadLeftStick.y;
                    movement.y -= mGamepadLeftTrigger;
                    movement.y += mGamepadRightTrigger;
                }

                float3 camPos = mpCamera->getPosition();
                float3 camTarget = mpCamera->getTarget();
                float3 camUp = mpCamera->getUpVector();

                float3 viewDir = normalize(camTarget - camPos);
                float3 sideway = glm::cross(viewDir, normalize(camUp));

                float curMove = mSpeedModifier * mSpeed * elapsedTime;
                camPos += movement.z * curMove * viewDir;
                camPos += movement.x * curMove * sideway;
                camPos += movement.y * curMove * camUp;

                if (mBounds.valid())
                    camPos = clamp(camPos, mBounds.minPoint, mBounds.maxPoint);

                camTarget = camPos + viewDir;

                mpCamera->setPosition(camPos);
                mpCamera->setTarget(camTarget);
                dirty = true;
            }
        }

        // Will be set true in next call to onGamepadState().
        mGamepadPresent = false;

        return dirty;
    }

    template<bool b6DoF>
    bool FirstPersonCameraControllerCommon<b6DoF>::onMouseEvent(const MouseEvent& event)
    {
        bool handled = false;
        switch(event.type)
        {
        case MouseEvent::Type::ButtonDown:
            if (event.button == Input::MouseButton::Left)
            {
                mLastMousePos = event.pos;
                mIsLeftButtonDown = true;
                handled = true;
            }
            else if (event.button == Input::MouseButton::Right)
            {
                mLastMousePos = event.pos;
                mIsRightButtonDown = true;
                handled = true;
            }
            break;
        case MouseEvent::Type::ButtonUp:
            if (event.button == Input::MouseButton::Left)
            {
                handled = mIsLeftButtonDown;
                mIsLeftButtonDown = false;
            }
            else if (event.button == Input::MouseButton::Right)
            {
                handled = mIsRightButtonDown;
                mIsRightButtonDown = false;
            }
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

    template<bool b6DoF>
    void FirstPersonCameraControllerCommon<b6DoF>::resetInputState()
    {
        mIsLeftButtonDown   = false;
        mIsRightButtonDown  = false;
        mShouldRotate       = false;
        mMovement.reset();

        mGamepadLeftStick       = float2(0.f);
        mGamepadRightStick      = float2(0.f);
        mGamepadLeftTrigger     = 0.f;
        mGamepadRightTrigger    = 0.f;
    }

    template class FirstPersonCameraControllerCommon < true > ;
    template class FirstPersonCameraControllerCommon < false > ;
}
