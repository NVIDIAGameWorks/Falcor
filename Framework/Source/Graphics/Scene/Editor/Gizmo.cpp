/***************************************************************************
# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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
#include "Graphics/Scene/Editor/Gizmo.h"
#include "glm/gtx/intersect.hpp"
#include "glm/gtc/epsilon.hpp"
#include "glm/gtx/matrix_interpolation.hpp"
#include "Utils/Math/FalcorMath.h"
#include <cmath>
#include "glm/gtx/projection.hpp"

namespace Falcor
{
    const std::array<glm::vec3, 3> Gizmo::kAxes = { glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f) };

    const float Gizmo::kGizmoSizeScale = 0.06f;

    void Gizmo::setTransform(const Camera::SharedPtr& pCamera, const Scene::ModelInstance::SharedPtr& pInstance)
    {
        const float distToCamera = glm::length(pInstance->getTranslation() - pCamera->getPosition());

        for (auto& axis : mpAxesInstances)
        {
            axis->setTranslation(pInstance->getTranslation(), true);
            axis->setScaling(glm::vec3(distToCamera * kGizmoSizeScale));
        }
    }

    bool Gizmo::beginAction(const Camera::SharedPtr& pCamera, const Scene::ModelInstance::SharedPtr& pAxisModelInstance)
    {
        for (uint32_t i = 0; i < 3; i++)
        {
            if (mpAxesInstances[i] == pAxisModelInstance)
            {
                mTransformAxis = (Axis)i;
                findBestPlane(pCamera);
                return true;
            }
        }

        return false;
    }

    void Gizmo::update(const Camera::SharedPtr& pCamera, const MouseEvent& mouseEvent)
    {
        glm::vec3 gizmoPos = mpAxesInstances[0]->getTranslation();
        glm::vec3 rayDir = mousePosToWorldRay(mouseEvent.pos, pCamera->getViewMatrix(), pCamera->getProjMatrix());

        float intersectDist = 0.0f;
        bool succeeded = glm::intersectRayPlane(pCamera->getPosition(), rayDir, gizmoPos, mGizmoAxes[(uint32_t)mBestPlaneAxis], intersectDist);

        // If failed, try again with reversed normal
        if (intersectDist < 0 || succeeded == false)
        {
            succeeded = glm::intersectRayPlane(pCamera->getPosition(), rayDir, gizmoPos, -mGizmoAxes[(uint32_t)mBestPlaneAxis], intersectDist);
        }

        if (succeeded)
        {
            mLastMousePos = mCurrMousePos;
            mCurrMousePos = pCamera->getPosition() + rayDir * intersectDist;
        }
    }

    void Gizmo::setVisible(bool visible)
    {
        for (auto& axis : mpAxesInstances)
        {
            axis->setVisible(visible);
        }
    }

    bool Gizmo::isPartOfGizmo(const Scene::ModelInstance* pInstance) const
    {
        for (auto& axis : mpAxesInstances)
        {
            if (axis.get() == pInstance)
            {
                return true;
            }
        }

        return false;
    }

    // static
    Gizmo::Type Gizmo::getGizmoType(const Gizmos& gizmos, const Model* pModel)
    {
        for (uint32_t i = 0; i < 3; i++)
        {
            if (gizmos[i] != nullptr)
            {
                if(gizmos[i]->getModel().get() == pModel)
                {
                    return (Gizmo::Type)i;
                }
            }
        }

        return Type::Invalid;
    }

    Gizmo::Gizmo(const Scene::SharedPtr& pScene, const char* modelFilename)
        : mGizmoType(Type::Invalid)
    {
        // Add model instances to scene
        Model::SharedPtr pModel = Model::createFromFile(modelFilename);
        pScene->addModelInstance(pModel, "X", glm::vec3(), glm::radians(glm::vec3(0.0f, 0.0f, -90.0f)));
        pScene->addModelInstance(pModel, "Y");
        pScene->addModelInstance(pModel, "Z", glm::vec3(), glm::radians(glm::vec3(0.0f, 90.0f, 0.0f)));

        const uint32_t sceneModelID = pScene->getModelCount() - 1;

        // Save shared copy of model instances locally
        for (uint32_t i = 0; i < 3; i++)
        {
            mpAxesInstances[i] = pScene->getModelInstance(sceneModelID, i);
            mpAxesInstances[i]->setVisible(false);
        }
    }

    void Gizmo::findBestPlane(const Camera::SharedPtr& pCamera)
    {
        const glm::vec3 camForward = glm::normalize(pCamera->getTarget() - pCamera->getPosition());

        float maxDot = -1.0f;

        for (uint32_t i = 0; i < 3; i++)
        {
            float result = std::fabs(glm::dot(mGizmoAxes[(uint32_t)i], camForward));

            // Reject plane with same normal as the gizmo transformation (for translation and scaling)
            // You wouldn't be able to detect movement in that direction if you intersect with a perpendicular plane
            if (result > maxDot && i != (uint32_t)mTransformAxis)
            {
                maxDot = result;
                mBestPlaneAxis = (Axis)i;
            }
        }
    }

    //***************************************************************************
    // Translation Gizmo
    //***************************************************************************

    TranslateGizmo::SharedPtr TranslateGizmo::create(const Scene::SharedPtr& pScene, const char* modelFilename)
    {
        return SharedPtr(new TranslateGizmo(pScene, modelFilename));
    }

    void TranslateGizmo::applyDelta(const Scene::ModelInstance::SharedPtr& pInstance) const
    {
        pInstance->setTranslation(pInstance->getTranslation() + calculateMovementDelta(), true);
    }

    void TranslateGizmo::applyDelta(const Camera::SharedPtr& pCamera) const
    {
        auto delta = calculateMovementDelta();
        pCamera->setPosition(pCamera->getPosition() + delta);
        pCamera->setTarget(pCamera->getTarget() + delta);
    }

    void TranslateGizmo::applyDelta(const PointLight::SharedPtr& pLight) const
    {
        pLight->setWorldPosition(pLight->getWorldPosition() + calculateMovementDelta());
    }

    glm::vec3 TranslateGizmo::calculateMovementDelta() const
    {
        glm::vec3 worldDelta = mCurrMousePos - mLastMousePos;
        return worldDelta * kAxes[(uint32_t)mTransformAxis];
    }

    //***************************************************************************
    // Rotation Gizmo
    //***************************************************************************

    RotateGizmo::SharedPtr RotateGizmo::create(const Scene::SharedPtr& pScene, const char* modelFilename)
    {
        return SharedPtr(new RotateGizmo(pScene, modelFilename));
    }

    void RotateGizmo::applyDelta(const Scene::ModelInstance::SharedPtr& pInstance) const
    {
        glm::mat3 rotMtx = calculateDeltaRotation() * createMatrixFromLookAt(pInstance->getTranslation(), pInstance->getTarget(), pInstance->getUpVector());

        pInstance->setTarget(pInstance->getTranslation() + rotMtx[2]);
        pInstance->setUpVector(rotMtx[1]);
    }

    void RotateGizmo::applyDelta(const Camera::SharedPtr& pCamera) const
    {
        // #TODO do we want to maintain target distance? do we want to do this on models too?
        float targetDist = glm::length(pCamera->getTarget() - pCamera->getPosition());
        glm::mat3 rotMtx = calculateDeltaRotation() * createMatrixFromLookAt(pCamera->getPosition(), pCamera->getTarget(), pCamera->getUpVector());

        pCamera->setTarget(pCamera->getPosition() + rotMtx[2] * targetDist);
        pCamera->setUpVector(rotMtx[1]);
    }

    void RotateGizmo::applyDelta(const PointLight::SharedPtr& pLight) const
    {
        glm::vec3 newDir = calculateDeltaRotation() * pLight->getWorldDirection();
        pLight->setWorldDirection(newDir);
    }

    void RotateGizmo::findBestPlane(const Camera::SharedPtr& pCamera)
    {
        mBestPlaneAxis = mTransformAxis;
    }

    glm::mat3 RotateGizmo::calculateDeltaRotation() const
    {
        glm::vec3 toMousePrev = glm::normalize(mLastMousePos - mpAxesInstances[0]->getTranslation());
        glm::vec3 toMouseCurr = glm::normalize(mCurrMousePos - mpAxesInstances[0]->getTranslation());

        // Calculate angle between mouse movement
        float angle = glm::acos(glm::clamp(glm::dot(toMousePrev, toMouseCurr), -1.0f, 1.0f));
        angle *= glm::sign(glm::dot(mGizmoAxes[(uint32_t)mTransformAxis], glm::cross(toMousePrev, toMouseCurr)));

        // Build basis from instance transform
        return glm::axisAngleMatrix(mGizmoAxes[(uint32_t)mTransformAxis], angle);
    }

    //***************************************************************************
    // Scaling Gizmo
    //***************************************************************************

    ScaleGizmo::ScaleGizmo(const Scene::SharedPtr& pScene, const char* modelFilename)
        : Gizmo(pScene, modelFilename)
    {
        mGizmoType = Gizmo::Type::Scale;

        for (uint32_t i = 0; i < 3; i++)
        {
            const auto& axis = mpAxesInstances[i];
            mDefaultGizmoRotation[i] = createMatrixFromLookAt(axis->getTranslation(), axis->getTarget(), axis->getUpVector());
        }
    }

    ScaleGizmo::SharedPtr ScaleGizmo::create(const Scene::SharedPtr& pScene, const char* modelFilename)
    {
        return SharedPtr(new ScaleGizmo(pScene, modelFilename));
    }

    void ScaleGizmo::setTransform(const Camera::SharedPtr& pCamera, const Scene::ModelInstance::SharedPtr& pInstance)
    {
        Gizmo::setTransform(pCamera, pInstance);

        glm::mat3 instanceRotMtx = createMatrixFromLookAt(pInstance->getTranslation(), pInstance->getTarget(), pInstance->getUpVector());

        for (uint32_t i = 0; i < 3; i++)
        {
            // Update rotation of model instance representing axis
            glm::mat3 finalRotMtx = instanceRotMtx * mDefaultGizmoRotation[i];
            mpAxesInstances[i]->setTarget(mpAxesInstances[i]->getTranslation() + finalRotMtx[2]);
            mpAxesInstances[i]->setUpVector(finalRotMtx[1]);

            // Update gizmo "basis"
            mGizmoAxes[i] = instanceRotMtx * kAxes[i];
        }
    }

    void ScaleGizmo::applyDelta(const PointLight::SharedPtr& pLight) const
    {
        glm::vec3 worldDelta = mCurrMousePos - mLastMousePos;
        float intensityDelta = glm::length(worldDelta) * glm::sign(glm::dot(worldDelta, mGizmoAxes[(uint32_t)mTransformAxis]));

        pLight->setIntensity(pLight->getIntensity() + glm::vec3(intensityDelta));
    }

    void ScaleGizmo::applyDelta(const Scene::ModelInstance::SharedPtr& pInstance) const
    {
        glm::vec3 worldDelta = mCurrMousePos - mLastMousePos;

        float distance = glm::length(glm::proj(worldDelta, mGizmoAxes[(uint32_t)mTransformAxis]));
        distance *= glm::sign(glm::dot(worldDelta, mGizmoAxes[(uint32_t)mTransformAxis]));

        // Use kAxes to apply scaling in model space
        glm::vec3 modelScaleDelta = kAxes[(uint32_t)mTransformAxis] * distance;
        pInstance->setScaling(pInstance->getScaling() + modelScaleDelta);
    }

}
