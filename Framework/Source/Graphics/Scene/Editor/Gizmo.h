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

#pragma once

#include "Graphics/Scene/Scene.h"
#include "Graphics/Camera/Camera.h"
#include <array>

namespace Falcor
{
    /** Used by the Scene Editor to allow users to modify object transforms through mouse interactions.
    */
    class Gizmo
    {
    public:
        using SharedPtr = std::shared_ptr<Gizmo>;
        using SharedConstPtr = std::shared_ptr<const Gizmo>;

        using Gizmos = std::array<Gizmo::SharedPtr, 3>;

        /** Size multiplier for gizmos. At 1 unit distance, size equals kGizmoSizeScale.
        */
        static const float kGizmoSizeScale;

        enum class Type
        {
            Translate,
            Rotate,
            Scale,
            Invalid
        };

        enum class Axis
        {
            XAxis,
            YAxis,
            ZAxis
        };

        /** Set gizmo transform to attach to another object, sets scale based on distance from camera.
            \param[in] pCamera Camera in the scene this gizmo is being viewed from.
            \param[in] pInstance Model instance to attach gizmo to.
        */
        virtual void setTransform(const Camera::SharedPtr& pCamera, const Scene::ModelInstance::SharedPtr& pInstance);

        /** Check if pInstance is part of the gizmo. If true initializes gizmo for a click-and-drag action.
            \param[in] pCamera Camera in the scene this gizmo is being viewed from.
            \param[in] pAxisModelInstance Model instance to compare with gizmo.
            \return Whether pAxisModelInstance is part of the gizmo
        */
        virtual bool beginAction(const Camera::SharedPtr& pCamera, const Scene::ModelInstance::SharedPtr& pAxisModelInstance);

        /** Updates current mouse state during a click-and-drag.
            \param[in] pCamera Camera in the scene this gizmo is being viewed from.
            \param[in] mouseEvent Mouse event.
        */
        void update(const Camera::SharedPtr& pCamera, const MouseEvent& mouseEvent);

        /** Applies the delta transform between the last two update calls to a model instance.
            \param[in] pInstance Model instance to update transform of.
        */
        virtual void applyDelta(const Scene::ModelInstance::SharedPtr& pInstance) const = 0;

        /** Applies the delta transform between the last two update calls to a camera.
            \param[in] pCamera Camera to update transform of.
        */
        virtual void applyDelta(const Camera::SharedPtr& pCamera) const = 0;

        /** Applies the delta transform between the last two update calls to a point light.
            \param[in] pLight Point light to update transform of.
        */
        virtual void applyDelta(const PointLight::SharedPtr& pLight) const = 0;

        /** Sets visibility of gizmo
            \param[in] visible Visibility.
        */
        void setVisible(bool visible);

        /** Gets the model used by the gizmo.
            \return Model used by the gizmo.
        */
        const Model::SharedPtr& getModel() const { return mpAxesInstances[0]->getObject(); }

        /** Gets the gizmo type.
            \return Type of the gizmo.
        */
        Type getType() const { return mGizmoType; };

        /** Check if model instance is part of the gizmo.
            \param[in] pInstance Model instance to check.
            \return Whether pInstance is part of the gizmo.
        */
        bool isPartOfGizmo(const Scene::ModelInstance* pInstance) const;

        /** Check if model is used in a set of gizmos
            \param[in] gizmos Array of three gizmos.
            \param[in] pInstance Model to check.
            \return If pModel is used by a gizmo, returns the gizmo type. Otherwise returns Type::Invalid.
        */
        static Type getGizmoType(const Gizmos& gizmos, const Model* pModel);

    protected:

        Gizmo(const Scene::SharedPtr& pScene, const char* modelFilename);

        // Finds the best plane to track mouse movements on
        virtual void findBestPlane(const Camera::SharedPtr& pCamera);

        // Default XYZ basis
        static const std::array<glm::vec3, 3> kAxes;

        Type mGizmoType;
        Scene::ModelInstance::SharedPtr mpAxesInstances[3];

        Axis mTransformAxis = Axis::XAxis;
        glm::vec3 mLastMousePos;
        glm::vec3 mCurrMousePos;

        Axis mBestPlaneAxis = Axis::XAxis;
        std::array<glm::vec3, 3> mGizmoAxes = kAxes;
    };


    class TranslateGizmo : public Gizmo, public inherit_shared_from_this<Gizmo, TranslateGizmo>
    {
    public:
        using SharedPtr = std::shared_ptr<TranslateGizmo>;
        using SharedConstPtr = std::shared_ptr<const TranslateGizmo>;

        static SharedPtr create(const Scene::SharedPtr& pScene, const char* modelFilename);

        virtual void applyDelta(const Scene::ModelInstance::SharedPtr& pInstance) const override;
        virtual void applyDelta(const Camera::SharedPtr& pCamera) const override;
        virtual void applyDelta(const PointLight::SharedPtr& pLight) const override;

    private:
        TranslateGizmo(const Scene::SharedPtr& pScene, const char* modelFilename)
            : Gizmo(pScene, modelFilename) { mGizmoType = Gizmo::Type::Translate; }

        glm::vec3 calculateMovementDelta() const;
    };


    class RotateGizmo : public Gizmo, public inherit_shared_from_this<Gizmo, RotateGizmo>
    {
    public:
        using SharedPtr = std::shared_ptr<RotateGizmo>;
        using SharedConstPtr = std::shared_ptr<const RotateGizmo>;

        static SharedPtr create(const Scene::SharedPtr& pScene, const char* modelFilename);

        virtual void applyDelta(const Scene::ModelInstance::SharedPtr& pInstance) const override;
        virtual void applyDelta(const Camera::SharedPtr& pCamera) const override;
        virtual void applyDelta(const PointLight::SharedPtr& pLight) const override;

    private:
        virtual void findBestPlane(const Camera::SharedPtr& pCamera) override;

        glm::mat3 calculateDeltaRotation() const;

        RotateGizmo(const Scene::SharedPtr& pScene, const char* modelFilename)
            : Gizmo(pScene, modelFilename) { mGizmoType = Gizmo::Type::Rotate; }
    };


    class ScaleGizmo : public Gizmo, public inherit_shared_from_this<Gizmo, ScaleGizmo>
    {
    public:
        using SharedPtr = std::shared_ptr<ScaleGizmo>;
        using SharedConstPtr = std::shared_ptr<const ScaleGizmo>;

        static SharedPtr create(const Scene::SharedPtr& pScene, const char* modelFilename);

        // Rotates axes to align with model
        virtual void setTransform(const Camera::SharedPtr& pCamera, const Scene::ModelInstance::SharedPtr& pInstance) override;

        virtual void applyDelta(const Scene::ModelInstance::SharedPtr& pInstance) const override;
        virtual void applyDelta(const Camera::SharedPtr& pCamera) const override {};
        virtual void applyDelta(const PointLight::SharedPtr& pLight) const override;

    private:
        ScaleGizmo(const Scene::SharedPtr& pScene, const char* modelFilename);

        std::array<glm::mat3, 3> mDefaultGizmoRotation;
    };


}
