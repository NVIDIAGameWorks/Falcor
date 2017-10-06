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

#include "Graphics/Paths/MovableObject.h"
#include "Utils/AABB.h"
#include "glm/mat4x4.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtx/euler_angles.hpp"
#include "Utils/Math/FalcorMath.h"

namespace Falcor
{
    class SceneRenderer;
    class Model;

    /** Handles transformations for Mesh and Model instances. Primary transform is stored in the "Base" transform. An additional "Movable"
        transform is applied after the Base transform can be set through the IMovableObject interface. This is currently used by paths.
    */
    template<typename ObjectType>
    class ObjectInstance : public IMovableObject, public inherit_shared_from_this<IMovableObject, ObjectInstance<ObjectType>>
    {
    public:
        using SharedPtr = std::shared_ptr<ObjectInstance<ObjectType>>;
        using SharedConstPtr = std::shared_ptr<const ObjectInstance<ObjectType>>;

        /** Constructs an object instance with a transform
            \param[in] pObject Object to create an instance of
            \param[in] baseTransform Base transform matrix of the instance
            \param[in] name Name of the instance
            \return A new instance of the object if pObject
        */
        static SharedPtr create(const typename ObjectType::SharedPtr& pObject, const glm::mat4& baseTransform, const std::string& name = "")
        {
            assert(pObject);
            return SharedPtr(new ObjectInstance<ObjectType>(pObject, baseTransform, name));
        }

        /** Constructs an object instance with a transform
            \param[in] pObject Object to create an instance of
            \param[in] translation Base translation of the instance
            \param[in] target Base look-at target of the instance
            \param[in] up Base up vector of the instance
            \param[in] scale Base scale of the instance
            \param[in] name Name of the instance
            \return A new instance of the object
        */
        static SharedPtr create(const typename ObjectType::SharedPtr& pObject, const glm::vec3& translation, const glm::vec3& target, const glm::vec3& up, const glm::vec3& scale, const std::string& name = "")
        {
             return SharedPtr(new ObjectInstance<ObjectType>(pObject, translation, target, up, scale, name));
        }

        /** Constructs an object instance with a transform
            \param[in] pObject Object to create an instance of
            \param[in] translation Base translation of the instance
            \param[in] yawPitchRoll Rotation of the instance in radians
            \param[in] scale Base scale of the instance
            \param[in] name Name of the instance
            \return A new instance of the object
        */
        static SharedPtr create(const typename ObjectType::SharedPtr& pObject, const glm::vec3& translation, const glm::vec3& yawPitchRoll, const glm::vec3& scale, const std::string& name = "")
        {
            return SharedPtr(new ObjectInstance<ObjectType>(pObject, translation, yawPitchRoll, scale, name));
        }

        /** Gets object for which this is an instance of
            \return Object for this instance
        */
        const typename ObjectType::SharedPtr& getObject() const { return mpObject; };

        /** Sets visibility of this instance
            \param[in] visible Visibility of this instance
        */
        void setVisible(bool visible) { mVisible = visible; };

        /** Gets whether this instance is visible
            \return Whether this instance is visible
        */
        bool isVisible() const { return mVisible; };

        /** Gets instance name
            \return Instance name
        */
        const std::string& getName() const { return mName; }

        /** Sets instance name
            \param[in] name Instance name
        */
        void setName(const std::string& name) { mName = name; }

        /** Sets position/translation of the instance
            \param[in] translation Instance translation
            \param[in] updateLookAt If true, translates the look-at target as well to maintain rotation
        */
        void setTranslation(const glm::vec3& translation, bool updateLookAt)
        {
            if (updateLookAt)
            {
                glm::vec3 toLookAt = mBase.target - mBase.translation;
                mBase.target = translation + toLookAt;
            }

            mBase.translation = translation;
            mBase.matrixDirty = true;
        };

        /** Gets the position/translation of the instance
            \return Translation of the instance
        */
        const glm::vec3& getTranslation() const { return mBase.translation; };

        /** Sets scale of the instance
            \param[in] scaling Instance scale
        */
        void setScaling(const glm::vec3& scaling) { mBase.scale = scaling; mBase.matrixDirty = true; }

        /** Gets scale of the instance
            \return Scale of the instance
        */
        const glm::vec3& getScaling() const { return mBase.scale; }

        /** Sets orientation of the instance
            \param[in] yawPitchRoll Yaw-Pitch-Roll rotation in radians
        */
        void setRotation(const glm::vec3& yawPitchRoll)
        {
            // Construct matrix from angles and take upper 3x3
            const glm::mat3 rotMtx(glm::yawPitchRoll(yawPitchRoll[0], yawPitchRoll[1], yawPitchRoll[2]));

            // Get look-at info
            mBase.up = rotMtx[1];
            mBase.target = mBase.translation + rotMtx[2]; // position + forward

            mBase.matrixDirty = true;
        }

        /** Gets rotation for the instance
            \return Yaw-Pitch-Roll rotations in radians
        */
        glm::vec3 getRotation() const 
        {
            glm::vec3 result;

            glm::mat4 rotationMtx = createMatrixFromLookAt(mBase.translation, mBase.target, mBase.up);
            glm::extractEulerAngleXYZ(rotationMtx, result[1], result[0], result[2]); // YawPitchRoll is YXZ

            return result;
        }

        /** Sets the up vector orientation
        */
        void setUpVector(const glm::vec3& up) { mBase.up = glm::normalize(up); mBase.matrixDirty = true; }

        /** Sets the look-at target
        */
        void setTarget(const glm::vec3& target) { mBase.target = target; mBase.matrixDirty = true; }

        /** Gets the up vector of the instance
            \return Up vector
        */
        const glm::vec3& getUpVector() const { return mBase.up; }

        /** Gets look-at target of the instance's orientation
            \return Look-at target position
        */
        const glm::vec3& getTarget() const { return mBase.target; }

        /** Gets the transform matrix
            \return Transform matrix
        */
        const glm::mat4& getTransformMatrix() const
        {
            updateInstanceProperties();
            return mFinalTransformMatrix;
        }

        /** Gets the bounding box
            \return Bounding box
        */
        const BoundingBox& getBoundingBox() const
        {
            updateInstanceProperties();
            return mBoundingBox;
        }

        /** IMovableObject interface
        */
        virtual void move(const glm::vec3& position, const glm::vec3& target, const glm::vec3& up) override
        {
            mMovable.translation = position;
            mMovable.target = target;
            mMovable.up = up;
            mMovable.scale = glm::vec3(1.0f);
            mMovable.matrixDirty = true;
        }

        SharedPtr shared_from_this()
        {
            return inherit_shared_from_this < IMovableObject, ObjectInstance>::shared_from_this();
        }

        SharedConstPtr shared_from_this() const
        {
            return inherit_shared_from_this < IMovableObject, ObjectInstance>::shared_from_this();
        }
    private:

        void updateInstanceProperties() const
        {
            if (mBase.matrixDirty || mMovable.matrixDirty)
            {
                if (mBase.matrixDirty)
                {
                    mBase.matrix = calculateTransformMatrix(mBase.translation, mBase.target, mBase.up, mBase.scale);
                    mBase.matrixDirty = false;
                }

                if (mMovable.matrixDirty)
                {
                    mMovable.matrix = calculateTransformMatrix(mMovable.translation, mMovable.target, mMovable.up, mMovable.scale);
                    mMovable.matrixDirty = false;
                }

                mFinalTransformMatrix = mMovable.matrix * mBase.matrix;
                mBoundingBox = mpObject->getBoundingBox().transform(mFinalTransformMatrix);
            }
        }

        static glm::mat4 calculateTransformMatrix(const glm::vec3& translation, const glm::vec3& target, const glm::vec3& up, const glm::vec3& scale)
        {
            glm::mat4 translationMtx = glm::translate(glm::mat4(), translation);
            glm::mat4 rotationMtx = createMatrixFromLookAt(translation, target, up);
            glm::mat4 scalingMtx = glm::scale(glm::mat4(), scale);

            return translationMtx * rotationMtx * scalingMtx;
        }

        static glm::mat4 calculateTransformMatrix(const glm::vec3& translation, const glm::vec3& yawPitchRoll, const glm::vec3& scale)
        {
            glm::mat4 translationMtx = glm::translate(glm::mat4(), translation);
            glm::mat4 rotationMtx = glm::yawPitchRoll(yawPitchRoll[0], yawPitchRoll[1], yawPitchRoll[2]);
            glm::mat4 scalingMtx = glm::scale(glm::mat4(), scale);

            return translationMtx * rotationMtx * scalingMtx;
        }

        ObjectInstance(const typename ObjectType::SharedPtr& pObject, const std::string& name)
            : mpObject(pObject), mName(name) { }

        ObjectInstance(const typename ObjectType::SharedPtr& pObject, const glm::mat4& baseTransform, const std::string& name)
            : ObjectInstance(pObject, name)
        {
            // #TODO Decompose matrix

            mBase.matrix = baseTransform;
            mBase.matrixDirty = false;
        }

        ObjectInstance(const typename ObjectType::SharedPtr& pObject, const glm::vec3& translation, const glm::vec3& target, const glm::vec3& up, const glm::vec3& scale, const std::string& name = "")
            : ObjectInstance(pObject, name)
        {
            mBase.translation = translation;
            mBase.target = target;
            mBase.up = up;
            mBase.scale = scale;
        }

        ObjectInstance(const typename ObjectType::SharedPtr& pObject, const glm::vec3& translation, const glm::vec3& yawPitchRoll, const glm::vec3& scale, const std::string& name = "")
            : ObjectInstance(pObject, name)
        {
            mBase.translation = translation;
            setRotation(yawPitchRoll);
            mBase.scale = scale;
        }

        friend class Model;

        std::string mName;
        bool mVisible = true;

        typename ObjectType::SharedPtr mpObject;

        struct Transform
        {
            glm::vec3 translation;
            glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
            glm::vec3 target = glm::vec3(0.0f, 0.0f, 1.0f);
            glm::vec3 scale = glm::vec3(1.0f);

            // Matrix containing the above transforms
            glm::mat4 matrix;
            bool matrixDirty = true;
        };

        mutable Transform mBase;
        mutable Transform mMovable;

        mutable glm::mat4 mFinalTransformMatrix;
        mutable BoundingBox mBoundingBox;
    };
}
