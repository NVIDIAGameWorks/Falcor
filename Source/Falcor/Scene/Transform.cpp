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
#include "Transform.h"
#include "Utils/Scripting/ScriptBindings.h"
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/transform.hpp>

namespace Falcor
{
    Transform::Transform() {}

    void Transform::setTranslation(const float3& translation)
    {
        mTranslation = translation;
        mDirty = true;
    }

    void Transform::setScaling(const float3& scaling)
    {
        mScaling = scaling;
        mDirty = true;
    }

    void Transform::setRotation(const glm::quat& rotation)
    {
        mRotation = rotation;
        mDirty = true;
    }

    float3 Transform::getRotationEuler() const
    {
        return glm::eulerAngles(mRotation);
    }

    void Transform::setRotationEuler(const float3& angles)
    {
        setRotation(glm::quat(angles));
    }

    float3 Transform::getRotationEulerDeg() const
    {
        return glm::degrees(getRotationEuler());
    }

    void Transform::setRotationEulerDeg(const float3& angles)
    {
        setRotationEuler(glm::radians(angles));
    }

    void Transform::lookAt(const float3& position, const float3& target, const float3& up)
    {
        mTranslation = position;
        float3 dir = normalize(target - position);
        mRotation = glm::quatLookAt(dir, up);
    }

    const rmcv::mat4& Transform::getMatrix() const
    {
        if (mDirty)
        {
            rmcv::mat4 T = rmcv::translate(mTranslation);
            rmcv::mat4 R = rmcv::mat4_cast(mRotation);
            rmcv::mat4 S = rmcv::scale(mScaling);
            mMatrix = T * R * S;
            mDirty = false;
        }

        return mMatrix;
    }

    bool Transform::operator==(const Transform& other) const
    {
        if (mTranslation != other.mTranslation) return false;
        if (mScaling != other.mScaling) return false;
        if (mRotation != other.mRotation) return false;
        return true;
    }

    FALCOR_SCRIPT_BINDING(Transform)
    {
        using namespace pybind11::literals;

        auto init = [](const pybind11::kwargs& args)
        {
            Transform transform;

            std::optional<float3> position;
            std::optional<float3> target;
            std::optional<float3> up;

            for (auto a : args)
            {
                auto key = a.first.cast<std::string>();
                const auto& value = a.second;

                float3 float3Value;
                float floatValue;

                bool isFloat3 = pybind11::isinstance<float3>(value);
                bool isNumber = pybind11::isinstance<pybind11::int_>(value) || pybind11::isinstance<pybind11::float_>(value);

                if (isFloat3) float3Value = pybind11::cast<float3>(value);
                if (isNumber) floatValue = pybind11::cast<float>(value);

                if (key == "translation")
                {
                    if (isFloat3) transform.setTranslation(float3Value);
                }
                else if (key == "scaling")
                {
                    if (isFloat3) transform.setScaling(float3Value);
                    if (isNumber) transform.setScaling(float3(floatValue));
                }
                else if (key == "rotationEuler")
                {
                    if (isFloat3) transform.setRotationEuler(float3Value);
                }
                else if (key == "rotationEulerDeg")
                {
                    if (isFloat3) transform.setRotationEulerDeg(float3Value);
                }
                else if (key == "position")
                {
                    if (isFloat3) position = float3Value;
                }
                else if (key == "target")
                {
                    if (isFloat3) target = float3Value;
                }
                else if (key == "up")
                {
                    if (isFloat3) up = float3Value;
                }
            }

            if (position && target && up)
            {
                transform.lookAt(*position, *target, *up);
            }

            return transform;
        };

        pybind11::class_<Transform> transform(m, "Transform");
        transform.def(pybind11::init(init));
        transform.def_property("translation", &Transform::getTranslation, &Transform::setTranslation);
        transform.def_property("rotationEuler", &Transform::getRotationEuler, &Transform::setRotationEuler);
        transform.def_property("rotationEulerDeg", &Transform::getRotationEulerDeg, &Transform::setRotationEulerDeg);
        transform.def_property("scaling", &Transform::getScaling, &Transform::setScaling);
        transform.def_property_readonly("matrix", &Transform::getMatrix);
        transform.def("lookAt", &Transform::lookAt, "position"_a, "target"_a, "up"_a);
    }
}
