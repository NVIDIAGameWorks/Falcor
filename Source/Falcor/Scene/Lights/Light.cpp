/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
#include "Light.h"
#include "Core/Program/ShaderVar.h"
#include "Utils/Logger.h"
#include "Utils/UI/Gui.h"
#include "Utils/Color/ColorHelpers.slang"
#include "Utils/Scripting/ScriptBindings.h"

namespace Falcor
{
    static_assert(sizeof(LightData) % 16 == 0, "LightData size should be a multiple of 16B");

    // Light

    void Light::setActive(bool active)
    {
        if (active != mActive)
        {
            mActive = active;
            mActiveChanged = true;
        }
    }

    void Light::setIntensity(const float3& intensity)
    {
        mData.intensity = intensity;
    }

    Light::Changes Light::beginFrame()
    {
        mChanges = Changes::None;
        if (mActiveChanged) mChanges |= Changes::Active;
        if (any(mPrevData.posW != mData.posW)) mChanges |= Changes::Position;
        if (any(mPrevData.dirW != mData.dirW)) mChanges |= Changes::Direction;
        if (any(mPrevData.intensity != mData.intensity)) mChanges |= Changes::Intensity;
        if (mPrevData.openingAngle != mData.openingAngle) mChanges |= Changes::SurfaceArea;
        if (mPrevData.penumbraAngle != mData.penumbraAngle) mChanges |= Changes::SurfaceArea;
        if (mPrevData.cosSubtendedAngle != mData.cosSubtendedAngle) mChanges |= Changes::SurfaceArea;
        if (mPrevData.surfaceArea != mData.surfaceArea) mChanges |= Changes::SurfaceArea;
        if (mPrevData.transMat != mData.transMat) mChanges |= (Changes::Position | Changes::Direction);

        FALCOR_ASSERT(all(mPrevData.tangent == mData.tangent));
        FALCOR_ASSERT(all(mPrevData.bitangent == mData.bitangent));

        mPrevData = mData;
        mActiveChanged = false;

        return getChanges();
    }

    void Light::bindShaderData(const ShaderVar& var)
    {
#define check_offset(_a) FALCOR_ASSERT(var.getType()->getMemberOffset(#_a).getByteOffset() == offsetof(LightData, _a))
        check_offset(dirW);
        check_offset(intensity);
        check_offset(penumbraAngle);
#undef check_offset

        var.setBlob(mData);
    }

    float3 Light::getColorForUI()
    {
        if (any((mUiLightIntensityColor * mUiLightIntensityScale) != mData.intensity))
        {
            float mag = std::max(mData.intensity.x, std::max(mData.intensity.y, mData.intensity.z));
            if (mag <= 1.f)
            {
                mUiLightIntensityColor = mData.intensity;
                mUiLightIntensityScale = 1.0f;
            }
            else
            {
                mUiLightIntensityColor = mData.intensity / mag;
                mUiLightIntensityScale = mag;
            }
        }

        return mUiLightIntensityColor;
    }

    void Light::setColorFromUI(const float3& uiColor)
    {
        mUiLightIntensityColor = uiColor;
        setIntensity(mUiLightIntensityColor * mUiLightIntensityScale);
    }

    float Light::getIntensityForUI()
    {
        if (any((mUiLightIntensityColor * mUiLightIntensityScale) != mData.intensity))
        {
            float mag = std::max(mData.intensity.x, std::max(mData.intensity.y, mData.intensity.z));
            if (mag <= 1.f)
            {
                mUiLightIntensityColor = mData.intensity;
                mUiLightIntensityScale = 1.0f;
            }
            else
            {
                mUiLightIntensityColor = mData.intensity / mag;
                mUiLightIntensityScale = mag;
            }
        }

        return mUiLightIntensityScale;
    }

    void Light::setIntensityFromUI(float intensity)
    {
        mUiLightIntensityScale = intensity;
        setIntensity(mUiLightIntensityColor * mUiLightIntensityScale);
    }

    void Light::renderUI(Gui::Widgets& widget)
    {
        bool active = isActive();
        if (widget.checkbox("Active", active)) setActive(active);

        if (mHasAnimation) widget.checkbox("Animated", mIsAnimated);

        float3 color = getColorForUI();
        if (widget.rgbColor("Color", color))
        {
            setColorFromUI(color);
        }

        float intensity = getIntensityForUI();
        if (widget.var("Intensity", intensity))
        {
            setIntensityFromUI(intensity);
        }
    }

    Light::Light(const std::string& name, LightType type)
        : mName(name)
    {
        mData.type = (uint32_t)type;
    }

    // PointLight

    PointLight::PointLight(const std::string& name)
        : Light(name, LightType::Point)
    {
        mPrevData = mData;
    }

    void PointLight::setWorldDirection(const float3& dir)
    {
        if (!(length(dir) > 0.f)) // NaNs propagate
        {
            logWarning("Can't set light direction to zero length vector. Ignoring call.");
            return;
        }
        mData.dirW = normalize(dir);
    }

    void PointLight::setWorldPosition(const float3& pos)
    {
        mData.posW = pos;
    }

    float PointLight::getPower() const
    {
        return luminance(mData.intensity) * 4.f * (float)M_PI;
    }

    void PointLight::renderUI(Gui::Widgets& widget)
    {
        Light::renderUI(widget);

        widget.var("World Position", mData.posW, -FLT_MAX, FLT_MAX);
        widget.direction("Direction", mData.dirW);

        float openingAngle = getOpeningAngle();
        if (widget.var("Opening Angle", openingAngle, 0.f, (float)M_PI)) setOpeningAngle(openingAngle);
        float penumbraAngle = getPenumbraAngle();
        if (widget.var("Penumbra Width", penumbraAngle, 0.f, (float)M_PI)) setPenumbraAngle(penumbraAngle);
    }

    void PointLight::setOpeningAngle(float openingAngle)
    {
        openingAngle = std::clamp(openingAngle, 0.f, (float)M_PI);
        if (openingAngle == mData.openingAngle) return;

        mData.openingAngle = openingAngle;
        mData.penumbraAngle = std::min(mData.penumbraAngle, openingAngle);

        // Prepare an auxiliary cosine of the opening angle to quickly check whether we're within the cone of a spot light.
        mData.cosOpeningAngle = std::cos(openingAngle);
    }

    void PointLight::setPenumbraAngle(float angle)
    {
        angle = std::clamp(angle, 0.0f, mData.openingAngle);
        if (mData.penumbraAngle == angle) return;
        mData.penumbraAngle = angle;
    }

    void PointLight::updateFromAnimation(const float4x4& transform)
    {
        float3 fwd = -transform.getCol(2).xyz();
        float3 pos = transform.getCol(3).xyz();
        setWorldPosition(pos);
        setWorldDirection(fwd);
    }

    // DirectionalLight

    DirectionalLight::DirectionalLight(const std::string& name)
        : Light(name, LightType::Directional)
    {
        mPrevData = mData;
    }

    void DirectionalLight::renderUI(Gui::Widgets& widget)
    {
        Light::renderUI(widget);

        if (widget.direction("Direction", mData.dirW))
        {
            setWorldDirection(mData.dirW);
        }
    }

    void DirectionalLight::setWorldDirection(const float3& dir)
    {
        if (!(length(dir) > 0.f)) // NaNs propagate
        {
            logWarning("Can't set light direction to zero length vector. Ignoring call.");
            return;
        }
        mData.dirW = normalize(dir);
    }

    void DirectionalLight::updateFromAnimation(const float4x4& transform)
    {
        float3 fwd = -transform.getCol(2).xyz();
        setWorldDirection(fwd);
    }

    // DistantLight

    DistantLight::DistantLight(const std::string& name)
        : Light(name, LightType::Distant)
    {
        mData.dirW = float3(0.f, -1.f, 0.f);
        setAngle(0.5f * 0.53f * (float)M_PI / 180.f);   // Approximate sun half-angle
        update();
        mPrevData = mData;
    }

    void DistantLight::renderUI(Gui::Widgets& widget)
    {
        Light::renderUI(widget);

        if (widget.direction("Direction", mData.dirW))
        {
            setWorldDirection(mData.dirW);
        }

        if (widget.var("Half-angle", mAngle, 0.f, (float)M_PI_2))
        {
            setAngle(mAngle);
        }
        widget.tooltip("Half-angle subtended by the light, in radians.");
    }

    void DistantLight::setAngle(float angle)
    {
        mAngle = std::clamp(angle, 0.f, (float)M_PI_2);

        mData.cosSubtendedAngle = std::cos(mAngle);
    }

    void DistantLight::setWorldDirection(const float3& dir)
    {
        if (!(length(dir) > 0.f)) // NaNs propagate
        {
            logWarning("Can't set light direction to zero length vector. Ignoring call.");
            return;
        }
        mData.dirW = normalize(dir);
        update();
    }

    void DistantLight::update()
    {
        // Update transformation matrices
        // Assumes that mData.dirW is normalized
        const float3 up(0.f, 0.f, 1.f);
        float3 vec = cross(up, -mData.dirW);
        float sinTheta = length(vec);
        if (sinTheta > 0.f)
        {
            float cosTheta = dot(up, -mData.dirW);
            mData.transMat = math::matrixFromRotation(std::acos(cosTheta), vec);
        }
        else
        {
            mData.transMat = float4x4::identity();
        }
        mData.transMatIT = inverse(transpose(mData.transMat));
    }

    void DistantLight::updateFromAnimation(const float4x4& transform)
    {
        float3 fwd = -transform.getCol(2).xyz();
        setWorldDirection(fwd);
    }

    // AnalyticAreaLight

    AnalyticAreaLight::AnalyticAreaLight(const std::string& name, LightType type)
        : Light(name, type)
    {
        mData.tangent = float3(1, 0, 0);
        mData.bitangent = float3(0, 1, 0);
        mData.surfaceArea = 4.0f;

        mScaling = float3(1, 1, 1);
        update();
        mPrevData = mData;
    }

    float AnalyticAreaLight::getPower() const
    {
        return luminance(mData.intensity) * (float)M_PI * mData.surfaceArea;
    }

    void AnalyticAreaLight::update()
    {
        // Update matrix
        mData.transMat = mul(mTransformMatrix, math::matrixFromScaling(mScaling));
        mData.transMatIT = inverse(transpose(mData.transMat));
    }

    // RectLight

    void RectLight::update()
    {
        AnalyticAreaLight::update();

        float rx = length(transformVector(mData.transMat, float3(1.0f, 0.0f, 0.0f)));
        float ry = length(transformVector(mData.transMat, float3(0.0f, 1.0f, 0.0f)));
        mData.surfaceArea = 4.0f * rx * ry;
    }

    // DiscLight

    void DiscLight::update()
    {
        AnalyticAreaLight::update();

        float rx = length(transformVector(mData.transMat, float3(1.0f, 0.0f, 0.0f)));
        float ry = length(transformVector(mData.transMat, float3(0.0f, 1.0f, 0.0f)));

        mData.surfaceArea = (float)M_PI * rx * ry;
    }

    // SphereLight

    void SphereLight::update()
    {
        AnalyticAreaLight::update();

        float rx = length(transformVector(mData.transMat, float3(1.0f, 0.0f, 0.0f)));
        float ry = length(transformVector(mData.transMat, float3(0.0f, 1.0f, 0.0f)));
        float rz = length(transformVector(mData.transMat, float3(0.0f, 0.0f, 1.0f)));

        mData.surfaceArea = 4.0f * (float)M_PI * std::pow(std::pow(rx * ry, 1.6f) + std::pow(ry * rz, 1.6f) + std::pow(rx * rz, 1.6f) / 3.0f, 1.0f / 1.6f);
    }


    FALCOR_SCRIPT_BINDING(Light)
    {
        using namespace pybind11::literals;

        FALCOR_SCRIPT_BINDING_DEPENDENCY(Animatable)

        pybind11::class_<Light, Animatable, ref<Light>> light(m, "Light");
        light.def_property("name", &Light::getName, &Light::setName);
        light.def_property("active", &Light::isActive, &Light::setActive);
        light.def_property("animated", &Light::isAnimated, &Light::setIsAnimated);
        light.def_property("intensity", &Light::getIntensity, &Light::setIntensity);

        pybind11::class_<PointLight, Light, ref<PointLight>> pointLight(m, "PointLight");
        pointLight.def(pybind11::init(&PointLight::create), "name"_a = "");
        pointLight.def_property("position", &PointLight::getWorldPosition, &PointLight::setWorldPosition);
        pointLight.def_property("direction", &PointLight::getWorldDirection, &PointLight::setWorldDirection);
        pointLight.def_property("openingAngle", &PointLight::getOpeningAngle, &PointLight::setOpeningAngle);
        pointLight.def_property("penumbraAngle", &PointLight::getPenumbraAngle, &PointLight::setPenumbraAngle);

        pybind11::class_<DirectionalLight, Light, ref<DirectionalLight>> directionalLight(m, "DirectionalLight");
        directionalLight.def(pybind11::init(&DirectionalLight::create), "name"_a = "");
        directionalLight.def_property("direction", &DirectionalLight::getWorldDirection, &DirectionalLight::setWorldDirection);

        pybind11::class_<DistantLight, Light, ref<DistantLight>> distantLight(m, "DistantLight");
        distantLight.def(pybind11::init(&DistantLight::create), "name"_a = "");
        distantLight.def_property("direction", &DistantLight::getWorldDirection, &DistantLight::setWorldDirection);
        distantLight.def_property("angle", &DistantLight::getAngle, &DistantLight::setAngle);

        pybind11::class_<AnalyticAreaLight, Light, ref<AnalyticAreaLight>> analyticLight(m, "AnalyticAreaLight");

        pybind11::class_<RectLight, AnalyticAreaLight, ref<RectLight>> rectLight(m, "RectLight");
        rectLight.def(pybind11::init(&RectLight::create), "name"_a = "");

        pybind11::class_<DiscLight, AnalyticAreaLight, ref<DiscLight>> discLight(m, "DiscLight");
        discLight.def(pybind11::init(&DiscLight::create), "name"_a = "");

        pybind11::class_<SphereLight, AnalyticAreaLight, ref<SphereLight>> sphereLight(m, "SphereLight");
        sphereLight.def(pybind11::init(&SphereLight::create), "name"_a = "");
   }
}
