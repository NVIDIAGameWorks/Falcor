/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
 **************************************************************************/
#include "stdafx.h"
#include "Light.h"
#include "Utils/UI/Gui.h"
#include "Utils/Color/ColorHelpers.slang"

namespace Falcor
{
    static bool checkOffset(const std::string& structName, UniformShaderVarOffset cbOffset, size_t cppOffset, const char* field)
    {
        if (cbOffset.getByteOffset() != cppOffset)
        {
            logError("Light::" + std::string(structName) + ":: " + std::string(field) + " CB offset mismatch. CB offset is " + std::to_string(cbOffset.getByteOffset()) + ", C++ data offset is " + std::to_string(cppOffset));
            return false;
        }
        return true;
    }

    void Light::setIntensity(const float3& intensity)
    {
        mData.intensity = intensity;
    }

    Light::Changes Light::beginFrame()
    {
        mChanges = Changes::None;
        if (mPrevData.posW != mData.posW) mChanges |= Changes::Position;
        if (mPrevData.dirW != mData.dirW) mChanges |= Changes::Direction;
        if (mPrevData.intensity != mData.intensity) mChanges |= Changes::Intensity;
        if (mPrevData.openingAngle != mData.openingAngle) mChanges |= Changes::SurfaceArea;
        if (mPrevData.penumbraAngle != mData.penumbraAngle) mChanges |= Changes::SurfaceArea;
        if (mPrevData.surfaceArea != mData.surfaceArea) mChanges |= Changes::SurfaceArea;

        assert(mPrevData.tangent == mData.tangent);
        assert(mPrevData.bitangent == mData.bitangent);
        assert(mPrevData.transMat == mData.transMat);

        mPrevData = mData;

        return getChanges();
    }

    void Light::setShaderData(const ShaderVar& var)
    {
#if _LOG_ENABLED
#define check_offset(_a) {static bool b = true; if(b) {assert(checkOffset("LightData", var.getType()->getMemberOffset(#_a), offsetof(LightData, _a), #_a));} b = false;}
        check_offset(dirW);
        check_offset(intensity);
        check_offset(penumbraAngle);
#undef check_offset
#endif

        var.setBlob(mData);
    }

    float3 Light::getColorForUI()
    {
        if ((mUiLightIntensityColor * mUiLightIntensityScale) != mData.intensity)
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
        if ((mUiLightIntensityColor * mUiLightIntensityScale) != mData.intensity)
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

    void Light::renderUI(Gui* pGui, const char* group)
    {
        if (!group) group = "General Light Settings";
        Gui::Group g(pGui, group);
        if (g.open())
        {
            float3 color = getColorForUI();
            if (g.rgbColor("Color", color))
            {
                setColorFromUI(color);
            }
            float intensity = getIntensityForUI();
            if (g.var("Intensity", intensity))
            {
                setIntensityFromUI(intensity);
            }

            g.release();
        }
    }

    DirectionalLight::DirectionalLight() : mDistance(-1.0f)
    {
        mData.type = LightType::Directional;
    }

    DirectionalLight::SharedPtr DirectionalLight::create()
    {
        DirectionalLight* pLight = new DirectionalLight;
        return SharedPtr(pLight);
    }

    DirectionalLight::~DirectionalLight() = default;

    void DirectionalLight::renderUI(Gui* pGui, const char* group)
    {
        if (!group) group = "Directional Light Settings";
        Gui::Group g(pGui, group);
        if (g.open())
        {
            if (g.direction("Direction", mData.dirW))
            {
                setWorldDirection(mData.dirW);
            }
            Light::renderUI(pGui);
            g.release();
        }
    }

    void DirectionalLight::setWorldDirection(const float3& dir)
    {
        if (!(glm::length(dir) > 0.f)) // NaNs propagate
        {
            logWarning("Can't set light direction to zero length vector. Ignoring call.");
            return;
        }
        mData.dirW = normalize(dir);
        mData.posW = mCenter - mData.dirW * mDistance; // Move light's position sufficiently far away
    }

    void DirectionalLight::setWorldParams(const float3& center, float radius)
    {
        mDistance = radius;
        mCenter = center;
        mData.posW = mCenter - mData.dirW * mDistance; // Move light's position sufficiently far away
    }

    float DirectionalLight::getPower() const
    {
        const float surfaceArea = (float)M_PI * mDistance * mDistance;
        return luminance(mData.intensity) * surfaceArea;
    }

    PointLight::SharedPtr PointLight::create()
    {
        PointLight* pLight = new PointLight;
        return SharedPtr(pLight);
    }

    PointLight::PointLight()
    {
        mData.type = LightType::Point;
    }

    PointLight::~PointLight() = default;

    void PointLight::setWorldDirection(const float3& dir)
    {
        if (!(glm::length(dir) > 0.f)) // NaNs propagate
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

    void PointLight::renderUI(Gui* pGui, const char* group)
    {
        if (!group) group = "Point Light Settings";
        Gui::Group g(pGui, group);
        if (g.open())
        {
            g.var("World Position", mData.posW, -FLT_MAX, FLT_MAX);
            g.direction("Direction", mData.dirW);

            if (g.var("Opening Angle", mData.openingAngle, 0.f, (float)M_PI))
            {
                setOpeningAngle(mData.openingAngle);
            }
            if (g.var("Penumbra Width", mData.penumbraAngle, 0.f, (float)M_PI))
            {
                setPenumbraAngle(mData.penumbraAngle);
            }
            Light::renderUI(pGui);

            g.release();
        }
    }

    void PointLight::setOpeningAngle(float openingAngle)
    {
        openingAngle = glm::clamp(openingAngle, 0.f, (float)M_PI);
        if (openingAngle == mData.openingAngle) return;

        mData.openingAngle = openingAngle;
        /* Prepare an auxiliary cosine of the opening angle to quickly check whether we're within the cone of a spot light */
        mData.cosOpeningAngle = cos(openingAngle);
    }

    void PointLight::setPenumbraAngle(float angle)
    {
        angle = glm::clamp(angle, 0.0f, mData.openingAngle);
        if (mData.penumbraAngle == angle) return;
        mData.penumbraAngle = angle;
    }

    // Code for analytic area lights.
    AnalyticAreaLight::SharedPtr AnalyticAreaLight::create(LightType type)
    {
        AnalyticAreaLight* pLight = new AnalyticAreaLight(type);
        return SharedPtr(pLight);
    }

    AnalyticAreaLight::AnalyticAreaLight(LightType type)
    {
        mData.type = type;
        mData.tangent = float3(1, 0, 0);
        mData.bitangent = float3(0, 1, 0);
        mData.surfaceArea = 4.0f;

        mScaling = float3(1, 1, 1);
        update();
    }

    AnalyticAreaLight::~AnalyticAreaLight() = default;

    float AnalyticAreaLight::getPower() const
    {
        return luminance(mData.intensity) * (float)M_PI * mData.surfaceArea;
    }

    void AnalyticAreaLight::renderUI(Gui* pGui, const char* group)
    {
        if (!group) group = "Analytic Area Light Settings";
        Gui::Group g(pGui, group);
        if (g.open())
        {
            Light::renderUI(pGui);

            g.release();
        }
    }

    void AnalyticAreaLight::update()
    {
        // Update matrix
        mData.transMat = mTransformMatrix * glm::scale(glm::mat4(), mScaling);
        mData.transMatIT = glm::inverse(glm::transpose(mData.transMat));

        switch (mData.type)
        {

        case LightType::Rect:
        {
            float rx = glm::length(mData.transMat * float4(1.0f, 0.0f, 0.0f, 0.0f));
            float ry = glm::length(mData.transMat * float4(0.0f, 1.0f, 0.0f, 0.0f));
            mData.surfaceArea = 4.0f * rx * ry;
        }
        break;

        case LightType::Sphere:
        {
            float rx = glm::length(mData.transMat * float4(1.0f, 0.0f, 0.0f, 0.0f));
            float ry = glm::length(mData.transMat * float4(0.0f, 1.0f, 0.0f, 0.0f));
            float rz = glm::length(mData.transMat * float4(0.0f, 0.0f, 1.0f, 0.0f));

            mData.surfaceArea = 4.0f * (float)M_PI * pow(pow(rx * ry, 1.6f) + pow(ry * rz, 1.6f) + pow(rx * rz, 1.6f) / 3.0f, 1.0f / 1.6f);
        }
        break;

        case LightType::Disc:
        {
            float rx = glm::length(mData.transMat * float4(1.0f, 0.0f, 0.0f, 0.0f));
            float ry = glm::length(mData.transMat * float4(0.0f, 1.0f, 0.0f, 0.0f));

            mData.surfaceArea = (float)M_PI * rx * ry;
        }
        break;

        default:
            break;
        }
    }

    SCRIPT_BINDING(Light)
    {
        auto light = m.regClass(Light);
        light.roProperty("name", &Light::getName);
        light.property("intensity", &Light::getIntensityForScript, &Light::setIntensityFromScript);
        light.property("color", &Light::getColorForScript, &Light::setColorFromScript);

        auto directionalLight = m.class_<DirectionalLight, Light>("DirectionalLight");
        directionalLight.property("direction", &DirectionalLight::getWorldDirection, &DirectionalLight::setWorldDirection);

        auto pointLight = m.class_<PointLight, Light>("PointLight");
        pointLight.property("position", &PointLight::getWorldPosition, &PointLight::setWorldPosition);
        pointLight.property("direction", &PointLight::getWorldDirection, &PointLight::setWorldDirection);
        pointLight.property("openingAngle", &PointLight::getOpeningAngle, &PointLight::setOpeningAngle);
        pointLight.property("penumbraAngle", &PointLight::getPenumbraAngle, &PointLight::setPenumbraAngle);

        m.class_<AnalyticAreaLight, Light>("AnalyticAreaLight");
    }
}
