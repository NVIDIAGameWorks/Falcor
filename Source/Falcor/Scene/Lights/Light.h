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
#pragma once
#include "LightData.slang"
#include "Core/Macros.h"
#include "Utils/Math/Vector.h"
#include "Utils/Math/Matrix.h"
#include "Utils/UI/Gui.h"
#include "Scene/Animation/Animatable.h"
#include <memory>
#include <string>

namespace Falcor
{
    class Scene;
    struct ShaderVar;

    /** Base class for light sources. All light sources should inherit from this.
    */
    class FALCOR_API Light : public Animatable
    {
    public:
        using SharedPtr = std::shared_ptr<Light>;
        using SharedConstPtr = std::shared_ptr<const Light>;

        virtual ~Light() = default;

        /** Set the light parameters into a shader variable. To use this you need to include/import 'ShaderCommon' inside your shader.
        */
        virtual void setShaderData(const ShaderVar& var);

        /** Render UI elements for this light.
        */
        virtual void renderUI(Gui::Widgets& widget);

        /** Get total light power
        */
        virtual float getPower() const = 0;

        /** Get the light type
        */
        LightType getType() const { return (LightType)mData.type; }

        /** Get the light data
        */
        inline const LightData& getData() const { return mData; }

        /** Name the light
        */
        void setName(const std::string& Name) { mName = Name; }

        /** Get the light's name
        */
        const std::string& getName() const { return mName; }

        /** Activate/deactivate the light
        */
        void setActive(bool active);

        /** Check if light is active
        */
        bool isActive() const { return mActive; }

        /** Gets the size of a single light data struct in bytes
        */
        static uint32_t getShaderStructSize() { return kDataSize; }

        /** Set the light intensity.
        */
        virtual void setIntensity(const float3& intensity);

        /** Get the light intensity.
        */
        const float3& getIntensity() const { return mData.intensity; }

        enum class Changes
        {
            None = 0x0,
            Active = 0x1,
            Position = 0x2,
            Direction = 0x4,
            Intensity = 0x8,
            SurfaceArea = 0x10,
        };

        /** Begin a new frame. Returns the changes from the previous frame
        */
        Changes beginFrame();

        /** Returns the changes from the previous frame
        */
        Changes getChanges() const { return mChanges; }

        void updateFromAnimation(const rmcv::mat4& transform) override {}

    protected:
        Light(const std::string& name, LightType type);

        static const size_t kDataSize = sizeof(LightData);

        // UI callbacks for keeping the intensity in-sync.
        float3 getColorForUI();
        void setColorFromUI(const float3& uiColor);
        float getIntensityForUI();
        void setIntensityFromUI(float intensity);

        std::string mName;
        bool mActive = true;
        bool mActiveChanged = false;

        // These two variables track mData values for consistent UI operation.
        float3 mUiLightIntensityColor = float3(0.5f, 0.5f, 0.5f);
        float mUiLightIntensityScale = 1.0f;
        LightData mData;
        LightData mPrevData;
        Changes mChanges = Changes::None;

        friend class SceneCache;
    };

    /** Point light source.
        Simple infinitely-small point light with quadratic attenuation.
    */
    class FALCOR_API PointLight : public Light
    {
    public:
        using SharedPtr = std::shared_ptr<PointLight>;
        using SharedConstPtr = std::shared_ptr<const PointLight>;

        static SharedPtr create(const std::string& name = "");
        ~PointLight() = default;

        /** Render UI elements for this light.
        */
        void renderUI(Gui::Widgets& widget) override;

        /** Get total light power (needed for light picking)
        */
        float getPower() const override;

        /** Set the light's world-space position
        */
        void setWorldPosition(const float3& pos);

        /** Set the light's world-space direction.
            \param[in] dir Light direction. Does not have to be normalized.
        */
        void setWorldDirection(const float3& dir);

        /** Set the cone opening half-angle for use as a spot light
            \param[in] openingAngle Angle in radians.
        */
        void setOpeningAngle(float openingAngle);

        /** Get the light's world-space position
        */
        const float3& getWorldPosition() const { return mData.posW; }

        /** Get the light's world-space direction
        */
        const float3& getWorldDirection() const { return mData.dirW; }

        /** Get the penumbra half-angle
        */
        float getPenumbraAngle() const { return mData.penumbraAngle; }

        /** Set the penumbra half-angle
            \param[in] angle Angle in radians
        */
        void setPenumbraAngle(float angle);

        /** Get the cone opening half-angle
        */
        float getOpeningAngle() const { return mData.openingAngle; }

        void updateFromAnimation(const rmcv::mat4& transform) override;

    private:
        PointLight(const std::string& name);
    };


    /** Directional light source.
    */
    class FALCOR_API DirectionalLight : public Light
    {
    public:
        using SharedPtr = std::shared_ptr<DirectionalLight>;
        using SharedConstPtr = std::shared_ptr<const DirectionalLight>;

        static SharedPtr create(const std::string& name = "");
        ~DirectionalLight() = default;

        /** Render UI elements for this light.
        */
        void renderUI(Gui::Widgets& widget) override;

        /** Set the light's world-space direction.
            \param[in] dir Light direction. Does not have to be normalized.
        */
        void setWorldDirection(const float3& dir);

        /** Set the scene parameters
        */
        void setWorldParams(const float3& center, float radius);

        /** Get the light's world-space direction.
        */
        const float3& getWorldDirection() const { return mData.dirW; }

        /** Get total light power (needed for light picking)
        */
        float getPower() const override { return 0.f; }

        void updateFromAnimation(const rmcv::mat4& transform) override;

    private:
        DirectionalLight(const std::string& name);
    };

    /** Distant light source.
        Same as directional light source but subtending a non-zero solid angle.
    */
    class FALCOR_API DistantLight : public Light
    {
    public:
        using SharedPtr = std::shared_ptr<DistantLight>;
        using SharedConstPtr = std::shared_ptr<const DistantLight>;

        static SharedPtr create(const std::string& name = "");
        ~DistantLight() = default;

        /** Render UI elements for this light.
        */
        void renderUI(Gui::Widgets& widget) override;

        /** Set the half-angle subtended by the light
            \param[in] theta Light angle
        */
        void setAngle(float theta);

        /** Get the half-angle subtended by the light
        */
        float getAngle() const { return mAngle; }

        /** Set the light's world-space direction.
            \param[in] dir Light direction. Does not have to be normalized.
        */
        void setWorldDirection(const float3& dir);

        /** Get the light's world-space direction.
        */
        const float3& getWorldDirection() const { return mData.dirW; }

        /** Get total light power
        */
        float getPower() const override { return 0.f; }

        void updateFromAnimation(const rmcv::mat4& transform) override;

    private:
        DistantLight(const std::string& name);
        void update();
        float mAngle;       ///<< Half-angle subtended by the source.

        friend class SceneCache;
    };

    /** Analytic area light source.
    */
    class FALCOR_API AnalyticAreaLight : public Light
    {
    public:
        using SharedPtr = std::shared_ptr<AnalyticAreaLight>;
        using SharedConstPtr = std::shared_ptr<const AnalyticAreaLight>;

        ~AnalyticAreaLight() = default;

        /** Set light source scaling
            \param[in] scale x,y,z scaling factors
        */
        void setScaling(float3 scale) { mScaling = scale; update(); }

        /** Set light source scale
          */
        float3 getScaling() const { return mScaling; }

        /** Get total light power (needed for light picking)
        */
        float getPower() const override;

        /** Set transform matrix
            \param[in] mtx object to world space transform matrix
        */
        void setTransformMatrix(const rmcv::mat4& mtx) { mTransformMatrix = mtx; update();  }

        /** Get transform matrix
        */
        rmcv::mat4 getTransformMatrix() const { return mTransformMatrix; }

        void updateFromAnimation(const rmcv::mat4& transform) override { setTransformMatrix(transform); }

    protected:
        AnalyticAreaLight(const std::string& name, LightType type);

        virtual void update();

        float3 mScaling;                ///< Scaling, controls the size of the light
        rmcv::mat4 mTransformMatrix;     ///< Transform matrix minus scaling component

        friend class SceneCache;
    };

    /** Rectangular area light source.
    */
    class FALCOR_API RectLight : public AnalyticAreaLight
    {
    public:
        using SharedPtr = std::shared_ptr<RectLight>;
        using SharedConstPtr = std::shared_ptr<const RectLight>;

        static SharedPtr create(const std::string& name = "");
        ~RectLight() = default;

    private:
        RectLight(const std::string& name) : AnalyticAreaLight(name, LightType::Rect) {}

        virtual void update() override;
    };

    /** Disc area light source.
    */
    class FALCOR_API DiscLight : public AnalyticAreaLight
    {
    public:
        using SharedPtr = std::shared_ptr<DiscLight>;
        using SharedConstPtr = std::shared_ptr<const DiscLight>;

        static SharedPtr create(const std::string& name = "");
        ~DiscLight() = default;

    private:
        DiscLight(const std::string& name) : AnalyticAreaLight(name, LightType::Disc) {}

        virtual void update() override;
    };

    /** Sphere area light source.
    */
    class FALCOR_API SphereLight : public AnalyticAreaLight
    {
    public:
        using SharedPtr = std::shared_ptr<SphereLight>;
        using SharedConstPtr = std::shared_ptr<const SphereLight>;

        static SharedPtr create(const std::string& name = "");
        ~SphereLight() = default;

    private:
        SphereLight(const std::string& name) : AnalyticAreaLight(name, LightType::Sphere) {}

        virtual void update() override;
    };

    FALCOR_ENUM_CLASS_OPERATORS(Light::Changes);
}
