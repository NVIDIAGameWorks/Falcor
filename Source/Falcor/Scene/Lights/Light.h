/***************************************************************************
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
#include "Data/HostDeviceData.h"

namespace Falcor
{
    class Scene;

    /** Base class for light sources. All light sources should inherit from this.
    */
    class dlldecl Light
    {
    public:
        using SharedPtr = std::shared_ptr<Light>;
        using SharedConstPtr = std::shared_ptr<const Light>;
        using ConstSharedPtrRef = const SharedPtr&;

        virtual ~Light() = default;

        /** Set the light parameters into a program. To use this you need to include/import 'ShaderCommon' inside your shader.
            \param[in] pBlock The parameter block to set the parameters into.
            \param[in] varName The name of the LightData variable in the parameter block.
        */
        virtual void setIntoParameterBlock(const ParameterBlock::SharedPtr& pBlock, const std::string& varName);

        /** Set the light parameters into a program. To use this you need to include/import 'ShaderCommon' inside your shader.
            \param[in] pCb The constant buffer to set the parameters into.
            \param[in] offset Byte offset into the constant buffer to set data to.
        */
        virtual void setIntoVariableBuffer(VariablesBuffer* pBuffer, size_t offset);

        /** Render UI elements for this light.
            \param[in] pGui The GUI to create the elements with
            \param[in] group Optional. If specified, creates a UI group to display elements within
        */
        virtual void renderUI(Gui* pGui, const char* group = nullptr);

        /** Get total light power
        */
        virtual float getPower() const = 0;

        /** Get the light type
        */
        uint32_t getType() const { return mData.type; }

        /** Get the light Type
        */
        inline const LightData& getData() const { return mData; }

        /** Name the light
        */
        const void setName(const std::string& Name) { mName = Name; }

        /** Get the light's name
        */
        const std::string& getName() const { return mName; }

        /** Gets the size of a single light data struct in bytes
        */
        static uint32_t getShaderStructSize() { return kDataSize; }

        /** Set the light intensity.
        */
        virtual void setIntensity(const glm::vec3& intensity);

        enum class Changes
        {
            None = 0x0,
            Position = 0x1,
            Direction = 0x2,
            Intensity = 0x4,
            SurfaceArea = 0x8,
        };

        /** Begin a new frame. Returns the changes from the previous frame
        */
        Changes beginFrame();

        /** Returns the changes from the previous frame
        */
        Changes getChanges() const { return mChanges; }

    protected:
        Light() = default;

        static const size_t kDataSize = sizeof(LightData);

        /* UI callbacks for keeping the intensity in-sync */
        glm::vec3 getColorForUI();
        void setColorFromUI(const glm::vec3& uiColor);
        float getIntensityForUI();
        void setIntensityFromUI(float intensity);

        std::string mName;

        /* These two variables track mData values for consistent UI operation.*/
        glm::vec3 mUiLightIntensityColor = glm::vec3(0.5f, 0.5f, 0.5f);
        float     mUiLightIntensityScale = 1.0f;
        LightData mData, mPrevData;
        Changes mChanges = Changes::None;
    };

    /** Directional light source.
    */
    class dlldecl DirectionalLight : public Light
    {
    public:
        using SharedPtr = std::shared_ptr<DirectionalLight>;
        using SharedConstPtr = std::shared_ptr<const DirectionalLight>;

        static SharedPtr create();
        ~DirectionalLight();

        /** Render UI elements for this light.
        \param[in] pGui The GUI to create the elements with
        \param[in] group Optional. If specified, creates a UI group to display elements within
        */
        void renderUI(Gui* pGui, const char* group = nullptr) override;

        /** Set the light's world-space direction.
        */
        void setWorldDirection(const glm::vec3& dir);

        /** Set the scene parameters
        */
        void setWorldParams(const glm::vec3& center, float radius);

        /** Get the light's world-space direction.
        */
        const glm::vec3& getWorldDirection() const { return mData.dirW; }

        /** Get total light power (needed for light picking)
        */
        float getPower() const override;

    private:
        DirectionalLight();
        float mDistance = 1e3f; ///< Scene bounding radius is required to move the light position sufficiently far away
        vec3 mCenter;
    };

    /** Simple infinitely-small point light with quadratic attenuation
    */
    class dlldecl PointLight : public Light
    {
    public:
        using SharedPtr = std::shared_ptr<PointLight>;
        using SharedConstPtr = std::shared_ptr<const PointLight>;

        static SharedPtr create();
        ~PointLight();

        /** Render UI elements for this light.
            \param[in] pGui The GUI to create the elements with
            \param[in] group Optional. If specified, creates a UI group to display elements within
        */
        void renderUI(Gui* pGui, const char* group = nullptr) override;

        /** Get total light power (needed for light picking)
        */
        float getPower() const override;

        /** Set the light's world-space position
        */
        void setWorldPosition(const glm::vec3& pos);

        /** Set the light's world-space direction.
        */
        void setWorldDirection(const glm::vec3& dir);

        /** Set the cone opening angle for use as a spot light
            \param[in] openingAngle Angle in radians.
        */
        void setOpeningAngle(float openingAngle);

        /** Get the light's world-space position
        */
        const glm::vec3& getWorldPosition() const { return mData.posW; }

        /** Get the light's world-space direction
        */
        const glm::vec3& getWorldDirection() const { return mData.dirW; }

        /** Get the light intensity.
        */
        const glm::vec3& getIntensity() const { return mData.intensity; }

        /** Get the penumbra angle
        */
        float getPenumbraAngle() const { return mData.penumbraAngle; }

        /** Set the penumbra angle
            \param[in] angle Angle in radians
        */
        void setPenumbraAngle(float angle);

        /** Get the opening angle
        */
        float getOpeningAngle() const { return mData.openingAngle; }

    private:
        PointLight();
    };

    /**
        Area light source

        This class is used to simulate area light sources. All emissive
        materials are treated as area light sources.
    */
    class dlldecl AreaLight : public Light
    {
    public:
        using SharedPtr = std::shared_ptr<AreaLight>;
        using SharedConstPtr = std::shared_ptr<const AreaLight>;

        static SharedPtr create();

        ~AreaLight();

        /** Get total light power (needed for light picking)
        */
        float getPower() const override;

        /** Set the light parameters into a program. To use this you need to include/import 'ShaderCommon' inside your shader
            and declare a constant buffer to bind the values to using the AREA_LIGHTS() macro defined in HostDeviceSharedMacros.h
            \param[in] pBlock The parameter block to set the parameters into.
            \param[in] varName The name of the declared variable in the parameter block.
        */
        virtual void setIntoParameterBlock(const ParameterBlock::SharedPtr& pBlock, const std::string& varName) override;

        /** Do not use this overload for area lights. Area light data contains resources that cannot be bound using an offset when
            there is an array of light data. Calling this function will do nothing except log a warning.
        */
        virtual void setIntoVariableBuffer(VariablesBuffer* pBuffer, size_t offset) override;

        /** Render UI elements for this light.
            \param[in] pGui The GUI to create the elements with
            \param[in] group Optional. If specified, creates a UI group to display elements within
        */
        void renderUI(Gui* pGui, const char* group = nullptr) override;

        /** Set the geometry mesh for this light
            \param[in] pScene The scene the light is in
            \param[in] instanceId Mesh instance id
        */
        void setMeshData(const std::shared_ptr<Scene>& pScene, uint32_t instanceId);

        /** Get surface area of the mesh
        */
        float getSurfaceArea() const { return mAreaLightData.surfaceArea; }

        /** Get the probability distribution of the mesh
        */
        const std::vector<float>& getMeshCDF() const { return mMeshCDF; }
        /** Set the mesh CDF buffer.
            \param[in] meshCDF Buffer containing mesh CDF data
        */
        void setMeshCDFBuffer(const Buffer::SharedPtr& meshCDFBuf) { mpMeshCDFBuffer = meshCDFBuf; }

        /** Get the mesh CDF buffer
        */
        const Buffer::SharedPtr& getMeshCDFBuffer() const { return mpMeshCDFBuffer; }

        /** Gets the size of a single light data struct in bytes
        */
        static uint32_t getShaderStructSize() { return kAreaLightDataSize; }

    private:
        AreaLight();
        void computeSurfaceArea(const std::shared_ptr<Scene>& pScene);

        static const size_t kAreaLightDataSize = sizeof(AreaLightData) - sizeof(AreaLightResources);
        AreaLightData mAreaLightData;

        std::weak_ptr<Scene> mpScene;       ///< Scene this area light is a part of
        uint32_t mInstanceId;               ///< Mesh Instance Id from the scene this light was created from
        MeshDesc mMeshDesc;                 ///< Where the actual mesh is inside the buffers
        Buffer::SharedPtr mpIndexBuffer;    ///< Buffer for indices
        Buffer::SharedPtr mpVertexBuffer;   ///< Buffer for vertices
        Buffer::SharedPtr mpMeshCDFBuffer;  ///< Buffer for mesh Cumulative distribution function (CDF)

        std::vector<float> mMeshCDF; ///< CDF function for importance sampling a triangle mesh
    };

    dlldecl AreaLight::SharedPtr createAreaLight(const std::shared_ptr<Scene>& pScene, uint32_t instanceId);
    dlldecl std::vector<AreaLight::SharedPtr> createAreaLightsForScene(const std::shared_ptr<Scene>& pScene);

    /**
        Analytic area light source.
    */
    class dlldecl AnalyticAreaLight : public Light
    {
    public:
        using SharedPtr = std::shared_ptr<AnalyticAreaLight>;
        using SharedConstPtr = std::shared_ptr<const AnalyticAreaLight>;

        /** Creates an analytic area light.
            \param[in] type The type of analytic area light (rectangular, sphere, disc etc). See HostDeviceSharedMacros.h
        */
        static SharedPtr create(uint32_t type);

        ~AnalyticAreaLight();

        /** Set light source scaling
            \param[in] scale x,y,z scaling factors
        */
        void setScaling(vec3 scale) { mScaling = scale; }

        /** Set light source scale
          */
        vec3 getScaling() const { return mScaling; }

        /** Get total light power (needed for light picking)
        */
        float getPower() const override;

        /** Set transform matrix
            \param[in] mtx object to world space transform matrix
        */
        void setTransformMatrix(const glm::mat4 &mtx) { mTransformMatrix = mtx; }

        /** Get transform matrix
        */
        glm::mat4 getTransformMatrix() const { return mTransformMatrix; }

        /** Render UI elements for this light.
            \param[in] pGui The GUI to create the elements with
            \param[in] group Optional. If specified, creates a UI group to display elements within
        */
        void renderUI(Gui* pGui, const char* group = nullptr) override;

    private:
        AnalyticAreaLight(uint32_t type);
        void update();

        bool mDirty = true;
        glm::vec3 mScaling;              ///< Scaling, controls the size of the light
        glm::mat4 mTransformMatrix;      ///< Transform matrix minus scaling component
    };

    inline std::string light_type_string(uint32_t type)
    {
        switch (type)
        {
        case LightPoint: return "Point Light";
        case LightDirectional: return "Directional Light";
        case LightArea: return "Area Light";
        case LightAreaRect: return "Rectangular Light";
        case LightAreaSphere: return "Spherical Light";
        case LightAreaDisc: return "Disc Light";
        default:
            should_not_get_here();
            return "";
        }
    }

    enum_class_operators(Light::Changes);
}
