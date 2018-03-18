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
#pragma once
#include <string>
#include <glm/common.hpp>
#include "glm/geometric.hpp"
#include "API/Texture.h"
#include "glm/mat4x4.hpp"
#include "Data/HostDeviceData.h"
#include "Utils/Gui.h"
#include "Graphics/Paths/MovableObject.h"
#include "Graphics/Model/Model.h"

namespace Falcor
{
    class ConstantBuffer;
    class Gui;
    class Scene;
    class CascadedShadowMaps;

    // Sequential ID used to note that something has been changed
    typedef int64_t VersionID;

    /** Base class for light sources. All light sources should inherit from this.
    */
    class Light : public IMovableObject, std::enable_shared_from_this<Light>
    {
    public:
        using SharedPtr = std::shared_ptr<Light>;
        using SharedConstPtr = std::shared_ptr<const Light>;

        Light() = default;
        virtual ~Light() = default;

        /** Set the light parameters into a program. To use this you need to include/import 'ShaderCommon' inside your shader.
            \param[in] pVars The program vars to set the parameters into.
            \param[in] pBuffer The constant buffer to set the parameters into.
            \param[in] varName The name of the light variable in the program.
        */
        virtual void setIntoProgramVars(ProgramVars* pVars, ConstantBuffer* pCb, const std::string& varName);

        /** Set the light parameters into a program. To use this you need to include/import 'ShaderCommon' inside your shader.
            \param[in] pVars The program vars to set the parameters into.
            \param[in] pBuffer The constant buffer to set the parameters into.
            \param[in] offset Byte offset into the constant buffer to set data to.
        */
        virtual void setIntoProgramVars(ProgramVars* pVars, ConstantBuffer* pCb, size_t offset);

        /** Set the light parameters into a parameter block. 
            This is called by LightEnv::getParameterBlock() method.
            \param[in] pBlock The parameter block to set the parameters into.
            \param[in] offset Byte offset into the constant buffer of the parameter block to set data to.
            \param[in] varName The name of the light variable in the parameter block.
        */
        virtual void setIntoParameterBlock(ParameterBlock* pBlock, size_t offset, const std::string& varName);

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
        virtual uint32_t getTypeId() const = 0;
        uint32_t getType() const { return getTypeId() & (~LightType_ShadowBit); }

        /** Name the light
        */
        const void setName(const std::string& Name) { mName = Name; }

        virtual void setSampler(const Sampler::SharedPtr& pSampler) {}

        virtual vec3 getPosW() const { return glm::vec3(0.0f); }

        /** Get the light's name
        */
        const std::string& getName() const { return mName; }

        /** Gets the size of a single light data struct in bytes
        */
        static uint32_t getShaderStructSize() { return kDataSize; }

        VersionID getVersionID() const { return mVersionID; }
        
        virtual const char * getShaderTypeName() { return "InfinitessimalLight"; };

    protected:

        static const size_t kDataSize = sizeof(LightData);

        /* UI callbacks for keeping the intensity in-sync */
        glm::vec3 getColorForUI();
        void setColorFromUI(const glm::vec3& uiColor);
        float getIntensityForUI();
        void setIntensityFromUI(float intensity);

        std::string mName;

        virtual void * getRawData() = 0;
        virtual glm::vec3& getIntensityData() = 0;

        /* These two variables track mData values for consistent UI operation.*/
        glm::vec3 mUiLightIntensityColor = glm::vec3(0.5f, 0.5f, 0.5f);
        float     mUiLightIntensityScale = 1.0f;
        VersionID mVersionID = 0;
        void makeDirty() { mVersionID++; }
    };

    class InfinitesimalLight : public Light, public std::enable_shared_from_this<InfinitesimalLight>
    {
    protected:
        bool isShadowed = false;
        std::unique_ptr<CascadedShadowMaps> mCsm;
        LightData mData;
        virtual glm::vec3& getIntensityData() { return mData.intensity; }
    public:
        ~InfinitesimalLight();

        /** Get the light's world-space direction.
        */
        const glm::vec3& getWorldDirection() const { return mData.dirW; }
        void enableShadowMap(std::shared_ptr<Scene> pScene, int width, int height, int numCascades);
        void disableShadowMap();
        CascadedShadowMaps* getCsm() { return mCsm.get(); }
        bool getShadowed() const { return isShadowed; }
        virtual vec3 getPosW() const override { return vec3(mData.posW.x, mData.posW.y, mData.posW.z); }
        virtual void setIntoParameterBlock(ParameterBlock* pBlock, size_t offset, const std::string& varName) override;
        void updateShadowParameters(ParameterBlock* pBlock, const std::string& varName);
        virtual void renderUI(Gui* pGui, const char* group = nullptr);

    };

    /** Directional light source.
    */
    class DirectionalLight : public InfinitesimalLight, public std::enable_shared_from_this<DirectionalLight>
    {
    public:
        using SharedPtr = std::shared_ptr<DirectionalLight>;
        using SharedConstPtr = std::shared_ptr<const DirectionalLight>;

        static SharedPtr create();

        DirectionalLight();
        ~DirectionalLight();

        /** Render UI elements for this light.
        \param[in] pGui The GUI to create the elements with
        \param[in] group Optional. If specified, creates a UI group to display elements within
        */
        void renderUI(Gui* pGui, const char* group = nullptr) override;

        virtual uint32_t getTypeId() const { return LightDirectional | (isShadowed ? LightType_ShadowBit : 0); ; }
        
        /** Set the light's world-space direction.
        */
        void setWorldDirection(const glm::vec3& dir);

        /** Set the light intensity.
            \param[in] intensity Vec3 corresponding to RGB intensity
        */
        void setIntensity(const glm::vec3& intensity) { mData.intensity = intensity; makeDirty(); }

        /** Set the scene parameters
        */
        void setWorldParams(const glm::vec3& center, float radius);

        /** Get the light intensity.
        */
        const glm::vec3& getIntensity() const { return mData.intensity; }

        /** Get total light power (needed for light picking)
        */
        float getPower() const override;

        /** IMovableObject interface
        */
        void move(const glm::vec3& position, const glm::vec3& target, const glm::vec3& up) override;

        virtual const char * getShaderTypeName() override { return isShadowed ? "DirectionalLight<CsmShadow> " : "DirectionalLight<NoShadow> "; };

    private:

        float mDistance = 1e3f; ///< Scene bounding radius is required to move the light position sufficiently far away
        vec3 mCenter;
    protected:
        void * getRawData() override { return &mData; }
    };

    /** Simple infinitely-small point light with quadratic attenuation
    */
    class PointLight : public InfinitesimalLight, public std::enable_shared_from_this<PointLight>
    {
    public:
        using SharedPtr = std::shared_ptr<PointLight>;
        using SharedConstPtr = std::shared_ptr<const PointLight>;

        static SharedPtr create();

        PointLight();
        ~PointLight();

        /** Render UI elements for this light.
            \param[in] pGui The GUI to create the elements with
            \param[in] group Optional. If specified, creates a UI group to display elements within
        */
        void renderUI(Gui* pGui, const char* group = nullptr) override;
        
        virtual uint32_t getTypeId() const { return LightPoint | (isShadowed ? LightType_ShadowBit : 0); }

        /** Get total light power (needed for light picking)
        */
        float getPower() const override;

        /** Set the light's world-space position
        */
        void setWorldPosition(const glm::vec3& pos) { mData.posW = pos; makeDirty(); }

        /** Set the light's world-space position
        */
        void setWorldDirection(const glm::vec3& dir) { mData.dirW = dir; makeDirty(); }

        /** Set the light intensity.
        */
        void setIntensity(const glm::vec3& intensity) { mData.intensity = intensity; makeDirty(); }

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
        void setPenumbraAngle(float angle) { mData.penumbraAngle = glm::clamp(angle, 0.0f, mData.openingAngle);; makeDirty(); }

        /** Get the opening angle
        */
        float getOpeningAngle() const { return mData.openingAngle; }

        /** IMovableObject interface
        */
        void move(const glm::vec3& position, const glm::vec3& target, const glm::vec3& up) override;

        virtual const char * getShaderTypeName() override { return isShadowed?"PointLight<CsmShadow> " : "PointLight<NoShadow> "; };
    protected:
        void * getRawData() override { return &mData; }
    };

    /**
        Area light source

        This class is used to simulate area light sources. All emissive
        materials are treated as area light sources.
    */
    class AreaLight : public Light, public std::enable_shared_from_this<AreaLight>
    {
    public:
        using SharedPtr = std::shared_ptr<AreaLight>;
        using SharedConstPtr = std::shared_ptr<const AreaLight>;

        static SharedPtr create();

        AreaLight();
        ~AreaLight();
        
        virtual uint32_t getTypeId() const { return LightArea; }

        /** Get total light power (needed for light picking)
        */
        float getPower() const override;

        /** Set the light parameters into a program. To use this you need to include/import 'ShaderCommon' inside your shader
            and declare a constant buffer to bind the values to using the AREA_LIGHTS() macro defined in HostDeviceSharedMacros.h
            \param[in] pVars The program vars to set the parameters into.
            \param[in] pBuffer The constant buffer to set the parameters into.
            \param[in] varName The name of the declared variable in the program. "gAreaLights" by default.
        */
        virtual void setIntoProgramVars(ProgramVars* pVars, ConstantBuffer* pCb, const std::string& varName);

        /** Do not use this overload for area lights. Area light data contains resources that cannot be bound using an offset when
            there is an array of light data. Calling this function will do nothing except log a warning.
        */
        virtual void setIntoProgramVars(ProgramVars* pVars, ConstantBuffer* pCb, size_t offset);

        /** Used by LightEnv class.
        */
        virtual void setIntoParameterBlock(ParameterBlock* pBlock, size_t offset, const std::string& varName) override;


        /** Render UI elements for this light.
            \param[in] pGui The GUI to create the elements with
            \param[in] group Optional. If specified, creates a UI group to display elements within
        */
        void renderUI(Gui* pGui, const char* group = nullptr) override;

        /** Set the geometry mesh for this light
            \param[in] pModel Model that contains the geometry mesh for this light
            \param[in] meshId Geometry mesh id within the model
            \param[in] instanceId Geometry mesh instance id
        */
        void setMeshData(const Model::MeshInstance::SharedPtr& pMeshInstance);

        /** Obtain the geometry mesh for this light
            \return Mesh instance for this light
        */
        const Model::MeshInstance::SharedPtr& getMeshData() const { return mpMeshInstance; }

        /** Compute surface area of the mesh
        */
        void computeSurfaceArea();

        /** Get surface area of the mesh
        */
        float getSurfaceArea() const { return mAreaLightData.surfaceArea; }

        /** Get the probability distribution of the mesh
        */
        const std::vector<float>& getMeshCDF() const { return mMeshCDF; }

        /** Set the index buffer
            \param[in] indexBuf Buffer containing mesh indices
        */
        void setIndexBuffer(const Buffer::SharedPtr& indexBuf) { mpIndexBuffer = indexBuf; makeDirty(); }

        /** Get the index buffer.
        */
        const Buffer::SharedPtr& getIndexBuffer() const { return mpIndexBuffer; }

        /** Set the vertex buffer.
            \param[in] vertexBuf Buffer containing mesh vertices
        */
        void setPositionsBuffer(const Buffer::SharedPtr& vertexBuf) { mpVertexBuffer = vertexBuf; makeDirty(); }

        /** Get the vertex buffer.
        */
        const Buffer::SharedPtr& getPositionsBuffer() const { return mpVertexBuffer; }

        /** Set the texture coordinate/UV buffer.
            \param[in] texCoordBuf Buffer containing texture coordinates
        */
        void setTexCoordBuffer(const Buffer::SharedPtr& texCoordBuf) { mpTexCoordBuffer = texCoordBuf; makeDirty(); }

        /** Get texture coordinate buffer.
        */
        const Buffer::SharedPtr& getTexCoordBuffer() const { return mpTexCoordBuffer; }

        /** Set the mesh CDF buffer.
            \param[in] meshCDF Buffer containing mesh CDF data
        */
        void setMeshCDFBuffer(const Buffer::SharedPtr& meshCDFBuf) { mpMeshCDFBuffer = meshCDFBuf; makeDirty(); }

        /** Get the mesh CDF buffer
        */
        const Buffer::SharedPtr& getMeshCDFBuffer() const { return mpMeshCDFBuffer; }

        /** IMovableObject interface
        */
        void move(const glm::vec3& position, const glm::vec3& target, const glm::vec3& up) override;

        virtual vec3 getPosW() const override { return vec3(mAreaLightData.posW.x, mAreaLightData.posW.y, mAreaLightData.posW.z); }

        /** Gets the size of a single light data struct in bytes
        */
        static uint32_t getShaderStructSize() { return kAreaLightDataSize; }

        virtual const char * getShaderTypeName() override { return "AreaLight"; };

    private:

        static const size_t kAreaLightDataSize = sizeof(AreaLightData) - sizeof(AreaLightResources);
        AreaLightData mAreaLightData;

        Model::MeshInstance::SharedPtr mpMeshInstance; ///< Geometry mesh data
        Buffer::SharedPtr mpIndexBuffer;    ///< Buffer for indices
        Buffer::SharedPtr mpVertexBuffer;   ///< Buffer for vertices
        Buffer::SharedPtr mpTexCoordBuffer; ///< Buffer for texcoord
        Buffer::SharedPtr mpMeshCDFBuffer;  ///< Buffer for mesh Cumulative distribution function (CDF)

        std::vector<float> mMeshCDF; ///< CDF function for importance sampling a triangle mesh
    protected:
        void * getRawData() override { return &mAreaLightData; }
        glm::vec3& getIntensityData() { return mAreaLightData.intensity; }
    };

    // Represent a lighting environment, which is effectively an array of lights
    class LightEnv : public std::enable_shared_from_this<LightEnv>
    {
    private:
        Sampler::SharedPtr lightProbeSampler;
    public:
        using SharedPtr = std::shared_ptr<LightEnv>;
        using SharedConstPtr = std::shared_ptr<const LightEnv>;
        static LightEnv::SharedPtr create();
        struct LightTypeInfo
        {
            const char* typeName;
            std::string variableName;
            size_t cbOffset = 0;
            std::vector<Light::SharedPtr> lights;
        };
        std::string shaderTypeName, lightCollectionTypeName;
        std::map<uint32_t, LightTypeInfo> lightTypes;

        void merge(LightEnv const* lightEnv);

        uint32_t addLight(const Light::SharedPtr& pLight);
        void deleteLight(uint32_t lightID);
        uint32_t getLightCount() const { return (uint32_t)mpLights.size(); }
        const Light::SharedPtr& getLight(uint32_t index) const { return mpLights[index]; }

        std::vector<Light::SharedPtr>& getLights() { return mpLights; }
        const std::vector<Light::SharedPtr>& getLights() const { return mpLights; }

        /**
        This routine deletes area light(s) from the environment.
        */
        void deleteAreaLights();
        void deleteLightProbes();
        /** Get the ParameterBlock object for the lighting environment.
        */
        ParameterBlock::SharedConstPtr getParameterBlock() const;
        void setIntoProgramVars(ProgramVars* vars);
        void bindSampler(Sampler::SharedPtr sampler);
    private:
        LightEnv();
        LightEnv(LightEnv const&) = delete;
        std::vector<Light::SharedPtr> mpLights;
        mutable std::vector<VersionID> mLightVersionIDs;

        mutable VersionID mVersionID = 0;
        mutable bool lightCollectionChanged = false;
        VersionID getVersionID() const;

        mutable ParameterBlock::SharedPtr mpParamBlock;
        mutable VersionID mParamBlockVersionID = -1;
        static ParameterBlockReflection::SharedConstPtr spBlockReflection;
        static size_t sLightCountOffset;
        static size_t sLightArrayOffset;
        static size_t sAmbientLightOffset;
    };

    AreaLight::SharedPtr createAreaLight(const Model::MeshInstance::SharedPtr& pMeshInstance);
    std::vector<AreaLight::SharedPtr> createAreaLightsForModel(const Model* pModel);
}
