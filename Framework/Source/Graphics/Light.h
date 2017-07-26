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
#include "Graphics/Model/Model.h"
#include "Graphics/Paths/MovableObject.h"

namespace Falcor
{
    class ConstantBuffer;
    class Gui;

    /** Base class for light sources. All light sources should inherit from this.
    */
    class Light : public IMovableObject, std::enable_shared_from_this<Light>
    {
    public:
        using SharedPtr = std::shared_ptr<Light>;
        using SharedConstPtr = std::shared_ptr<const Light>;

        Light() = default;
        virtual ~Light() = default;

        /** Set the light parameters into a program. To use this you need to include 'Falcor.h' inside your shader.
            \param[in] pBuffer The constant buffer to set the parameters into.
            \param[in] varName The name of the light variable in the program.
        */
        virtual void setIntoConstantBuffer(ConstantBuffer* pBuffer, const std::string& varName);
        virtual void setIntoConstantBuffer(ConstantBuffer* pBuffer, size_t offset);

        /** create UI elements for this light.
            \param[in] pGui The GUI to create the elements with
        */
        virtual void renderUI(Gui* pGui, const char* group = nullptr);

        /**
            Prepare GPU data
        */
        virtual void prepareGPUData() = 0;

        /**
            Unload GPU data
        */
        virtual void unloadGPUData() = 0;

		/**
		    Get total light power (needed for light picking)
		*/
        virtual float getPower() = 0;

        /** Get the light Type
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

        static uint32_t getShaderStructSize() { return kDataSize; }

    protected:

        static const size_t kDataSize = sizeof(LightData); //TODO(tfoley) HACK:SPIRE - sizeof(MaterialData);

        /* UI callbacks for keeping the intensity in-sync */
        glm::vec3 getColorForUI();
        void setColorFromUI(const glm::vec3& uiColor);
        float getIntensityForUI();
        void setIntensityFromUI(float intensity);

        std::string mName;

        /* These two variables track mData values for consistent UI operation.*/
        glm::vec3 mUiLightIntensityColor = glm::vec3(0.5f, 0.5f, 0.5f);
        float     mUiLightIntensityScale = 1.0f;
        LightData mData;
    };

    /** Directional light source.
    */
    class DirectionalLight : public Light, public std::enable_shared_from_this<DirectionalLight>
    {
    public:
        using SharedPtr = std::shared_ptr<DirectionalLight>;
        using SharedConstPtr = std::shared_ptr<const DirectionalLight>;

        static SharedPtr create();

        /**
            Default Constructor
        */
        DirectionalLight();

        ~DirectionalLight();

        /** create UI elements for this light.
            \param[in] pGui The GUI to create the elements with
        */
        void renderUI(Gui* pGui, const char* group = nullptr) override;

        /**
            Prepare GPU data
        */
        void prepareGPUData() override;

        /**
            Unload GPU data
        */
        void unloadGPUData() override;

        /** Set the light's world-space direction.
        */
        void setWorldDirection(const glm::vec3& dir);
        /** Set the light intensity.
        */
        void setIntensity(const glm::vec3& intensity) { mData.intensity = intensity; }
                
        /** Set the scene parameters
        */
        void setWorldParams(const glm::vec3& center, float radius);

        /** Get the light's world-space direction.
        */
        const glm::vec3& getWorldDirection() const { return mData.worldDir; }
        /** Get the light intensity.
        */
        const glm::vec3& getIntensity() const { return mData.intensity; }

        /**
		    Get total light power (needed for light picking)
		*/
        float getPower() override;

		/**
            IMovableObject interface
        */
        void move(const glm::vec3& position, const glm::vec3& target, const glm::vec3& up) override;

    private:

        float     mDistance = 1e3f;       ///< Scene bounding radius is required to move the light position sufficiently far away
        vec3      mCenter;
    };

    /** Simple infinitely-small point light with quadratic attenuation
    */
    class PointLight : public Light, public std::enable_shared_from_this<PointLight>
    {
    public:
        using SharedPtr = std::shared_ptr<PointLight>;
        using SharedConstPtr = std::shared_ptr<const PointLight>;

        static SharedPtr create();

        /**
            Default Constructor
        */
        PointLight();

        ~PointLight();

        /** create UI elements for this light.
            \param[in] pGui The GUI to create the elements with
        */
        void renderUI(Gui* pGui, const char* group = nullptr) override;

        /**
            Prepare GPU data
        */
        void prepareGPUData() override;

        /**
            Unload GPU data
        */
        void unloadGPUData() override;
        
		/**
		    Get total light power (needed for light picking)
		*/
        float getPower() override;

        /** Set the light's world-space position
        */
        void setWorldPosition(const glm::vec3& pos) { mData.worldPos = pos; }

        /** Set the light's world-space position
        */
        void setWorldDirection(const glm::vec3& dir) { mData.worldDir = dir; }

        /** Set the light intensity.
        */
        void setIntensity(const glm::vec3& intensity) { mData.intensity = intensity; }

        /** Set the cone opening angle (for spot lights), in radians.
        */
        void setOpeningAngle(float openingAngle);

        /** Get the light's world-space position
        */
        const glm::vec3& getWorldPosition() const { return mData.worldPos; }

        /** Get the light's world-space direction
        */
        const glm::vec3& getWorldDirection() const { return mData.worldDir; }

        /** Get the light intensity.
        */
        const glm::vec3& getIntensity() const { return mData.intensity; }

        /** Get the penumbra angle
        */
        float getPenumbraAngle() const { return mData.penumbraAngle; }

        /** Set the penumbra angle
        */
        void setPenumbraAngle(float angle) { mData.penumbraAngle = glm::clamp(angle, 0.0f, mData.openingAngle);; }

        /** Get the opening angle
        */
        float getOpeningAngle() const { return mData.openingAngle; }

        /**
            IMovableObject interface
        */
        void move(const glm::vec3& position, const glm::vec3& target, const glm::vec3& up) override;

    private:
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

        /**
            Default constructor
        */
        AreaLight();

        /**
            Default destructor
        */
        ~AreaLight();
        
		/**
		    Get total light power (needed for light picking)
		*/
        float getPower() override;

        /**
            Set the light parameters into a program. To use this you need to
            include 'Falcor.h' inside your shader.

            \param[in] pBuffer The constant buffer to set the parameters into.
            \param[in] varName The name of the light variable in the program.
        */
        void setIntoConstantBuffer(ConstantBuffer* pBuffer, const std::string& varName) override;

        /**
            Create UI elements for this light.

            \param[in] pGui The GUI to create the elements with
        */
        void renderUI(Gui* pGui, const char* group = nullptr) override;

        /**
            Prepare GPU data
        */
        void prepareGPUData() override;

        /**
            Unload GPU data
        */
        void unloadGPUData() override;

        /**
            Set the geometry mesh for this light

            \param[in] pModel Model that contains the geometry mesh for this light
            \param[in] meshId Geometry mesh id within the model
            \param[in] instanceId Geometry mesh instance id
        */
        void setMeshData(const Model::MeshInstance::SharedPtr& pMeshInstance);

        /**
            Obtain the geometry mesh for this light

            \return Mesh instance for this light
        */
        const Model::MeshInstance::SharedPtr& getMeshData() const { return mpMeshInstance; }

        /**
            Compute surface area of the mesh
        */
        void computeSurfaceArea();

        /**
            Get surface area of the mesh

            \return Surface area of the mesh
        */
        float getSurfaceArea() const { return mSurfaceArea; }

        /**
            Gather probability distribution of the mesh

            \return Probability distribution of the mesh
        */
        const std::vector<float>& getMeshCDF() const { return mMeshCDF; }

        /**
            Set buffer id for indices

            \param[in] indexId buffer id for indices
        */
        void setIndexBuffer(const Buffer::SharedPtr& indexBuf) { mIndexBuf = indexBuf; }

        /**
            Get buffer id for indices

            \return buffer id for indices
        */
        const Buffer::SharedPtr& getIndexBuffer() const { return mIndexBuf; }

        /**
            Set buffer id for vertices

            \param[in] vertex buffer id for vertices
        */
        void setPositionsBuffer(const Buffer::SharedPtr& vertexBuf) { mVertexBuf = vertexBuf; }

        /**
            Get buffer id for vertices

            \return buffer id for vertices
        */
        const Buffer::SharedPtr& getPositionsBuffer() const { return mVertexBuf; }

        /**
            Set buffer id for texcoord

            \param[in] texCoord buffer id for texcoord
        */
        void setTexCoordBuffer(const Buffer::SharedPtr& texCoordBuf) { mTexCoordBuf = texCoordBuf; }

        /**
            Get buffer id for texcoord

            \return buffer id for texcoord
        */
        const Buffer::SharedPtr& getTexCoordBuffer() const { return mTexCoordBuf; }

        /**
            Set buffer id for mesh CDF

            \param[in] meshCDF buffer id for mesh CDF
        */
        void setMeshCDFBuffer(const Buffer::SharedPtr& meshCDFBuf) { mMeshCDFBuf = meshCDFBuf; }

        /**
            Get Buffer id for mesh CDF

            \return Buffer id for mesh CDF
        */
        const Buffer::SharedPtr& getMeshCDFBuffer() const { return mMeshCDFBuf; }

        /**
            IMovableObject interface
        */
        void move(const glm::vec3& position, const glm::vec3& target, const glm::vec3& up) override;

        /**
            This routine creates area light(s) for the given model.

            \param[in] pModel Model
            \param[out] areaLights Vector to store area lights
        */
        static void createAreaLightsForModel(const Model::SharedPtr& pModel, std::vector<Light::SharedPtr>& areaLights);

    private:

        /**
            This is a utility function that creates an area light for the geometry mesh.

            \param[in] pMeshInstance Instance of geometry mesh
        */
        static Light::SharedPtr createAreaLight(const Model::MeshInstance::SharedPtr& pMeshInstance);

        Model::MeshInstance::SharedPtr mpMeshInstance;      ///< Geometry mesh data
        Buffer::SharedPtr         mIndexBuf;           ///< Buffer id for indices
        Buffer::SharedPtr         mVertexBuf;          ///< Buffer id for vertices
        Buffer::SharedPtr         mTexCoordBuf;        ///< Buffer id for texcoord
        Buffer::SharedPtr         mMeshCDFBuf;         ///< Buffer id for mesh Cumulative distribution function (CDF)

        float                          mSurfaceArea;        ///< Surface area of the mesh
		vec3                           mTangent;            ///< Unnormalized tangent vector of the light
		vec3                           mBitangent;          ///< Unnormalized bitangent vector of the light
        std::vector<float>             mMeshCDF;            ///< CDF function for importance sampling a triangle mesh
    };
}
