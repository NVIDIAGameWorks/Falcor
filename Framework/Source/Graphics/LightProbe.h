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
#include "Graphics/Paths/MovableObject.h"
#include "API/Texture.h"
#include "Data/HostDeviceData.h"
#include "API/Sampler.h"

namespace Falcor
{
    class ProgramVars;
    class ConstantBuffer;
    class Gui;

    class LightProbe : public IMovableObject, public inherit_shared_from_this<IMovableObject, LightProbe>
    {
    public:
        using SharedPtr = std::shared_ptr<LightProbe>;
        using SharedConstPtr = std::shared_ptr<const LightProbe>;
        SharedPtr shared_from_this() { return inherit_shared_from_this<IMovableObject, LightProbe>::shared_from_this(); }

        static const uint32_t kDataSize = sizeof(LightProbeData) - sizeof(LightProbeResources);
        static const uint32_t kDefaultDiffSamples = 4096;
        static const uint32_t kDefaultSpecSamples = 1024;
        static const uint32_t kDefaultDiffSize = 128;
        static const uint32_t kDefaultSpecSize = 1024;

        /** Create a light-probe from a file
            \param[in] pContext The current render context to be used for pre-integration.
            \param[in] filename Texture filename
            \param[in] loadAsSrgb Indicates whether the source texture is in sRGB or linear color space
            \param[in] overrideFormat Override the format of the original texture. ResourceFormat::Unknown means keep the original format. Useful in cases where generateMips is true, but the original format doesn't support automatic mip generation
            \param[in] diffSampleCount How many times to sample when generating diffuse texture.
            \param[in] specSampleCount How many times to sample when generating specular texture.
            \param[in] diffSize The width and height of the pre-filtered diffuse texture. We always create a square texture.
            \param[in] specSize The width and height of the pre-filtered specular texture. We always create a square texture.
            \param[in] preFilteredFormat The format of the pre-filtered texture
        */
        static SharedPtr create(RenderContext* pContext, const std::string& filename, bool loadAsSrgb, ResourceFormat overrideFormat = ResourceFormat::Unknown, uint32_t diffSampleCount = kDefaultDiffSamples, uint32_t specSampleCount = kDefaultSpecSamples, uint32_t diffSize = kDefaultDiffSize, uint32_t specSize = kDefaultSpecSize, ResourceFormat preFilteredFormat = ResourceFormat::RGBA16Float);

        /** Create a light-probe from a texture
            \param[in] pContext The current render context to be used for pre-integration.
            \param[in] pTexture The source texture
            \param[in] diffSampleCount How many times to sample when generating diffuse texture.
            \param[in] specSampleCount How many times to sample when generating specular texture.
            \param[in] diffSize The width and height of the pre-filtered diffuse texture. We always create a square texture.
            \param[in] specSize The width and height of the pre-filtered specular texture. We always create a square texture.
            \param[in] preFilteredFormat The format of the pre-filtered texture
        */
        static SharedPtr create(RenderContext* pContext, const Texture::SharedPtr& pTexture, uint32_t diffSampleCount = kDefaultDiffSamples, uint32_t specSampleCount = kDefaultSpecSamples, uint32_t diffSize = kDefaultDiffSize, uint32_t specSize = kDefaultSpecSize, ResourceFormat preFilteredFormat = ResourceFormat::RGBA16Float);

        ~LightProbe();

        /** Render UI elements for this light.
            \param[in] pGui The GUI to create the elements with
            \param[in] group Optional. If specified, creates a UI group to display elements within
        */
        void renderUI(Gui* pGui, const char* group = nullptr);

        /** Set the light probe's world-space position
        */
        void setPosW(const vec3& posW) { mData.posW = posW; }

        /** Get the light probe's world-space position
        */
        const vec3& getPosW() const { return mData.posW; }

        /** Set the spherical radius the light probe encompasses. Set radius to negative to sample as an infinite-distance global light probe.
        */
        void setRadius(float radius) { mData.radius = radius; }

        /** Get the light probe's radius.
        */
        float getRadius() const { return mData.radius; }

        /** Get the sample count used to generate the diffuse texture.
        */
        uint32_t getDiffSampleCount() const { return mDiffSampleCount; }

        /** Get the sample count used to generate the specular texture.
        */
        uint32_t getSpecSampleCount() const { return mSpecSampleCount; }

        /** Set the light probe's light intensity
        */
        void setIntensity(const vec3& intensity) { mData.intensity = intensity; }

        /** Get the light probe's light intensity
        */
        const vec3& getIntensity() const { return mData.intensity; }

        /** Attach a sampler to the light probe
        */
        void setSampler(const Sampler::SharedPtr& pSampler) { mData.resources.sampler = pSampler; }

        /** Get the sampler state
        */
        const Sampler::SharedPtr& getSampler() const { return mData.resources.sampler; }

        /** Get the light probe's source texture.
        */
        const Texture::SharedPtr& getOrigTexture() const { return mData.resources.origTexture; }

        /** Get the light probe's diffuse texture.
        */
        const Texture::SharedPtr& getDiffuseTexture() const { return mData.resources.diffuseTexture; }

        /** Get the light probe's specular texture.
        */
        const Texture::SharedPtr& getSpecularTexture() const { return mData.resources.specularTexture; }

        /** Get the texture storing the pre-integrated DFG term shared by all light probes.
        */
        static const Texture::SharedPtr& getDfgTexture() { return sSharedData.dfgTexture; }

        /** Bind the light data into a ProgramVars object
        */
        void setIntoProgramVars(ProgramVars* pVars, ConstantBuffer* pBuffer, const std::string& varName);

        /** Bind common light probe resources into a ProgramVars object
        */
        static void setCommonIntoProgramVars(ProgramVars* pVars, const std::string& varName);

    private:
        static uint32_t sLightProbeCount;
        static LightProbeSharedResources sSharedData;

        LightProbeData mData;
        uint32_t mDiffSampleCount;
        uint32_t mSpecSampleCount;
        void move(const glm::vec3& position, const glm::vec3& target, const glm::vec3& up) override;
        LightProbe(RenderContext* pContext, const Texture::SharedPtr& pTexture, uint32_t diffSamples, uint32_t specSamples, uint32_t diffSize, uint32_t specSize, ResourceFormat preFilteredFormat);
    };
}