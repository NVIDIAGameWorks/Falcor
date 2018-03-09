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

    class LightProbe : public IMovableObject, std::enable_shared_from_this<LightProbe>
    {
    public:
        using SharedPtr = std::shared_ptr<LightProbe>;
        using SharedConstPtr = std::shared_ptr<const LightProbe>;

        static const uint32_t kDataSize = sizeof(LightProbeData) - sizeof(LightProbeResources);

        /** Create a light-probe from a file
            \param[in] pContext The current render context to be used for pre-integration.
            \param[in] filename Texture filename
            \param[in] loadAsSrgb Indicates whether the source texture is in sRGB or linear color space
            \param[in] generateMips Generate mip-chain for the unfiltered texture
            \param[in] overrideFormat Override the format of the original texture. ResourceFormat::Unknown means keep the original format. Useful in cases where generateMips is true, but the original format doesn't support automatic mip generation
            \param[in] size The width and height of the pre-filtered texture. We always create a square texture.
            \param[in] diffSampleCount How many times to sample when generating diffuse texture.
            \param[in] preFilteredFormat The format of the pre-filtered texture
            */
        static SharedPtr create(RenderContext* pContext, const std::string& filename, bool loadAsSrgb, bool generateMips, ResourceFormat overrideFormat = ResourceFormat::Unknown, uint32_t size = 128, uint32_t diffSampleCount = 1024, ResourceFormat preFilteredFormat = ResourceFormat::RGBA16Float);

        /** Create a light-probe from a texture
            \param[in] pContext The current render context to be used for pre-integration.
            \param[in] pTexture The source texture
            \param[in] size The width and height of the pre-filtered texture. We always create a square texture.
            \param[in] diffSampleCount How many times to sample when generating diffuse texture.
            \param[in] preFilteredFormat The format of the pre-filtered texture
        */
        static SharedPtr create(RenderContext* pContext, const Texture::SharedPtr& pTexture, uint32_t size = 128, uint32_t diffSampleCount = 1024, ResourceFormat preFilteredFormat = ResourceFormat::RGBA16Float);

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

        /** Set the light probe's light intensity
        */
        void setIntensity(const vec3& intensity) { mData.intensity = intensity; }

        /** Get the light probe's light intensity
        */
        const vec3& getIntensity() const { return mData.intensity; }

        /** Attach a sampler to the light probe
        */
        void setSampler(const Sampler::SharedPtr& pSampler) { mData.resources.samplerState = pSampler; }

        /** Get the sampler state
        */
        const Sampler::SharedPtr& getSampler() const { return mData.resources.samplerState; }

        /** Get the light probe's source texture.
        */
        const Texture::SharedPtr& getOrigTexture() const { return mData.resources.origTexture; }

        /** Get the light probe's diffuse texture.
        */
        const Texture::SharedPtr& getDiffuseTexture() const { return mData.resources.diffuseTexture; }

        const Texture::SharedPtr& getDfgTexture() const { return mData.resources.dfgTexture; }

        const Texture::SharedPtr& getSpecularTexture() const { return mData.resources.specularTexture; }

        /** Bind the light data into a ProgramVars object
        */
        void setIntoProgramVars(ProgramVars* pVars, ConstantBuffer* pBuffer, const std::string& varName);

    private:
        LightProbeData mData;
        uint32_t mDiffSampleCount;
        void move(const glm::vec3& position, const glm::vec3& target, const glm::vec3& up) override;
        LightProbe(RenderContext* pContext, const Texture::SharedPtr& pTexture, uint32_t size, uint32_t diffuseSamples, ResourceFormat preFilteredFormat);
    };
}