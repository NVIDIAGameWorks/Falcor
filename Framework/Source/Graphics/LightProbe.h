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
#include "Graphics/Light.h"

namespace Falcor
{
    class ProgramVars;
    class ConstantBuffer;
    class Gui;

    class LightProbe : public Light, std::enable_shared_from_this<LightProbe>
    {
    public:
        using SharedPtr = std::shared_ptr<LightProbe>;
        using SharedConstPtr = std::shared_ptr<const LightProbe>;

        static const uint32_t kDataSize = sizeof(LightProbeData) - sizeof(LightProbeResources);

        /** The type of the filtering that will be applied to the source texture
        */
        enum class PreFilterMode
        {
            None,                   ///< No filtering. The light probe will have a single mip-level
            PreIntegration,         ///< Pre-filter the textures and generate mip-chain using the technique described in https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf
        };

        /** Create a light-probe from a file
            \param[in] filename Texture filename
            \param[in] loadAsSrgb Indicates whether the source texture is in sRGB or linear color space
            \param[in] generateMips Generate mip-chain for the unfiltered texture
            \param[in] overrideFormat Override the format of the original texture. ResourceFormat::Unknown means keep the original format. Useful in cases where generateMips is true, but the original format doesn't support automatic mip generation
            \param[in] filter The pre-filtering mode. If this value equals PreFilterMode::None, then a pre-filtering texture will not be created
            \param[in] size The width and height of the pre-filtered texture. We always create a square texture. If this value equals Texture::kMaxPossible, the size will chosen automatically
            \param[in] preFilteredFormat The format of the pre-filtered texture
            */
        static SharedPtr create(const std::string& filename, bool loadAsSrgb, bool generateMips, ResourceFormat overrideFormat = ResourceFormat::Unknown, PreFilterMode filter = PreFilterMode::None, uint32_t size = Texture::kMaxPossible, ResourceFormat preFilteredFormat = ResourceFormat::RGBA16Float);

        /** Create a light-probe from a texture
            \param[in] pTexture The source texture
            \param[in] filter The pre-filtering mode. If this value equals PreFilterMode::None, then a pre-filtering texture will not be created
            \param[in] size The width and height of the pre-filtered texture. We always create a square texture. If this value equals Texture::kMaxPossible, the size will chosen automatically
            \param[in] preFilteredFormat The format of the pre-filtered texture
        */
        static SharedPtr create(const Texture::SharedPtr& pTexture, PreFilterMode filter = PreFilterMode::None, uint32_t size = Texture::kMaxPossible, ResourceFormat preFilteredFormat = ResourceFormat::RGBA16Float);

        /** Render UI elements for this light.
            \param[in] pGui The GUI to create the elements with
            \param[in] group Optional. If specified, creates a UI group to display elements within
        */
        void renderUI(Gui* pGui, const char* group = nullptr);

        /** Set the light-probe's world-space position
        */
        void setPosW(const vec3& posW) { mData.posW = posW; }

        /** Get the light-probe's world-space position
        */
        vec3 getPosW() const { return vec3(mData.posW.x, mData.posW.y, mData.posW.z); }

        /** Set the light-probe's light intensity
        */
        void setIntensity(const vec3& intensity) { mData.intensity = intensity; }

        /** Get the light-probe's light intensity
        */
        const vec3& getIntensity() const { return mData.intensity; }

        /** Attach a sampler to the light-probe
        */
        virtual void setSampler(const Sampler::SharedPtr& pSampler) override { mData.resources.samplerState = pSampler; }

        float getPower() const { return 0.0f; }
        /** Get the sampler state
        */
        const Sampler::SharedPtr& getSampler() const { return mData.resources.samplerState; }

        /** Bind the light-data into a ProgramVars object
        */
        void setIntoProgramVars(ProgramVars* pVars, ConstantBuffer* pBuffer, const std::string& varName);

        void setIntoParameterBlock(ParameterBlock * pBlock, ConstantBuffer* pBuffer, size_t offset, const std::string & varName);

        virtual void setIntoParameterBlock(ParameterBlock* pBlock, size_t offset, const std::string& varName) override;

        static uint32_t getShaderStructSize() { return sizeof(LightProbeData); }

        virtual const char * getShaderTypeName() override { return "ProbeLight"; };
        virtual uint32_t getType() const override;
        virtual void * getRawData() override { return &mData; }

    private:
        void move(const glm::vec3& position, const glm::vec3& target, const glm::vec3& up) override;
        LightProbe(const Texture::SharedPtr& pTexture, PreFilterMode filter, uint32_t size, ResourceFormat preFilteredFormat);
        LightProbeData mData;
    protected:
        glm::vec3& getIntensityData();
    };
}