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

    class LightProbe : public IMovableObject, std::enable_shared_from_this<LightProbe>
    {
    public:
        using SharedPtr = std::shared_ptr<LightProbe>;
        using SharedConstPtr = std::shared_ptr<const LightProbe>;

        static const uint32_t kDataSize = sizeof(LightProbeData) - sizeof(LightProbeResources);

        /** The type of the filtering that will be applied to the source texture
        */
        enum class MipFilter
        {
            None,                   ///< No filtering. The light probe will have a single mip-level
            Linear,                 ///< Generate mip-chain using bilinear filtering
            PreIntegration,         ///< Pre-filter the textures and generate mip-chain using the technique described in https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf
        };

        /** Create a light-probe from a file
            \param[in] filename Texture filename
            \param[in] size The width and height of the destination texture. We always create a square texture
            \param[in] loadAsSrgb Indicates whether the source texture is in sRGB or linear color space
            \param[in] format The format of the light-probe texture
            \param[in] mipFilter The filter to use when creating the light-probe
        */
        static SharedPtr create(const std::string& filename, uint32_t size, bool loadAsSrgb, ResourceFormat format = ResourceFormat::RGBA16Float, MipFilter mipFilter = MipFilter::Linear);

        /** Create a light-probe from a texture
        \param[in] pTexture The source texture
        \param[in] size The width and height of the destination texture. We always create a square texture
        \param[in] format The format of the light-probe texture
        \param[in] mipFilter The filter to use when creating the light-probe
        */
        static SharedPtr create(const Texture::SharedPtr& pTexture, uint32_t size, ResourceFormat format = ResourceFormat::RGBA16Float, MipFilter mipFilter = MipFilter::Linear);

        /** Set the light-probe's world-space position
        */
        void setPosW(const vec3& posW) { mData.posW = posW; }

        /** Get the light-probe's world-space position
        */
        const vec3& getPosW() const { return mData.posW; }

        /** Set the light-probe's light intensity
        */
        void setIntensity(const vec3& intensity) { mData.intensity = intensity; }

        /** Get the light-probe's light intensity
        */
        const vec3& getIntensity() const { return mData.intensity; }

        /** Attach a sampler to the light-probe
        */
        void setSampler(const Sampler::SharedPtr& pSampler) { mData.resources.samplerState = pSampler; }

        /** Get the sampler state
        */
        const Sampler::SharedPtr& getSampler() const { return mData.resources.samplerState; }

        /** Bind the light-data into a ProgramVars object
        */
        void setIntoProgramVars(ProgramVars* pVars, ConstantBuffer* pBuffer, const std::string& varName);

    private:
        void move(const glm::vec3& position, const glm::vec3& target, const glm::vec3& up) override;
        LightProbe(const Texture::SharedPtr& pTexture, uint32_t size, ResourceFormat format, MipFilter mipFilter);
        LightProbeData mData;
    };
}