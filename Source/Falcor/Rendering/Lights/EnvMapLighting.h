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
#include "Core/Macros.h"
#include "Scene/Lights/EnvMap.h"
#include <memory>

namespace Falcor
{
    class RenderContext;

    /** Helper for image based lighting using an environment map radiance probe.
    */
    class FALCOR_API EnvMapLighting
    {
    public:
        using SharedPtr = std::shared_ptr<EnvMapLighting>;
        using SharedConstPtr = std::shared_ptr<const EnvMapLighting>;

        static const uint32_t kDefaultDiffSamples = 4096;
        static const uint32_t kDefaultSpecSamples = 1024;
        static const uint32_t kDefaultDiffSize = 128;
        static const uint32_t kDefaultSpecSize = 1024;

        /** Create a environment map lighting helper.
            \param[in] pContext The current render context to be used for pre-integration.
            \param[in] pEnvMap Environment map.
            \param[in] diffSampleCount How many times to sample when generating diffuse texture.
            \param[in] specSampleCount How many times to sample when generating specular texture.
            \param[in] diffSize The width and height of the pre-filtered diffuse texture. We always create a square texture.
            \param[in] specSize The width and height of the pre-filtered specular texture. We always create a square texture.
            \param[in] preFilteredFormat The format of the pre-filtered texture
        */
        static SharedPtr create(RenderContext* pContext, const EnvMap::SharedPtr& pEnvMap, uint32_t diffSampleCount = kDefaultDiffSamples, uint32_t specSampleCount = kDefaultSpecSamples, uint32_t diffSize = kDefaultDiffSize, uint32_t specSize = kDefaultSpecSize, ResourceFormat preFilteredFormat = ResourceFormat::RGBA16Float);

        /** Get the associated environment map.
        */
        const EnvMap::SharedPtr& getEnvMap() const { return mpEnvMap; }

        /** Bind the environment map lighting helper into a shader var.
        */
        void setShaderData(const ShaderVar& var);

        /** Get the total GPU memory usage in bytes.
        */
        uint64_t getMemoryUsageInBytes() const;

    private:
        EnvMapLighting(RenderContext* pContext, const EnvMap::SharedPtr& pEnvMap, uint32_t diffSamples, uint32_t specSamples, uint32_t diffSize, uint32_t specSize, ResourceFormat preFilteredFormat);

        EnvMap::SharedPtr mpEnvMap;
        uint32_t mDiffSampleCount;
        uint32_t mSpecSampleCount;

        Texture::SharedPtr mpDFGTexture;
        Sampler::SharedPtr mpDFGSampler;
        Texture::SharedPtr mpDiffuseTexture;
        Texture::SharedPtr mpSpecularTexture;
        Sampler::SharedPtr mpSampler;
    };
}
