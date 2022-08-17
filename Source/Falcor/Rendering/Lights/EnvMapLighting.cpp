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
#include "EnvMapLighting.h"
#include "RenderGraph/BasePasses/FullScreenPass.h"
#include "Core/API/Device.h"
#include "Core/API/RenderContext.h"
#include "Utils/Logger.h"

namespace Falcor
{
    namespace
    {
        const char* kShader = "Rendering/Lights/EnvMapIntegration.ps.slang";

        Texture::SharedPtr executeSingleMip(RenderContext* pContext, const FullScreenPass::SharedPtr& pPass, const Texture::SharedPtr& pTexture, const Sampler::SharedPtr& pSampler, uint32_t size, ResourceFormat format, uint32_t sampleCount)
        {
            pPass["gInputTex"] = pTexture;
            pPass["gSampler"] = pSampler;
            pPass["DataCB"]["gSampleCount"] = sampleCount;

            // Output texture
            Fbo::SharedPtr pFbo = Fbo::create2D(size, size, Fbo::Desc().setColorTarget(0, format));

            // Execute
            pPass->execute(pContext, pFbo);
            return pFbo->getColorTexture(0);
        }

        Texture::SharedPtr integrateDFG(RenderContext* pContext, uint32_t size, ResourceFormat format, uint32_t sampleCount)
        {
            auto pPass = FullScreenPass::create(std::string(kShader), Program::DefineList().add("_INTEGRATE_DFG"));
            return executeSingleMip(pContext, pPass, nullptr, nullptr, size, format, sampleCount);
        }

        Texture::SharedPtr integrateDiffuseLD(RenderContext* pContext, const Texture::SharedPtr& pTexture, const Sampler::SharedPtr& pSampler, uint32_t size, ResourceFormat format, uint32_t sampleCount)
        {
            auto pPass = FullScreenPass::create(std::string(kShader), Program::DefineList().add("_INTEGRATE_DIFFUSE_LD"));
            return executeSingleMip(pContext, pPass, pTexture, pSampler, size, format, sampleCount);
        }

        Texture::SharedPtr integrateSpecularLD(RenderContext* pContext, const Texture::SharedPtr& pTexture, const Sampler::SharedPtr& pSampler, uint32_t size, ResourceFormat format, uint32_t sampleCount)
        {
            auto pPass = FullScreenPass::create(std::string(kShader), Program::DefineList().add("_INTEGRATE_SPECULAR_LD"));
            pPass["gInputTex"] = pTexture;
            pPass["gSampler"] = pSampler;
            pPass["DataCB"]["gSampleCount"] = sampleCount;

            Texture::SharedPtr pOutput = Texture::create2D(size, size, format, 1, Texture::kMaxPossible, nullptr, Resource::BindFlags::ShaderResource | Resource::BindFlags::RenderTarget);

            // Execute on each mip level
            uint32_t mipCount = pOutput->getMipCount();
            for (uint32_t i = 0; i < mipCount; i++)
            {
                Fbo::SharedPtr pFbo = Fbo::create();
                pFbo->attachColorTarget(pOutput, 0, i);

                // Roughness to integrate for on current mip level
                pPass["DataCB"]["gRoughness"] = float(i) / float(mipCount - 1);
                pPass->execute(pContext, pFbo);
            }

            return pOutput;
        }
    }

    EnvMapLighting::EnvMapLighting(RenderContext* pContext, const EnvMap::SharedPtr& pEnvMap, uint32_t diffSamples, uint32_t specSamples, uint32_t diffSize, uint32_t specSize, ResourceFormat preFilteredFormat)
        : mpEnvMap(pEnvMap)
        , mDiffSampleCount(diffSamples)
        , mSpecSampleCount(specSamples)
    {
        mpDFGSampler = Sampler::create(Sampler::Desc().setFilterMode(Sampler::Filter::Point, Sampler::Filter::Point, Sampler::Filter::Point).setAddressingMode(Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp));
        mpDFGTexture = integrateDFG(pContext, 128, ResourceFormat::RGBA16Float, 128);

        auto pEnvTexture = pEnvMap->getEnvMap();
        auto pEnvSampler = pEnvMap->getEnvSampler();

        mpSampler = Sampler::create(Sampler::Desc().setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear).setAddressingMode(Sampler::AddressMode::Wrap, Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp));
        mpDiffuseTexture = integrateDiffuseLD(pContext, pEnvTexture, pEnvSampler, diffSize, preFilteredFormat, diffSamples);
        mpSpecularTexture = integrateSpecularLD(pContext, pEnvTexture, pEnvSampler, specSize, preFilteredFormat, specSamples);
    }

    EnvMapLighting::SharedPtr EnvMapLighting::create(RenderContext* pContext, const EnvMap::SharedPtr& pEnvMap, uint32_t diffSampleCount, uint32_t specSampleCount, uint32_t diffSize, uint32_t specSize, ResourceFormat preFilteredFormat)
    {
        if (pEnvMap->getEnvMap()->getMipCount() == 1)
        {
            logWarning("Environment map texture sould have a valid mip chain.");
        }

        return SharedPtr(new EnvMapLighting(pContext, pEnvMap, diffSampleCount, specSampleCount, diffSize, specSize, preFilteredFormat));
    }

    void EnvMapLighting::setShaderData(const ShaderVar& var)
    {
        if(!var.isValid()) return;

        var["dfgTexture"] = mpDFGTexture;
        var["dfgSampler"] = mpDFGSampler;

        var["diffuseTexture"] = mpDiffuseTexture;
        var["specularTexture"] = mpSpecularTexture;
        var["sampler"] = mpSampler;
    }

    uint64_t EnvMapLighting::getMemoryUsageInBytes() const
    {
        uint64_t m = 0;
        m += mpDFGTexture->getTextureSizeInBytes();
        m += mpDiffuseTexture->getTextureSizeInBytes();
        m += mpSpecularTexture->getTextureSizeInBytes();
        return m;
    }
}
