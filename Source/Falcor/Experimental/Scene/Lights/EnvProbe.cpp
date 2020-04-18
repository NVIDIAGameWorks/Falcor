/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
 **************************************************************************/
#include "stdafx.h"
#include "EnvProbe.h"
#include "glm/gtc/integer.hpp"

namespace Falcor
{
    namespace
    {
        const char kShaderFilenameSetup[] = "Experimental/Scene/Lights/EnvProbeSetup.cs.slang";

        // The defaults are 512x512 @ 64spp in the resampling step.
        const uint32_t kDefaultDimension = 512;
        const uint32_t kDefaultSpp = 64;

        // Default variable name used by setShaderData().
        const char kDefaultCbVar[] = "gEnvProbe";
    }

    EnvProbe::SharedPtr EnvProbe::create(RenderContext* pRenderContext, const std::string& filename)
    {
        SharedPtr ptr = SharedPtr(new EnvProbe());
        return ptr->init(pRenderContext, filename) ? ptr : nullptr;
    }

    bool EnvProbe::setShaderData(const ShaderVar& var) const
    {
        assert(var.isValid());

        // Set variables.
        float2 invDim = 1.f / float2(mpImportanceMap->getWidth(), mpImportanceMap->getHeight());
        if (!var["importanceBaseMip"].set(mpImportanceMap->getMipCount() - 1)) return false;   // The base mip is 1x1 texels
        if (!var["importanceInvDim"].set(invDim)) return false;

        // Bind resources.
        if (!var["envMap"].setTexture(mpEnvMap) ||
            !var["importanceMap"].setTexture(mpImportanceMap) ||
            !var["envSampler"].setSampler(mpEnvSampler) ||
            !var["importanceSampler"].setSampler(mpImportanceSampler))
        {
            return false;
        }

        return true;
    }

    bool EnvProbe::init(RenderContext* pRenderContext, const std::string& filename)
    {
        // Create compute program for the setup phase.
        mpSetupPass = ComputePass::create(kShaderFilenameSetup, "main");

        // Create sampler.
        // The lat-long map wraps around horizontally, but not vertically. Set the sampler to only wrap in U.
        Sampler::Desc samplerDesc;
        samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear);
        samplerDesc.setAddressingMode(Sampler::AddressMode::Wrap, Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp);
        mpEnvSampler = Sampler::create(samplerDesc);
        samplerDesc.setFilterMode(Sampler::Filter::Point, Sampler::Filter::Point, Sampler::Filter::Point);
        samplerDesc.setAddressingMode(Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp);
        mpImportanceSampler = Sampler::create(samplerDesc);

        // Load environment map from file. Set it to generate mips and use linear color.
        mpEnvMap = Texture::createFromFile(filename, true, false);
        if (!mpEnvMap)
        {
            logError("EnvProbe::init() - Failed to load texture " + filename);
            return false;
        }

        // Create hierarchical importance map for sampling.
        if (!createImportanceMap(pRenderContext, kDefaultDimension, kDefaultSpp))
        {
            logError("EnvProbe::init() - Failed to create importance map" + filename);
            return false;
        }

        return true;
    }

    bool EnvProbe::createImportanceMap(RenderContext* pRenderContext, uint32_t dimension, uint32_t samples)
    {
        assert(isPowerOf2(dimension));
        assert(isPowerOf2(samples));

        // We create log2(N)+1 mips from NxN...1x1 texels resolution.
        uint32_t mips = glm::log2(dimension) + 1;
        assert((1u << (mips - 1)) == dimension);
        assert(mips > 1 && mips <= 12);     // Shader constant limits max resolution, increase if needed.

        // Create importance map. We have to set the RTV flag to be able to use generateMips().
        mpImportanceMap = Texture::create2D(dimension, dimension, ResourceFormat::R32Float, 1, mips, nullptr, Resource::BindFlags::ShaderResource | Resource::BindFlags::RenderTarget | Resource::BindFlags::UnorderedAccess);
        assert(mpImportanceMap);

        mpSetupPass["gEnvMap"] = mpEnvMap;
        mpSetupPass["gImportanceMap"] = mpImportanceMap;
        mpSetupPass["gEnvSampler"] = mpEnvSampler;

        uint32_t samplesX = std::max(1u, (uint32_t)std::sqrt(samples));
        uint32_t samplesY = samples / samplesX;
        assert(samples == samplesX * samplesY);

        mpSetupPass["CB"]["outputDim"] = uint2(dimension);
        mpSetupPass["CB"]["outputDimInSamples"] = uint2(dimension * samplesX, dimension * samplesY);
        mpSetupPass["CB"]["numSamples"] = uint2(samplesX, samplesY);
        mpSetupPass["CB"]["invSamples"] = 1.f / (samplesX * samplesY);

        // Execute setup pass to compute the square importance map (base mip).
        mpSetupPass->execute(pRenderContext, dimension, dimension);

        // Populate mip hierarchy. We rely on the default mip generation for this.
        mpImportanceMap->generateMips(pRenderContext);

        return true;
    }

}
