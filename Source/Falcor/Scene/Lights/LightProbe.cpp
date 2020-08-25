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
#include "stdafx.h"
#include "LightProbe.h"
#include "RenderGraph/BasePasses/FullScreenPass.h"
#include "Utils/UI/Gui.h"
#include "Core/API/RenderContext.h"
#include "Core/API/Device.h"

namespace Falcor
{
    uint32_t LightProbe::sLightProbeCount = 0;
    LightProbeSharedResources LightProbe::sSharedResources;

    class PreIntegration
    {
    public:
        const char* kShader = "Scene/Lights/LightProbeIntegration.ps.slang";

        bool isInitialized() const { return mInitialized; }

        void init()
        {
            mpDiffuseLDPass = FullScreenPass::create(std::string(kShader), Program::DefineList().add("_INTEGRATE_DIFFUSE_LD"));
            mpSpecularLDPass = FullScreenPass::create(std::string(kShader), Program::DefineList().add("_INTEGRATE_SPECULAR_LD"));
            mpDFGPass = FullScreenPass::create(std::string(kShader), Program::DefineList().add("_INTEGRATE_DFG"));

            // Shared
            mpSampler = Sampler::create(Sampler::Desc().setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear));
            mpDiffuseLDPass["gSampler"] = mpSampler;
            mpSpecularLDPass["gSampler"] = mpSampler;
            mpDFGPass["gSampler"] = mpSampler;

            mInitialized = true;
        }

        void release()
        {
            mpDiffuseLDPass = nullptr;
            mpSpecularLDPass = nullptr;
            mpDFGPass = nullptr;
            mpSampler = nullptr;

            mInitialized = false;
        }

        Texture::SharedPtr integrateDFG(RenderContext* pContext, const Texture::SharedPtr& pTexture, uint32_t size, ResourceFormat format, uint32_t sampleCount)
        {
            return executeSingleMip(pContext, mpDFGPass, pTexture, size, format, sampleCount);
        }

        Texture::SharedPtr integrateDiffuseLD(RenderContext* pContext, const Texture::SharedPtr& pTexture, uint32_t size, ResourceFormat format, uint32_t sampleCount)
        {
            return executeSingleMip(pContext, mpDiffuseLDPass, pTexture, size, format, sampleCount);
        }

        Texture::SharedPtr integrateSpecularLD(RenderContext* pContext, const Texture::SharedPtr& pTexture, uint32_t size, ResourceFormat format, uint32_t sampleCount)
        {
            mpSpecularLDPass["gInputTex"] = pTexture;
            mpSpecularLDPass["DataCB"]["gSampleCount"] = sampleCount;

            Texture::SharedPtr pOutput = Texture::create2D(size, size, format, 1, Texture::kMaxPossible, nullptr, Resource::BindFlags::ShaderResource | Resource::BindFlags::RenderTarget);

            // Execute on each mip level
            uint32_t mipCount = pOutput->getMipCount();
            for (uint32_t i = 0; i < mipCount; i++)
            {
                Fbo::SharedPtr pFbo = Fbo::create();
                pFbo->attachColorTarget(pOutput, 0, i);

                // Roughness to integrate for on current mip level
                mpSpecularLDPass["DataCB"]["gRoughness"] = float(i) / float(mipCount - 1);
                mpSpecularLDPass->execute(pContext, pFbo);
            }

            return pOutput;
        }

    private:

        Texture::SharedPtr executeSingleMip(RenderContext* pContext, const FullScreenPass::SharedPtr& pPass, const Texture::SharedPtr& pTexture, uint32_t size, ResourceFormat format, uint32_t sampleCount)
        {
            pPass["gInputTex"] = pTexture;
            pPass["DataCB"]["gSampleCount"] = sampleCount;

            // Output texture
            Fbo::SharedPtr pFbo = Fbo::create2D(size, size, Fbo::Desc().setColorTarget(0, format));

            // Execute
            pPass->execute(pContext, pFbo);
            return pFbo->getColorTexture(0);
        }


        bool mInitialized = false;
        FullScreenPass::SharedPtr mpDiffuseLDPass;
        FullScreenPass::SharedPtr mpSpecularLDPass;
        FullScreenPass::SharedPtr mpDFGPass;
        Sampler::SharedPtr mpSampler;
    };

    static PreIntegration sIntegration;

    LightProbe::LightProbe(RenderContext* pContext, const Texture::SharedPtr& pTexture, uint32_t diffSamples, uint32_t specSamples, uint32_t diffSize, uint32_t specSize, ResourceFormat preFilteredFormat)
        : mDiffSampleCount(diffSamples)
        , mSpecSampleCount(specSamples)
    {
        if (sIntegration.isInitialized() == false)
        {
            assert(sLightProbeCount == 0);
            sIntegration.init();
            sSharedResources.dfgTexture = sIntegration.integrateDFG(pContext, pTexture, 128, ResourceFormat::RGBA16Float, 128);
            sSharedResources.dfgSampler = Sampler::create(Sampler::Desc().setFilterMode(Sampler::Filter::Point, Sampler::Filter::Point, Sampler::Filter::Point).setAddressingMode(Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp));
        }

        mData.resources.origTexture = pTexture;
        mData.resources.diffuseTexture = sIntegration.integrateDiffuseLD(pContext, pTexture, diffSize, preFilteredFormat, diffSamples);
        mData.resources.specularTexture = sIntegration.integrateSpecularLD(pContext, pTexture, specSize, preFilteredFormat, specSamples);
        mData.sharedResources = sSharedResources;
        sLightProbeCount++;
    }

    LightProbe::~LightProbe()
    {
        sLightProbeCount--;
        if (sLightProbeCount == 0)
        {
            sSharedResources.dfgTexture = nullptr;
            sSharedResources.dfgSampler = nullptr;
            sIntegration.release();
        }
    }

    LightProbe::SharedPtr LightProbe::create(RenderContext* pContext, const std::string& filename, bool loadAsSrgb, ResourceFormat overrideFormat, uint32_t diffSampleCount, uint32_t specSampleCount, uint32_t diffSize, uint32_t specSize, ResourceFormat preFilteredFormat)
    {
        assert(gpDevice);
        Texture::SharedPtr pTexture;
        if (overrideFormat != ResourceFormat::Unknown)
        {
            Texture::SharedPtr pOrigTex = Texture::createFromFile(filename, false, loadAsSrgb);
            if (pOrigTex)
            {
                pTexture = Texture::create2D(pOrigTex->getWidth(), pOrigTex->getHeight(), overrideFormat, 1, Texture::kMaxPossible, nullptr, Resource::BindFlags::RenderTarget | Resource::BindFlags::ShaderResource);
                pTexture->setSourceFilename(pOrigTex->getSourceFilename());
                gpDevice->getRenderContext()->blit(pOrigTex->getSRV(0, 1, 0, 1), pTexture->getRTV(0, 0, 1));
                pTexture->generateMips(gpDevice->getRenderContext());
            }
        }
        else
        {
            pTexture = Texture::createFromFile(filename, true, loadAsSrgb);
        }

        if (!pTexture) throw std::exception("Failed to create light probe");

        return create(pContext, pTexture, diffSampleCount, specSampleCount, diffSize, specSize, preFilteredFormat);
    }

    LightProbe::SharedPtr LightProbe::create(RenderContext* pContext, const Texture::SharedPtr& pTexture, uint32_t diffSampleCount, uint32_t specSampleCount, uint32_t diffSize, uint32_t specSize, ResourceFormat preFilteredFormat)
    {
        if (pTexture->getMipCount() == 1)
        {
            logWarning("Source textures used for generating light probes should have a valid mip chain.");
        }

        return SharedPtr(new LightProbe(pContext, pTexture, diffSampleCount, specSampleCount, diffSize, specSize, preFilteredFormat));
    }

    void LightProbe::renderUI(Gui* pGui, const char* group)
    {
        Gui::Group g(pGui, group);
        if (!group || g.open())
        {
            g.var("World Position", mData.posW, -FLT_MAX, FLT_MAX);

            float intensity = mData.intensity.r;
            if (g.var("Intensity", intensity, 0.0f))
            {
                mData.intensity = float3(intensity);
            }

            g.var("Radius", mData.radius, -1.0f);

            if (g.open()) g.release();
        }
    }

    static bool checkOffset(UniformShaderVarOffset cbOffset, size_t cppOffset, const char* field)
    {
        if (cbOffset.getByteOffset() != cppOffset)
        {
            logError("LightProbe::setShaderData() = LightProbeData::" + std::string(field) + " CB offset mismatch. CB offset is " + std::to_string(cbOffset.getByteOffset()) + ", C++ data offset is " + std::to_string(cppOffset));
            return false;
        }
        return true;
    }

#if _LOG_ENABLED
#define check_offset(_a) {static bool b = true; if(b) {assert(checkOffset(var.getType()->getMemberOffset(#_a), offsetof(LightProbeData, _a), #_a));} b = false;}
#else
#define check_offset(_a)
#endif

    void LightProbe::setShaderData(const ShaderVar& var)
    {

        // Set the data into the constant buffer
        check_offset(posW);
        check_offset(intensity);
        static_assert(kDataSize % sizeof(float4) == 0, "LightProbeData size should be a multiple of 16");

        if(!var.isValid()) return;

        // Set everything except for the resources
        var.setBlob(&mData, kDataSize);

        // Bind the textures
        auto resources = var["resources"];
        resources["origTexture"] = mData.resources.origTexture;
        resources["diffuseTexture"] = mData.resources.diffuseTexture;
        resources["specularTexture"] = mData.resources.specularTexture;
        resources["sampler"] = mData.resources.sampler;

        auto sharedResources = var["sharedResources"];
        sharedResources["dfgTexture"] = mData.sharedResources.dfgTexture;
        sharedResources["dfgSampler"] = mData.sharedResources.dfgSampler;
    }

    SCRIPT_BINDING(LightProbe)
    {
        pybind11::class_<LightProbe, LightProbe::SharedPtr>(m, "LightProbe");
    }
}
