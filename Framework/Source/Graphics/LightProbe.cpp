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
#include "Framework.h"
#include "LightProbe.h"
#include "API/Device.h"
#include "TextureHelper.h"
#include "Utils/Gui.h"
#include "Graphics/FboHelper.h"

namespace Falcor
{
    uint32_t LightProbe::sLightProbeCount = 0;
    LightProbeSharedResources LightProbe::sSharedData;

    class PreIntegration
    {
    public:
        const char* kShader = "Framework/Shaders/LightProbeIntegration.ps.slang";

        bool isInitialized() const { return mInitialized; }

        void init()
        {
            mpDiffuseLDPass = FullScreenPass::create(std::string(kShader), Program::DefineList().add("_INTEGRATE_DIFFUSE_LD"));
            mpSpecularLDPass = FullScreenPass::create(std::string(kShader), Program::DefineList().add("_INTEGRATE_SPECULAR_LD"));
            mpDFGPass = FullScreenPass::create(std::string(kShader), Program::DefineList().add("_INTEGRATE_DFG"));

            // Shared
            mpVars = GraphicsVars::create(mpDiffuseLDPass->getProgram()->getReflector());
            mpSampler = Sampler::create(Sampler::Desc().setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear));
            mpVars->getDefaultBlock()->setSampler("gSampler", mpSampler);

            mInitialized = true;
        }

        void release()
        {
            mpDiffuseLDPass = nullptr;
            mpSpecularLDPass = nullptr;
            mpDFGPass = nullptr;
            mpVars = nullptr;
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
            mpVars->getDefaultBlock()->setTexture("gInputTex", pTexture);
            mpVars["DataCB"]["gSampleCount"] = sampleCount;

            Texture::SharedPtr pOutput = Texture::create2D(size, size, format, 1, Texture::kMaxPossible, nullptr, Resource::BindFlags::ShaderResource | Resource::BindFlags::RenderTarget);

            GraphicsState::SharedPtr pState = pContext->getGraphicsState();
            pContext->pushGraphicsVars(mpVars);
            // Execute on each mip level
            uint32_t mipCount = pOutput->getMipCount();
            for (uint32_t i = 0; i < mipCount; i++)
            {
                Fbo::SharedPtr pFbo = Fbo::create();
                pFbo->attachColorTarget(pOutput, 0, i);

                // Roughness to integrate for on current mip level
                mpVars["DataCB"]["gRoughness"] = float(i) / float(mipCount - 1);

                pState->pushFbo(pFbo);
                mpSpecularLDPass->execute(pContext);
                pState->popFbo();
            }

            pContext->popGraphicsVars();
            return pOutput;
        }

    private:

        Texture::SharedPtr executeSingleMip(RenderContext* pContext, const FullScreenPass::UniquePtr& pPass, const Texture::SharedPtr& pTexture, uint32_t size, ResourceFormat format, uint32_t sampleCount)
        {
            mpVars->getDefaultBlock()->setTexture("gInputTex", pTexture);
            mpVars["DataCB"]["gSampleCount"] = sampleCount;

            // Output texture
            Fbo::SharedPtr pFbo = FboHelper::create2D(size, size, Fbo::Desc().setColorTarget(0, format));

            // Execute
            GraphicsState::SharedPtr pState = pContext->getGraphicsState();
            pState->pushFbo(pFbo);
            pContext->pushGraphicsVars(mpVars);
            pPass->execute(pContext);
            pContext->popGraphicsVars();
            pState->popFbo();

            return pFbo->getColorTexture(0);
        }


        bool mInitialized = false;
        FullScreenPass::UniquePtr mpDiffuseLDPass;
        FullScreenPass::UniquePtr mpSpecularLDPass;
        FullScreenPass::UniquePtr mpDFGPass;
        GraphicsVars::SharedPtr mpVars;
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
            sSharedData.dfgTexture = sIntegration.integrateDFG(pContext, pTexture, 128, ResourceFormat::RGBA16Float, 128);
            sSharedData.dfgSampler = Sampler::create(Sampler::Desc().setFilterMode(Sampler::Filter::Point, Sampler::Filter::Point, Sampler::Filter::Point).setAddressingMode(Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp));
        }

        mData.resources.origTexture = pTexture;
        mData.resources.diffuseTexture = sIntegration.integrateDiffuseLD(pContext, pTexture, diffSize, preFilteredFormat, diffSamples);
        mData.resources.specularTexture = sIntegration.integrateSpecularLD(pContext, pTexture, specSize, preFilteredFormat, specSamples);
        sLightProbeCount++;
    }

    LightProbe::~LightProbe()
    {
        sLightProbeCount--;
        if (sLightProbeCount == 0)
        {
            sSharedData.dfgTexture = nullptr;
            sSharedData.dfgSampler = nullptr;
            sIntegration.release();
        }
    }

    LightProbe::SharedPtr LightProbe::create(RenderContext* pContext, const std::string& filename, bool loadAsSrgb, ResourceFormat overrideFormat, uint32_t diffSampleCount, uint32_t specSampleCount, uint32_t diffSize, uint32_t specSize, ResourceFormat preFilteredFormat)
    {
        Texture::SharedPtr pTexture;
        if (overrideFormat != ResourceFormat::Unknown)
        {
            Texture::SharedPtr pOrigTex = createTextureFromFile(filename, false, loadAsSrgb);
            pTexture = Texture::create2D(pOrigTex->getWidth(), pOrigTex->getHeight(), overrideFormat, 1, Texture::kMaxPossible, nullptr, Resource::BindFlags::RenderTarget | Resource::BindFlags::ShaderResource);
            pTexture->setSourceFilename(pOrigTex->getSourceFilename());
            gpDevice->getRenderContext()->blit(pOrigTex->getSRV(0, 1, 0, 1), pTexture->getRTV(0, 0, 1));
            pTexture->generateMips(gpDevice->getRenderContext().get());
        }
        else
        {
            pTexture = createTextureFromFile(filename, true, loadAsSrgb);
        }

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
        if (group == nullptr || pGui->beginGroup(group))
        {
            pGui->addFloat3Var("World Position", mData.posW, -FLT_MAX, FLT_MAX);

            float intensity = mData.intensity.r;
            if (pGui->addFloatVar("Intensity", intensity, 0.0f))
            {
                mData.intensity = vec3(intensity);
            }

            pGui->addFloatVar("Radius", mData.radius, -1.0f);

            if (group != nullptr)
            {
                pGui->endGroup();
            }
        }
    }

    void LightProbe::move(const glm::vec3& position, const glm::vec3& target, const glm::vec3& up)
    {
        logWarning("Light probes don't support paths. Expect absolutely nothing to happen");
    }

    static bool checkOffset(size_t cbOffset, size_t cppOffset, const char* field)
    {
        if (cbOffset != cppOffset)
        {
            logError("LightProbe::setIntoProgramVars() = LightProbeData::" + std::string(field) + " CB offset mismatch. CB offset is " + std::to_string(cbOffset) + ", C++ data offset is " + std::to_string(cppOffset));
            return false;
        }
        return true;
    }

#if _LOG_ENABLED
#define check_offset(_a) {static bool b = true; if(b) {assert(checkOffset(pBuffer->getVariableOffset(varName + '.' + #_a) - offset, offsetof(LightProbeData, _a), #_a));} b = false;}
#else
#define check_offset(_a)
#endif

    void LightProbe::setIntoProgramVars(ProgramVars* pVars, ConstantBuffer* pBuffer, const std::string& varName)
    {
        size_t offset = pBuffer->getVariableOffset(varName);

        // Set the data into the constant buffer
        check_offset(posW);
        check_offset(intensity);
        static_assert(kDataSize % sizeof(vec4) == 0, "LightProbeData size should be a multiple of 16");

        if (offset == ConstantBuffer::kInvalidOffset)
        {
            logWarning("LightProbe::setIntoProgramVars() - variable \"" + varName + "\"not found in constant buffer\n");
            return;
        }

        assert(offset + kDataSize <= pBuffer->getSize());

        // Set everything except for the resources
        pBuffer->setBlob(&mData, offset, kDataSize);

        // Bind the textures
        pVars->setTexture(varName + ".resources.origTexture", mData.resources.origTexture);
        pVars->setTexture(varName + ".resources.diffuseTexture", mData.resources.diffuseTexture);
        pVars->setTexture(varName + ".resources.specularTexture", mData.resources.specularTexture);
        pVars->setSampler(varName + ".resources.sampler", mData.resources.sampler);
    }

    void LightProbe::setCommonIntoProgramVars(ProgramVars* pVars, const std::string& varName)
    {
        pVars->setTexture(varName + ".dfgTexture", sSharedData.dfgTexture);
        pVars->setSampler(varName + ".dfgSampler", sSharedData.dfgSampler);
    }
}
