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
    Texture::SharedPtr integrateDiffuse(RenderContext* pContext, const Texture::SharedPtr& pTexture, uint32_t size, ResourceFormat preFilteredFormat, uint32_t diffSampleCount)
    {
        static FullScreenPass::UniquePtr pIntegration;
        static GraphicsVars::SharedPtr pVars;
        static Sampler::SharedPtr pSampler;

        // Initialize fullscreen pass
        if (pIntegration == nullptr)
        {
            Program::DefineList defines;
            defines.add("_SAMPLE_COUNT", std::to_string(diffSampleCount));
            pIntegration = FullScreenPass::create("Framework/Shaders/LightProbeIntegration.ps.slang", defines);
            pVars = GraphicsVars::create(pIntegration->getProgram()->getActiveVersion()->getReflector());
            pSampler = Sampler::create(Sampler::Desc().setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear));
            pVars->getDefaultBlock()->setSampler("gSampler", pSampler);
        }

        // Set inputs
        pIntegration->getProgram()->addDefine("_SAMPLE_COUNT", std::to_string(diffSampleCount));
        pVars->getDefaultBlock()->setTexture("gInputTex", pTexture);

        // Output texture
        Fbo::SharedPtr pFbo = FboHelper::create2D(size, size, Fbo::Desc().setColorTarget(0, preFilteredFormat));

        // Execute
        GraphicsState::SharedPtr pState = pContext->getGraphicsState();
        pState->pushFbo(pFbo);
        pContext->pushGraphicsVars(pVars);
        pIntegration->execute(pContext);
        pContext->popGraphicsVars();
        pState->popFbo();

        return pFbo->getColorTexture(0);
    }

    LightProbe::LightProbe(RenderContext* pContext, const Texture::SharedPtr& pTexture, uint32_t size, uint32_t diffSampleCount, ResourceFormat preFilteredFormat)
        : mDiffSampleCount(diffSampleCount)
    {
        mData.resources.origTexture = pTexture;
        mData.resources.diffuseTexture = integrateDiffuse(pContext, pTexture, size, preFilteredFormat, diffSampleCount);
    }

    LightProbe::SharedPtr LightProbe::create(RenderContext* pContext, const std::string& filename, bool loadAsSrgb, bool generateMips, ResourceFormat overrideFormat, uint32_t size, uint32_t diffSampleCount, ResourceFormat preFilteredFormat)
    {
        Texture::SharedPtr pTexture;
        if (overrideFormat != ResourceFormat::Unknown)
        {
            Texture::SharedPtr pOrigTex = createTextureFromFile(filename, false, loadAsSrgb);
            pTexture = Texture::create2D(pOrigTex->getWidth(), pOrigTex->getHeight(), overrideFormat, 1, generateMips ? Texture::kMaxPossible : 1, nullptr, Resource::BindFlags::RenderTarget | Resource::BindFlags::ShaderResource);
            gpDevice->getRenderContext()->blit(pOrigTex->getSRV(0, 1, 0, 1), pTexture->getRTV(0, 0, 1));
            pTexture->generateMips(gpDevice->getRenderContext().get());
        }
        else
        {
            pTexture = createTextureFromFile(filename, generateMips, loadAsSrgb);
        }
        
        return create(pContext, pTexture, size, diffSampleCount, preFilteredFormat);
    }

    LightProbe::SharedPtr LightProbe::create(RenderContext* pContext, const Texture::SharedPtr& pTexture, uint32_t size, uint32_t diffSampleCount, ResourceFormat preFilteredFormat)
    {
        return SharedPtr(new LightProbe(pContext, pTexture, size, diffSampleCount, preFilteredFormat));
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
        check_offset(type);
        static_assert(kDataSize % sizeof(float) * 4 == 0, "LightProbeData size should be a multiple of 16");

        if (offset == ConstantBuffer::kInvalidOffset)
        {
            logWarning("LightProbe::setIntoProgramVars() - variable \"" + varName + "\"not found in constant buffer\n");
            return;
        }

        assert(offset + kDataSize <= pBuffer->getSize());

        // Set everything except for the material
        pBuffer->setBlob(&mData, offset, kDataSize);

        // Bind the textures
        pVars->setTexture(varName + ".resources.origTexture", mData.resources.origTexture);
        pVars->setTexture(varName + ".resources.diffuseTexture", mData.resources.diffuseTexture);
        pVars->setSampler(varName + ".resources.samplerState", mData.resources.samplerState);
    }
}
