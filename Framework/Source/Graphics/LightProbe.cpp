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

namespace Falcor
{
    LightProbe::LightProbe(const Texture::SharedPtr& pTexture, PreFilterMode filter, uint32_t size, ResourceFormat preFilteredFormat)
    {
        assert(filter == PreFilterMode::None);
        mData.type = LightProbeLinear2D;
        mData.resources.origTexture = pTexture;
    }

    glm::vec3 dummyIntensityData;
    glm::vec3 & LightProbe::getIntensityData()
    {
        return dummyIntensityData;
    }

    LightProbe::SharedPtr LightProbe::create(const std::string& filename, bool loadAsSrgb, bool generateMips, ResourceFormat overrideFormat, PreFilterMode filter, uint32_t size, ResourceFormat preFilteredFormat)
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
        
        return create(pTexture, filter, size, preFilteredFormat);
    }

    LightProbe::SharedPtr LightProbe::create(const Texture::SharedPtr& pTexture, PreFilterMode filter, uint32_t size, ResourceFormat preFilteredFormat)
    {
        return SharedPtr(new LightProbe(pTexture, filter, size, preFilteredFormat));
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

    uint32_t LightProbe::getType() const
    {
        return mData.type;
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

    void LightProbe::setIntoParameterBlock(ParameterBlock * pBlock, ConstantBuffer * pBuffer, size_t offset, const std::string& lightVarName)
    {
        auto varName = lightVarName + ".probeData";
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
        pBlock->setTexture(varName + ".resources.origTexture", mData.resources.origTexture);
    }
    
    void LightProbe::setIntoProgramVars(ProgramVars* pVars, ConstantBuffer* pBuffer, const std::string& varName)
    {
        size_t offset = pBuffer->getVariableOffset(varName);
        setIntoParameterBlock(pVars->getDefaultBlock().get(), pBuffer, offset, varName);
    }

    void LightProbe::setIntoParameterBlock(ParameterBlock * pBlock, size_t offset, const std::string & varName)
    {
        auto pBuffer = pBlock->getConstantBuffer(pBlock->getReflection()->getName()).get();
        setIntoParameterBlock(pBlock, pBuffer, offset, varName);
    }
}
