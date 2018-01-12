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

namespace Falcor
{
    LightProbe::LightProbe(const Texture::SharedPtr& pTexture, uint32_t size, ResourceFormat format, MipFilter mipFilter)
    {
        mData.type = LightProbeLinear2D;

        // Create the texture
        assert(pTexture->getType() == Texture::Type::Texture2D);
        uint32_t mipLevels = (mipFilter == MipFilter::None) ? 1 : Texture::kMaxPossible;
        Texture::BindFlags bindFlags = Texture::BindFlags::ShaderResource;
        if (mipFilter != MipFilter::None)
        {
            bindFlags |= Texture::BindFlags::RenderTarget;
        }
        mData.resources.diffuseProbe2D = Texture::create2D(size, size, format, 1, mipLevels, nullptr, bindFlags);
        if (mipFilter == MipFilter::PreIntegration)
        {
            should_not_get_here();
            //            mpSpecularTex = Texture::create2D(size, size, format, 1, mipLevels, nullptr, bindFlags);
        }
        else
        {
            mData.resources.specularProbe2D = mData.resources.diffuseProbe2D;
        }

        RenderContext* pContext = gpDevice->getRenderContext().get();
        pContext->blit(pTexture->getSRV(), mData.resources.diffuseProbe2D->getRTV());
        // Filter
        if (mipFilter == MipFilter::Linear)
        {
            mData.resources.diffuseProbe2D->generateMips(pContext);
        }
    }

    LightProbe::SharedPtr LightProbe::create(const std::string& filename, uint32_t size, bool loadAsSrgb, ResourceFormat format, MipFilter mipFilter)
    {
        Texture::SharedPtr pTexture = createTextureFromFile(filename, false, loadAsSrgb);
        return create(pTexture, size, format, mipFilter);
    }

    LightProbe::SharedPtr LightProbe::create(const Texture::SharedPtr& pTexture, uint32_t size, ResourceFormat format, MipFilter mipFilter)
    {
        return SharedPtr(new LightProbe(pTexture, size, format, mipFilter));
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
        pVars->setTexture(varName + ".resources.diffuseProbe2D", mData.resources.diffuseProbe2D);
        pVars->setTexture(varName + ".resources.specularProbe2D", mData.resources.specularProbe2D);
        pVars->setSampler(varName + ".resources.samplerState", mData.resources.samplerState);
    }
}
