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
#include "TextureAnalyzer.h"
#include "Core/API/RenderContext.h"

namespace Falcor
{
    namespace
    {
        // Verify channel flags use same bit pattern as the shader.
        static_assert((uint32_t)TextureChannelFlags::Red == 0x1);
        static_assert((uint32_t)TextureChannelFlags::Green == 0x2);
        static_assert((uint32_t)TextureChannelFlags::Blue == 0x4);
        static_assert((uint32_t)TextureChannelFlags::Alpha == 0x8);

        const char kShaderFilename[] = "Utils/Image/TextureAnalyzer.cs.slang";
    }

    // Verify that the result struct matches the size expected by the shader.
    static_assert(sizeof(TextureAnalyzer::Result) == 64, "TextureAnalyzer::Result struct size mismatch");

    size_t TextureAnalyzer::getResultSize() { return sizeof(TextureAnalyzer::Result); }

    TextureAnalyzer::SharedPtr TextureAnalyzer::create()
    {
        return SharedPtr(new TextureAnalyzer());
    }

    TextureAnalyzer::TextureAnalyzer()
    {
        mpClearPass = ComputePass::create(kShaderFilename, "clear");
        mpAnalyzePass = ComputePass::create(kShaderFilename, "analyze");
    }

    void TextureAnalyzer::analyze(RenderContext* pRenderContext, const Texture::SharedPtr pInput, uint32_t mipLevel, uint32_t arraySlice, Buffer::SharedPtr pResult, uint64_t resultOffset, bool clearResult)
    {
        FALCOR_ASSERT(pRenderContext && pInput);
        FALCOR_ASSERT(pResult && resultOffset + getResultSize() <= pResult->getSize());
        FALCOR_ASSERT(resultOffset < std::numeric_limits<uint32_t>::max());

        checkFormatSupport(pInput, mipLevel, arraySlice);

        if (clearResult)
        {
            clear(pRenderContext, pResult, resultOffset, 1);
        }

        auto pSRV = pInput->getSRV(mipLevel, 1, arraySlice, 1);
        FALCOR_ASSERT(pSRV);

        uint2 dim = { pInput->getWidth(mipLevel), pInput->getHeight(mipLevel) };
        FALCOR_ASSERT(dim.x > 0 && dim.y > 0);

        // Bind resources.
        auto var = mpAnalyzePass->getRootVar()["gTextureAnalyzer"];
        var["input"].setSrv(pSRV);
        var["result"] = pResult;
        var["resultOffset"] = (uint32_t)resultOffset;
        var["inputDim"] = dim;

        mpAnalyzePass->execute(pRenderContext, uint3(dim, 1));
    }

    void TextureAnalyzer::analyze(RenderContext* pRenderContext, const std::vector<Texture::SharedPtr>& inputs, Buffer::SharedPtr pResult, bool clearResult)
    {
        FALCOR_ASSERT(pRenderContext && !inputs.empty());
        FALCOR_ASSERT(pResult && inputs.size() * getResultSize() <= pResult->getSize());

        if (clearResult)
        {
            clear(pRenderContext, pResult, 0, inputs.size());
        }

        // Iterate over the textures to analyze them one by one.
        // Note that Falcor inserts a UAV barrier between each dispatch. This is unnecessary as the writes are non-overlapping.
        // TODO: Update this code when there is a an interface for disabling UAV barriers.
        for (size_t i = 0; i < inputs.size(); i++)
        {
            analyze(pRenderContext, inputs[i], 0, 0, pResult, i * getResultSize(), false);
        }
    }

    void TextureAnalyzer::clear(RenderContext* pRenderContext, Buffer::SharedPtr pResult, uint64_t resultOffset, size_t resultCount) const
    {
        FALCOR_ASSERT(pRenderContext);
        FALCOR_ASSERT(pResult && resultOffset + resultCount * getResultSize() <= pResult->getSize());
        FALCOR_ASSERT(resultCount > 0 && resultOffset < std::numeric_limits<uint32_t>::max());

        // Bind resources.
        auto var = mpClearPass->getRootVar()["gTextureAnalyzer"];
        var["result"] = pResult;
        var["resultOffset"] = (uint32_t)resultOffset;
        var["inputDim"] = uint2((uint32_t)resultCount, 1);

        mpClearPass->execute(pRenderContext, uint3(resultCount, 1, 1));
    }

    void TextureAnalyzer::checkFormatSupport(const Texture::SharedPtr pInput, uint32_t mipLevel, uint32_t arraySlice) const
    {
        // Validate that input is supported.
        if (pInput->getDepth() > 1)
        {
            throw RuntimeError("3D textures are not supported");
        }
        if (mipLevel >= pInput->getMipCount() || arraySlice >= pInput->getArraySize())
        {
            throw RuntimeError("Mip level and/or array slice is out of range");
        }
        if (pInput->getSampleCount() != 1)
        {
            throw RuntimeError("Multi-sampled textures are not supported");
        }

        auto format = pInput->getFormat();
        switch (getFormatType(format))
        {
        case FormatType::Float:
        case FormatType::Snorm:
        case FormatType::Unorm:
        case FormatType::UnormSrgb:
            break;
        case FormatType::Sint:
        case FormatType::Uint:
            throw RuntimeError("Format {} is not supported", to_string(format));
        default:
            FALCOR_ASSERT(false);
            throw RuntimeError("Unknown format type");
        }
    }
}
