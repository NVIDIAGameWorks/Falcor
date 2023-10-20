/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
#include "TextRenderer.h"
#include "Core/API/RenderContext.h"
#include "Core/Pass/RasterPass.h"
#include "Utils/UI/Font.h"

namespace Falcor
{
namespace
{
struct Vertex
{
    float2 screenPos;
    float2 texCoord;
};

const float2 kVertexPos[] = {
    float2(0, 0),
    float2(0, 1),
    float2(1, 0),

    float2(1, 0),
    float2(0, 1),
    float2(1, 1),
};

const uint32_t kMaxCharCount = 1000;

ref<Vao> createVAO(const ref<Buffer>& pVB)
{
    ref<VertexLayout> pLayout = VertexLayout::create();
    ref<VertexBufferLayout> pBufLayout = VertexBufferLayout::create();
    pBufLayout->addElement("POSITION", 0, ResourceFormat::RG32Float, 1, 0);
    pBufLayout->addElement("TEXCOORD", 8, ResourceFormat::RG32Float, 1, 1);
    pLayout->addBufferLayout(0, pBufLayout);
    Vao::BufferVec buffers{pVB};

    return Vao::create(Vao::Topology::TriangleList, pLayout, buffers);
}

} // namespace

TextRenderer::TextRenderer(ref<Device> pDevice) : mpDevice(pDevice)
{
    for (uint32_t i = 0; i < kVaoCount; ++i)
    {
        // Create a vertex buffer
        const uint32_t vbSize = (uint32_t)(sizeof(Vertex) * kMaxCharCount * std::size(kVertexPos));
        ref<Buffer> pVb = mpDevice->createBuffer(vbSize, ResourceBindFlags::Vertex, MemoryType::Upload, nullptr);
        mpVaos[i] = createVAO(pVb);
    }

    // Create the RenderState
    mpPass = RasterPass::create(mpDevice, "Utils/UI/TextRenderer.3d.slang", "vsMain", "psMain");
    auto& pState = mpPass->getState();

    // create the depth-state
    DepthStencilState::Desc dsDesc;
    dsDesc.setDepthEnabled(false);
    pState->setDepthStencilState(DepthStencilState::create(dsDesc));

    // Rasterizer state
    RasterizerState::Desc rsState;
    rsState.setCullMode(RasterizerState::CullMode::None);
    pState->setRasterizerState(RasterizerState::create(rsState));

    // Blend state
    BlendState::Desc blendDesc;
    blendDesc.setRtBlend(0, true).setRtParams(
        0,
        BlendState::BlendOp::Add,
        BlendState::BlendOp::Add,
        BlendState::BlendFunc::SrcAlpha,
        BlendState::BlendFunc::OneMinusSrcAlpha,
        BlendState::BlendFunc::One,
        BlendState::BlendFunc::One
    );
    pState->setBlendState(BlendState::create(blendDesc));
    mpFont = std::make_unique<Font>(mpDevice, getRuntimeDirectory() / "data/framework/fonts/dejavu-sans-mono-14");

    // Initialize the buffer
    mpPass->getRootVar()["gFontTex"] = mpFont->getTexture();
}

TextRenderer::~TextRenderer() = default;

void TextRenderer::render(RenderContext* pRenderContext, const std::string& text, const ref<Fbo>& pDstFbo, float2 pos)
{
    if (is_set(mFlags, TextRenderer::Flags::Shadowed))
    {
        float3 oldColor = getColor();
        setColor(float3(0));
        renderText(pRenderContext, text, pDstFbo, pos + float2(1));
        setColor(oldColor);
    }
    renderText(pRenderContext, text, pDstFbo, pos);
}

void TextRenderer::setCbData(const ref<Fbo>& pDstFbo)
{
    float width = (float)pDstFbo->getWidth();
    float height = (float)pDstFbo->getHeight();

    // Set the matrix
    float4x4 vpTransform = float4x4::identity();
    vpTransform[0][0] = 2 / width;
    vpTransform[1][1] = -2 / height;
    vpTransform[0][3] = -1;
    vpTransform[1][3] = 1;
#ifdef FALCOR_FLIP_Y
    vpTransform[1][1] *= -1.0f;
    vpTransform[3][1] *= -1.0f;
#endif
    // Update the program variables
    auto var = mpPass->getRootVar()["PerFrameCB"];
    var["gvpTransform"] = vpTransform;
    var["gFontColor"] = mColor;
}

void TextRenderer::renderText(RenderContext* pRenderContext, const std::string& text, const ref<Fbo>& pDstFbo, float2 pos)
{
    // Make sure we enough space for the next char
    FALCOR_ASSERT(text.size() < kMaxCharCount);
    setCbData(pDstFbo);
    const auto& pVao = mpVaos[mVaoIndex];
    const auto& pVb = pVao->getVertexBuffer(0);
    Vertex* verts = reinterpret_cast<Vertex*>(pVb->map(Buffer::MapType::Write));

    float startX = pos.x;
    uint32_t vertexCount = 0; // Not the same as text.size(), since some special characters are ignored

    // Create the vertex-buffer
    for (const auto& c : text)
    {
        if (c == '\n')
        {
            pos.y += mpFont->getFontHeight();
            pos.x = startX;
        }
        else if (c == '\t')
            pos.x += mpFont->getTabWidth();
        else if (c == ' ')
            pos.x += mpFont->getLettersSpacing();
        else
        {
            // Regular character
            const Font::CharTexCrdDesc& desc = mpFont->getCharDesc(c);
            for (uint32_t i = 0; i < std::size(kVertexPos); i++, vertexCount++)
            {
                float2 posScale = kVertexPos[i];
                float2 charPos = desc.size * posScale;
                charPos += pos;
                verts[vertexCount].screenPos = charPos;
                verts[vertexCount].texCoord = desc.topLeft + desc.size * kVertexPos[i];
            }
            pos.x += mpFont->getLettersSpacing();
        }
    }

    // Submit
    pVb->unmap();
    mpPass->getState()->setVao(pVao);
    mpPass->getState()->setFbo(pDstFbo);
    mpPass->draw(pRenderContext, vertexCount, 0);

    mVaoIndex = (mVaoIndex + 1) % kVaoCount;
}

} // namespace Falcor
