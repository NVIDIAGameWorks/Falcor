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
#include "TextRenderer.h"
#include "Core/API/RenderContext.h"

namespace Falcor
{
    namespace
    {
        struct Vertex
        {
            float2 screenPos;
            float2 texCoord;
        };

        const float2 kVertexPos[] =
        {
            float2(0, 0),
            float2(0, 1),
            float2(1, 0),

            float2(1, 0),
            float2(0, 1),
            float2(1, 1),
        };

        const uint32_t kMaxCharCount = 1000;

        Vao::SharedPtr createVAO(const Buffer::SharedPtr& pVB)
        {
            VertexLayout::SharedPtr pLayout = VertexLayout::create();
            VertexBufferLayout::SharedPtr pBufLayout = VertexBufferLayout::create();
            pBufLayout->addElement("POSITION", 0, ResourceFormat::RG32Float, 1, 0);
            pBufLayout->addElement("TEXCOORD", 8, ResourceFormat::RG32Float, 1, 1);
            pLayout->addBufferLayout(0, pBufLayout);
            Vao::BufferVec buffers{ pVB };

            return Vao::create(Vao::Topology::TriangleList, pLayout, buffers);
        }

        struct TextData
        {
            bool init = false;
            TextRenderer::Flags flags = TextRenderer::Flags::Shadowed;
            float3 color = float3(1, 1, 1);
            Buffer::SharedPtr pVb;
            RasterPass::SharedPtr pPass;
            Font::UniquePtr pFont;
        } gTextData;

        void setCbData(const Fbo::SharedPtr& pDstFbo)
        {
            float width = (float)pDstFbo->getWidth();
            float height = (float)pDstFbo->getHeight();

            // Set the matrix
            glm::mat4 vpTransform;
            vpTransform[0][0] = 2 / width;
            vpTransform[1][1] = -2 / height;
            vpTransform[3][0] = -1;
            vpTransform[3][1] = 1;
#ifdef FALCOR_VK
            vpTransform[1][1] *= -1.0f;
            vpTransform[3][1] *= -1.0f;
#endif

            // Update the program variables
            gTextData.pPass["PerFrameCB"]["gvpTransform"] = vpTransform;
            gTextData.pPass["PerFrameCB"]["gFontColor"] = gTextData.color;
        }

        void renderText(RenderContext* pRenderContext, const std::string& text, const Fbo::SharedPtr& pDstFbo, float2 pos)
        {
            // Make sure we enough space for the next char
            assert(text.size() < kMaxCharCount);
            setCbData(pDstFbo);
            Vertex* verts = (Vertex*)gTextData.pVb->map(Buffer::MapType::WriteDiscard);

            float startX = pos.x;
            uint32_t vertexCount = 0; // Not the same as text.size(), since some special characters are ignored

            // Create the vertex-buffer
            for (const auto& c : text)
            {
                if (c == '\n')
                {
                    pos.y += gTextData.pFont->getFontHeight();
                    pos.x = startX;
                }
                else if (c == '\t') pos.x += gTextData.pFont->getTabWidth();
                else if (c == ' ') pos.x += gTextData.pFont->getLettersSpacing();
                else
                {
                    // Regular character
                    const Font::CharTexCrdDesc& desc = gTextData.pFont->getCharDesc(c);
                    for (uint32_t i = 0; i < arraysize(kVertexPos); i++, vertexCount++)
                    {
                        float2 posScale = kVertexPos[i];
                        float2 charPos = desc.size * posScale;
                        charPos += pos;
                        verts[vertexCount].screenPos = charPos;
                        verts[vertexCount].texCoord = desc.topLeft + desc.size * kVertexPos[i];
                    }
                    pos.x += gTextData.pFont->getLettersSpacing();
                }
            }

            // Submit
            gTextData.pVb->unmap();
            gTextData.pPass->getState()->setFbo(pDstFbo);
            gTextData.pPass->draw(pRenderContext, vertexCount, 0);
        }
    }

    const float3& TextRenderer::getColor() { return gTextData.color; }
    void TextRenderer::setColor(const float3& color) { gTextData.color = color; }
    TextRenderer::Flags TextRenderer::getFlags() { return gTextData.flags; }
    void TextRenderer::setFlags(Flags f) { gTextData.flags = f; }

    void TextRenderer::start()
    {
        if (gTextData.init) return;

        static const std::string kShaderFile("Utils/UI/TextRenderer.slang");

        // Create a vertex buffer
        const uint32_t vbSize = (uint32_t)(sizeof(Vertex)*kMaxCharCount*arraysize(kVertexPos));
        gTextData.pVb = Buffer::create(vbSize, Buffer::BindFlags::Vertex, Buffer::CpuAccess::Write, nullptr);

        // Create the RenderState
        gTextData.pPass = RasterPass::create(kShaderFile, "vs", "ps");
        auto& pState = gTextData.pPass->getState();
        pState->setVao(createVAO(gTextData.pVb));

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
        blendDesc.setRtBlend(0, true).setRtParams(0, BlendState::BlendOp::Add,
            BlendState::BlendOp::Add,
            BlendState::BlendFunc::SrcAlpha,
            BlendState::BlendFunc::OneMinusSrcAlpha,
            BlendState::BlendFunc::One,
            BlendState::BlendFunc::One);
        pState->setBlendState(BlendState::create(blendDesc));
        gTextData.pFont = Font::create();

        // Initialize the buffer
        gTextData.pPass["gFontTex"] = gTextData.pFont->getTexture();

        gTextData.init = true;
    }

    void TextRenderer::shutdown()
    {
        gTextData = {};
    }

    void TextRenderer::render(RenderContext* pRenderContext, const std::string& text, const Fbo::SharedPtr& pDstFbo, float2 pos)
    {
        if (is_set(gTextData.flags, TextRenderer::Flags::Shadowed))
        {
            float3 oldColor = getColor();
            setColor(float3(0));
            renderText(pRenderContext, text, pDstFbo, pos + float2(1));
            setColor(oldColor);
        }
        renderText(pRenderContext, text, pDstFbo, pos);
    }
}
