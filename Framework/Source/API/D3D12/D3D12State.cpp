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
#include "D3D12State.h"
#include "API/BlendState.h"
#include "API/RasterizerState.h"
#include "API/VertexLayout.h"
#include "glm/gtc/type_ptr.hpp"

namespace Falcor
{
    D3D12_BLEND getD3D12BlendFunc(BlendState::BlendFunc func)
    {
        switch (func)
        {
        case BlendState::BlendFunc::Zero:
            return D3D12_BLEND_ZERO;
        case BlendState::BlendFunc::One:
            return D3D12_BLEND_ONE;
        case BlendState::BlendFunc::SrcColor:
            return D3D12_BLEND_SRC_COLOR;
        case BlendState::BlendFunc::OneMinusSrcColor:
            return D3D12_BLEND_INV_SRC_COLOR;
        case BlendState::BlendFunc::DstColor:
            return D3D12_BLEND_DEST_COLOR;
        case BlendState::BlendFunc::OneMinusDstColor:
            return D3D12_BLEND_INV_DEST_COLOR;
        case BlendState::BlendFunc::SrcAlpha:
            return D3D12_BLEND_SRC_ALPHA;
        case BlendState::BlendFunc::OneMinusSrcAlpha:
            return D3D12_BLEND_INV_SRC_ALPHA;
        case BlendState::BlendFunc::DstAlpha:
            return D3D12_BLEND_DEST_ALPHA;
        case BlendState::BlendFunc::OneMinusDstAlpha:
            return D3D12_BLEND_INV_DEST_ALPHA;
        case BlendState::BlendFunc::BlendFactor:
            return D3D12_BLEND_BLEND_FACTOR;
        case BlendState::BlendFunc::OneMinusBlendFactor:
            return D3D12_BLEND_INV_BLEND_FACTOR;
        case BlendState::BlendFunc::SrcAlphaSaturate:
            return D3D12_BLEND_SRC_ALPHA_SAT;
        case BlendState::BlendFunc::Src1Color:
            return D3D12_BLEND_INV_SRC1_COLOR;
        case BlendState::BlendFunc::OneMinusSrc1Color:
            return D3D12_BLEND_INV_SRC1_COLOR;
        case BlendState::BlendFunc::Src1Alpha:
            return D3D12_BLEND_SRC1_ALPHA;
        case BlendState::BlendFunc::OneMinusSrc1Alpha:
            return D3D12_BLEND_INV_SRC1_ALPHA;
        default:
            should_not_get_here();
            return (D3D12_BLEND)0;
        }

    }

    D3D12_BLEND_OP getD3D12BlendOp(BlendState::BlendOp op)
    {
        switch (op)
        {
        case BlendState::BlendOp::Add:
            return D3D12_BLEND_OP_ADD;
        case BlendState::BlendOp::Subtract:
            return D3D12_BLEND_OP_SUBTRACT;
        case BlendState::BlendOp::ReverseSubtract:
            return D3D12_BLEND_OP_REV_SUBTRACT;
        case BlendState::BlendOp::Min:
            return D3D12_BLEND_OP_MIN;
        case BlendState::BlendOp::Max:
            return D3D12_BLEND_OP_MAX;
        default:
            return (D3D12_BLEND_OP)0;
        }
    }

    void initD3D12BlendDesc(const BlendState* pState, D3D12_BLEND_DESC& desc)
    {
        desc.AlphaToCoverageEnable = dxBool(pState->isAlphaToCoverageEnabled());
        desc.IndependentBlendEnable = dxBool(pState->isIndependentBlendEnabled());
        for (uint32_t rt = 0; rt < pState->getRtCount(); rt++)
        {
            const BlendState::Desc::RenderTargetDesc& rtDesc = pState->getRtDesc(rt);
            D3D12_RENDER_TARGET_BLEND_DESC& d3dRtDesc = desc.RenderTarget[rt];

            d3dRtDesc.BlendEnable = dxBool(rtDesc.blendEnabled);
            d3dRtDesc.SrcBlend = getD3D12BlendFunc(rtDesc.srcRgbFunc);
            d3dRtDesc.DestBlend = getD3D12BlendFunc(rtDesc.dstRgbFunc);
            d3dRtDesc.BlendOp = getD3D12BlendOp(rtDesc.rgbBlendOp);
            d3dRtDesc.SrcBlendAlpha = getD3D12BlendFunc(rtDesc.srcAlphaFunc);
            d3dRtDesc.DestBlendAlpha = getD3D12BlendFunc(rtDesc.dstAlphaFunc);
            d3dRtDesc.BlendOpAlpha = getD3D12BlendOp(rtDesc.alphaBlendOp);

            d3dRtDesc.RenderTargetWriteMask = rtDesc.writeMask.writeRed ? D3D12_COLOR_WRITE_ENABLE_RED : 0;
            d3dRtDesc.RenderTargetWriteMask |= rtDesc.writeMask.writeGreen ? D3D12_COLOR_WRITE_ENABLE_GREEN : 0;
            d3dRtDesc.RenderTargetWriteMask |= rtDesc.writeMask.writeBlue ? D3D12_COLOR_WRITE_ENABLE_BLUE : 0;
            d3dRtDesc.RenderTargetWriteMask |= rtDesc.writeMask.writeAlpha ? D3D12_COLOR_WRITE_ENABLE_ALPHA : 0;
        }
    }

    D3D12_FILL_MODE getD3D12FillMode(RasterizerState::FillMode fill)
    {
        switch (fill)
        {
        case RasterizerState::FillMode::Wireframe:
            return D3D12_FILL_MODE_WIREFRAME;
        case RasterizerState::FillMode::Solid:
            return D3D12_FILL_MODE_SOLID;
        default:
            should_not_get_here();
            return (D3D12_FILL_MODE)0;
        }
    }

    D3D12_CULL_MODE getD3D12CullMode(RasterizerState::CullMode cull)
    {
        switch (cull)
        {
        case Falcor::RasterizerState::CullMode::None:
            return D3D12_CULL_MODE_NONE;
        case Falcor::RasterizerState::CullMode::Front:
            return D3D12_CULL_MODE_FRONT;
        case Falcor::RasterizerState::CullMode::Back:
            return D3D12_CULL_MODE_BACK;
        default:
            should_not_get_here();
            return (D3D12_CULL_MODE)0;
        }
    }

    void initD3D12RasterizerDesc(const RasterizerState* pState, D3D12_RASTERIZER_DESC& desc)
    {
        desc = {};
        desc.FillMode = getD3D12FillMode(pState->getFillMode());
        desc.CullMode = getD3D12CullMode(pState->getCullMode());
        desc.FrontCounterClockwise = dxBool(pState->isFrontCounterCW());
        desc.DepthBias = pState->getDepthBias();
        desc.DepthBiasClamp = 0;
        desc.SlopeScaledDepthBias = pState->getSlopeScaledDepthBias();
        desc.DepthClipEnable = dxBool(!pState->isDepthClampEnabled()); // Depth-clamp disables depth-clip
                                                                       // Set the line anti-aliasing mode
        desc.AntialiasedLineEnable = dxBool(pState->isLineAntiAliasingEnabled());
        desc.MultisampleEnable = desc.AntialiasedLineEnable;

        desc.ConservativeRaster = pState->isConservativeRasterizationEnabled() ? D3D12_CONSERVATIVE_RASTERIZATION_MODE_ON : D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;
        desc.ForcedSampleCount = pState->getForcedSampleCount();
    }

    D3D12_INPUT_CLASSIFICATION getD3D12InputClass(VertexBufferLayout::InputClass inClass)
    {
        switch (inClass)
        {
        case VertexBufferLayout::InputClass::PerVertexData:
            return D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA;
        case VertexBufferLayout::InputClass::PerInstanceData:
            return D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA;
        default:
            should_not_get_here();
            return (D3D12_INPUT_CLASSIFICATION) - 1;
        }
    }

    void initD3D12VertexLayout(const VertexLayout* pLayout, InputLayoutDesc& layoutDesc)
    {
        layoutDesc.elements.clear();
        layoutDesc.names.clear();

        for (size_t vb = 0; vb < pLayout->getBufferCount(); vb++)
        {
            auto& pVB = pLayout->getBufferLayout(vb);
            if (pVB)
            {
                for (uint32_t elemID = 0; elemID < pVB->getElementCount(); elemID++)
                {
                    D3D12_INPUT_ELEMENT_DESC element;
                    element.AlignedByteOffset = pVB->getElementOffset(elemID);
                    element.Format = getDxgiFormat(pVB->getElementFormat(elemID));
                    element.InputSlot = (uint32_t)vb;
                    element.InputSlotClass = getD3D12InputClass(pVB->getInputClass());
                    element.InstanceDataStepRate = pVB->getInstanceStepRate();
                    const auto& SemanticName = pVB->getElementName(elemID);
                    layoutDesc.names.push_back(std::make_unique<char[]>(SemanticName.size() + 1));
                    char* name = layoutDesc.names.back().get();
                    memcpy(name, SemanticName.c_str(), SemanticName.size());
                    name[SemanticName.size()] = 0;

                    for (uint32_t arrayIndex = 0; arrayIndex < pVB->getElementArraySize(elemID); arrayIndex++)
                    {
                        element.SemanticName = name;
                        element.SemanticIndex = arrayIndex;
                        layoutDesc.elements.push_back(element);

                        element.AlignedByteOffset += getFormatBytesPerBlock(pVB->getElementFormat(elemID));
                    }
                }
            }
        }
    }

    template<typename FalcorType>
    D3D12_COMPARISON_FUNC getD3D12ComparisonFunc(FalcorType func)
    {
        switch (func)
        {
        case FalcorType::Disabled:
            return D3D12_COMPARISON_FUNC_ALWAYS;
        case FalcorType::Never:
            return D3D12_COMPARISON_FUNC_NEVER;
        case FalcorType::Always:
            return D3D12_COMPARISON_FUNC_ALWAYS;
        case FalcorType::Less:
            return D3D12_COMPARISON_FUNC_LESS;
        case FalcorType::Equal:
            return D3D12_COMPARISON_FUNC_EQUAL;
        case FalcorType::NotEqual:
            return D3D12_COMPARISON_FUNC_NOT_EQUAL;
        case FalcorType::LessEqual:
            return D3D12_COMPARISON_FUNC_LESS_EQUAL;
        case FalcorType::Greater:
            return D3D12_COMPARISON_FUNC_GREATER;
        case FalcorType::GreaterEqual:
            return D3D12_COMPARISON_FUNC_GREATER_EQUAL;
        default:
            should_not_get_here();
            return (D3D12_COMPARISON_FUNC)0;
        }
    }

    D3D12_STENCIL_OP getD3D12StencilOp(DepthStencilState::StencilOp op)
    {
        switch (op)
        {
        case DepthStencilState::StencilOp::Keep:
            return D3D12_STENCIL_OP_KEEP;
        case DepthStencilState::StencilOp::Zero:
            return D3D12_STENCIL_OP_ZERO;
        case DepthStencilState::StencilOp::Replace:
            return D3D12_STENCIL_OP_REPLACE;
        case DepthStencilState::StencilOp::Increase:
            return D3D12_STENCIL_OP_INCR;
        case DepthStencilState::StencilOp::IncreaseSaturate:
            return D3D12_STENCIL_OP_INCR_SAT;
        case DepthStencilState::StencilOp::Decrease:
            return D3D12_STENCIL_OP_DECR;
        case DepthStencilState::StencilOp::DecreaseSaturate:
            return D3D12_STENCIL_OP_DECR_SAT;
        case DepthStencilState::StencilOp::Invert:
            return D3D12_STENCIL_OP_INVERT;
        default:
            should_not_get_here();
            return (D3D12_STENCIL_OP)0;
        }
    }

    D3D12_DEPTH_STENCILOP_DESC getD3D12StencilOpDesc(const DepthStencilState::StencilDesc& desc)
    {
        D3D12_DEPTH_STENCILOP_DESC dxDesc;
        dxDesc.StencilFunc = getD3D12ComparisonFunc(desc.func);
        dxDesc.StencilDepthFailOp = getD3D12StencilOp(desc.depthFailOp);
        dxDesc.StencilFailOp = getD3D12StencilOp(desc.stencilFailOp);
        dxDesc.StencilPassOp = getD3D12StencilOp(desc.depthStencilPassOp);

        return dxDesc;
    }

    void initD3DDepthStencilDesc(const DepthStencilState* pState, D3D12_DEPTH_STENCIL_DESC& desc)
    {
        desc.DepthEnable = dxBool(pState->isDepthTestEnabled());
        desc.DepthFunc = getD3D12ComparisonFunc(pState->getDepthFunc());
        desc.DepthWriteMask = pState->isDepthWriteEnabled() ? D3D12_DEPTH_WRITE_MASK_ALL : D3D12_DEPTH_WRITE_MASK_ZERO;
        desc.StencilEnable = dxBool(pState->isStencilTestEnabled());
        desc.StencilReadMask = pState->getStencilReadMask();
        desc.StencilWriteMask = pState->getStencilWriteMask();
        desc.FrontFace = getD3D12StencilOpDesc(pState->getStencilDesc(DepthStencilState::Face::Front));
        desc.BackFace = getD3D12StencilOpDesc(pState->getStencilDesc(DepthStencilState::Face::Back));
    }

    D3D12_FILTER_TYPE getFilterType(Sampler::Filter filter)
    {
        switch (filter)
        {
        case Sampler::Filter::Point:
            return D3D12_FILTER_TYPE_POINT;
        case Sampler::Filter::Linear:
            return D3D12_FILTER_TYPE_LINEAR;
        default:
            should_not_get_here();
            return (D3D12_FILTER_TYPE)-1;
        }
    }

    D3D12_FILTER getD3D12Filter(Sampler::Filter minFilter, Sampler::Filter magFilter, Sampler::Filter mipFilter, bool isComparison, bool isAnisotropic)
    {
        D3D12_FILTER filter;
        D3D12_FILTER_REDUCTION_TYPE reduction = isComparison ? D3D12_FILTER_REDUCTION_TYPE_COMPARISON : D3D12_FILTER_REDUCTION_TYPE_STANDARD;

        if (isAnisotropic)
        {
            filter = D3D12_ENCODE_ANISOTROPIC_FILTER(reduction);
        }
        else
        {
            D3D12_FILTER_TYPE dxMin = getFilterType(minFilter);
            D3D12_FILTER_TYPE dxMag = getFilterType(magFilter);
            D3D12_FILTER_TYPE dxMip = getFilterType(mipFilter);
            filter = D3D12_ENCODE_BASIC_FILTER(dxMin, dxMag, dxMip, reduction);
        }

        return filter;
    };

    D3D12_TEXTURE_ADDRESS_MODE getD3D12AddressMode(Sampler::AddressMode mode)
    {
        switch (mode)
        {
        case Sampler::AddressMode::Wrap:
            return D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        case Sampler::AddressMode::Mirror:
            return D3D12_TEXTURE_ADDRESS_MODE_MIRROR;
        case Sampler::AddressMode::Clamp:
            return D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        case Sampler::AddressMode::Border:
            return D3D12_TEXTURE_ADDRESS_MODE_BORDER;
        case Sampler::AddressMode::MirrorOnce:
            return D3D12_TEXTURE_ADDRESS_MODE_MIRROR_ONCE;
        default:
            should_not_get_here();
            return (D3D12_TEXTURE_ADDRESS_MODE)-1;
        }
    }

    void initD3D12SamplerDesc(const Sampler* pSampler, D3D12_SAMPLER_DESC& desc)
    {
        desc.Filter = getD3D12Filter(pSampler->getMinFilter(), pSampler->getMagFilter(), pSampler->getMipFilter(), (pSampler->getComparisonMode() != Sampler::ComparisonMode::Disabled), (pSampler->getMaxAnisotropy() > 1));
        desc.AddressU = getD3D12AddressMode(pSampler->getAddressModeU());
        desc.AddressV = getD3D12AddressMode(pSampler->getAddressModeV());
        desc.AddressW = getD3D12AddressMode(pSampler->getAddressModeW());
        desc.MipLODBias = pSampler->getLodBias();;
        desc.MaxAnisotropy = pSampler->getMaxAnisotropy();
        desc.ComparisonFunc = getD3D12ComparisonFunc(pSampler->getComparisonMode());
        desc.MinLOD = pSampler->getMinLod();
        desc.MaxLOD = pSampler->getMaxLod();

        const glm::vec4& borderColor = pSampler->getBorderColor();
        memcpy(desc.BorderColor, glm::value_ptr(borderColor), sizeof(borderColor));
    }
}