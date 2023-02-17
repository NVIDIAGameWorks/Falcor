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
#include "GraphicsStateObject.h"
#include "Device.h"
#include "GFXHelpers.h"
#include "GFXAPI.h"

namespace Falcor
{
gfx::BlendFactor getGFXBlendFactor(BlendState::BlendFunc func)
{
    switch (func)
    {
    case Falcor::BlendState::BlendFunc::Zero:
        return gfx::BlendFactor::Zero;
    case Falcor::BlendState::BlendFunc::One:
        return gfx::BlendFactor::One;
    case Falcor::BlendState::BlendFunc::SrcColor:
        return gfx::BlendFactor::SrcColor;
    case Falcor::BlendState::BlendFunc::OneMinusSrcColor:
        return gfx::BlendFactor::InvSrcColor;
    case Falcor::BlendState::BlendFunc::DstColor:
        return gfx::BlendFactor::DestColor;
    case Falcor::BlendState::BlendFunc::OneMinusDstColor:
        return gfx::BlendFactor::InvDestColor;
    case Falcor::BlendState::BlendFunc::SrcAlpha:
        return gfx::BlendFactor::SrcAlpha;
    case Falcor::BlendState::BlendFunc::OneMinusSrcAlpha:
        return gfx::BlendFactor::InvSrcAlpha;
    case Falcor::BlendState::BlendFunc::DstAlpha:
        return gfx::BlendFactor::DestAlpha;
    case Falcor::BlendState::BlendFunc::OneMinusDstAlpha:
        return gfx::BlendFactor::InvDestAlpha;
    case Falcor::BlendState::BlendFunc::BlendFactor:
        return gfx::BlendFactor::BlendColor;
    case Falcor::BlendState::BlendFunc::OneMinusBlendFactor:
        return gfx::BlendFactor::InvBlendColor;
    case Falcor::BlendState::BlendFunc::SrcAlphaSaturate:
        return gfx::BlendFactor::SrcAlphaSaturate;
    case Falcor::BlendState::BlendFunc::Src1Color:
        return gfx::BlendFactor::SecondarySrcColor;
    case Falcor::BlendState::BlendFunc::OneMinusSrc1Color:
        return gfx::BlendFactor::InvSecondarySrcColor;
    case Falcor::BlendState::BlendFunc::Src1Alpha:
        return gfx::BlendFactor::SecondarySrcAlpha;
    case Falcor::BlendState::BlendFunc::OneMinusSrc1Alpha:
        return gfx::BlendFactor::InvSecondarySrcAlpha;
    default:
        FALCOR_UNREACHABLE();
        return gfx::BlendFactor::Zero;
    }
}

gfx::BlendOp getGFXBlendOp(BlendState::BlendOp op)
{
    switch (op)
    {
    case Falcor::BlendState::BlendOp::Add:
        return gfx::BlendOp::Add;
    case Falcor::BlendState::BlendOp::Subtract:
        return gfx::BlendOp::Subtract;
    case Falcor::BlendState::BlendOp::ReverseSubtract:
        return gfx::BlendOp::ReverseSubtract;
    case Falcor::BlendState::BlendOp::Min:
        return gfx::BlendOp::Min;
    case Falcor::BlendState::BlendOp::Max:
        return gfx::BlendOp::Max;
    default:
        FALCOR_UNREACHABLE();
        return gfx::BlendOp::Add;
    }
}

gfx::StencilOp getGFXStencilOp(DepthStencilState::StencilOp op)
{
    switch (op)
    {
    case Falcor::DepthStencilState::StencilOp::Keep:
        return gfx::StencilOp::Keep;
    case Falcor::DepthStencilState::StencilOp::Zero:
        return gfx::StencilOp::Zero;
    case Falcor::DepthStencilState::StencilOp::Replace:
        return gfx::StencilOp::Replace;
    case Falcor::DepthStencilState::StencilOp::Increase:
        return gfx::StencilOp::IncrementWrap;
    case Falcor::DepthStencilState::StencilOp::IncreaseSaturate:
        return gfx::StencilOp::IncrementSaturate;
    case Falcor::DepthStencilState::StencilOp::Decrease:
        return gfx::StencilOp::DecrementWrap;
    case Falcor::DepthStencilState::StencilOp::DecreaseSaturate:
        return gfx::StencilOp::DecrementSaturate;
    case Falcor::DepthStencilState::StencilOp::Invert:
        return gfx::StencilOp::Invert;
    default:
        FALCOR_UNREACHABLE();
        return gfx::StencilOp::Keep;
    }
}

gfx::ComparisonFunc getGFXComparisonFunc(ComparisonFunc func)
{
    switch (func)
    {
    case Falcor::ComparisonFunc::Disabled:
        return gfx::ComparisonFunc::Never;
    case Falcor::ComparisonFunc::Never:
        return gfx::ComparisonFunc::Never;
    case Falcor::ComparisonFunc::Always:
        return gfx::ComparisonFunc::Always;
    case Falcor::ComparisonFunc::Less:
        return gfx::ComparisonFunc::Less;
    case Falcor::ComparisonFunc::Equal:
        return gfx::ComparisonFunc::Equal;
    case Falcor::ComparisonFunc::NotEqual:
        return gfx::ComparisonFunc::NotEqual;
    case Falcor::ComparisonFunc::LessEqual:
        return gfx::ComparisonFunc::LessEqual;
    case Falcor::ComparisonFunc::Greater:
        return gfx::ComparisonFunc::Greater;
    case Falcor::ComparisonFunc::GreaterEqual:
        return gfx::ComparisonFunc::GreaterEqual;
    default:
        FALCOR_UNREACHABLE();
        return gfx::ComparisonFunc::Never;
    }
}

void getGFXStencilDesc(gfx::DepthStencilOpDesc& gfxDesc, DepthStencilState::StencilDesc desc)
{
    gfxDesc.stencilDepthFailOp = getGFXStencilOp(desc.depthFailOp);
    gfxDesc.stencilFailOp = getGFXStencilOp(desc.stencilFailOp);
    gfxDesc.stencilPassOp = getGFXStencilOp(desc.depthStencilPassOp);
    gfxDesc.stencilFunc = getGFXComparisonFunc(desc.func);
}

gfx::PrimitiveType getGFXPrimitiveType(GraphicsStateObject::PrimitiveType primitiveType)
{
    switch (primitiveType)
    {
    case Falcor::GraphicsStateObject::PrimitiveType::Undefined:
        return gfx::PrimitiveType::Triangle;
    case Falcor::GraphicsStateObject::PrimitiveType::Point:
        return gfx::PrimitiveType::Point;
    case Falcor::GraphicsStateObject::PrimitiveType::Line:
        return gfx::PrimitiveType::Line;
    case Falcor::GraphicsStateObject::PrimitiveType::Triangle:
        return gfx::PrimitiveType::Triangle;
    case Falcor::GraphicsStateObject::PrimitiveType::Patch:
        return gfx::PrimitiveType::Patch;
    default:
        FALCOR_UNREACHABLE();
        return gfx::PrimitiveType::Triangle;
    }
}

gfx::CullMode getGFXCullMode(RasterizerState::CullMode mode)
{
    switch (mode)
    {
    case Falcor::RasterizerState::CullMode::None:
        return gfx::CullMode::None;
    case Falcor::RasterizerState::CullMode::Front:
        return gfx::CullMode::Front;
    case Falcor::RasterizerState::CullMode::Back:
        return gfx::CullMode::Back;
    default:
        FALCOR_UNREACHABLE();
        return gfx::CullMode::None;
    }
}

gfx::FillMode getGFXFillMode(RasterizerState::FillMode mode)
{
    switch (mode)
    {
    case Falcor::RasterizerState::FillMode::Wireframe:
        return gfx::FillMode::Wireframe;
    case Falcor::RasterizerState::FillMode::Solid:
        return gfx::FillMode::Solid;
    default:
        FALCOR_UNREACHABLE();
        return gfx::FillMode::Solid;
    }
}

gfx::InputSlotClass getGFXInputSlotClass(VertexBufferLayout::InputClass cls)
{
    switch (cls)
    {
    case Falcor::VertexBufferLayout::InputClass::PerVertexData:
        return gfx::InputSlotClass::PerVertex;
    case Falcor::VertexBufferLayout::InputClass::PerInstanceData:
        return gfx::InputSlotClass::PerInstance;
    default:
        FALCOR_UNREACHABLE();
        return gfx::InputSlotClass::PerVertex;
    }
}

BlendState::SharedPtr GraphicsStateObject::spDefaultBlendState;
RasterizerState::SharedPtr GraphicsStateObject::spDefaultRasterizerState;
DepthStencilState::SharedPtr GraphicsStateObject::spDefaultDepthStencilState;

bool GraphicsStateObject::Desc::operator==(const GraphicsStateObject::Desc& other) const
{
    bool b = true;
    b = b && (mpLayout == other.mpLayout);
    b = b && (mFboDesc == other.mFboDesc);
    b = b && (mpProgram == other.mpProgram);
    b = b && (mSampleMask == other.mSampleMask);
    b = b && (mPrimType == other.mPrimType);

    if (mpRasterizerState)
    {
        b = b && (mpRasterizerState == other.mpRasterizerState);
    }
    else
    {
        b = b && (other.mpRasterizerState == nullptr || other.mpRasterizerState == spDefaultRasterizerState);
    }

    if (mpBlendState)
    {
        b = b && (mpBlendState == other.mpBlendState);
    }
    else
    {
        b = b && (other.mpBlendState == nullptr || other.mpBlendState == spDefaultBlendState);
    }

    if (mpDepthStencilState)
    {
        b = b && (mpDepthStencilState == other.mpDepthStencilState);
    }
    else
    {
        b = b && (other.mpDepthStencilState == nullptr || other.mpDepthStencilState == spDefaultDepthStencilState);
    }

    return b;
}

GraphicsStateObject::~GraphicsStateObject()
{
    mpDevice->releaseResource(mGfxPipelineState);
    mpDevice->releaseResource(mpGFXRenderPassLayout);
}

GraphicsStateObject::GraphicsStateObject(std::shared_ptr<Device> pDevice, const Desc& desc) : mpDevice(std::move(pDevice)), mDesc(desc)
{
    if (spDefaultBlendState == nullptr)
    {
        // Create default objects
        spDefaultBlendState = BlendState::create(BlendState::Desc());
        spDefaultDepthStencilState = DepthStencilState::create(DepthStencilState::Desc());
        spDefaultRasterizerState = RasterizerState::create(RasterizerState::Desc());
    }

    // Initialize default objects
    if (!mDesc.mpBlendState)
        mDesc.mpBlendState = spDefaultBlendState;
    if (!mDesc.mpRasterizerState)
        mDesc.mpRasterizerState = spDefaultRasterizerState;
    if (!mDesc.mpDepthStencilState)
        mDesc.mpDepthStencilState = spDefaultDepthStencilState;

    gfx::GraphicsPipelineStateDesc gfxDesc = {};
    // Set blend state.
    auto blendState = mDesc.getBlendState();
    FALCOR_ASSERT(blendState->getRtCount() <= gfx::kMaxRenderTargetCount);
    auto& targetBlendDescs = gfxDesc.blend.targets;
    {
        gfxDesc.blend.targetCount = blendState->getRtCount();
        for (gfx::UInt i = 0; i < gfxDesc.blend.targetCount; ++i)
        {
            auto& rtDesc = blendState->getRtDesc(i);
            auto& gfxRtDesc = targetBlendDescs[i];
            gfxRtDesc.enableBlend = rtDesc.blendEnabled;

            gfxRtDesc.alpha.dstFactor = getGFXBlendFactor(rtDesc.dstAlphaFunc);
            gfxRtDesc.alpha.srcFactor = getGFXBlendFactor(rtDesc.srcAlphaFunc);
            gfxRtDesc.alpha.op = getGFXBlendOp(rtDesc.alphaBlendOp);

            gfxRtDesc.color.dstFactor = getGFXBlendFactor(rtDesc.dstRgbFunc);
            gfxRtDesc.color.srcFactor = getGFXBlendFactor(rtDesc.srcRgbFunc);
            gfxRtDesc.color.op = getGFXBlendOp(rtDesc.rgbBlendOp);

            targetBlendDescs[i].writeMask = gfx::RenderTargetWriteMask::EnableNone;
            if (rtDesc.writeMask.writeAlpha)
                gfxRtDesc.writeMask |= gfx::RenderTargetWriteMask::EnableAlpha;
            if (rtDesc.writeMask.writeBlue)
                gfxRtDesc.writeMask |= gfx::RenderTargetWriteMask::EnableBlue;
            if (rtDesc.writeMask.writeGreen)
                gfxRtDesc.writeMask |= gfx::RenderTargetWriteMask::EnableGreen;
            if (rtDesc.writeMask.writeRed)
                gfxRtDesc.writeMask |= gfx::RenderTargetWriteMask::EnableRed;
        }
    }

    // Set depth stencil state.
    {
        auto depthStencilState = mDesc.getDepthStencilState();
        getGFXStencilDesc(gfxDesc.depthStencil.backFace, depthStencilState->getStencilDesc(Falcor::DepthStencilState::Face::Back));
        getGFXStencilDesc(gfxDesc.depthStencil.frontFace, depthStencilState->getStencilDesc(Falcor::DepthStencilState::Face::Front));
        gfxDesc.depthStencil.depthFunc = getGFXComparisonFunc(depthStencilState->getDepthFunc());
        gfxDesc.depthStencil.depthTestEnable = depthStencilState->isDepthTestEnabled();
        gfxDesc.depthStencil.depthWriteEnable = depthStencilState->isDepthWriteEnabled();
        gfxDesc.depthStencil.stencilEnable = depthStencilState->isStencilTestEnabled();
        gfxDesc.depthStencil.stencilReadMask = depthStencilState->getStencilReadMask();
        gfxDesc.depthStencil.stencilWriteMask = depthStencilState->getStencilWriteMask();
        gfxDesc.depthStencil.stencilRef = depthStencilState->getStencilRef();
    }

    // Set raterizer state.
    {
        auto rasterState = mDesc.getRasterizerState();
        gfxDesc.rasterizer.antialiasedLineEnable = rasterState->isLineAntiAliasingEnabled();
        gfxDesc.rasterizer.cullMode = getGFXCullMode(rasterState->getCullMode());
        gfxDesc.rasterizer.depthBias = rasterState->getDepthBias();
        gfxDesc.rasterizer.slopeScaledDepthBias = rasterState->getSlopeScaledDepthBias();
        gfxDesc.rasterizer.depthBiasClamp = 0.0f;
        gfxDesc.rasterizer.depthClipEnable = !rasterState->isDepthClampEnabled(); // Depth-clamp disables depth-clip
        gfxDesc.rasterizer.fillMode = getGFXFillMode(rasterState->getFillMode());
        gfxDesc.rasterizer.frontFace =
            rasterState->isFrontCounterCW() ? gfx::FrontFaceMode::CounterClockwise : gfx::FrontFaceMode::Clockwise;
        gfxDesc.rasterizer.multisampleEnable = mDesc.getFboDesc().getSampleCount() != 1;
        gfxDesc.rasterizer.scissorEnable = rasterState->isScissorTestEnabled();
        gfxDesc.rasterizer.enableConservativeRasterization = rasterState->isConservativeRasterizationEnabled();
        gfxDesc.rasterizer.forcedSampleCount = rasterState->getForcedSampleCount();
    }

    // Create input layout.
    {
        auto vertexLayout = mDesc.getVertexLayout();
        if (vertexLayout)
        {
            std::vector<gfx::VertexStreamDesc> vertexStreams(vertexLayout->getBufferCount());
            std::vector<gfx::InputElementDesc> inputElements;
            for (size_t i = 0; i < vertexLayout->getBufferCount(); ++i)
            {
                auto& bufferLayout = mDesc.getVertexLayout()->getBufferLayout(i);
                vertexStreams[i].instanceDataStepRate = bufferLayout->getInstanceStepRate();
                vertexStreams[i].slotClass = getGFXInputSlotClass(bufferLayout->getInputClass());
                vertexStreams[i].stride = bufferLayout->getStride();

                for (uint32_t j = 0; j < bufferLayout->getElementCount(); ++j)
                {
                    gfx::InputElementDesc elementDesc = {};
                    gfx::VertexStreamDesc vertexStreamDesc = {};

                    elementDesc.format = getGFXFormat(bufferLayout->getElementFormat(j));
                    elementDesc.offset = bufferLayout->getElementOffset(j);
                    elementDesc.semanticName = bufferLayout->getElementName(j).c_str();
                    elementDesc.bufferSlotIndex = static_cast<gfx::GfxIndex>(i);

                    for (uint32_t arrayIndex = 0; arrayIndex < bufferLayout->getElementArraySize(j); arrayIndex++)
                    {
                        elementDesc.semanticIndex = arrayIndex;
                        inputElements.push_back(elementDesc);
                        elementDesc.offset += getFormatBytesPerBlock(bufferLayout->getElementFormat(j));
                    }
                }
            }

            gfx::IInputLayout::Desc inputLayoutDesc = {};
            inputLayoutDesc.inputElementCount = static_cast<gfx::GfxCount>(inputElements.size());
            inputLayoutDesc.inputElements = inputElements.data();
            inputLayoutDesc.vertexStreamCount = static_cast<gfx::GfxCount>(vertexStreams.size());
            inputLayoutDesc.vertexStreams = vertexStreams.data();
            FALCOR_GFX_CALL(mpDevice->getGfxDevice()->createInputLayout(inputLayoutDesc, mpGFXInputLayout.writeRef()));
        }
        gfxDesc.inputLayout = mpGFXInputLayout;
    }

    // Create framebuffer layout.
    gfx::IFramebufferLayout::Desc gfxFbDesc = {};
    {
        auto fboDesc = mDesc.getFboDesc();
        gfx::IFramebufferLayout::TargetLayout depthAttachment = {};
        std::vector<gfx::IFramebufferLayout::TargetLayout> attachments(Fbo::getMaxColorTargetCount());
        if (mDesc.getFboDesc().getDepthStencilFormat() != ResourceFormat::Unknown)
        {
            depthAttachment.format = getGFXFormat(fboDesc.getDepthStencilFormat());
            depthAttachment.sampleCount = fboDesc.getSampleCount();
            gfxFbDesc.depthStencil = &depthAttachment;
        }
        for (uint32_t i = 0; i < (uint32_t)attachments.size(); ++i)
        {
            attachments[i].format = getGFXFormat(fboDesc.getColorTargetFormat(i));
            attachments[i].sampleCount = fboDesc.getSampleCount();
            if (attachments[i].format != gfx::Format::Unknown)
            {
                gfxFbDesc.renderTargetCount = i + 1;
            }
        }
        gfxFbDesc.renderTargets = attachments.data();
        FALCOR_GFX_CALL(mpDevice->getGfxDevice()->createFramebufferLayout(gfxFbDesc, mpGFXFramebufferLayout.writeRef()));
        gfxDesc.framebufferLayout = mpGFXFramebufferLayout;
    }

    // Create render pass layout.
    {
        gfx::IRenderPassLayout::Desc renderPassDesc = {};
        gfx::IRenderPassLayout::TargetAccessDesc depthAccess = {};
        depthAccess.initialState = gfx::ResourceState::DepthWrite;
        depthAccess.finalState = gfx::ResourceState::DepthWrite;
        depthAccess.loadOp = gfx::IRenderPassLayout::TargetLoadOp::Load;
        depthAccess.stencilLoadOp = gfx::IRenderPassLayout::TargetLoadOp::Load;
        depthAccess.stencilStoreOp = gfx::IRenderPassLayout::TargetStoreOp::Store;
        depthAccess.storeOp = gfx::IRenderPassLayout::TargetStoreOp::Store;
        if (this->mDesc.getFboDesc().getDepthStencilFormat() != ResourceFormat::Unknown)
        {
            renderPassDesc.depthStencilAccess = &depthAccess;
        }
        renderPassDesc.framebufferLayout = mpGFXFramebufferLayout.get();
        renderPassDesc.renderTargetCount = gfxFbDesc.renderTargetCount;
        std::vector<gfx::IRenderPassLayout::TargetAccessDesc> colorAccesses(renderPassDesc.renderTargetCount);
        for (auto& colorAccess : colorAccesses)
        {
            colorAccess.initialState = gfx::ResourceState::RenderTarget;
            colorAccess.finalState = gfx::ResourceState::RenderTarget;
            colorAccess.loadOp = gfx::IRenderPassLayout::TargetLoadOp::Load;
            colorAccess.storeOp = gfx::IRenderPassLayout::TargetStoreOp::Store;
        }
        renderPassDesc.renderTargetAccess = colorAccesses.data();
        FALCOR_GFX_CALL(mpDevice->getGfxDevice()->createRenderPassLayout(renderPassDesc, mpGFXRenderPassLayout.writeRef()));
    }

    gfxDesc.primitiveType = getGFXPrimitiveType(mDesc.getPrimitiveType());
    gfxDesc.program = mDesc.getProgramKernels()->getGfxProgram();

    FALCOR_GFX_CALL(mpDevice->getGfxDevice()->createGraphicsPipelineState(gfxDesc, mGfxPipelineState.writeRef()));
}

GraphicsStateObject::SharedPtr GraphicsStateObject::create(Device* pDevice, const Desc& desc)
{
    return SharedPtr(new GraphicsStateObject(pDevice->shared_from_this(), desc));
}
} // namespace Falcor
