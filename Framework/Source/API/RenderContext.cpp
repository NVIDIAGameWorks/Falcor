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
#include "RenderContext.h"
#include "RasterizerState.h"
#include "DepthStencilState.h"
#include "BlendState.h"
#include "FBO.h"
#include "Device.h"

namespace Falcor
{
    CommandSignatureHandle RenderContext::spDrawCommandSig = nullptr;
    CommandSignatureHandle RenderContext::spDrawIndexCommandSig = nullptr;

    RenderContext::RenderContext()
    {
        if (spDrawCommandSig == nullptr)
        {
            initDrawCommandSignatures();
        }
    }

    void RenderContext::pushGraphicsState(const GraphicsState::SharedPtr& pState)
    {
        mPipelineStateStack.push(mpGraphicsState);
        setGraphicsState(pState);
    }

    void RenderContext::popGraphicsState()
    {
        if (mPipelineStateStack.empty())
        {
            logWarning("Can't pop from the PipelineState stack. The stack is empty");
            return;
        }

        setGraphicsState(mPipelineStateStack.top());
        mPipelineStateStack.pop();
    }

    void RenderContext::pushGraphicsVars(const GraphicsVars::SharedPtr& pVars)
    {
        mpGraphicsVarsStack.push(mpGraphicsVars);
        setGraphicsVars(pVars);
    }

    void RenderContext::popGraphicsVars()
    {
        if (mpGraphicsVarsStack.empty())
        {
            logWarning("Can't pop from the graphics vars stack. The stack is empty");
            return;
        }

        setGraphicsVars(mpGraphicsVarsStack.top());
        mpGraphicsVarsStack.pop();
    }

    void RenderContext::clearFbo(const Fbo* pFbo, const glm::vec4& color, float depth, uint8_t stencil, FboAttachmentType flags)
    {
        bool hasDepthStencilTexture = pFbo->getDepthStencilTexture() != nullptr;
        ResourceFormat depthStencilFormat = hasDepthStencilTexture ? pFbo->getDepthStencilTexture()->getFormat() : ResourceFormat::Unknown;

        bool clearColor = (flags & FboAttachmentType::Color) != FboAttachmentType::None;
        bool clearDepth = hasDepthStencilTexture && ((flags & FboAttachmentType::Depth) != FboAttachmentType::None);
        bool clearStencil = hasDepthStencilTexture && ((flags & FboAttachmentType::Stencil) != FboAttachmentType::None) && isStencilFormat(depthStencilFormat);

        if (clearColor)
        {
            for (uint32_t i = 0; i < Fbo::getMaxColorTargetCount(); i++)
            {
                if (pFbo->getColorTexture(i))
                {
                    clearRtv(pFbo->getRenderTargetView(i).get(), color);
                }
            }
        }

        if (clearDepth || clearStencil)
        {
            clearDsv(pFbo->getDepthStencilView().get(), depth, stencil, clearDepth, clearStencil);
        }
    }

    void RenderContext::applyGraphicsVars()
    {
        if (mpGraphicsVars->apply(const_cast<RenderContext*>(this), mBindGraphicsRootSig) == false)
        {
            logWarning("RenderContext::prepareForDraw() - applying GraphicsVars failed, most likely because we ran out of descriptors. Flushing the GPU and retrying");
            flush(true);
            bool b = mpGraphicsVars->apply(const_cast<RenderContext*>(this), mBindGraphicsRootSig);
            assert(b);
        }
    }

    void RenderContext::reset()
    {
        ComputeContext::reset();
        mBindGraphicsRootSig = true;
    }

    void RenderContext::flush(bool wait)
    {
        ComputeContext::flush(wait);
        mBindGraphicsRootSig = true;
    }
}
