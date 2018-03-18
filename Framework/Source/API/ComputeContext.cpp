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
#include "API/ComputeContext.h"

namespace Falcor
{
    ComputeContext::ComputeContext()
    {
        if (spDispatchCommandSig == nullptr)
        {
            initDispatchCommandSignature();
        }
    }

    ComputeContext::~ComputeContext() = default;
    CommandSignatureHandle ComputeContext::spDispatchCommandSig = nullptr;

    ComputeContext::SharedPtr ComputeContext::create(CommandQueueHandle queue)
    {
        SharedPtr pCtx = SharedPtr(new ComputeContext());
        pCtx->mpLowLevelData = LowLevelContextData::create(LowLevelContextData::CommandQueueType::Compute, queue);
        if (pCtx->mpLowLevelData == nullptr)
        {
            return nullptr;
        }
        pCtx->bindDescriptorHeaps();
        return pCtx;
    }
    
    void ComputeContext::pushComputeVars(const ComputeVars::SharedPtr& pVars)
    {
        mpComputeVarsStack.push(mpComputeVars);
        setComputeVars(pVars);
    }

    void ComputeContext::popComputeVars()
    {
        if (mpComputeVarsStack.empty())
        {
            logWarning("Can't pop from the compute vars stack. The stack is empty");
            return;
        }

        setComputeVars(mpComputeVarsStack.top());
        mpComputeVarsStack.pop();
    }

    void ComputeContext::pushComputeState(const ComputeState::SharedPtr& pState)
    {
        mpComputeStateStack.push(mpComputeState);
        setComputeState(pState);
    }

    void ComputeContext::popComputeState()
    {
        if (mpComputeStateStack.empty())
        {
            logWarning("Can't pop from the compute state stack. The stack is empty");
            return;
        }

        setComputeState(mpComputeStateStack.top());
        mpComputeStateStack.pop();
    }

    void ComputeContext::applyComputeVars(RootSignature* rootSignature) 
    {
        if (mpComputeVars->apply(const_cast<ComputeContext*>(this), mBindComputeRootSig, rootSignature) == false)
        {
            logWarning("ComputeContext::prepareForDispatch() - applying ComputeVars failed, most likely because we ran out of descriptors. Flushing the GPU and retrying");
            flush(true);
            rootSignature->bindForCompute(this);
            bool b = mpComputeVars->apply(const_cast<ComputeContext*>(this), true, rootSignature);
            assert(b);
        }
    }

    void ComputeContext::flush(bool wait)
    {
        CopyContext::flush(wait);
        mBindComputeRootSig = true;
    }
}
