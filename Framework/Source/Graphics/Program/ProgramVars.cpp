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
#include "ProgramVars.h"
#include "API/Buffer.h"
#include "API/CopyContext.h"
#include "API/RenderContext.h"
#include "API/DescriptorSet.h"
#include "API/Device.h"
#include "Utils/StringUtils.h"

namespace Falcor
{
    ProgramVars::ProgramVars(const ProgramReflection::SharedConstPtr& pReflector, bool createBuffers, const RootSignature::SharedPtr& pRootSig) : mpReflector(pReflector)
    {
        mpRootSignature = pRootSig ? pRootSig : RootSignature::create(pReflector.get());
        ParameterBlockReflection::SharedConstPtr pDefaultBlock = pReflector->getDefaultParameterBlock();
        mpGlobalBlock = ParameterBlock::create(pDefaultBlock, mpRootSignature.get(), createBuffers);
    }


    GraphicsVars::SharedPtr GraphicsVars::create(const ProgramReflection::SharedConstPtr& pReflector, bool createBuffers, const RootSignature::SharedPtr& pRootSig)
    {
        return SharedPtr(new GraphicsVars(pReflector, createBuffers, pRootSig));
    }

    ComputeVars::SharedPtr ComputeVars::create(const ProgramReflection::SharedConstPtr& pReflector, bool createBuffers, const RootSignature::SharedPtr& pRootSig)
    {
        return SharedPtr(new ComputeVars(pReflector, createBuffers, pRootSig));
    }

    ConstantBuffer::SharedPtr ProgramVars::getConstantBuffer(const std::string& name) const
    {
        return mpGlobalBlock->getConstantBuffer(name);
    }

    ConstantBuffer::SharedPtr ProgramVars::getConstantBuffer(uint32_t regSpace, uint32_t baseRegIndex, uint32_t arrayIndex) const
    {
        return mpGlobalBlock->getConstantBuffer(regSpace, baseRegIndex, arrayIndex);
    }

    bool ProgramVars::setConstantBuffer(uint32_t regSpace, uint32_t baseRegIndex, uint32_t arrayIndex, const ConstantBuffer::SharedPtr& pCB)
    {
        return mpGlobalBlock->setConstantBuffer(regSpace, baseRegIndex, arrayIndex, pCB);
    }

    bool ProgramVars::setConstantBuffer(const std::string& name, const ConstantBuffer::SharedPtr& pCB)
    {
        return mpGlobalBlock->setConstantBuffer(name, pCB);
    }

    bool ProgramVars::setRawBuffer(const std::string& name, Buffer::SharedPtr pBuf)
    {
        return mpGlobalBlock->setRawBuffer(name, pBuf);
    }

    bool ProgramVars::setTypedBuffer(const std::string& name, TypedBufferBase::SharedPtr pBuf)
    {
        return mpGlobalBlock->setTypedBuffer(name, pBuf);
    }
    
    bool ProgramVars::setStructuredBuffer(const std::string& name, StructuredBuffer::SharedPtr pBuf)
    {
        return mpGlobalBlock->setStructuredBuffer(name, pBuf);
    }
    
    Buffer::SharedPtr ProgramVars::getRawBuffer(const std::string& name) const
    {
        return mpGlobalBlock->getRawBuffer(name);
    }

    TypedBufferBase::SharedPtr ProgramVars::getTypedBuffer(const std::string& name) const
    {
        return mpGlobalBlock->getTypedBuffer(name);
   }

    StructuredBuffer::SharedPtr ProgramVars::getStructuredBuffer(const std::string& name) const
    {
        return mpGlobalBlock->getStructuredBuffer(name);
    }

    bool ProgramVars::setSampler(uint32_t regSpace, uint32_t baseRegIndex, uint32_t arrayIndex, const Sampler::SharedPtr& pSampler)
    {
        return mpGlobalBlock->setSampler(regSpace, baseRegIndex, arrayIndex, pSampler);
    }

    bool ProgramVars::setSampler(const std::string& name, const Sampler::SharedPtr& pSampler)
    {
        return mpGlobalBlock->setSampler(name, pSampler);
    }

    Sampler::SharedPtr ProgramVars::getSampler(const std::string& name) const
    {
        return mpGlobalBlock->getSampler(name);
    }

    Sampler::SharedPtr ProgramVars::getSampler(uint32_t regSpace, uint32_t baseRegIndex, uint32_t arrayIndex) const
    {
        return mpGlobalBlock->getSampler(regSpace, baseRegIndex, arrayIndex);
    }

    ShaderResourceView::SharedPtr ProgramVars::getSrv(uint32_t regSpace, uint32_t baseRegIndex, uint32_t arrayIndex) const
    {
        return mpGlobalBlock->getSrv(regSpace, baseRegIndex, arrayIndex);
    }

    UnorderedAccessView::SharedPtr ProgramVars::getUav(uint32_t regSpace, uint32_t baseRegIndex, uint32_t arrayIndex) const
    {
        return mpGlobalBlock->getUav(regSpace, baseRegIndex, arrayIndex);
    }

    bool ProgramVars::setTexture(const std::string& name, const Texture::SharedPtr& pTexture)
    {
        return mpGlobalBlock->setTexture(name, pTexture);
    }

    Texture::SharedPtr ProgramVars::getTexture(const std::string& name) const
    {
        return mpGlobalBlock->getTexture(name);
    }

    bool ProgramVars::setSrv(uint32_t regSpace, uint32_t baseRegIndex, uint32_t arrayIndex, const ShaderResourceView::SharedPtr& pSrv)
    {
        return mpGlobalBlock->setSrv(regSpace, baseRegIndex, arrayIndex, pSrv);
    }

    bool ProgramVars::setUav(uint32_t regSpace, uint32_t baseRegIndex, uint32_t arrayIndex, const UnorderedAccessView::SharedPtr& pUav)
    {
        return mpGlobalBlock->setUav(regSpace, baseRegIndex, arrayIndex, pUav);
    }

    template<bool forGraphics>
    bool applyProgramVarsCommon(const ProgramVars* pVars, ParameterBlock* pGlobalBlock, CopyContext* pContext, bool bindRootSig)
    {
        if (bindRootSig)
        {
            if (forGraphics)
            {
                pVars->getRootSignature()->bindForGraphics(pContext);
            }
            else
            {
                pVars->getRootSignature()->bindForCompute(pContext);
            }
        }

        if (pGlobalBlock->prepareForDraw(pContext) == false) return false;

        // Bind the sets
        auto& rootSets = pGlobalBlock->getRootSets();

        for (uint32_t i = 0; i < rootSets.size(); i++)
        {
            if (rootSets[i].dirty || bindRootSig)
            {
                rootSets[i].dirty = false;
                if (forGraphics)
                {
                    rootSets[i].pSet->bindForGraphics(pContext, pVars->getRootSignature().get(), i);
                }
                else
                {
                    rootSets[i].pSet->bindForCompute(pContext, pVars->getRootSignature().get(), i);
                }
            }
        }
        return true;
    }

    bool ComputeVars::apply(ComputeContext* pContext, bool bindRootSig)
    {
        return applyProgramVarsCommon<false>(this, mpGlobalBlock.get(), pContext, bindRootSig);
    }

    bool GraphicsVars::apply(RenderContext* pContext, bool bindRootSig)
    {
        return applyProgramVarsCommon<true>(this, mpGlobalBlock.get(), pContext, bindRootSig);
    }
}