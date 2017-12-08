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
    static bool compareRootSets(const DescriptorSet::Layout& a, const DescriptorSet::Layout& b)
    {
        if (a.getRangeCount() != b.getRangeCount()) return false;
        if (a.getVisibility() != b.getVisibility()) return false;
        for (uint32_t i = 0; i < a.getRangeCount(); i++)
        {
            const auto& rangeA = a.getRange(i);
            const auto& rangeB = b.getRange(i);
            if (rangeA.baseRegIndex != rangeB.baseRegIndex) return false;
            if (rangeA.descCount != rangeB.descCount) return false;
            if (rangeA.regSpace != rangeB.regSpace) return false;
            if (rangeA.type != rangeB.type) return false;
        }
        return true;
    }

    static uint32_t findRootIndex(const DescriptorSet::Layout& blockSet, const RootSignature::SharedPtr& pRootSig)
    {
        for (uint32_t i = 0; i < pRootSig->getDescriptorSetCount(); i++)
        {
            const auto& rootSet = pRootSig->getDescriptorSet(i);
            if (compareRootSets(rootSet, blockSet))
            {
                return i;
            }
        }
        should_not_get_here();
        return -1;
    }

    void ProgramVars::initParameterBlock(const ParameterBlockReflection::SharedConstPtr& pBlockReflection, bool createBuffers, const RootSignature::SharedPtr& pRootSig)
    {
        BlockData data;
        data.pBlock = ParameterBlock::create(pBlockReflection, mpRootSignature.get(), createBuffers);
        mParamBlockNameToIndex[pBlockReflection->getName()] = (uint32_t)mParameterBlocks.size();
        // For each set, find the matching root-index. 
        const auto& sets = pBlockReflection->getDescriptorSetLayouts();
        data.rootIndex.resize(sets.size());
        for (size_t i = 0; i < sets.size(); i++)
        {
            data.rootIndex[i] = findRootIndex(sets[i], pRootSig);
        }

        mParameterBlocks.push_back(data);
    }

    ProgramVars::ProgramVars(const ProgramReflection::SharedConstPtr& pReflector, bool createBuffers, const RootSignature::SharedPtr& pRootSig) : mpReflector(pReflector)
    {
        mpRootSignature = pRootSig ? pRootSig : RootSignature::create(pReflector.get());
        ParameterBlockReflection::SharedConstPtr pDefaultBlock = pReflector->getDefaultParameterBlock();
        // Initialize the global-block first so that it's the first entry in the vector
        initParameterBlock(pDefaultBlock, createBuffers, mpRootSignature);
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
        return mParameterBlocks[0].pBlock->getConstantBuffer(name);
    }

    ConstantBuffer::SharedPtr ProgramVars::getConstantBuffer(uint32_t regSpace, uint32_t baseRegIndex, uint32_t arrayIndex) const
    {
        return mParameterBlocks[0].pBlock->getConstantBuffer(regSpace, baseRegIndex, arrayIndex);
    }

    bool ProgramVars::setConstantBuffer(uint32_t regSpace, uint32_t baseRegIndex, uint32_t arrayIndex, const ConstantBuffer::SharedPtr& pCB)
    {
        return mParameterBlocks[0].pBlock->setConstantBuffer(regSpace, baseRegIndex, arrayIndex, pCB);
    }

    bool ProgramVars::setConstantBuffer(const std::string& name, const ConstantBuffer::SharedPtr& pCB)
    {
        return mParameterBlocks[0].pBlock->setConstantBuffer(name, pCB);
    }

    bool ProgramVars::setRawBuffer(const std::string& name, Buffer::SharedPtr pBuf)
    {
        return mParameterBlocks[0].pBlock->setRawBuffer(name, pBuf);
    }

    bool ProgramVars::setTypedBuffer(const std::string& name, TypedBufferBase::SharedPtr pBuf)
    {
        return mParameterBlocks[0].pBlock->setTypedBuffer(name, pBuf);
    }
    
    bool ProgramVars::setStructuredBuffer(const std::string& name, StructuredBuffer::SharedPtr pBuf)
    {
        return mParameterBlocks[0].pBlock->setStructuredBuffer(name, pBuf);
    }
    
    Buffer::SharedPtr ProgramVars::getRawBuffer(const std::string& name) const
    {
        return mParameterBlocks[0].pBlock->getRawBuffer(name);
    }

    TypedBufferBase::SharedPtr ProgramVars::getTypedBuffer(const std::string& name) const
    {
        return mParameterBlocks[0].pBlock->getTypedBuffer(name);
   }

    StructuredBuffer::SharedPtr ProgramVars::getStructuredBuffer(const std::string& name) const
    {
        return mParameterBlocks[0].pBlock->getStructuredBuffer(name);
    }

    bool ProgramVars::setSampler(uint32_t regSpace, uint32_t baseRegIndex, uint32_t arrayIndex, const Sampler::SharedPtr& pSampler)
    {
        return mParameterBlocks[0].pBlock->setSampler(regSpace, baseRegIndex, arrayIndex, pSampler);
    }

    bool ProgramVars::setSampler(const std::string& name, const Sampler::SharedPtr& pSampler)
    {
        return mParameterBlocks[0].pBlock->setSampler(name, pSampler);
    }

    Sampler::SharedPtr ProgramVars::getSampler(const std::string& name) const
    {
        return mParameterBlocks[0].pBlock->getSampler(name);
    }

    Sampler::SharedPtr ProgramVars::getSampler(uint32_t regSpace, uint32_t baseRegIndex, uint32_t arrayIndex) const
    {
        return mParameterBlocks[0].pBlock->getSampler(regSpace, baseRegIndex, arrayIndex);
    }

    ShaderResourceView::SharedPtr ProgramVars::getSrv(uint32_t regSpace, uint32_t baseRegIndex, uint32_t arrayIndex) const
    {
        return mParameterBlocks[0].pBlock->getSrv(regSpace, baseRegIndex, arrayIndex);
    }

    UnorderedAccessView::SharedPtr ProgramVars::getUav(uint32_t regSpace, uint32_t baseRegIndex, uint32_t arrayIndex) const
    {
        return mParameterBlocks[0].pBlock->getUav(regSpace, baseRegIndex, arrayIndex);
    }

    bool ProgramVars::setTexture(const std::string& name, const Texture::SharedPtr& pTexture)
    {
        return mParameterBlocks[0].pBlock->setTexture(name, pTexture);
    }

    Texture::SharedPtr ProgramVars::getTexture(const std::string& name) const
    {
        return mParameterBlocks[0].pBlock->getTexture(name);
    }

    bool ProgramVars::setSrv(uint32_t regSpace, uint32_t baseRegIndex, uint32_t arrayIndex, const ShaderResourceView::SharedPtr& pSrv)
    {
        return mParameterBlocks[0].pBlock->setSrv(regSpace, baseRegIndex, arrayIndex, pSrv);
    }

    bool ProgramVars::setUav(uint32_t regSpace, uint32_t baseRegIndex, uint32_t arrayIndex, const UnorderedAccessView::SharedPtr& pUav)
    {
        return mParameterBlocks[0].pBlock->setUav(regSpace, baseRegIndex, arrayIndex, pUav);
    }

    template<bool forGraphics>
    bool applyProgramVarsCommon(const ProgramVars* pVars, CopyContext* pContext, bool bindRootSig)
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

        // Bind the sets
        for(uint32_t b = 0 ; b < pVars->getParameterBlockCount() ; b++)
        {
            ParameterBlock* pBlock = pVars->getParameterBlock(b).get();
            if (pBlock->prepareForDraw(pContext) == false) return false; // #PARAMBLOCK Get rid of it. getRootSets() should have a dirty flag

            const auto& rootIndices = pVars->getParameterBlockRootIndices(b);

            auto& rootSets = pBlock->getRootSets();

            for (uint32_t s = 0; s < rootSets.size(); s++)
            {
                if (rootSets[s].dirty || bindRootSig)
                {
                    rootSets[s].dirty = false;
                    uint32_t rootIndex = rootIndices[s];
                    if (forGraphics)
                    {
                        rootSets[s].pSet->bindForGraphics(pContext, pVars->getRootSignature().get(), rootIndex);
                    }
                    else
                    {
                        rootSets[s].pSet->bindForCompute(pContext, pVars->getRootSignature().get(), rootIndex);
                    }
                }
            }
        }
        return true;
    }

    bool ComputeVars::apply(ComputeContext* pContext, bool bindRootSig)
    {
        return applyProgramVarsCommon<false>(this, pContext, bindRootSig);
    }

    bool GraphicsVars::apply(RenderContext* pContext, bool bindRootSig)
    {
        return applyProgramVarsCommon<true>(this, pContext, bindRootSig);
    }
}