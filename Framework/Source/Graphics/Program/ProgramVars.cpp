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
    ProgramVars::BlockData ProgramVars::initParameterBlock(const ParameterBlockReflection::SharedConstPtr& pBlockReflection, bool createBuffers)
    {
        BlockData data;
        data.pBlock = ParameterBlock::create(pBlockReflection, createBuffers);
        return data;
    }

    ProgramVars::ProgramVars(const ProgramReflection::SharedConstPtr& pReflector, bool createBuffers) : mpReflector(pReflector)
    {
        ParameterBlockReflection::SharedConstPtr pDefaultBlock = pReflector->getDefaultParameterBlock();
        // Initialize the global-block first so that it's the first entry in the vector
        for (uint32_t i = 0; i < pReflector->getParameterBlockCount(); i++)
        {
            const auto& pBlock = pReflector->getParameterBlock(i);
            BlockData data = initParameterBlock(pBlock, createBuffers);
            mParameterBlocks.push_back(data);
        }
        mDefaultBlock = mParameterBlocks[mpReflector->getParameterBlockIndex("")];
    }

    const ParameterBlock::SharedConstPtr ProgramVars::getParameterBlock(const std::string& name) const
    {
        uint32_t index = mpReflector->getParameterBlockIndex(name);
        if (index == ProgramReflection::kInvalidLocation)
        {
            logWarning("Can't find parameter block named " + name + ". Ignoring getParameterBlock() call");
            return nullptr;
        }
        return mParameterBlocks[index].pBlock;
    }

    const ParameterBlock::SharedConstPtr ProgramVars::getParameterBlock(uint32_t blockIndex) const
    {
        return (blockIndex < mParameterBlocks.size()) ? mParameterBlocks[blockIndex].pBlock : nullptr;
    }

    void ProgramVars::setParameterBlock(const std::string& name, const ParameterBlock::SharedConstPtr& pBlock)
    {
        uint32_t index = mpReflector->getParameterBlockIndex(name);
        if (index == ProgramReflection::kInvalidLocation)
        {
            logWarning("Can't find parameter block named " + name + ". Ignoring setParameterBlock() call");
            return;
        }
        mParameterBlocks[index].bind = true;
        mParameterBlocks[index].pBlock = pBlock ? std::const_pointer_cast<ParameterBlock>(pBlock) : ParameterBlock::create(mpReflector->getParameterBlock(index), true);   // #PARAMBLOCK
    }

    void ProgramVars::setParameterBlock(uint32_t blockIndex, const ParameterBlock::SharedConstPtr& pBlock)
    {
        if (blockIndex >= mParameterBlocks.size())
        {
            logWarning("setParameterBlock() - block index out-of-bounds");
            return;
        }
        mParameterBlocks[blockIndex].bind = true;
        mParameterBlocks[blockIndex].pBlock = pBlock ? std::const_pointer_cast<ParameterBlock>(pBlock) : ParameterBlock::create(mpReflector->getParameterBlock(blockIndex), true);   // #PARAMBLOCK
    }

    GraphicsVars::SharedPtr GraphicsVars::create(const ProgramReflection::SharedConstPtr& pReflector, bool createBuffers)
    {
        return SharedPtr(new GraphicsVars(pReflector, createBuffers));
    }

    ComputeVars::SharedPtr ComputeVars::create(const ProgramReflection::SharedConstPtr& pReflector, bool createBuffers)
    {
        return SharedPtr(new ComputeVars(pReflector, createBuffers));
    }

    ConstantBuffer::SharedPtr ProgramVars::getConstantBuffer(const std::string& name) const
    {
        return mDefaultBlock.pBlock->getConstantBuffer(name);
    }

    ConstantBuffer::SharedPtr ProgramVars::getConstantBuffer(uint32_t regSpace, uint32_t baseRegIndex, uint32_t arrayIndex) const
    {
        const auto& loc = mpReflector->translateRegisterIndicesToBindLocation(regSpace, baseRegIndex, ProgramReflection::BindType::Cbv);
        return mDefaultBlock.pBlock->getConstantBuffer(loc, arrayIndex);
    }

    bool ProgramVars::setConstantBuffer(uint32_t regSpace, uint32_t baseRegIndex, uint32_t arrayIndex, const ConstantBuffer::SharedPtr& pCB)
    {
        const auto& loc = mpReflector->translateRegisterIndicesToBindLocation(regSpace, baseRegIndex, ProgramReflection::BindType::Cbv);
        return mDefaultBlock.pBlock->setConstantBuffer(loc, arrayIndex, pCB);
    }

    bool ProgramVars::setConstantBuffer(const std::string& name, const ConstantBuffer::SharedPtr& pCB)
    {
        return mDefaultBlock.pBlock->setConstantBuffer(name, pCB);
    }

    bool ProgramVars::setRawBuffer(const std::string& name, Buffer::SharedPtr pBuf)
    {
        return mDefaultBlock.pBlock->setRawBuffer(name, pBuf);
    }

    bool ProgramVars::setTypedBuffer(const std::string& name, TypedBufferBase::SharedPtr pBuf)
    {
        return mDefaultBlock.pBlock->setTypedBuffer(name, pBuf);
    }
    
    bool ProgramVars::setStructuredBuffer(const std::string& name, StructuredBuffer::SharedPtr pBuf)
    {
        return mDefaultBlock.pBlock->setStructuredBuffer(name, pBuf);
    }
    
    Buffer::SharedPtr ProgramVars::getRawBuffer(const std::string& name) const
    {
        return mDefaultBlock.pBlock->getRawBuffer(name);
    }

    TypedBufferBase::SharedPtr ProgramVars::getTypedBuffer(const std::string& name) const
    {
        return mDefaultBlock.pBlock->getTypedBuffer(name);
   }

    StructuredBuffer::SharedPtr ProgramVars::getStructuredBuffer(const std::string& name) const
    {
        return mDefaultBlock.pBlock->getStructuredBuffer(name);
    }

    bool ProgramVars::setSampler(uint32_t regSpace, uint32_t baseRegIndex, uint32_t arrayIndex, const Sampler::SharedPtr& pSampler)
    {
        const auto& loc = mpReflector->translateRegisterIndicesToBindLocation(regSpace, baseRegIndex, ProgramReflection::BindType::Sampler);
        return mDefaultBlock.pBlock->setSampler(loc, arrayIndex, pSampler);
    }

    bool ProgramVars::setSampler(const std::string& name, const Sampler::SharedPtr& pSampler)
    {
        return mDefaultBlock.pBlock->setSampler(name, pSampler);
    }

    Sampler::SharedPtr ProgramVars::getSampler(const std::string& name) const
    {
        return mDefaultBlock.pBlock->getSampler(name);
    }

    Sampler::SharedPtr ProgramVars::getSampler(uint32_t regSpace, uint32_t baseRegIndex, uint32_t arrayIndex) const
    {
        const auto& loc = mpReflector->translateRegisterIndicesToBindLocation(regSpace, baseRegIndex, ProgramReflection::BindType::Sampler);
        return mDefaultBlock.pBlock->getSampler(loc, arrayIndex);
    }

    ShaderResourceView::SharedPtr ProgramVars::getSrv(uint32_t regSpace, uint32_t baseRegIndex, uint32_t arrayIndex) const
    {
        const auto& loc = mpReflector->translateRegisterIndicesToBindLocation(regSpace, baseRegIndex, ProgramReflection::BindType::Srv);
        return mDefaultBlock.pBlock->getSrv(loc, arrayIndex);
    }

    UnorderedAccessView::SharedPtr ProgramVars::getUav(uint32_t regSpace, uint32_t baseRegIndex, uint32_t arrayIndex) const
    {
        const auto& loc = mpReflector->translateRegisterIndicesToBindLocation(regSpace, baseRegIndex, ProgramReflection::BindType::Uav);
        return mDefaultBlock.pBlock->getUav(loc, arrayIndex);
    }

    bool ProgramVars::setTexture(const std::string& name, const Texture::SharedPtr& pTexture)
    {
        return mDefaultBlock.pBlock->setTexture(name, pTexture);
    }

    Texture::SharedPtr ProgramVars::getTexture(const std::string& name) const
    {
        return mDefaultBlock.pBlock->getTexture(name);
    }

    bool ProgramVars::setSrv(uint32_t regSpace, uint32_t baseRegIndex, uint32_t arrayIndex, const ShaderResourceView::SharedPtr& pSrv)
    {
        const auto& loc = mpReflector->translateRegisterIndicesToBindLocation(regSpace, baseRegIndex, ProgramReflection::BindType::Srv);
        return mDefaultBlock.pBlock->setSrv(loc, arrayIndex, pSrv);
    }
    
    bool ProgramVars::setUav(uint32_t regSpace, uint32_t baseRegIndex, uint32_t arrayIndex, const UnorderedAccessView::SharedPtr& pUav)
    {
        const auto& loc = mpReflector->translateRegisterIndicesToBindLocation(regSpace, baseRegIndex, ProgramReflection::BindType::Uav);
        return mDefaultBlock.pBlock->setUav(loc, arrayIndex, pUav);
    }

    template<bool forGraphics>
    bool ProgramVars::bindRootSetsCommon(CopyContext* pContext, bool bindRootSig, const ProgramKernels* pKernels)
    {
        auto pRootSig = pKernels->getRootSignature().get();

        // Bind the sets
        for(uint32_t b = 0 ; b < getParameterBlockCount() ; b++)
        {
            ParameterBlock* pBlock = mParameterBlocks[b].pBlock.get(); // #PARAMBLOCK getParameterBlock() because we don't want the user to change blocks directly, but we need it non-const here
            if (pBlock->prepareForDraw(pContext) == false) return false; // #PARAMBLOCK Get rid of it. getRootSets() should have a dirty flag

            const auto& rootIndices = pKernels->getParameterBlockRootIndices(b);
            auto& rootSets = pBlock->getRootSets();
            bool forceBind = bindRootSig || mParameterBlocks[b].bind;
            mParameterBlocks[b].bind = false;

            for (uint32_t s = 0; s < rootSets.size(); s++)
            {
                if (rootSets[s].dirty || forceBind)
                {
                    rootSets[s].dirty = false;
                    uint32_t rootIndex = rootIndices[s];
                    if (forGraphics)
                    {
                        rootSets[s].pSet->bindForGraphics(pContext, pRootSig, rootIndex);
                    }
                    else
                    {
                        rootSets[s].pSet->bindForCompute(pContext, pRootSig, rootIndex);
                    }
                }
            }
        }
        return true;
    }

    template<bool forGraphics>
    bool ProgramVars::applyProgramVarsCommon(CopyContext* pContext, bool bindRootSig, const ProgramKernels* pKernels)
    {
        if (bindRootSig)
        {
            auto pRootSig = pKernels->getRootSignature();
            if (forGraphics)
            {
                pRootSig->bindForGraphics(pContext);
            }
            else
            {
                pRootSig->bindForCompute(pContext);
            }
        }

        return bindRootSetsCommon<forGraphics>(pContext, bindRootSig, pKernels);
    }


    bool ComputeVars::apply(ComputeContext* pContext, bool bindRootSig, const ProgramKernels* pKernels)
    {
        return applyProgramVarsCommon<false>(pContext, bindRootSig, pKernels);
    }

    bool GraphicsVars::apply(RenderContext* pContext, bool bindRootSig, const ProgramKernels* pKernels)
    {
        return applyProgramVarsCommon<true>(pContext, bindRootSig, pKernels);
    }
}