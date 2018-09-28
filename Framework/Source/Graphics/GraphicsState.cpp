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
#include "GraphicsState.h"
#include "Graphics/Program/ProgramVars.h"

namespace Falcor
{
    static GraphicsStateObject::PrimitiveType topology2Type(Vao::Topology t)
    {
        switch (t)
        {
        case Vao::Topology::PointList:
            return GraphicsStateObject::PrimitiveType::Point;
        case Vao::Topology::LineList:
        case Vao::Topology::LineStrip:
            return GraphicsStateObject::PrimitiveType::Line;
        case Vao::Topology::TriangleList:
        case Vao::Topology::TriangleStrip:
            return GraphicsStateObject::PrimitiveType::Triangle;
        default:
            should_not_get_here();
            return GraphicsStateObject::PrimitiveType::Undefined;
        }
    }

    GraphicsState::GraphicsState()
    {
        uint32_t vpCount = getMaxViewportCount();
        // create the viewports
        mViewports.resize(vpCount);
        mScissors.resize(vpCount);
        mVpStack.resize(vpCount);
        mScStack.resize(vpCount);
        for (uint32_t i = 0; i < vpCount; i++)
        {
            setViewport(i, mViewports[i], true);
        }

        mpGsoGraph = StateGraph::create();
    }

    GraphicsState::~GraphicsState() = default;

    GraphicsStateObject::SharedPtr GraphicsState::getGSO(const GraphicsVars* pVars)
    {
        assert(mpVao);
        if (mpProgram && mpVao->getVertexLayout() != nullptr)
        {
            mpVao->getVertexLayout()->addVertexAttribDclToProg(mpProgram.get());
        }
        const ProgramVersion::SharedConstPtr pProgVersion = mpProgram ? mpProgram->getActiveVersion() : nullptr;
        bool newProgVersion = pProgVersion.get() != mCachedData.pProgramVersion;
        if (newProgVersion)
        {
            mCachedData.pProgramVersion = pProgVersion.get();
            mpGsoGraph->walk((void*)pProgVersion.get());
        }
    
        RootSignature::SharedPtr pRoot = pVars ? pVars->getRootSignature() : RootSignature::getEmpty();

        if (mCachedData.pRootSig != pRoot.get())
        {
            mCachedData.pRootSig = pRoot.get();
            mpGsoGraph->walk((void*)mCachedData.pRootSig);
        }

        const Fbo::Desc* pFboDesc = mpFbo ? &mpFbo->getDesc() : nullptr;
        if(mCachedData.pFboDesc != pFboDesc)
        {
            mpGsoGraph->walk((void*)pFboDesc);
            mCachedData.pFboDesc = pFboDesc;
        }

        GraphicsStateObject::SharedPtr pGso = mpGsoGraph->getCurrentNode();
        if(pGso == nullptr)
        {
            mDesc.setProgramVersion(pProgVersion);
            mDesc.setFboFormats(mpFbo ? mpFbo->getDesc() : Fbo::Desc());
#ifdef FALCOR_VK
            mDesc.setRenderPass(mpFbo ? (VkRenderPass)mpFbo->getApiHandle() : VK_NULL_HANDLE);
#endif
            mDesc.setVertexLayout(mpVao->getVertexLayout());
            mDesc.setPrimitiveType(topology2Type(mpVao->getPrimitiveTopology()));
            mDesc.setRootSignature(pRoot);

            mDesc.setSinglePassStereoEnable(mEnableSinglePassStereo);
            
            StateGraph::CompareFunc cmpFunc = [&desc = mDesc](GraphicsStateObject::SharedPtr pGso) -> bool
            {
                return pGso && (desc == pGso->getDesc());
            };

            if (mpGsoGraph->scanForMatchingNode(cmpFunc))
            {
                pGso = mpGsoGraph->getCurrentNode();
            }
            else
            {
                pGso = GraphicsStateObject::create(mDesc);
                mpGsoGraph->setCurrentNodeData(pGso);
            }
        }
        return pGso;
    }

    GraphicsState& GraphicsState::setFbo(const Fbo::SharedPtr& pFbo, bool setVp0Sc0)
    {
        mpFbo = pFbo;

        if (setVp0Sc0 && pFbo)
        {
            uint32_t w = pFbo->getWidth();
            uint32_t h = pFbo->getHeight();
            GraphicsState::Viewport vp(0, 0, float(w), float(h), 0, 1);
            setViewport(0, vp, true);
        }
        return *this;
    }

    void GraphicsState::pushFbo(const Fbo::SharedPtr& pFbo, bool setVp0Sc0)
    {
        mFboStack.push(mpFbo);
        setFbo(pFbo, setVp0Sc0);
    }

    void GraphicsState::popFbo(bool setVp0Sc0)
    {
        if (mFboStack.empty())
        {
            logError("PipelineState::popFbo() - can't pop FBO since the viewport stack is empty.");
            return;
        }
        setFbo(mFboStack.top(), setVp0Sc0);
        mFboStack.pop();
    }

    GraphicsState& GraphicsState::setVao(const Vao::SharedConstPtr& pVao)
    {
        if(mpVao != pVao)
        {
            mpVao = pVao;

#ifdef FALCOR_VK
            mDesc.setVao(pVao);
#endif

            mpGsoGraph->walk(pVao ? (void*)pVao->getVertexLayout().get() : nullptr);
        }
        return *this;
    }

    GraphicsState& GraphicsState::setBlendState(BlendState::SharedPtr pBlendState)
    {
        if(mDesc.getBlendState() != pBlendState)
        {
            mDesc.setBlendState(pBlendState);
            mpGsoGraph->walk((void*)pBlendState.get());
        }
        return *this;
    }

    GraphicsState& GraphicsState::setRasterizerState(RasterizerState::SharedPtr pRasterizerState)
    {
        if(mDesc.getRasterizerState() != pRasterizerState)
        {
            mDesc.setRasterizerState(pRasterizerState);
            mpGsoGraph->walk((void*)pRasterizerState.get());
        }
        return *this;
    }

    GraphicsState& GraphicsState::setSampleMask(uint32_t sampleMask)
    { 
        if(mDesc.getSampleMask() != sampleMask)
        {
            mDesc.setSampleMask(sampleMask);
            mpGsoGraph->walk((void*)(uint64_t)sampleMask);
        }
        return *this; 
    }

    GraphicsState& GraphicsState::setDepthStencilState(DepthStencilState::SharedPtr pDepthStencilState)
    {
        if(mDesc.getDepthStencilState() != pDepthStencilState)
        {
            mDesc.setDepthStencilState(pDepthStencilState);
            mpGsoGraph->walk((void*)pDepthStencilState.get());
        }
        return *this;
    }

    void GraphicsState::pushViewport(uint32_t index, const GraphicsState::Viewport& vp, bool setScissors)
    {
        mVpStack[index].push(mViewports[index]);
        setViewport(index, vp, setScissors);
    }

    void GraphicsState::popViewport(uint32_t index, bool setScissors)
    {
        if (mVpStack[index].empty())
        {
            logError("PipelineState::popViewport() - can't pop viewport since the viewport stack is empty.");
            return;
        }
        const auto& VP = mVpStack[index].top();
        setViewport(index, VP, setScissors);
        mVpStack[index].pop();
    }

    void GraphicsState::pushScissors(uint32_t index, const GraphicsState::Scissor& sc)
    {
        mScStack[index].push(mScissors[index]);
        setScissors(index, sc);
    }

    void GraphicsState::popScissors(uint32_t index)
    {
        if (mScStack[index].empty())
        {
            logError("PipelineState::popScissors() - can't pop scissors since the scissors stack is empty.");
            return;
        }
        const auto& sc = mScStack[index].top();
        setScissors(index, sc);
        mScStack[index].pop();
    }

    void GraphicsState::setViewport(uint32_t index, const GraphicsState::Viewport& vp, bool setScissors)
    {
        mViewports[index] = vp;

        if (setScissors)
        {
            GraphicsState::Scissor sc;
            sc.left = (int32_t)vp.originX;
            sc.right = sc.left + (int32_t)vp.width;
            sc.top = (int32_t)vp.originY;
            sc.bottom = sc.top + (int32_t)vp.height;
            this->setScissors(index, sc);
        }
    }

    void GraphicsState::setScissors(uint32_t index, const GraphicsState::Scissor& sc)
    {
        mScissors[index] = sc;
    }

    void GraphicsState::toggleSinglePassStereo(bool enable)
    {
#if _ENABLE_NVAPI
        mEnableSinglePassStereo = enable;
        mpGsoGraph->walk((void*)enable);
#else
        if (enable)
        {
            logWarning("NVAPI support is missing. Can't enable Single-Pass-Stereo");
        }
#endif
    }
}