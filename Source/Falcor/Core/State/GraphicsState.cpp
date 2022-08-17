/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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
#include "GraphicsState.h"
#include "Core/API/Device.h"
#include "Core/Program/ProgramVars.h"
#include "Utils/Scripting/ScriptBindings.h"

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
            FALCOR_UNREACHABLE();
            return GraphicsStateObject::PrimitiveType::Undefined;
        }
    }

    GraphicsState::GraphicsState()
    {
        uint32_t vpCount = getMaxViewportCount();

        // Create the viewports
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
        auto pProgramKernels = mpProgram ? mpProgram->getActiveVersion()->getKernels(pVars) : nullptr;
        bool newProgVersion = pProgramKernels.get() != mCachedData.pProgramKernels;
        if (newProgVersion)
        {
            mCachedData.pProgramKernels = pProgramKernels.get();
            mpGsoGraph->walk((void*)pProgramKernels.get());
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
            mDesc.setProgramKernels(pProgramKernels);
            mDesc.setFboFormats(mpFbo ? mpFbo->getDesc() : Fbo::Desc());
            mDesc.setVertexLayout(mpVao->getVertexLayout());
            mDesc.setPrimitiveType(topology2Type(mpVao->getPrimitiveTopology()));

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
        checkInvariant(!mFboStack.empty(), "Empty stack.");

        setFbo(mFboStack.top(), setVp0Sc0);
        mFboStack.pop();
    }

    GraphicsState& GraphicsState::setVao(const Vao::SharedConstPtr& pVao)
    {
        if(mpVao != pVao)
        {
            mpVao = pVao;
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
        if (mDesc.getRasterizerState() != pRasterizerState)
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
        checkArgument(index < mVpStack.size(), "'index' is out of range.");

        mVpStack[index].push(mViewports[index]);
        setViewport(index, vp, setScissors);
    }

    void GraphicsState::popViewport(uint32_t index, bool setScissors)
    {
        checkArgument(index < mVpStack.size(), "'index' is out of range.");
        checkInvariant(!mVpStack[index].empty(), "Empty stack.");

        const auto& VP = mVpStack[index].top();
        setViewport(index, VP, setScissors);
        mVpStack[index].pop();
    }

    void GraphicsState::pushScissors(uint32_t index, const GraphicsState::Scissor& sc)
    {
        checkArgument(index < mScStack.size(), "'index' is out of range.");

        mScStack[index].push(mScissors[index]);
        setScissors(index, sc);
    }

    void GraphicsState::popScissors(uint32_t index)
    {
        checkArgument(index < mScStack.size(), "'index' is out of range.");
        checkInvariant(!mScStack[index].empty(), "Empty stack.");

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

    FALCOR_SCRIPT_BINDING(GraphicsState)
    {
        pybind11::class_<GraphicsState, GraphicsState::SharedPtr>(m, "GraphicsState");
    }
}
