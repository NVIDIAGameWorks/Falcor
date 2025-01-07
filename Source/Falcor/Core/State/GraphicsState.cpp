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
#include "GraphicsState.h"
#include "Core/ObjectPython.h"
#include "Core/API/Device.h"
#include "Core/Program/ProgramVars.h"
#include "Utils/Scripting/ScriptBindings.h"

namespace Falcor
{
static GraphicsStateObjectDesc::PrimitiveType topology2Type(Vao::Topology t)
{
    switch (t)
    {
    case Vao::Topology::PointList:
        return GraphicsStateObjectDesc::PrimitiveType::Point;
    case Vao::Topology::LineList:
    case Vao::Topology::LineStrip:
        return GraphicsStateObjectDesc::PrimitiveType::Line;
    case Vao::Topology::TriangleList:
    case Vao::Topology::TriangleStrip:
        return GraphicsStateObjectDesc::PrimitiveType::Triangle;
    default:
        FALCOR_UNREACHABLE();
        return GraphicsStateObjectDesc::PrimitiveType::Undefined;
    }
}

ref<GraphicsState> GraphicsState::create(ref<Device> pDevice)
{
    return ref<GraphicsState>(new GraphicsState(pDevice));
}

GraphicsState::GraphicsState(ref<Device> pDevice) : mpDevice(pDevice)
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

    mpGsoGraph = std::make_unique<GraphicsStateGraph>();
}

GraphicsState::~GraphicsState() = default;

ref<GraphicsStateObject> GraphicsState::getGSO(const ProgramVars* pVars)
{
    auto pProgramKernels = mpProgram ? mpProgram->getActiveVersion()->getKernels(mpDevice, pVars) : nullptr;
    bool newProgVersion = pProgramKernels.get() != mCachedData.pProgramKernels;
    if (newProgVersion)
    {
        mCachedData.pProgramKernels = pProgramKernels.get();
        mpGsoGraph->walk((void*)pProgramKernels.get());
    }

    const Fbo::Desc* pFboDesc = mpFbo ? &mpFbo->getDesc() : nullptr;
    if (mCachedData.pFboDesc != pFboDesc)
    {
        mpGsoGraph->walk((void*)pFboDesc);
        mCachedData.pFboDesc = pFboDesc;
    }

    ref<GraphicsStateObject> pGso = mpGsoGraph->getCurrentNode();
    if (pGso == nullptr)
    {
        mDesc.pProgramKernels = pProgramKernels;
        mDesc.fboDesc = mpFbo ? mpFbo->getDesc() : Fbo::Desc();
        mDesc.pVertexLayout = mpVao->getVertexLayout();
        mDesc.primitiveType = topology2Type(mpVao->getPrimitiveTopology());

        GraphicsStateGraph::CompareFunc cmpFunc = [&desc = mDesc](ref<GraphicsStateObject> pGso) -> bool
        { return pGso && (desc == pGso->getDesc()); };

        if (mpGsoGraph->scanForMatchingNode(cmpFunc))
        {
            pGso = mpGsoGraph->getCurrentNode();
        }
        else
        {
            pGso = mpDevice->createGraphicsStateObject(mDesc);
            mDesc = pGso->getDesc();
            pGso->breakStrongReferenceToDevice();
            mpGsoGraph->setCurrentNodeData(pGso);
        }
    }
    return pGso;
}

GraphicsState& GraphicsState::setFbo(const ref<Fbo>& pFbo, bool setVp0Sc0)
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

void GraphicsState::pushFbo(const ref<Fbo>& pFbo, bool setVp0Sc0)
{
    mFboStack.push(mpFbo);
    setFbo(pFbo, setVp0Sc0);
}

void GraphicsState::popFbo(bool setVp0Sc0)
{
    FALCOR_CHECK(!mFboStack.empty(), "Empty stack.");

    setFbo(mFboStack.top(), setVp0Sc0);
    mFboStack.pop();
}

GraphicsState& GraphicsState::setVao(const ref<Vao>& pVao)
{
    if (mpVao != pVao)
    {
        mpVao = pVao;
        mpGsoGraph->walk(pVao ? (void*)pVao->getVertexLayout().get() : nullptr);
    }
    return *this;
}

GraphicsState& GraphicsState::setBlendState(ref<BlendState> pBlendState)
{
    if (mDesc.pBlendState != pBlendState)
    {
        mDesc.pBlendState = pBlendState;
        mpGsoGraph->walk((void*)pBlendState.get());
    }
    return *this;
}

GraphicsState& GraphicsState::setRasterizerState(ref<RasterizerState> pRasterizerState)
{
    if (mDesc.pRasterizerState != pRasterizerState)
    {
        mDesc.pRasterizerState = pRasterizerState;
        mpGsoGraph->walk((void*)pRasterizerState.get());
    }
    return *this;
}

GraphicsState& GraphicsState::setSampleMask(uint32_t sampleMask)
{
    if (mDesc.sampleMask != sampleMask)
    {
        mDesc.sampleMask = sampleMask;
        mpGsoGraph->walk((void*)(uint64_t)sampleMask);
    }
    return *this;
}

GraphicsState& GraphicsState::setDepthStencilState(ref<DepthStencilState> pDepthStencilState)
{
    if (mDesc.pDepthStencilState != pDepthStencilState)
    {
        mDesc.pDepthStencilState = pDepthStencilState;
        mpGsoGraph->walk((void*)pDepthStencilState.get());
    }
    return *this;
}

void GraphicsState::pushViewport(uint32_t index, const GraphicsState::Viewport& vp, bool setScissors)
{
    FALCOR_CHECK(index < mVpStack.size(), "'index' is out of range.");

    mVpStack[index].push(mViewports[index]);
    setViewport(index, vp, setScissors);
}

void GraphicsState::popViewport(uint32_t index, bool setScissors)
{
    FALCOR_CHECK(index < mVpStack.size(), "'index' is out of range.");
    FALCOR_CHECK(!mVpStack[index].empty(), "Empty stack.");

    const auto& VP = mVpStack[index].top();
    setViewport(index, VP, setScissors);
    mVpStack[index].pop();
}

void GraphicsState::pushScissors(uint32_t index, const GraphicsState::Scissor& sc)
{
    FALCOR_CHECK(index < mScStack.size(), "'index' is out of range.");

    mScStack[index].push(mScissors[index]);
    setScissors(index, sc);
}

void GraphicsState::popScissors(uint32_t index)
{
    FALCOR_CHECK(index < mScStack.size(), "'index' is out of range.");
    FALCOR_CHECK(!mScStack[index].empty(), "Empty stack.");

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

void GraphicsState::breakStrongReferenceToDevice()
{
    mpDevice.breakStrongReference();
}

FALCOR_SCRIPT_BINDING(GraphicsState)
{
    pybind11::class_<GraphicsState, ref<GraphicsState>>(m, "GraphicsState");
}
} // namespace Falcor
