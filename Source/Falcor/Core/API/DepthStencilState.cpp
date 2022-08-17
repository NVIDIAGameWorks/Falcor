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
#include "DepthStencilState.h"
#include "Core/Assert.h"
#include "Utils/Scripting/ScriptBindings.h"

namespace Falcor
{
    DepthStencilState::SharedPtr DepthStencilState::create(const Desc& desc)
    {
        return SharedPtr(new DepthStencilState(desc));
    }

    DepthStencilState::~DepthStencilState() = default;

    DepthStencilState::Desc& DepthStencilState::Desc::setStencilWriteMask(uint8_t mask)
    {
        mStencilWriteMask = mask;
        return *this;
    }

    DepthStencilState::Desc& DepthStencilState::Desc::setStencilReadMask(uint8_t mask)
    {
        mStencilReadMask = mask;
        return *this;
    }

    DepthStencilState::Desc& DepthStencilState::Desc::setStencilFunc(Face face, Func func)
    {
        if(face == Face::FrontAndBack)
        {
            setStencilFunc(Face::Front, func);
            setStencilFunc(Face::Back, func);
            return *this;
        }
        StencilDesc& Desc = (face == Face::Front) ? mStencilFront : mStencilBack;
        Desc.func = func;
        return *this;
    }

    DepthStencilState::Desc& DepthStencilState::Desc::setStencilOp(Face face, StencilOp stencilFail, StencilOp depthFail, StencilOp depthStencilPass)
    {
        if(face == Face::FrontAndBack)
        {
            setStencilOp(Face::Front, stencilFail, depthFail, depthStencilPass);
            setStencilOp(Face::Back, stencilFail, depthFail, depthStencilPass);
            return *this;
        }
        StencilDesc& Desc = (face == Face::Front) ? mStencilFront : mStencilBack;
        Desc.stencilFailOp = stencilFail;
        Desc.depthFailOp = depthFail;
        Desc.depthStencilPassOp = depthStencilPass;

        return *this;
    }

    const DepthStencilState::StencilDesc& DepthStencilState::getStencilDesc(Face face) const
    {
        FALCOR_ASSERT(face != Face::FrontAndBack);
        return (face == Face::Front) ? mDesc.mStencilFront : mDesc.mStencilBack;
    }

    FALCOR_SCRIPT_BINDING(DepthStencilState)
    {
        pybind11::class_<DepthStencilState, DepthStencilState::SharedPtr>(m, "DepthStencilState");
    }
}
