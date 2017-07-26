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
#include "API/RasterizerState.h"

namespace Falcor
{
    RasterizerState::~RasterizerState() = default;
    
    D3D11_FILL_MODE getD3DFillMode(RasterizerState::FillMode fill)
    {
        switch(fill)
        {
        case RasterizerState::FillMode::Wireframe:
            return D3D11_FILL_WIREFRAME;
        case RasterizerState::FillMode::Solid:
            return D3D11_FILL_SOLID;
        default:
            should_not_get_here();
            return (D3D11_FILL_MODE)0;
        }
    }

    D3D11_CULL_MODE getD3DCullMode(RasterizerState::CullMode cull)
    {
        switch(cull)
        {
        case Falcor::RasterizerState::CullMode::None:
            return D3D11_CULL_NONE;
        case Falcor::RasterizerState::CullMode::Front:
            return D3D11_CULL_FRONT;
        case Falcor::RasterizerState::CullMode::Back:
            return D3D11_CULL_BACK;
        default:
            should_not_get_here();
            return (D3D11_CULL_MODE)0;
        }
    }

    RasterizerState::SharedPtr RasterizerState::create(const Desc& desc)
    {
        D3D11_RASTERIZER_DESC dxDesc;
        dxDesc.FillMode = getD3DFillMode(desc.mFillMode);
        dxDesc.CullMode = getD3DCullMode(desc.mCullMode);
        dxDesc.FrontCounterClockwise = dxBool(desc.mIsFrontCcw);
        dxDesc.DepthBias = desc.mDepthBias;
        dxDesc.DepthBiasClamp = 0;
        dxDesc.SlopeScaledDepthBias = desc.mSlopeScaledDepthBias;
        dxDesc.DepthClipEnable = dxBool(!desc.mClampDepth); // Depth-clamp disables depth-clip
        dxDesc.ScissorEnable = dxBool(desc.mScissorEnabled);

        // Set the line anti-aliasing mode
        dxDesc.AntialiasedLineEnable = dxBool(desc.mEnableLinesAA);
        dxDesc.MultisampleEnable = dxBool(desc.mEnableLinesAA);

        SharedPtr pState = SharedPtr(new RasterizerState(desc));
        d3d_call(getD3D11Device()->CreateRasterizerState(&dxDesc, &pState->mApiHandle));
        return pState;
    }

    RasterizerStateHandle RasterizerState::getApiHandle() const
    {
        return mApiHandle;
    }
}