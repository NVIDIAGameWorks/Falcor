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
#include "GraphicsStateObject.h"
#include "Device.h"

namespace Falcor
{
    BlendState::SharedPtr GraphicsStateObject::spDefaultBlendState;
    RasterizerState::SharedPtr GraphicsStateObject::spDefaultRasterizerState;
    DepthStencilState::SharedPtr GraphicsStateObject::spDefaultDepthStencilState;

    bool GraphicsStateObject::Desc::operator==(const GraphicsStateObject::Desc& other) const
    {
        bool b = true;
        b = b && (mpLayout                  == other.mpLayout);
        b = b && (mFboDesc                  == other.mFboDesc);
        b = b && (mpProgram                 == other.mpProgram);
        b = b && (mSampleMask               == other.mSampleMask);
        b = b && (mPrimType                 == other.mPrimType);

        if (mpRasterizerState)
        {
            b = b && (mpRasterizerState == other.mpRasterizerState);
        }
        else
        {
            b = b && (other.mpRasterizerState == nullptr || other.mpRasterizerState == spDefaultRasterizerState);
        }

        if (mpBlendState)
        {
            b = b && (mpBlendState == other.mpBlendState);
        }
        else
        {
            b = b && (other.mpBlendState == nullptr || other.mpBlendState == spDefaultBlendState);
        }

        if (mpDepthStencilState)
        {
            b = b && (mpDepthStencilState == other.mpDepthStencilState);
        }
        else
        {
            b = b && (other.mpDepthStencilState == nullptr || other.mpDepthStencilState == spDefaultDepthStencilState);
        }

        return b;
    }

    GraphicsStateObject::~GraphicsStateObject()
    {
        gpDevice->releaseResource(mApiHandle);
#ifdef FALCOR_GFX
        gpDevice->releaseResource(mpGFXRenderPassLayout);
#endif
    }

    GraphicsStateObject::GraphicsStateObject(const Desc& desc)
        : mDesc(desc)
    {
        if (spDefaultBlendState == nullptr)
        {
            // Create default objects
            spDefaultBlendState = BlendState::create(BlendState::Desc());
            spDefaultDepthStencilState = DepthStencilState::create(DepthStencilState::Desc());
            spDefaultRasterizerState = RasterizerState::create(RasterizerState::Desc());
        }

        // Initialize default objects
        if (!mDesc.mpBlendState) mDesc.mpBlendState = spDefaultBlendState;
        if (!mDesc.mpRasterizerState) mDesc.mpRasterizerState = spDefaultRasterizerState;
        if (!mDesc.mpDepthStencilState) mDesc.mpDepthStencilState = spDefaultDepthStencilState;

        apiInit();
    }

    GraphicsStateObject::SharedPtr GraphicsStateObject::create(const Desc& desc)
    {
        return SharedPtr(new GraphicsStateObject(desc));
    }
}
