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
#pragma once
#include "Handles.h"
#include "Core/Macros.h"
#include "Core/API/VertexLayout.h"
#include "Core/API/FBO.h"
#include "Core/API/RasterizerState.h"
#include "Core/API/DepthStencilState.h"
#include "Core/API/BlendState.h"
#include "Core/Program/ProgramVersion.h"
#include <memory>

namespace Falcor
{
    class FALCOR_API GraphicsStateObject
    {
    public:
        using SharedPtr = std::shared_ptr<GraphicsStateObject>;
        using SharedConstPtr = std::shared_ptr<const GraphicsStateObject>;
        using ApiHandle = GraphicsStateHandle;

        static const uint32_t kSampleMaskAll = -1;

        /** Primitive topology
        */
        enum class PrimitiveType
        {
            Undefined,
            Point,
            Line,
            Triangle,
            Patch,
        };

        class FALCOR_API Desc
        {
        public:
            Desc& setVertexLayout(VertexLayout::SharedConstPtr pLayout) { mpLayout = pLayout; return *this; }
            Desc& setFboFormats(const Fbo::Desc& fboFormats) { mFboDesc = fboFormats; return *this; }
            Desc& setProgramKernels(ProgramKernels::SharedConstPtr pProgram) { mpProgram = pProgram; return *this; }
            Desc& setBlendState(BlendState::SharedPtr pBlendState) { mpBlendState = pBlendState; return *this; }
            Desc& setRasterizerState(RasterizerState::SharedPtr pRasterizerState) { mpRasterizerState = pRasterizerState; return *this; }
            Desc& setDepthStencilState(DepthStencilState::SharedPtr pDepthStencilState) { mpDepthStencilState = pDepthStencilState; return *this; }
            Desc& setSampleMask(uint32_t sampleMask) { mSampleMask = sampleMask; return *this; }
            Desc& setPrimitiveType(PrimitiveType type) { mPrimType = type; return *this; }

            BlendState::SharedPtr getBlendState() const { return mpBlendState; }
            RasterizerState::SharedPtr getRasterizerState() const { return mpRasterizerState; }
            DepthStencilState::SharedPtr getDepthStencilState() const { return mpDepthStencilState; }
            ProgramKernels::SharedConstPtr getProgramKernels() const { return mpProgram; }
            ProgramVersion::SharedConstPtr getProgramVersion() const { return mpProgram->getProgramVersion(); }
            uint32_t getSampleMask() const { return mSampleMask; }
            VertexLayout::SharedConstPtr getVertexLayout() const { return mpLayout; }
            PrimitiveType getPrimitiveType() const { return mPrimType; }
            Fbo::Desc getFboDesc() const { return mFboDesc; }

            bool operator==(const Desc& other) const;

        private:
            friend class GraphicsStateObject;
            Fbo::Desc mFboDesc;
            VertexLayout::SharedConstPtr mpLayout;
            ProgramKernels::SharedConstPtr mpProgram;
            RasterizerState::SharedPtr mpRasterizerState;
            DepthStencilState::SharedPtr mpDepthStencilState;
            BlendState::SharedPtr mpBlendState;
            uint32_t mSampleMask = kSampleMaskAll;
            PrimitiveType mPrimType = PrimitiveType::Undefined;
        };

        ~GraphicsStateObject();

        /** Create a graphics state object.
            \param[in] desc State object description.
            \return New object, or throws an exception if creation failed.
        */
        static SharedPtr create(const Desc& desc);

        const ApiHandle& getApiHandle() { return mApiHandle; }

        const Desc& getDesc() const { return mDesc; }

#ifdef FALCOR_GFX
        gfx::IRenderPassLayout* getGFXRenderPassLayout() const { return mpGFXRenderPassLayout.get(); }
#endif

    private:
        GraphicsStateObject(const Desc& desc);
        void apiInit();

        Desc mDesc;
        ApiHandle mApiHandle;

#ifdef FALCOR_GFX
        Slang::ComPtr<gfx::IInputLayout> mpGFXInputLayout;
        Slang::ComPtr<gfx::IFramebufferLayout> mpGFXFramebufferLayout;
        Slang::ComPtr<gfx::IRenderPassLayout> mpGFXRenderPassLayout;
#endif

        // Default state objects
        static BlendState::SharedPtr spDefaultBlendState;
        static RasterizerState::SharedPtr spDefaultRasterizerState;
        static DepthStencilState::SharedPtr spDefaultDepthStencilState;
    };
}
