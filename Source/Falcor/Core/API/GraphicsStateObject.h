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
#pragma once
#include "Handles.h"
#include "Core/Macros.h"
#include "Core/Object.h"
#include "Core/API/VertexLayout.h"
#include "Core/API/FBO.h"
#include "Core/API/RasterizerState.h"
#include "Core/API/DepthStencilState.h"
#include "Core/API/BlendState.h"
#include "Core/Program/ProgramVersion.h"

namespace Falcor
{
class FALCOR_API GraphicsStateObject : public Object
{
    FALCOR_OBJECT(GraphicsStateObject)
public:
    static constexpr uint32_t kSampleMaskAll = -1;

    /**
     * Primitive topology
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
        Desc& setVertexLayout(ref<const VertexLayout> pLayout)
        {
            mpLayout = pLayout;
            return *this;
        }

        Desc& setFboFormats(const Fbo::Desc& fboFormats)
        {
            mFboDesc = fboFormats;
            return *this;
        }

        Desc& setProgramKernels(ref<const ProgramKernels> pProgram)
        {
            mpProgram = pProgram;
            return *this;
        }

        Desc& setBlendState(ref<BlendState> pBlendState)
        {
            mpBlendState = pBlendState;
            return *this;
        }

        Desc& setRasterizerState(ref<RasterizerState> pRasterizerState)
        {
            mpRasterizerState = pRasterizerState;
            return *this;
        }

        Desc& setDepthStencilState(ref<DepthStencilState> pDepthStencilState)
        {
            mpDepthStencilState = pDepthStencilState;
            return *this;
        }

        Desc& setSampleMask(uint32_t sampleMask)
        {
            mSampleMask = sampleMask;
            return *this;
        }

        Desc& setPrimitiveType(PrimitiveType type)
        {
            mPrimType = type;
            return *this;
        }

        ref<BlendState> getBlendState() const { return mpBlendState; }
        ref<RasterizerState> getRasterizerState() const { return mpRasterizerState; }
        ref<DepthStencilState> getDepthStencilState() const { return mpDepthStencilState; }
        ref<const ProgramKernels> getProgramKernels() const { return mpProgram; }
        uint32_t getSampleMask() const { return mSampleMask; }
        ref<const VertexLayout> getVertexLayout() const { return mpLayout; }
        PrimitiveType getPrimitiveType() const { return mPrimType; }
        Fbo::Desc getFboDesc() const { return mFboDesc; }

        bool operator==(const Desc& other) const;

    private:
        friend class GraphicsStateObject;
        Fbo::Desc mFboDesc;
        ref<const VertexLayout> mpLayout;
        ref<const ProgramKernels> mpProgram;
        ref<RasterizerState> mpRasterizerState;
        ref<DepthStencilState> mpDepthStencilState;
        ref<BlendState> mpBlendState;
        uint32_t mSampleMask = kSampleMaskAll;
        PrimitiveType mPrimType = PrimitiveType::Undefined;
    };

    ~GraphicsStateObject();

    /**
     * Create a graphics state object.
     * @param[in] desc State object description.
     * @return New object, or throws an exception if creation failed.
     */
    static ref<GraphicsStateObject> create(ref<Device> pDevice, const Desc& desc);

    gfx::IPipelineState* getGfxPipelineState() const { return mGfxPipelineState; }

    const Desc& getDesc() const { return mDesc; }

    gfx::IRenderPassLayout* getGFXRenderPassLayout() const { return mpGFXRenderPassLayout.get(); }

    void breakStrongReferenceToDevice();

private:
    GraphicsStateObject(ref<Device> pDevice, const Desc& desc);

    BreakableReference<Device> mpDevice;
    Desc mDesc;
    Slang::ComPtr<gfx::IPipelineState> mGfxPipelineState;

    Slang::ComPtr<gfx::IInputLayout> mpGFXInputLayout;
    Slang::ComPtr<gfx::IFramebufferLayout> mpGFXFramebufferLayout;
    Slang::ComPtr<gfx::IRenderPassLayout> mpGFXRenderPassLayout;

    // Default state objects
    static ref<BlendState> spDefaultBlendState;               // TODO: REMOVEGLOBAL
    static ref<RasterizerState> spDefaultRasterizerState;     // TODO: REMOVEGLOBAL
    static ref<DepthStencilState> spDefaultDepthStencilState; // TODO: REMOVEGLOBAL
};
} // namespace Falcor
