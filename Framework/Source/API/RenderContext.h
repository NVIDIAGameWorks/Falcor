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
#pragma once
#include <stack>
#include <vector>
#include "API/Sampler.h"
#include "API/FBO.h"
#include "API/VAO.h"
#include "API/StructuredBuffer.h"
#include "API/Texture.h"
#include "Framework.h"
#include "API/GraphicsStateObject.h"
#include "Graphics/Program/ProgramVars.h"
#include "Graphics/GraphicsState.h"
#include "API/ComputeContext.h"
#include "Graphics/FullScreenPass.h"

namespace Falcor
{
    class RtProgramVars;
    class RtState;

    /** The rendering context. Use it to bind state and dispatch calls to the GPU
    */
    class RenderContext : public ComputeContext
    {
    public:
        using SharedPtr = std::shared_ptr<RenderContext>;
        using SharedConstPtr = std::shared_ptr<const RenderContext>;

        ~RenderContext();

        /**
            This flag control which aspects of the GraphicState will be bound into the pipeline before drawing.
            It is useful in cases where the user wants to set a specific object using a raw-API call before calling one of the draw functions
        */
        enum class StateBindFlags : uint32_t
        {
            None            = 0x0,              ///<Bind Nothing
            Vars            = 0x1,              ///<Bind Graphics Vars (root signature and sets)
            Topology        = 0x2,              ///<Bind Primitive Topology
            Vao             = 0x4,              ///<Bind Vao
            Fbo             = 0x8,              ///<Bind Fbo
            Viewports       = 0x10,             ///<Bind Viewport
            Scissors        = 0x20,             ///<Bind scissors
            PipelineState   = 0x40,             ///<Bind Pipeline State Object
            SamplePositions = 0x80,             ///<Set the programmable sample positions
            All             = uint32_t(-1)
        };

        /** Create a new object.
        */
        static SharedPtr create(CommandQueueHandle queue);

        /** Clear an FBO.
            \param[in] pFbo The FBO to clear
            \param[in] color The clear color for the bound render-targets
            \param[in] depth The depth clear value
            \param[in] stencil The stencil clear value
            \param[in] flags Optional. Which components of the FBO to clear. By default will clear all attached resource.
            If you'd like to clear a specific color target, you can use RenderContext#clearFboColorTarget().
        */
        void clearFbo(const Fbo* pFbo, const glm::vec4& color, float depth, uint8_t stencil, FboAttachmentType flags = FboAttachmentType::All);

        /** Clear a render-target view.
            \param[in] pRtv The RTV to clear
            \param[in] color The clear color
        */
        void clearRtv(const RenderTargetView* pRtv, const glm::vec4& color);

        /** Clear a depth-stencil view.
            \param[in] pDsv The DSV to clear
            \param[in] depth The depth clear value
            \param[in] stencil The stencil clear value
            \param[in] clearDepth Optional. Controls whether or not to clear the depth channel
            \param[in] clearStencil Optional. Controls whether or not to clear the stencil channel
        */
        void clearDsv(const DepthStencilView* pDsv, float depth, uint8_t stencil, bool clearDepth = true, bool clearStencil = true);

        /** Ordered draw call.
            \param[in] vertexCount Number of vertices to draw
            \param[in] startVertexLocation The location of the first vertex to read from the vertex buffers (offset in vertices)
        */
        void draw(uint32_t vertexCount, uint32_t startVertexLocation);

        /** Ordered instanced draw call.
            \param[in] vertexCount Number of vertices to draw
            \param[in] instanceCount Number of instances to draw
            \param[in] startVertexLocation The location of the first vertex to read from the vertex buffers (offset in vertices)
            \param[in] startInstanceLocation A value which is added to each index before reading per-instance data from the vertex buffer
        */
        void drawInstanced(uint32_t vertexCount, uint32_t instanceCount, uint32_t startVertexLocation, uint32_t startInstanceLocation);

        /** Indexed draw call.
            \param[in] indexCount Number of indices to draw
            \param[in] startIndexLocation The location of the first index to read from the index buffer (offset in indices)
            \param[in] baseVertexLocation A value which is added to each index before reading a vertex from the vertex buffer
        */
        void drawIndexed(uint32_t indexCount, uint32_t startIndexLocation, int32_t baseVertexLocation);

        /** Indexed instanced draw call.
            \param[in] indexCount Number of indices to draw per instance
            \param[in] instanceCount Number of instances to draw
            \param[in] startIndexLocation The location of the first index to read from the index buffer (offset in indices)
            \param[in] baseVertexLocation A value which is added to each index before reading a vertex from the vertex buffer
            \param[in] startInstanceLocation A value which is added to each index before reading per-instance data from the vertex buffer
        */
        void drawIndexedInstanced(uint32_t indexCount, uint32_t instanceCount, uint32_t startIndexLocation, int32_t baseVertexLocation, uint32_t startInstanceLocation);

        /** Executes an indirect draw call.
            \param[in] pArgBuffer Buffer containing draw arguments
            \param[in] argBufferOffset Offset into buffer to read arguments from
        */
        void drawIndirect(const Buffer* pArgBuffer, uint64_t argBufferOffset);

        /** Executes an indirect draw-indexed call.
            \param[in] pArgBuffer Buffer containing draw arguments
            \param[in] argBufferOffset Offset into buffer to read arguments from
        */
        void drawIndexedIndirect(const Buffer* pArgBuffer, uint64_t argBufferOffset);

        /** Blits (low-level copy) an SRV into an RTV.
            \param[in] pSrc Source view to copy from
            \param[in] pDst Target view to copy to
            \param[in] srcRect Source rectangle to blit from, specified by [left, up, right, down]
            \param[in] dstRect Target rectangle to blit to, specified by [left, up, right, down]
        */
        void blit(ShaderResourceView::SharedPtr pSrc, RenderTargetView::SharedPtr pDst, const uvec4& srcRect = uvec4(-1), const uvec4& dstRect = uvec4(-1), Sampler::Filter = Sampler::Filter::Linear);

        /** Set the program variables for graphics
        */
        void setGraphicsVars(const GraphicsVars::SharedPtr& pVars) { mBindGraphicsRootSig = true;/* mBindGraphicsRootSig || (mpGraphicsVars != pVars)*/; mpGraphicsVars = pVars; }
        
        /** Get the bound graphics program variables object
        */
        const GraphicsVars::SharedPtr& getGraphicsVars() const { return mpGraphicsVars; }

        /** Push the current graphics vars and sets a new one
        */
        void pushGraphicsVars(const GraphicsVars::SharedPtr& pVars);

        /** Pops the last ProgramVars from the stack and sets it
        */
        void popGraphicsVars();

        /** Set a graphics state
        */
        void setGraphicsState(const GraphicsState::SharedPtr& pState) { mpGraphicsState = pState; }
        
        /** Get the currently bound graphics state
        */
        GraphicsState::SharedPtr getGraphicsState() const { return mpGraphicsState; }

        /** Push the current graphics state and sets a new one
        */
        void pushGraphicsState(const GraphicsState::SharedPtr& pState);

        /** Pops the last graphics state from the stack and sets it
        */
        void popGraphicsState();
        
        /** Submit the command list
        */
        void flush(bool wait = false) override;

        /** Tell the render context what it should and shouldn't bind before drawing
        */
        void setBindFlags(StateBindFlags flags) { mBindFlags = flags; }

        /** Get the render context bind flags so the user can restore the state after setBindFlags()
        */
        StateBindFlags getBindFlags() const { return mBindFlags; }

        /** Resolve an entire multi-sampled resource. The dst and src resources must have the same dimensions, array-size, mip-count and format.
            If any of these properties don't match, you'll have to use `resolveSubresource`
        */
        void resolveResource(const Texture::SharedPtr& pSrc, const Texture::SharedPtr& pDst);

        /** Resolve a multi-sampled sub-resource
        */
        void resolveSubresource(const Texture::SharedPtr& pSrc, uint32_t srcSubresource, const Texture::SharedPtr& pDst, uint32_t dstSubresource);

        /** Submit a raytrace command. This function doesn't change the state of the render-context. Graphics/compute vars and state will stay the same
        */
        deprecate("3.3", "Ray dispatch now accepts depth as a parameter. Using the deprecated version will assume depth = 1.")
        void raytrace(std::shared_ptr<RtProgramVars> pVars, std::shared_ptr<RtState> pState, uint32_t width, uint32_t height);
        void raytrace(std::shared_ptr<RtProgramVars> pVars, std::shared_ptr<RtState> pState, uint32_t width, uint32_t height, uint32_t depth);

    private:
        RenderContext();
        GraphicsVars::SharedPtr mpGraphicsVars;
        GraphicsState::SharedPtr mpGraphicsState;
        bool mBindGraphicsRootSig = true;

        std::stack<GraphicsState::SharedPtr> mPipelineStateStack;
        std::stack<GraphicsVars::SharedPtr> mpGraphicsVarsStack;

        /** Creates command signatures for DrawIndirect, DrawIndexedIndirect. Also calls
        compute context's initDispatchCommandSignature() to create command signature for dispatchIndirect
        */
        static void initDrawCommandSignatures();
        void applyGraphicsVars();

        // Internal functions used by the API layers
        void prepareForDraw();
        StateBindFlags mBindFlags = StateBindFlags::All;
    };

    enum_class_operators(RenderContext::StateBindFlags);

#ifndef FALCOR_VK
    struct BlitData
    {
        FullScreenPass::UniquePtr pPass;
        GraphicsVars::SharedPtr pVars;
        GraphicsState::SharedPtr pState;

        Sampler::SharedPtr pLinearSampler;
        Sampler::SharedPtr pPointSampler;

        ConstantBuffer::SharedPtr pSrcRectBuffer;
        vec2 prevSrcRectOffset = vec2(0,0);
        vec2 prevSrcReftScale = vec2(0,0);

        // Variable offsets in constant buffer
        size_t offsetVarOffset = ConstantBuffer::kInvalidOffset;
        size_t scaleVarOffset = ConstantBuffer::kInvalidOffset;;

        ProgramReflection::BindLocation texBindLoc;
        ProgramReflection::BindLocation samplerBindLoc;
    };

    dlldecl BlitData gBlitData;
#endif
    dlldecl CommandSignatureHandle gpDrawCommandSig;
    dlldecl CommandSignatureHandle gpDrawIndexCommandSig;
}
