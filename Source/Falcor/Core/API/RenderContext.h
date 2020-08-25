/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include <stack>
#include <vector>
#include "ComputeContext.h"
#include "Sampler.h"
#include "Core/State/GraphicsState.h"

namespace Falcor
{
#ifdef FALCOR_D3D12
    class RtProgram;
    class RtProgramVars;
#endif

    class FullScreenPass;

    /** The rendering context. Use it to bind state and dispatch calls to the GPU
    */
    class dlldecl RenderContext : public ComputeContext
    {
    public:
        using SharedPtr = std::shared_ptr<RenderContext>;
        using SharedConstPtr = std::shared_ptr<const RenderContext>;

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

        ~RenderContext();

        /** Create a new render context.
            \param[in] queue The command queue.
            \return A new object, or throws an exception if creation failed.
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
        void clearFbo(const Fbo* pFbo, const float4& color, float depth, uint8_t stencil, FboAttachmentType flags = FboAttachmentType::All);

        /** Clear a render-target view.
            \param[in] pRtv The RTV to clear
            \param[in] color The clear color
        */
        void clearRtv(const RenderTargetView* pRtv, const float4& color);

        /** Clear a depth-stencil view.
            \param[in] pDsv The DSV to clear
            \param[in] depth The depth clear value
            \param[in] stencil The stencil clear value
            \param[in] clearDepth Optional. Controls whether or not to clear the depth channel
            \param[in] clearStencil Optional. Controls whether or not to clear the stencil channel
        */
        void clearDsv(const DepthStencilView* pDsv, float depth, uint8_t stencil, bool clearDepth = true, bool clearStencil = true);

        /** Clear a texture. The function will use the bind-flags to find the optimal API call to make
            \param[in] pTexture The texture to clear
            \param[in] clearColor The clear color
            The function only support floating-point and normalized color-formats and depth. For depth buffers, `clearColor.x` will be used. If there's a stencil-channel, `clearColor.y` must be zero
        */
        void clearTexture(Texture* pTexture, const float4& clearColor = float4(0, 0, 0, 1));

        /** Ordered draw call.
            \param[in] vertexCount Number of vertices to draw
            \param[in] startVertexLocation The location of the first vertex to read from the vertex buffers (offset in vertices)
        */
        void draw(GraphicsState* pState, GraphicsVars* pVars, uint32_t vertexCount, uint32_t startVertexLocation);

        /** Ordered instanced draw call.
            \param[in] vertexCount Number of vertices to draw
            \param[in] instanceCount Number of instances to draw
            \param[in] startVertexLocation The location of the first vertex to read from the vertex buffers (offset in vertices)
            \param[in] startInstanceLocation A value which is added to each index before reading per-instance data from the vertex buffer
        */
        void drawInstanced(GraphicsState* pState, GraphicsVars* pVars, uint32_t vertexCount, uint32_t instanceCount, uint32_t startVertexLocation, uint32_t startInstanceLocation);

        /** Indexed draw call.
            \param[in] indexCount Number of indices to draw
            \param[in] startIndexLocation The location of the first index to read from the index buffer (offset in indices)
            \param[in] baseVertexLocation A value which is added to each index before reading a vertex from the vertex buffer
        */
        void drawIndexed(GraphicsState* pState, GraphicsVars* pVars, uint32_t indexCount, uint32_t startIndexLocation, int32_t baseVertexLocation);

        /** Indexed instanced draw call.
            \param[in] indexCount Number of indices to draw per instance
            \param[in] instanceCount Number of instances to draw
            \param[in] startIndexLocation The location of the first index to read from the index buffer (offset in indices)
            \param[in] baseVertexLocation A value which is added to each index before reading a vertex from the vertex buffer
            \param[in] startInstanceLocation A value which is added to each index before reading per-instance data from the vertex buffer
        */
        void drawIndexedInstanced(GraphicsState* pState, GraphicsVars* pVars, uint32_t indexCount, uint32_t instanceCount, uint32_t startIndexLocation, int32_t baseVertexLocation, uint32_t startInstanceLocation);

        /** Executes an indirect draw call.
            \param[in] maxCommandCount If pCountBuffer is null, this specifies the command count. Otherwise, command count is minimum of maxCommandCount and the value contained in pCountBuffer
            \param[in] pArgBuffer Buffer containing draw arguments
            \param[in] argBufferOffset Offset into buffer to read arguments from
            \param[in] pCountBuffer Optional. A GPU buffer that contains a uint32 value specifying the command count. This can, but does not have to be a dedicated buffer
            \param[in] countBufferOffset Offset into pCountBuffer to read the value from
        */
        void drawIndirect(GraphicsState* pState, GraphicsVars* pVars, uint32_t maxCommandCount, const Buffer* pArgBuffer, uint64_t argBufferOffset, const Buffer* pCountBuffer, uint64_t countBufferOffset);

        /** Executes an indirect draw-indexed call.
            \param[in] maxCommandCount If pCountBuffer is null, this specifies the command count. Otherwise, command count is minimum of maxCommandCount and the value contained in pCountBuffer
            \param[in] pArgBuffer Buffer containing draw arguments
            \param[in] argBufferOffset Offset into buffer to read arguments from
            \param[in] pCountBuffer Optional. A GPU buffer that contains a uint32 value specifying the command count. This can, but does not have to be a dedicated buffer
            \param[in] countBufferOffset Offset into pCountBuffer to read the value from
        */
        void drawIndexedIndirect(GraphicsState* pState, GraphicsVars* pVars, uint32_t maxCommandCount, const Buffer* pArgBuffer, uint64_t argBufferOffset, const Buffer* pCountBuffer, uint64_t countBufferOffset);

        /** Blits (low-level copy) an SRV into an RTV.
            \param[in] pSrc Source view to copy from
            \param[in] pDst Target view to copy to
            \param[in] srcRect Source rectangle to blit from, specified by [left, up, right, down]
            \param[in] dstRect Target rectangle to blit to, specified by [left, up, right, down]
        */
        void blit(ShaderResourceView::SharedPtr pSrc, RenderTargetView::SharedPtr pDst, const uint4& srcRect = uint4(-1), const uint4& dstRect = uint4(-1), Sampler::Filter = Sampler::Filter::Linear);

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

#ifdef FALCOR_D3D12
        /** Submit a raytrace command. This function doesn't change the state of the render-context. Graphics/compute vars and state will stay the same.
        */
        void raytrace(RtProgram* pProgram, RtProgramVars* pVars, uint32_t width, uint32_t height, uint32_t depth);
#endif

    private:
        RenderContext(CommandQueueHandle queue);
        bool applyGraphicsVars(GraphicsVars* pVars, RootSignature* pRootSignature);
        bool prepareForDraw(GraphicsState* pState, GraphicsVars* pVars);

        StateBindFlags mBindFlags = StateBindFlags::All;
        GraphicsVars* mpLastBoundGraphicsVars = nullptr;
    };

    enum_class_operators(RenderContext::StateBindFlags);
}
