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
#include "Core/API/RenderContext.h"
#include "Core/API/BlitContext.h"
#include "Core/API/Device.h"
#include "Core/API/D3D12/D3D12State.h"
#include "Core/API/D3D12/D3D12RtAccelerationStructure.h"
#include "Core/API/D3D12/D3D12API.h"
#include "Core/Program/RtProgram.h"
#include "RenderGraph/BasePasses/FullScreenPass.h"
#include <glm/gtc/type_ptr.hpp>

namespace Falcor
{
    namespace
    {
        struct RenderContextApiData
        {
            size_t refCount = 0;

            CommandSignatureHandle pDrawCommandSig;
            CommandSignatureHandle pDrawIndexCommandSig;

            BlitContext blitData;

            static void init();
            static void release();
        };

        RenderContextApiData sApiData;

        void RenderContextApiData::init()
        {
            FALCOR_ASSERT(gpDevice);
            auto& blitData = sApiData.blitData;
            if (blitData.pPass == nullptr)
            {
                blitData.init();

                // Init the draw signature
                D3D12_COMMAND_SIGNATURE_DESC sigDesc;
                sigDesc.NumArgumentDescs = 1;
                sigDesc.NodeMask = 0;
                D3D12_INDIRECT_ARGUMENT_DESC argDesc;

                // Draw
                sigDesc.ByteStride = sizeof(D3D12_DRAW_ARGUMENTS);
                argDesc.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DRAW;
                sigDesc.pArgumentDescs = &argDesc;
                FALCOR_D3D_CALL(gpDevice->getApiHandle()->CreateCommandSignature(&sigDesc, nullptr, IID_PPV_ARGS(&sApiData.pDrawCommandSig)));

                // Draw index
                sigDesc.ByteStride = sizeof(D3D12_DRAW_INDEXED_ARGUMENTS);
                argDesc.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED;
                sigDesc.pArgumentDescs = &argDesc;
                FALCOR_D3D_CALL(gpDevice->getApiHandle()->CreateCommandSignature(&sigDesc, nullptr, IID_PPV_ARGS(&sApiData.pDrawIndexCommandSig)));
            }

            sApiData.refCount++;
        }

        void RenderContextApiData::release()
        {
            sApiData.refCount--;
            if (sApiData.refCount == 0)
            {
                sApiData.blitData.release();
                sApiData = {};
            }
        }
    }

    RenderContext::RenderContext(CommandQueueHandle queue)
        : ComputeContext(LowLevelContextData::CommandQueueType::Direct, queue)
    {
        RenderContextApiData::init();
    }

    RenderContext::~RenderContext()
    {
        RenderContextApiData::release();
    }

    BlitContext& RenderContext::getBlitContext() { return sApiData.blitData; }

    void RenderContext::clearRtv(const RenderTargetView* pRtv, const float4& color)
    {
        resourceBarrier(pRtv->getResource().get(), Resource::State::RenderTarget);
        mpLowLevelData->getCommandList()->ClearRenderTargetView(pRtv->getApiHandle()->getCpuHandle(0), glm::value_ptr(color), 0, nullptr);
        mCommandsPending = true;
    }

    void RenderContext::clearDsv(const DepthStencilView* pDsv, float depth, uint8_t stencil, bool clearDepth, bool clearStencil)
    {
        uint32_t flags = clearDepth ? D3D12_CLEAR_FLAG_DEPTH : 0;
        flags |= clearStencil ? D3D12_CLEAR_FLAG_STENCIL : 0;

        resourceBarrier(pDsv->getResource().get(), Resource::State::DepthStencil);
        mpLowLevelData->getCommandList()->ClearDepthStencilView(pDsv->getApiHandle()->getCpuHandle(0), D3D12_CLEAR_FLAGS(flags), depth, stencil, 0, nullptr);
        mCommandsPending = true;
    }

    static void D3D12SetVao(RenderContext* pCtx, ID3D12GraphicsCommandList* pList, const Vao* pVao)
    {
        D3D12_VERTEX_BUFFER_VIEW vb[D3D12_IA_VERTEX_INPUT_RESOURCE_SLOT_COUNT] = {};
        D3D12_INDEX_BUFFER_VIEW ib = {};

        if (pVao)
        {
            // Get the vertex buffers
            for (uint32_t i = 0; i < pVao->getVertexBuffersCount(); i++)
            {
                const Buffer* pVB = pVao->getVertexBuffer(i).get();
                if (pVB)
                {
                    vb[i].BufferLocation = pVB->getGpuAddress();
                    vb[i].SizeInBytes = (uint32_t)pVB->getSize();
                    vb[i].StrideInBytes = pVao->getVertexLayout()->getBufferLayout(i)->getStride();
                    pCtx->resourceBarrier(pVB, Resource::State::VertexBuffer);
                }
            }

            const Buffer* pIB = pVao->getIndexBuffer().get();
            if (pIB)
            {
                ib.BufferLocation = pIB->getGpuAddress();
                ib.SizeInBytes = (uint32_t)pIB->getSize();
                ib.Format = getDxgiFormat(pVao->getIndexBufferFormat());
                pCtx->resourceBarrier(pIB, Resource::State::IndexBuffer);
            }
        }

        pList->IASetVertexBuffers(0, (uint32_t)std::size(vb), vb);
        pList->IASetIndexBuffer(&ib);
    }

    static void D3D12SetFbo(RenderContext* pCtx, const Fbo* pFbo)
    {
        // We are setting the entire RTV array to make sure everything that was previously bound is detached.
        // We're using 2D null views for any unused slots.
        uint32_t colorTargets = Fbo::getMaxColorTargetCount();
        auto pNullRtv = RenderTargetView::getNullView(RenderTargetView::Dimension::Texture2D);
        std::vector<D3D12DescriptorCpuHandle> pRTV(colorTargets, pNullRtv->getApiHandle()->getCpuHandle(0));
        D3D12DescriptorCpuHandle pDSV = DepthStencilView::getNullView(DepthStencilView::Dimension::Texture2D)->getApiHandle()->getCpuHandle(0);

        if (pFbo)
        {
            for (uint32_t i = 0; i < colorTargets; i++)
            {
                auto pTexture = pFbo->getColorTexture(i);
                if (pTexture)
                {
                    pRTV[i] = pFbo->getRenderTargetView(i)->getApiHandle()->getCpuHandle(0);
                    pCtx->resourceBarrier(pTexture.get(), Resource::State::RenderTarget);
                }
            }

            auto& pTexture = pFbo->getDepthStencilTexture();
            if (pTexture)
            {
                pDSV = pFbo->getDepthStencilView()->getApiHandle()->getCpuHandle(0);
                if (pTexture)
                {
                    pCtx->resourceBarrier(pTexture.get(), Resource::State::DepthStencil);
                }
            }
        }
        ID3D12GraphicsCommandList* pCmdList = pCtx->getLowLevelData()->getCommandList().GetInterfacePtr();
        pCmdList->OMSetRenderTargets(colorTargets, pRTV.data(), FALSE, &pDSV);
    }

    static void D3D12SetSamplePositions(ID3D12GraphicsCommandList* pList, const Fbo* pFbo)
    {
        if (!pFbo) return;
        ID3D12GraphicsCommandList1* pList1;
        pList->QueryInterface(IID_PPV_ARGS(&pList1));

        bool featureSupported = gpDevice->isFeatureSupported(Device::SupportedFeatures::ProgrammableSamplePositionsPartialOnly) ||
            gpDevice->isFeatureSupported(Device::SupportedFeatures::ProgrammableSamplePositionsFull);

        const auto& samplePos = pFbo->getSamplePositions();

        if (featureSupported == false && samplePos.size() > 0)
        {
            throw RuntimeError("The FBO specifies programmable sample positions, but the hardware does not support it");
        }
        else if (gpDevice->isFeatureSupported(Device::SupportedFeatures::ProgrammableSamplePositionsPartialOnly) && samplePos.size() > 1)
        {
            throw RuntimeError("The FBO specifies multiple programmable sample positions, but the hardware only supports one");
        }

        if (featureSupported)
        {
            static_assert(offsetof(Fbo::SamplePosition, xOffset) == offsetof(D3D12_SAMPLE_POSITION, X), "SamplePosition.X");
            static_assert(offsetof(Fbo::SamplePosition, yOffset) == offsetof(D3D12_SAMPLE_POSITION, Y), "SamplePosition.Y");

            if (samplePos.size())
            {
                pList1->SetSamplePositions(pFbo->getSampleCount(), pFbo->getSamplePositionsPixelCount(), (D3D12_SAMPLE_POSITION*)samplePos.data());
            }
            else
            {
                pList1->SetSamplePositions(0, 0, nullptr);
            }
        }
    }

    static void D3D12SetViewports(ID3D12GraphicsCommandList* pList, const GraphicsState::Viewport* vp)
    {
        static_assert(offsetof(GraphicsState::Viewport, originX) == offsetof(D3D12_VIEWPORT, TopLeftX), "VP originX offset");
        static_assert(offsetof(GraphicsState::Viewport, originY) == offsetof(D3D12_VIEWPORT, TopLeftY), "VP originY offset");
        static_assert(offsetof(GraphicsState::Viewport, width) == offsetof(D3D12_VIEWPORT, Width), "VP Width offset");
        static_assert(offsetof(GraphicsState::Viewport, height) == offsetof(D3D12_VIEWPORT, Height), "VP Height offset");
        static_assert(offsetof(GraphicsState::Viewport, minDepth) == offsetof(D3D12_VIEWPORT, MinDepth), "VP MinDepth offset");
        static_assert(offsetof(GraphicsState::Viewport, maxDepth) == offsetof(D3D12_VIEWPORT, MaxDepth), "VP TopLeftX offset");

        pList->RSSetViewports(D3D12_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE, (D3D12_VIEWPORT*)vp);
    }

    static void D3D12SetScissors(ID3D12GraphicsCommandList* pList, const GraphicsState::Scissor* sc)
    {
        static_assert(offsetof(GraphicsState::Scissor, left) == offsetof(D3D12_RECT, left), "Scissor.left offset");
        static_assert(offsetof(GraphicsState::Scissor, top) == offsetof(D3D12_RECT, top), "Scissor.top offset");
        static_assert(offsetof(GraphicsState::Scissor, right) == offsetof(D3D12_RECT, right), "Scissor.right offset");
        static_assert(offsetof(GraphicsState::Scissor, bottom) == offsetof(D3D12_RECT, bottom), "Scissor.bottom offset");

        pList->RSSetScissorRects(D3D12_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE, (D3D12_RECT*)sc);
    }

    void RenderContext::applyGraphicsVars(GraphicsVars* pVars, const ProgramKernels* pProgramKernels)
    {
        bool bindRootSig = (pVars != mpLastBoundGraphicsVars);
        if (pVars->apply(this, bindRootSig, pProgramKernels) == false)
        {
            logWarning("RenderContext::prepareForDraw() - applying GraphicsVars failed, most likely because we ran out of descriptors. Flushing the GPU and retrying");
            flush(true);
            if (!pVars->apply(this, bindRootSig, pProgramKernels))
            {
                throw RuntimeError("RenderContext::applyGraphicsVars() - applying GraphicsVars failed, most likely because we ran out of descriptors");
            }
        }
    }

    void RenderContext::prepareForDraw(GraphicsState* pState, GraphicsVars* pVars)
    {
        FALCOR_ASSERT(pState);
        // Vao must be valid so at least primitive topology is known
        FALCOR_ASSERT(pState->getVao().get());

        auto pGSO = pState->getGSO(pVars);

        if (is_set(StateBindFlags::Vars, mBindFlags))
        {
            // Apply the vars. Must be first because applyGraphicsVars() might cause a flush
            if (pVars)
            {
                // TODO(tfoley): Need to find a way to pass the specialization information
                // from computing the GSO down into `applyGraphicsVars` so that parameters
                // can be bound using an appropriate layout.
                //
                applyGraphicsVars(pVars, pGSO->getDesc().getProgramKernels().get());
            }
            else
            {
                mpLowLevelData->getCommandList()->SetGraphicsRootSignature(D3D12RootSignature::getEmpty()->getApiHandle());
            }
            mpLastBoundGraphicsVars = pVars;
        }

        ID3D12GraphicsCommandList* pList = mpLowLevelData->getCommandList();


        if (is_set(StateBindFlags::Topology, mBindFlags))           pList->IASetPrimitiveTopology(getD3DPrimitiveTopology(pState->getVao()->getPrimitiveTopology()));
        if (is_set(StateBindFlags::Vao, mBindFlags))                D3D12SetVao(this, pList, pState->getVao().get());
        if (is_set(StateBindFlags::Fbo, mBindFlags))                D3D12SetFbo(this, pState->getFbo().get());
        if (is_set(StateBindFlags::SamplePositions, mBindFlags))    D3D12SetSamplePositions(pList, pState->getFbo().get());
        if (is_set(StateBindFlags::Viewports, mBindFlags))          D3D12SetViewports(pList, &pState->getViewport(0));
        if (is_set(StateBindFlags::Scissors, mBindFlags))           D3D12SetScissors(pList, &pState->getScissors(0));
        if (is_set(StateBindFlags::PipelineState, mBindFlags))      pList->SetPipelineState(pGSO->getApiHandle());

        BlendState::SharedPtr blendState = pState->getBlendState();
        if (blendState != nullptr)  pList->OMSetBlendFactor(glm::value_ptr(blendState->getBlendFactor()));

        const auto pDsState = pState->getDepthStencilState();
        pList->OMSetStencilRef(pDsState == nullptr ? 0 : pDsState->getStencilRef());

        mCommandsPending = true;
    }

    void RenderContext::drawInstanced(GraphicsState* pState, GraphicsVars* pVars, uint32_t vertexCount, uint32_t instanceCount, uint32_t startVertexLocation, uint32_t startInstanceLocation)
    {
        prepareForDraw(pState, pVars);
        mpLowLevelData->getCommandList()->DrawInstanced(vertexCount, instanceCount, startVertexLocation, startInstanceLocation);
    }

    void RenderContext::draw(GraphicsState* pState, GraphicsVars* pVars, uint32_t vertexCount, uint32_t startVertexLocation)
    {
        drawInstanced(pState, pVars, vertexCount, 1, startVertexLocation, 0);
    }

    void RenderContext::drawIndexedInstanced(GraphicsState* pState, GraphicsVars* pVars, uint32_t indexCount, uint32_t instanceCount, uint32_t startIndexLocation, int32_t baseVertexLocation, uint32_t startInstanceLocation)
    {
        prepareForDraw(pState, pVars);
        mpLowLevelData->getCommandList()->DrawIndexedInstanced(indexCount, instanceCount, startIndexLocation, baseVertexLocation, startInstanceLocation);
    }

    void RenderContext::drawIndexed(GraphicsState* pState, GraphicsVars* pVars, uint32_t indexCount, uint32_t startIndexLocation, int32_t baseVertexLocation)
    {
        drawIndexedInstanced(pState, pVars, indexCount, 1, startIndexLocation, baseVertexLocation, 0);
    }

    void drawIndirectCommon(RenderContext* pContext, const CommandListHandle& pCommandList, ID3D12CommandSignature* pCommandSig, uint32_t maxCommandCount, const Buffer* pArgBuffer, uint64_t argBufferOffset, const Buffer* pCountBuffer, uint64_t countBufferOffset)
    {
        pContext->resourceBarrier(pArgBuffer, Resource::State::IndirectArg);
        if (pCountBuffer != nullptr && pCountBuffer != pArgBuffer) pContext->resourceBarrier(pCountBuffer, Resource::State::IndirectArg);
        pCommandList->ExecuteIndirect(pCommandSig, maxCommandCount, pArgBuffer->getApiHandle(), argBufferOffset, (pCountBuffer != nullptr ? pCountBuffer->getApiHandle() : nullptr), countBufferOffset);
    }

    void RenderContext::drawIndirect(GraphicsState* pState, GraphicsVars* pVars, uint32_t maxCommandCount, const Buffer* pArgBuffer, uint64_t argBufferOffset, const Buffer* pCountBuffer, uint64_t countBufferOffset)
    {
        prepareForDraw(pState, pVars);
        drawIndirectCommon(this, mpLowLevelData->getCommandList(), sApiData.pDrawCommandSig, maxCommandCount, pArgBuffer, argBufferOffset, pCountBuffer, countBufferOffset);
    }

    void RenderContext::drawIndexedIndirect(GraphicsState* pState, GraphicsVars* pVars, uint32_t maxCommandCount, const Buffer* pArgBuffer, uint64_t argBufferOffset, const Buffer* pCountBuffer, uint64_t countBufferOffset)
    {
        prepareForDraw(pState, pVars);
        drawIndirectCommon(this, mpLowLevelData->getCommandList(), sApiData.pDrawIndexCommandSig, maxCommandCount, pArgBuffer, argBufferOffset, pCountBuffer, countBufferOffset);
    }

    void RenderContext::raytrace(RtProgram* pProgram, RtProgramVars* pVars, uint32_t width, uint32_t height, uint32_t depth)
    {
        auto pRtso = pProgram->getRtso(pVars);

        pVars->apply(this, pRtso.get());

        const auto& pShaderTable = pVars->getShaderTable();
        resourceBarrier(pShaderTable->getBuffer().get(), Resource::State::NonPixelShader);

        D3D12_GPU_VIRTUAL_ADDRESS startAddress = pShaderTable->getBuffer()->getGpuAddress();

        D3D12_DISPATCH_RAYS_DESC raytraceDesc = {};
        raytraceDesc.Width = width;
        raytraceDesc.Height = height;
        raytraceDesc.Depth = depth;

        // RayGen data
        //
        // TODO: We could easily support specifying the ray-gen program to invoke by an index in
        // the call to `raytrace()`.
        //
        raytraceDesc.RayGenerationShaderRecord.StartAddress = startAddress + pShaderTable->getRayGenTableOffset();
        raytraceDesc.RayGenerationShaderRecord.SizeInBytes = pShaderTable->getRayGenRecordSize();;

        // Miss data
        if (pShaderTable->getMissRecordCount() > 0)
        {
            raytraceDesc.MissShaderTable.StartAddress = startAddress + pShaderTable->getMissTableOffset();
            raytraceDesc.MissShaderTable.StrideInBytes = pShaderTable->getMissRecordSize();
            raytraceDesc.MissShaderTable.SizeInBytes = pShaderTable->getMissRecordSize() * pShaderTable->getMissRecordCount();
        }

        // Hit data
        if (pShaderTable->getHitRecordCount() > 0)
        {
            raytraceDesc.HitGroupTable.StartAddress = startAddress + pShaderTable->getHitTableOffset();
            raytraceDesc.HitGroupTable.StrideInBytes = pShaderTable->getHitRecordSize();
            raytraceDesc.HitGroupTable.SizeInBytes = pShaderTable->getHitRecordSize() * pShaderTable->getHitRecordCount();
        }

        auto pCmdList = getLowLevelData()->getCommandList();
        pCmdList->SetComputeRootSignature(pRtso->getKernels()->getD3D12RootSignature()->getApiHandle().GetInterfacePtr());

        // Dispatch
        FALCOR_GET_COM_INTERFACE(pCmdList, ID3D12GraphicsCommandList4, pList4);
        pList4->SetPipelineState1(pRtso->getApiHandle().GetInterfacePtr());
        pList4->DispatchRays(&raytraceDesc);
    }

    void RenderContext::resolveSubresource(const Texture::SharedPtr& pSrc, uint32_t srcSubresource, const Texture::SharedPtr& pDst, uint32_t dstSubresource)
    {
        DXGI_FORMAT format = getDxgiFormat(pDst->getFormat());
        mpLowLevelData->getCommandList()->ResolveSubresource(pDst->getApiHandle(), dstSubresource, pSrc->getApiHandle(), srcSubresource, format);
        mCommandsPending = true;
    }

    void RenderContext::resolveResource(const Texture::SharedPtr& pSrc, const Texture::SharedPtr& pDst)
    {
        bool match = true;
        match = match && (pSrc->getMipCount() == pDst->getMipCount());
        match = match && (pSrc->getArraySize() == pDst->getArraySize());
        if (!match)
        {
            logWarning("Can't resolve a resource. The src and dst textures have a different array-size or mip-count");
        }

        resourceBarrier(pSrc.get(), Resource::State::ResolveSource);
        resourceBarrier(pDst.get(), Resource::State::ResolveDest);

        uint32_t subresourceCount = pSrc->getMipCount() * pSrc->getArraySize();
        for (uint32_t s = 0; s < subresourceCount; s++)
        {
            resolveSubresource(pSrc, s, pDst, s);
        }
    }

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_TYPE translatePostBuildInfoType(RtAccelerationStructurePostBuildInfoQueryType);

    void RenderContext::buildAccelerationStructure(const RtAccelerationStructure::BuildDesc& desc, uint32_t postBuildInfoCount, RtAccelerationStructurePostBuildInfoDesc* pPostBuildInfoDescs)
    {
        // Translate BuildInputs.
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC buildDesc = {};
        D3D12AccelerationStructureBuildInputsTranslator translator;
        buildDesc.Inputs = translator.translate(desc.inputs);
        FALCOR_ASSERT(desc.dest);
        buildDesc.DestAccelerationStructureData = desc.dest->getGpuAddress();
        buildDesc.ScratchAccelerationStructureData = desc.scratchData;
        buildDesc.SourceAccelerationStructureData = desc.source ? desc.source->getGpuAddress() : 0;

        // Translate PostBuildInfoDesc.
        std::vector<D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC> postBuildInfoDescs;
        postBuildInfoDescs.resize(postBuildInfoCount);
        for (uint32_t i = 0; i < postBuildInfoCount; ++i)
        {
            postBuildInfoDescs[i].DestBuffer = pPostBuildInfoDescs[i].pool->getBufferAddress(pPostBuildInfoDescs[i].index);
            postBuildInfoDescs[i].InfoType = translatePostBuildInfoType(pPostBuildInfoDescs[i].type);
        }

        // Dispatch D3D12 command to build the acceleration structure.
        FALCOR_GET_COM_INTERFACE(getLowLevelData()->getCommandList(), ID3D12GraphicsCommandList4, pList4);
        pList4->BuildRaytracingAccelerationStructure(&buildDesc, (UINT)postBuildInfoDescs.size(), postBuildInfoDescs.size() > 0 ? postBuildInfoDescs.data() : nullptr);
    }

    void RenderContext::copyAccelerationStructure(RtAccelerationStructure* dest, RtAccelerationStructure* source, RenderContext::RtAccelerationStructureCopyMode mode)
    {
        FALCOR_GET_COM_INTERFACE(getLowLevelData()->getCommandList(), ID3D12GraphicsCommandList4, pList4);
        pList4->CopyRaytracingAccelerationStructure(
            dest->getGpuAddress(),
            source->getGpuAddress(),
            mode == RtAccelerationStructureCopyMode::Compact
                ? D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_COMPACT
                : D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_CLONE);
    }
}
