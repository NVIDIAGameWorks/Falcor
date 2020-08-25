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
#include "Core/API/CopyContext.h"
#include "Core/API/RootSignature.h"

namespace Falcor
{
    class dlldecl RtVarsCmdList : public ID3D12GraphicsCommandList4
    {
    public:
        using SharedPtr = std::shared_ptr<RtVarsCmdList>;

        /** Create a new ray tracing command list.
            \return New object, or throws an exception if creation failed.
        */
        static SharedPtr create() { return SharedPtr(new RtVarsCmdList); }

        void SetGraphicsRootSignature(ID3D12RootSignature *pRootSignature) {};
        void SetGraphicsRootDescriptorTable(UINT RootParameterIndex, D3D12_GPU_DESCRIPTOR_HANDLE BaseDescriptor);
        void SetGraphicsRoot32BitConstant(UINT RootParameterIndex, UINT SrcData, UINT DestOffsetIn32BitValues);
        void SetGraphicsRoot32BitConstants(UINT RootParameterIndex, UINT Num32BitValuesToSet, const void *pSrcData, UINT DestOffsetIn32BitValues);
        void SetGraphicsRootConstantBufferView(UINT RootParameterIndex, D3D12_GPU_VIRTUAL_ADDRESS BufferLocation);
        void SetGraphicsRootShaderResourceView(UINT RootParameterIndex, D3D12_GPU_VIRTUAL_ADDRESS BufferLocation);
        void SetGraphicsRootUnorderedAccessView(UINT RootParameterIndex, D3D12_GPU_VIRTUAL_ADDRESS BufferLocation);

        void setRootParams(RootSignature::SharedPtr pRoot, uint8_t* pBase) { mpRootBase = pBase; mpRootSignature = pRoot; }

        // The following functions should not be used
        HRESULT QueryInterface(REFIID riid, void **ppvObject);
        ULONG AddRef() { return 0; }
        ULONG Release() { return 0; }

        HRESULT GetPrivateData(REFGUID guid, UINT *pDataSize, void *pData) { should_not_get_here(); return E_FAIL; }
        HRESULT SetPrivateData(REFGUID guid, UINT DataSize, const void *pData) { should_not_get_here(); return E_FAIL; }
        HRESULT SetPrivateDataInterface(REFGUID guid, const IUnknown *pData) { should_not_get_here(); return E_FAIL; }
        HRESULT SetName(LPCWSTR name) { should_not_get_here(); return E_FAIL; }
        HRESULT GetDevice(REFIID riid, void **ppvDevice) { should_not_get_here(); return E_FAIL; }
        D3D12_COMMAND_LIST_TYPE GetType() { should_not_get_here(); return D3D12_COMMAND_LIST_TYPE_BUNDLE; }
        HRESULT Close() { should_not_get_here(); return E_FAIL; }
        HRESULT Reset(ID3D12CommandAllocator *pAllocator, ID3D12PipelineState *pInitialState) { should_not_get_here(); return E_FAIL; }
        void ClearState(ID3D12PipelineState *pPipelineState) { should_not_get_here(); }
        void DrawInstanced(UINT VertexCountPerInstance, UINT InstanceCount, UINT StartVertexLocation, UINT StartInstanceLocation) { should_not_get_here(); }
        void DrawIndexedInstanced(UINT IndexCountPerInstance, UINT InstanceCount, UINT StartIndexLocation, INT BaseVertexLocation, UINT StartInstanceLocation) { should_not_get_here(); }
        void Dispatch(UINT ThreadGroupCountX, UINT ThreadGroupCountY, UINT ThreadGroupCountZ) { should_not_get_here(); }
        void CopyBufferRegion(ID3D12Resource *pDstBuffer, UINT64 DstOffset, ID3D12Resource *pSrcBuffer, UINT64 SrcOffset, UINT64 NumBytes) { should_not_get_here(); }
        void CopyTextureRegion(const D3D12_TEXTURE_COPY_LOCATION *pDst, UINT DstX, UINT DstY, UINT DstZ, const D3D12_TEXTURE_COPY_LOCATION *pSrc, const D3D12_BOX *pSrcBox) { should_not_get_here(); }
        void CopyResource(ID3D12Resource *pDstResource, ID3D12Resource *pSrcResource) { should_not_get_here(); }
        void CopyTiles(ID3D12Resource *pTiledResource, const D3D12_TILED_RESOURCE_COORDINATE *pTileRegionStartCoordinate, const D3D12_TILE_REGION_SIZE *pTileRegionSize, ID3D12Resource *pBuffer, UINT64 BufferStartOffsetInBytes, D3D12_TILE_COPY_FLAGS Flags) { should_not_get_here(); }
        void ResolveSubresource(ID3D12Resource *pDstResource, UINT DstSubresource, ID3D12Resource *pSrcResource, UINT SrcSubresource, DXGI_FORMAT Format) { should_not_get_here(); }
        void IASetPrimitiveTopology(D3D12_PRIMITIVE_TOPOLOGY PrimitiveTopology) { should_not_get_here(); }
        void RSSetViewports(UINT NumViewports, const D3D12_VIEWPORT *pViewports) { should_not_get_here(); }
        void RSSetScissorRects(UINT NumRects, const D3D12_RECT *pRects) { should_not_get_here(); }
        void OMSetBlendFactor(const FLOAT BlendFactor[ 4 ]) { should_not_get_here(); }
        void OMSetRenderTargets(UINT NumRenderTargetDescriptors, const D3D12_CPU_DESCRIPTOR_HANDLE *pRenderTargetDescriptors, BOOL RTsSingleHandleToDescriptorRange, const D3D12_CPU_DESCRIPTOR_HANDLE *pDepthStencilDescriptor) { should_not_get_here(); }
        void OMSetStencilRef(UINT StencilRef) { should_not_get_here(); }
        void SetPipelineState(ID3D12PipelineState *pPipelineState) { should_not_get_here(); }
        void ResourceBarrier(UINT NumBarriers, const D3D12_RESOURCE_BARRIER *pBarriers) { should_not_get_here(); }
        void ExecuteBundle(ID3D12GraphicsCommandList *pCommandList) { should_not_get_here(); }
        void SetDescriptorHeaps(UINT NumDescriptorHeaps, ID3D12DescriptorHeap *const *ppDescriptorHeaps) { should_not_get_here(); }
        void SetComputeRootSignature(ID3D12RootSignature *pRootSignature) { should_not_get_here(); }
        void SetComputeRootDescriptorTable(UINT RootParameterIndex, D3D12_GPU_DESCRIPTOR_HANDLE BaseDescriptor) { should_not_get_here(); }
        void SetComputeRoot32BitConstant(UINT RootParameterIndex, UINT SrcData, UINT DestOffsetIn32BitValues) { should_not_get_here(); };
        void SetComputeRoot32BitConstants(UINT RootParameterIndex, UINT Num32BitValuesToSet, const void *pSrcData, UINT DestOffsetIn32BitValues) { should_not_get_here(); }
        void SetComputeRootConstantBufferView(UINT RootParameterIndex, D3D12_GPU_VIRTUAL_ADDRESS BufferLocation) { should_not_get_here(); }
        void SetComputeRootShaderResourceView(UINT RootParameterIndex, D3D12_GPU_VIRTUAL_ADDRESS BufferLocation) { should_not_get_here(); }
        void SetComputeRootUnorderedAccessView(UINT RootParameterIndex, D3D12_GPU_VIRTUAL_ADDRESS BufferLocation) { should_not_get_here(); }
        void IASetIndexBuffer(const D3D12_INDEX_BUFFER_VIEW *pView) { should_not_get_here(); }
        void IASetVertexBuffers(UINT StartSlot, UINT NumViews, const D3D12_VERTEX_BUFFER_VIEW *pViews) { should_not_get_here(); }
        void SOSetTargets(UINT StartSlot, UINT NumViews, const D3D12_STREAM_OUTPUT_BUFFER_VIEW *pViews) { should_not_get_here(); }
        void ClearDepthStencilView(D3D12_CPU_DESCRIPTOR_HANDLE DepthStencilView, D3D12_CLEAR_FLAGS ClearFlags, FLOAT Depth, UINT8 Stencil, UINT NumRects, const D3D12_RECT *pRects) { should_not_get_here(); }
        void ClearRenderTargetView(D3D12_CPU_DESCRIPTOR_HANDLE RenderTargetView, const FLOAT ColorRGBA[ 4 ], UINT NumRects, const D3D12_RECT *pRects) { should_not_get_here(); }
        void ClearUnorderedAccessViewFloat(D3D12_GPU_DESCRIPTOR_HANDLE ViewGPUHandleInCurrentHeap, D3D12_CPU_DESCRIPTOR_HANDLE ViewCPUHandle, ID3D12Resource *pResource, const FLOAT Values[4], UINT NumRects, const D3D12_RECT *pRects) { should_not_get_here(); }
        void ClearUnorderedAccessViewUint(D3D12_GPU_DESCRIPTOR_HANDLE ViewGPUHandleInCurrentHeap, D3D12_CPU_DESCRIPTOR_HANDLE ViewCPUHandle, ID3D12Resource *pResource, const UINT Values[4], UINT NumRects, const D3D12_RECT *pRects) { should_not_get_here(); }
        void DiscardResource(ID3D12Resource *pResource, const D3D12_DISCARD_REGION *pRegion) { should_not_get_here(); }
        void BeginQuery(ID3D12QueryHeap *pQueryHeap, D3D12_QUERY_TYPE Type, UINT Index) { should_not_get_here(); }
        void EndQuery(ID3D12QueryHeap *pQueryHeap, D3D12_QUERY_TYPE Type, UINT Index) { should_not_get_here(); }
        void ResolveQueryData(ID3D12QueryHeap *pQueryHeap, D3D12_QUERY_TYPE Type, UINT StartIndex, UINT NumQueries, ID3D12Resource *pDestinationBuffer, UINT64 AlignedDestinationBufferOffset) { should_not_get_here(); }
        void SetPredication(ID3D12Resource *pBuffer, UINT64 AlignedBufferOffset, D3D12_PREDICATION_OP Operation) { should_not_get_here(); }
        void SetMarker(UINT Metadata, const void *pData, UINT Size) { should_not_get_here(); }
        void BeginEvent(UINT Metadata, const void *pData, UINT Size) { should_not_get_here(); };
        void EndEvent() { should_not_get_here(); }
        void ExecuteIndirect(ID3D12CommandSignature *pCommandSignature, UINT MaxCommandCount, ID3D12Resource *pArgumentBuffer, UINT64 ArgumentBufferOffset, ID3D12Resource *pCountBuffer, UINT64 CountBufferOffset) { should_not_get_here(); }

        // ID3D12GraphicsCommandList1
        void AtomicCopyBufferUINT(ID3D12Resource *pDstBuffer, UINT64 DstOffset, ID3D12Resource *pSrcBuffer, UINT64 SrcOffset, UINT Dependencies, ID3D12Resource *const *ppDependentResources, const D3D12_SUBRESOURCE_RANGE_UINT64 *pDependentSubresourceRanges) { should_not_get_here(); }
        void AtomicCopyBufferUINT64(ID3D12Resource *pDstBuffer, UINT64 DstOffset, ID3D12Resource *pSrcBuffer, UINT64 SrcOffset, UINT Dependencies, ID3D12Resource *const *ppDependentResources, const D3D12_SUBRESOURCE_RANGE_UINT64 *pDependentSubresourceRanges) { should_not_get_here(); }
        void OMSetDepthBounds(FLOAT Min, FLOAT Max) { should_not_get_here(); }
        void SetSamplePositions(UINT NumSamplesPerPixel, UINT NumPixels, D3D12_SAMPLE_POSITION *pSamplePositions) { should_not_get_here(); }
        void ResolveSubresourceRegion(ID3D12Resource *pDstResource, UINT DstSubresource, UINT DstX, UINT DstY, ID3D12Resource *pSrcResource, UINT SrcSubresource, D3D12_RECT *pSrcRect, DXGI_FORMAT Format, D3D12_RESOLVE_MODE ResolveMode) { should_not_get_here(); }
        void SetViewInstanceMask(UINT Mask) { should_not_get_here(); }

        // ID3D12GraphicsCommandList2
        void WriteBufferImmediate(UINT Count, const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER *pParams, const D3D12_WRITEBUFFERIMMEDIATE_MODE *pModes) { should_not_get_here(); }

        // ID3D12GraphicsCommandList3
        void SetProtectedResourceSession(ID3D12ProtectedResourceSession *pProtectedResourceSession) { should_not_get_here(); }

        // ID3D12GraphicsCommandList4
        void BeginRenderPass(UINT NumRenderTargets, const D3D12_RENDER_PASS_RENDER_TARGET_DESC *pRenderTargets, const D3D12_RENDER_PASS_DEPTH_STENCIL_DESC *pDepthStencil, D3D12_RENDER_PASS_FLAGS Flags) { should_not_get_here(); }
        void EndRenderPass(void) { should_not_get_here(); }
        void InitializeMetaCommand(ID3D12MetaCommand *pMetaCommand, const void *pInitializationParametersData, SIZE_T InitializationParametersDataSizeInBytes) { should_not_get_here(); }
        void ExecuteMetaCommand(ID3D12MetaCommand *pMetaCommand, const void *pExecutionParametersData, SIZE_T ExecutionParametersDataSizeInBytes) { should_not_get_here(); }
        void BuildRaytracingAccelerationStructure(const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC *pDesc, UINT NumPostbuildInfoDescs, const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC *pPostbuildInfoDescs) { should_not_get_here(); }
        void EmitRaytracingAccelerationStructurePostbuildInfo(const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC *pDesc, UINT NumSourceAccelerationStructures, const D3D12_GPU_VIRTUAL_ADDRESS *pSourceAccelerationStructureData) { should_not_get_here(); }
        void CopyRaytracingAccelerationStructure(D3D12_GPU_VIRTUAL_ADDRESS DestAccelerationStructureData, D3D12_GPU_VIRTUAL_ADDRESS SourceAccelerationStructureData, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE Mode) { should_not_get_here(); }
        void SetPipelineState1(ID3D12StateObject *pStateObject) { should_not_get_here(); }
        void DispatchRays(const D3D12_DISPATCH_RAYS_DESC *pDesc) { should_not_get_here(); }

    private:
        RtVarsCmdList() = default;
        uint8_t* mpRootBase;
        RootSignature::SharedPtr mpRootSignature;
    };

    class dlldecl RtVarsContext : public CopyContext
    {
    public:
        using SharedPtr = std::shared_ptr<RtVarsContext>;
        ~RtVarsContext();

        /** Create a new ray tracing vars context object.
            \return A new object, or throws an exception if creation failed.
        */
        static SharedPtr create();

        const LowLevelContextData::SharedPtr& getLowLevelData() const override { return mpLowLevelData; }
        bool resourceBarrier(const Resource* pResource, Resource::State newState, const ResourceViewInfo* pViewInfo = nullptr) override;
        RtVarsCmdList::SharedPtr getRtVarsCmdList() const { return mpList; }

        void uavBarrier(const Resource* pResource) override;

    private:
        RtVarsContext();
        RtVarsCmdList::SharedPtr mpList;
    };
}
