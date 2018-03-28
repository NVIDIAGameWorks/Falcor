/************************************************************************************************************************************\
|*                                                                                                                                    *|
|*     Copyright © 2017 NVIDIA Corporation.  All rights reserved.                                                                     *|
|*                                                                                                                                    *|
|*  NOTICE TO USER:                                                                                                                   *|
|*                                                                                                                                    *|
|*  This software is subject to NVIDIA ownership rights under U.S. and international Copyright laws.                                  *|
|*                                                                                                                                    *|
|*  This software and the information contained herein are PROPRIETARY and CONFIDENTIAL to NVIDIA                                     *|
|*  and are being provided solely under the terms and conditions of an NVIDIA software license agreement                              *|
|*  and / or non-disclosure agreement.  Otherwise, you have no rights to use or access this software in any manner.                   *|
|*                                                                                                                                    *|
|*  If not covered by the applicable NVIDIA software license agreement:                                                               *|
|*  NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOFTWARE FOR ANY PURPOSE.                                            *|
|*  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.                                                           *|
|*  NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,                                                                     *|
|*  INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.                       *|
|*  IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES,                               *|
|*  OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT,                         *|
|*  NEGLIGENCE OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOURCE CODE.            *|
|*                                                                                                                                    *|
|*  U.S. Government End Users.                                                                                                        *|
|*  This software is a "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT 1995),                                       *|
|*  consisting  of "commercial computer  software"  and "commercial computer software documentation"                                  *|
|*  as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government only as a commercial end item.     *|
|*  Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995),                                          *|
|*  all U.S. Government End Users acquire the software with only those rights set forth herein.                                       *|
|*                                                                                                                                    *|
|*  Any use of this software in individual and commercial software must include,                                                      *|
|*  in the user documentation and internal comments to the code,                                                                      *|
|*  the above Disclaimer (as applicable) and U.S. Government End Users Notice.                                                        *|
|*                                                                                                                                    *|
 \************************************************************************************************************************************/
#pragma once
#include "API/CopyContext.h"
#include "API/LowLevel/RootSignature.h"

namespace Falcor
{
    class RtVarsCmdList : public ID3D12GraphicsCommandList
    {
    public:
        using SharedPtr = std::shared_ptr<RtVarsCmdList>;
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
        ID3D12GraphicsCommandList* AsID3D12GraphicsCommandList() { should_not_get_here(); return this; }

    private:
        RtVarsCmdList() = default;
        uint8_t* mpRootBase;
        RootSignature::SharedPtr mpRootSignature;
    };

    class RtVarsContext : public CopyContext, inherit_shared_from_this<CopyContext, RtVarsContext>
    {
    public:
        using SharedPtr = std::shared_ptr<RtVarsContext>;
        ~RtVarsContext();

        static SharedPtr create(CopyContext::SharedPtr pRtContext);

        LowLevelContextData::SharedPtr getLowLevelData() const override { return mpLowLevelData; }
        void resourceBarrier(const Resource* pResource, Resource::State newState) override { return mpRayTraceContext->resourceBarrier(pResource, newState); }
        RtVarsCmdList::SharedPtr getRtVarsCmdList() const { return mpList; }
    private:
        RtVarsContext(CopyContext::SharedPtr pRtContext);
        RtVarsCmdList::SharedPtr mpList;
        CopyContext::SharedPtr mpRayTraceContext;
    };
}
