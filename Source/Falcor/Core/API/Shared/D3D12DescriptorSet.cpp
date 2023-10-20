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
#include "D3D12DescriptorSet.h"
#include "D3D12DescriptorData.h"
#include "D3D12ConstantBufferView.h"
#include "Core/API/Device.h"
#include "Core/API/CopyContext.h"
#include "Core/API/GFXAPI.h"
#include "Core/API/NativeHandleTraits.h"

namespace Falcor
{
D3D12_DESCRIPTOR_HEAP_TYPE falcorToDxDescType(D3D12DescriptorPool::Type t);

static D3D12DescriptorHeap* getHeap(const D3D12DescriptorPool* pPool, D3D12DescriptorSet::Type type)
{
    auto dxType = falcorToDxDescType(type);
    D3D12DescriptorHeap* pHeap = pPool->getApiData()->pHeaps[dxType].get();
    FALCOR_ASSERT(pHeap);
    FALCOR_ASSERT(pHeap->getType() == dxType);
    return pHeap;
}

D3D12DescriptorSet::CpuHandle D3D12DescriptorSet::getCpuHandle(uint32_t rangeIndex, uint32_t descInRange) const
{
    uint32_t index = mpApiData->rangeBaseOffset[rangeIndex] + descInRange;
    return mpApiData->pAllocation->getCpuHandle(index);
}

D3D12DescriptorSet::GpuHandle D3D12DescriptorSet::getGpuHandle(uint32_t rangeIndex, uint32_t descInRange) const
{
    FALCOR_THROW("Not supported.");
}

void D3D12DescriptorSet::setCpuHandle(uint32_t rangeIndex, uint32_t descIndex, const CpuHandle& handle)
{
    auto dstHandle = getCpuHandle(rangeIndex, descIndex);
    ID3D12Device* pD3D12Device = mpDevice->getNativeHandle().as<ID3D12Device*>();
    pD3D12Device->CopyDescriptorsSimple(1, dstHandle, handle, falcorToDxDescType(getRange(rangeIndex).type));
}

void D3D12DescriptorSet::setSrv(uint32_t rangeIndex, uint32_t descIndex, const ShaderResourceView* pSrv)
{
    auto type = getRange(rangeIndex).type;
    FALCOR_CHECK(
        type == Type::TextureSrv || type == Type::RawBufferSrv || type == Type::TypedBufferSrv || type == Type::StructuredBufferSrv ||
            type == Type::AccelerationStructureSrv,
        "Unexpected descriptor range type in setSrv()"
    );
    setCpuHandle(rangeIndex, descIndex, pSrv->getNativeHandle().as<D3D12_CPU_DESCRIPTOR_HANDLE>());
}

void D3D12DescriptorSet::setUav(uint32_t rangeIndex, uint32_t descIndex, const UnorderedAccessView* pUav)
{
    auto type = getRange(rangeIndex).type;
    FALCOR_CHECK(
        type == Type::TextureUav || type == Type::RawBufferUav || type == Type::TypedBufferUav || type == Type::StructuredBufferUav,
        "Unexpected descriptor range type in setUav()"
    );
    setCpuHandle(rangeIndex, descIndex, pUav->getNativeHandle().as<D3D12_CPU_DESCRIPTOR_HANDLE>());
}

void D3D12DescriptorSet::setSampler(uint32_t rangeIndex, uint32_t descIndex, const Sampler* pSampler)
{
    FALCOR_CHECK(getRange(rangeIndex).type == Type::Sampler, "Unexpected descriptor range type in setSampler()");
    setCpuHandle(rangeIndex, descIndex, pSampler->getNativeHandle().as<D3D12_CPU_DESCRIPTOR_HANDLE>());
}

static D3D12_GPU_DESCRIPTOR_HANDLE copyDescriptorTableToGPU(
    CopyContext* pCtx,
    DescriptorSetApiData* apiData,
    const D3D12DescriptorSetLayout& layout
)
{
    Slang::ComPtr<gfx::ITransientResourceHeapD3D12> d3dTransientHeap;
    pCtx->getDevice()->getCurrentTransientResourceHeap()->queryInterface(
        SlangUUID(SLANG_UUID_ITransientResourceHeapD3D12), (void**)d3dTransientHeap.writeRef()
    );
    auto dxHeapType = falcorToDxDescType(layout.getRange(0).type);
    gfx::ITransientResourceHeapD3D12::DescriptorType gfxHeapType;
    switch (dxHeapType)
    {
    case D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV:
        gfxHeapType = gfx::ITransientResourceHeapD3D12::DescriptorType::ResourceView;
        break;
    case D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER:
        gfxHeapType = gfx::ITransientResourceHeapD3D12::DescriptorType::Sampler;
        break;
    default:
        FALCOR_UNREACHABLE();
        break;
    }
    uint64_t gpuTableOffset;
    ID3D12DescriptorHeap* gpuHeapHandle;
    FALCOR_GFX_CALL(
        d3dTransientHeap->allocateTransientDescriptorTable(gfxHeapType, apiData->descriptorCount, gpuTableOffset, (void**)&gpuHeapHandle)
    );
    auto gpuHandle = gpuHeapHandle->GetGPUDescriptorHandleForHeapStart();
    ID3D12Device* pD3D12Device = pCtx->getDevice()->getNativeHandle().as<ID3D12Device*>();
    uint32_t descriptorSize = pD3D12Device->GetDescriptorHandleIncrementSize(dxHeapType);
    gpuHandle.ptr += gpuTableOffset * descriptorSize;
    auto transientCpuHandle = gpuHeapHandle->GetCPUDescriptorHandleForHeapStart();
    transientCpuHandle.ptr += gpuTableOffset * descriptorSize;
    D3D12_CPU_DESCRIPTOR_HANDLE srcCPUHandle = apiData->pAllocation->getCpuHandle(0);
    pD3D12Device->CopyDescriptorsSimple(apiData->descriptorCount, transientCpuHandle, srcCPUHandle, dxHeapType);
    return gpuHandle;
}

void D3D12DescriptorSet::bindForGraphics(CopyContext* pCtx, const D3D12RootSignature* pRootSig, uint32_t rootIndex)
{
    FALCOR_ASSERT(
        mpApiData->pAllocation->getHeap()->getShaderVisible() == false &&
        "DescriptorSet must be created on CPU heap for bind operation in GFX."
    );
    Slang::ComPtr<gfx::ICommandBufferD3D12> commandBufferD3D12;
    pCtx->getLowLevelData()->getGfxCommandBuffer()->queryInterface(
        SlangUUID SLANG_UUID_ICommandBufferD3D12, (void**)commandBufferD3D12.writeRef()
    );
    FALCOR_ASSERT(commandBufferD3D12);
    commandBufferD3D12->ensureInternalDescriptorHeapsBound();
    auto gpuHandle = copyDescriptorTableToGPU(pCtx, mpApiData.get(), mLayout);
    pCtx->getLowLevelData()->getCommandBufferNativeHandle().as<ID3D12GraphicsCommandList*>()->SetGraphicsRootDescriptorTable(
        rootIndex, gpuHandle
    );
}

void D3D12DescriptorSet::bindForCompute(CopyContext* pCtx, const D3D12RootSignature* pRootSig, uint32_t rootIndex)
{
    FALCOR_ASSERT(
        mpApiData->pAllocation->getHeap()->getShaderVisible() == false &&
        "DescriptorSet must be created on CPU heap for bind operation in GFX."
    );
    Slang::ComPtr<gfx::ICommandBufferD3D12> commandBufferD3D12;
    pCtx->getLowLevelData()->getGfxCommandBuffer()->queryInterface(
        SlangUUID SLANG_UUID_ICommandBufferD3D12, (void**)commandBufferD3D12.writeRef()
    );
    FALCOR_ASSERT(commandBufferD3D12);
    commandBufferD3D12->ensureInternalDescriptorHeapsBound();
    auto gpuHandle = copyDescriptorTableToGPU(pCtx, mpApiData.get(), mLayout);
    pCtx->getLowLevelData()->getCommandBufferNativeHandle().as<ID3D12GraphicsCommandList*>()->SetComputeRootDescriptorTable(
        rootIndex, gpuHandle
    );
}

void D3D12DescriptorSet::setCbv(uint32_t rangeIndex, uint32_t descIndex, D3D12ConstantBufferView* pView)
{
    FALCOR_CHECK(getRange(rangeIndex).type == Type::Cbv, "Unexpected descriptor range type in setCbv()");
    setCpuHandle(rangeIndex, descIndex, pView->getD3D12CpuHeapHandle());
}

ref<D3D12DescriptorSet> D3D12DescriptorSet::create(
    ref<Device> pDevice,
    ref<D3D12DescriptorPool> pPool,
    const D3D12DescriptorSetLayout& layout
)
{
    FALCOR_ASSERT(pDevice);
    pDevice->requireD3D12();
    return ref<D3D12DescriptorSet>(new D3D12DescriptorSet(pDevice, pPool, layout));
}

ref<D3D12DescriptorSet> D3D12DescriptorSet::create(
    ref<Device> pDevice,
    const D3D12DescriptorSetLayout& layout,
    D3D12DescriptorSetBindingUsage bindingUsage
)
{
    FALCOR_ASSERT(pDevice);
    pDevice->requireD3D12();
    return ref<D3D12DescriptorSet>(new D3D12DescriptorSet(
        pDevice,
        bindingUsage == D3D12DescriptorSetBindingUsage::RootSignatureOffset ? pDevice->getD3D12GpuDescriptorPool()
                                                                            : pDevice->getD3D12CpuDescriptorPool(),
        layout
    ));
}

D3D12DescriptorSet::D3D12DescriptorSet(ref<Device> pDevice, ref<D3D12DescriptorPool> pPool, const D3D12DescriptorSetLayout& layout)
    : mpDevice(pDevice), mLayout(layout), mpPool(pPool)
{
    mpApiData = std::make_shared<DescriptorSetApiData>();
    uint32_t count = 0;
    const auto falcorType = mLayout.getRange(0).type;
    const auto d3dType = falcorToDxDescType(falcorType);

    // For each range we need to allocate a table from a heap
    mpApiData->rangeBaseOffset.resize(mLayout.getRangeCount());

    for (size_t i = 0; i < mLayout.getRangeCount(); i++)
    {
        const auto& range = mLayout.getRange(i);
        mpApiData->rangeBaseOffset[i] = count;
        FALCOR_ASSERT(d3dType == falcorToDxDescType(range.type)); // We can only allocate from a single heap
        count += range.descCount;
    }

    D3D12DescriptorHeap* pHeap = getHeap(mpPool.get(), falcorType);
    mpApiData->pAllocation = pHeap->allocateDescriptors(count);
    if (mpApiData->pAllocation == nullptr)
    {
        // Execute deferred releases and try again
        mpPool->executeDeferredReleases();
        mpApiData->pAllocation = pHeap->allocateDescriptors(count);
    }

    // Allocation failed again, there is nothing else we can do.
    if (mpApiData->pAllocation == nullptr)
        FALCOR_THROW("Failed to create descriptor set");

    mpApiData->descriptorCount = count;
}

D3D12DescriptorSet::~D3D12DescriptorSet()
{
    mpPool->releaseAllocation(mpApiData);
}

} // namespace Falcor
