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
#include "CopyContext.h"
#include "Device.h"
#include "Texture.h"
#include "Buffer.h"
#include "Fence.h"
#include "GFXAPI.h"
#include "GFXHelpers.h"
#include "NativeHandleTraits.h"
#include "Aftermath.h"
#if FALCOR_HAS_D3D12
#include "Shared/D3D12DescriptorPool.h"
#include "Shared/D3D12DescriptorData.h"
#endif
#include "Core/Error.h"
#include "Utils/Logger.h"
#include "Utils/Math/Common.h"

#if FALCOR_HAS_CUDA
#include "Utils/CudaUtils.h"
#endif

namespace Falcor
{
CopyContext::CopyContext(Device* pDevice, gfx::ICommandQueue* pQueue) : mpDevice(pDevice)
{
    FALCOR_ASSERT(mpDevice);
    FALCOR_ASSERT(pQueue);
    mpLowLevelData = std::make_unique<LowLevelContextData>(mpDevice, pQueue);
}

CopyContext::~CopyContext() = default;

ref<Device> CopyContext::getDevice() const
{
    return ref<Device>(mpDevice);
}

Profiler* CopyContext::getProfiler() const
{
    return mpDevice->getProfiler();
}

void CopyContext::submit(bool wait)
{
    if (mCommandsPending)
    {
        mpLowLevelData->submitCommandBuffer();
        mCommandsPending = false;
    }
    else
    {
        // We need to signal even if there are no commands to execute. We need this because some resources may have been released since the
        // last flush(), and unless we signal they will not be released
        signal(mpLowLevelData->getFence().get());
    }

    bindDescriptorHeaps();

    if (wait)
    {
        mpLowLevelData->getFence()->wait();
    }
}

uint64_t CopyContext::signal(Fence* pFence, uint64_t value)
{
    FALCOR_CHECK(pFence, "'fence' must not be null");
    uint64_t signalValue = pFence->updateSignaledValue(value);
    mpLowLevelData->getGfxCommandQueue()->executeCommandBuffers(0, nullptr, pFence->getGfxFence(), signalValue);
    return signalValue;
}

void CopyContext::wait(Fence* pFence, uint64_t value)
{
    FALCOR_CHECK(pFence, "'fence' must not be null");
    uint64_t waitValue = value == Fence::kAuto ? pFence->getSignaledValue() : value;
    gfx::IFence* fences[] = {pFence->getGfxFence()};
    uint64_t waitValues[] = {waitValue};
    FALCOR_GFX_CALL(mpLowLevelData->getGfxCommandQueue()->waitForFenceValuesOnDevice(1, fences, waitValues));
}

#if FALCOR_HAS_CUDA
void CopyContext::waitForCuda(cudaStream_t stream)
{
    if (mpDevice->getType() == Device::Type::D3D12)
    {
        mpLowLevelData->getCudaSemaphore()->waitForCuda(this, stream);
    }
    else
    {
        // In the past, we used to wait for all CUDA work to be done.
        // Since GFX with Vulkan doesn't support shared fences yet, we do the same here.
        cuda_utils::deviceSynchronize();
    }
}

void CopyContext::waitForFalcor(cudaStream_t stream)
{
    if (mpDevice->getType() == Device::Type::D3D12)
    {
        mpLowLevelData->submitCommandBuffer();
        mpLowLevelData->getCudaSemaphore()->waitForFalcor(this, stream);
    }
    else
    {
        // In the past, we used to wait for all work on the command queue to be done.
        // Since GFX with Vulkan doesn't support shared fences yet, we do the same here.
        submit(true);
    }
}
#endif

CopyContext::ReadTextureTask::SharedPtr CopyContext::asyncReadTextureSubresource(const Texture* pTexture, uint32_t subresourceIndex)
{
    return CopyContext::ReadTextureTask::create(this, pTexture, subresourceIndex);
}

std::vector<uint8_t> CopyContext::readTextureSubresource(const Texture* pTexture, uint32_t subresourceIndex)
{
    CopyContext::ReadTextureTask::SharedPtr pTask = asyncReadTextureSubresource(pTexture, subresourceIndex);
    return pTask->getData();
}

bool CopyContext::resourceBarrier(const Resource* pResource, Resource::State newState, const ResourceViewInfo* pViewInfo)
{
    const Texture* pTexture = dynamic_cast<const Texture*>(pResource);
    if (pTexture)
    {
        bool globalBarrier = pTexture->isStateGlobal();
        if (pViewInfo)
        {
            globalBarrier = globalBarrier && pViewInfo->firstArraySlice == 0;
            globalBarrier = globalBarrier && pViewInfo->mostDetailedMip == 0;
            globalBarrier = globalBarrier && pViewInfo->mipCount == pTexture->getMipCount();
            globalBarrier = globalBarrier && pViewInfo->arraySize == pTexture->getArraySize();
        }

        if (globalBarrier)
        {
            return textureBarrier(pTexture, newState);
        }
        else
        {
            return subresourceBarriers(pTexture, newState, pViewInfo);
        }
    }
    else
    {
        const Buffer* pBuffer = dynamic_cast<const Buffer*>(pResource);
        return bufferBarrier(pBuffer, newState);
    }
}

bool CopyContext::subresourceBarriers(const Texture* pTexture, Resource::State newState, const ResourceViewInfo* pViewInfo)
{
    ResourceViewInfo fullResource;
    bool setGlobal = false;
    if (pViewInfo == nullptr)
    {
        fullResource.arraySize = pTexture->getArraySize();
        fullResource.firstArraySlice = 0;
        fullResource.mipCount = pTexture->getMipCount();
        fullResource.mostDetailedMip = 0;
        setGlobal = true;
        pViewInfo = &fullResource;
    }

    bool entireViewTransitioned = true;

    for (uint32_t a = pViewInfo->firstArraySlice; a < pViewInfo->firstArraySlice + pViewInfo->arraySize; a++)
    {
        for (uint32_t m = pViewInfo->mostDetailedMip; m < pViewInfo->mipCount + pViewInfo->mostDetailedMip; m++)
        {
            Resource::State oldState = pTexture->getSubresourceState(a, m);
            if (oldState != newState)
            {
                apiSubresourceBarrier(pTexture, newState, oldState, a, m);
                if (setGlobal == false)
                    pTexture->setSubresourceState(a, m, newState);
                mCommandsPending = true;
            }
            else
                entireViewTransitioned = false;
        }
    }
    if (setGlobal)
        pTexture->setGlobalState(newState);
    return entireViewTransitioned;
}

void CopyContext::updateTextureData(const Texture* pTexture, const void* pData)
{
    mCommandsPending = true;
    uint32_t subresourceCount = pTexture->getArraySize() * pTexture->getMipCount();
    if (pTexture->getType() == Texture::Type::TextureCube)
    {
        subresourceCount *= 6;
    }
    updateTextureSubresources(pTexture, 0, subresourceCount, pData);
}

void CopyContext::updateSubresourceData(
    const Texture* pDst,
    uint32_t subresource,
    const void* pData,
    const uint3& offset,
    const uint3& size
)
{
    mCommandsPending = true;
    updateTextureSubresources(pDst, subresource, 1, pData, offset, size);
}

void CopyContext::bindDescriptorHeaps() {}

void CopyContext::bindCustomGPUDescriptorPool()
{
#if FALCOR_HAS_D3D12
    mpDevice->requireD3D12();

    const D3D12DescriptorPool* pGpuPool = mpDevice->getD3D12GpuDescriptorPool().get();
    const D3D12DescriptorPool::ApiData* pData = pGpuPool->getApiData();
    ID3D12DescriptorHeap* pHeaps[D3D12DescriptorPool::ApiData::kHeapCount];
    uint32_t heapCount = 0;
    for (uint32_t i = 0; i < std::size(pData->pHeaps); i++)
    {
        if (pData->pHeaps[i])
        {
            pHeaps[heapCount] = pData->pHeaps[i]->getApiHandle();
            heapCount++;
        }
    }
    mpLowLevelData->getCommandBufferNativeHandle().as<ID3D12GraphicsCommandList*>()->SetDescriptorHeaps(heapCount, pHeaps);
#endif
}

void CopyContext::unbindCustomGPUDescriptorPool()
{
#if FALCOR_HAS_D3D12
    mpDevice->requireD3D12();

    Slang::ComPtr<gfx::ICommandBufferD3D12> d3d12CommandBuffer;
    mpLowLevelData->getGfxCommandBuffer()->queryInterface(
        SlangUUID SLANG_UUID_ICommandBufferD3D12, reinterpret_cast<void**>(d3d12CommandBuffer.writeRef())
    );
    d3d12CommandBuffer->invalidateDescriptorHeapBinding();
#endif
}

void CopyContext::updateTextureSubresources(
    const Texture* pTexture,
    uint32_t firstSubresource,
    uint32_t subresourceCount,
    const void* pData,
    const uint3& offset,
    const uint3& size
)
{
    resourceBarrier(pTexture, Resource::State::CopyDest);

    bool copyRegion = any(offset != uint3(0)) || any(size != uint3(-1));
    FALCOR_ASSERT(subresourceCount == 1 || (copyRegion == false));
    uint8_t* dataPtr = (uint8_t*)pData;
    auto resourceEncoder = getLowLevelData()->getResourceCommandEncoder();
    gfx::ITextureResource::Offset3D gfxOffset = {
        static_cast<gfx::GfxIndex>(offset.x), static_cast<gfx::GfxIndex>(offset.y), static_cast<gfx::GfxIndex>(offset.z)};
    gfx::ITextureResource::Extents gfxSize = {
        static_cast<gfx::GfxCount>(size.x), static_cast<gfx::GfxCount>(size.y), static_cast<gfx::GfxCount>(size.z)};
    gfx::FormatInfo formatInfo = {};
    gfx::gfxGetFormatInfo(getGFXFormat(pTexture->getFormat()), &formatInfo);
    for (uint32_t index = firstSubresource; index < firstSubresource + subresourceCount; index++)
    {
        gfx::SubresourceRange subresourceRange = {};
        subresourceRange.baseArrayLayer = static_cast<gfx::GfxIndex>(pTexture->getSubresourceArraySlice(index));
        subresourceRange.mipLevel = static_cast<gfx::GfxIndex>(pTexture->getSubresourceMipLevel(index));
        subresourceRange.layerCount = 1;
        subresourceRange.mipLevelCount = 1;
        if (!copyRegion)
        {
            gfxSize.width = align_to(formatInfo.blockWidth, static_cast<gfx::GfxCount>(pTexture->getWidth(subresourceRange.mipLevel)));
            gfxSize.height = align_to(formatInfo.blockHeight, static_cast<gfx::GfxCount>(pTexture->getHeight(subresourceRange.mipLevel)));
            gfxSize.depth = static_cast<gfx::GfxCount>(pTexture->getDepth(subresourceRange.mipLevel));
        }
        gfx::ITextureResource::SubresourceData data = {};
        data.data = dataPtr;
        data.strideY = static_cast<int64_t>(gfxSize.width) / formatInfo.blockWidth * formatInfo.blockSizeInBytes;
        data.strideZ = data.strideY * (gfxSize.height / formatInfo.blockHeight);
        dataPtr += data.strideZ * gfxSize.depth;
        resourceEncoder->uploadTextureData(pTexture->getGfxTextureResource(), subresourceRange, gfxOffset, gfxSize, &data, 1);
    }
}

CopyContext::ReadTextureTask::SharedPtr CopyContext::ReadTextureTask::create(
    CopyContext* pCtx,
    const Texture* pTexture,
    uint32_t subresourceIndex
)
{
    SharedPtr pThis = SharedPtr(new ReadTextureTask);
    pThis->mpContext = pCtx;
    // Get footprint
    gfx::ITextureResource* srcTexture = pTexture->getGfxTextureResource();
    gfx::FormatInfo formatInfo;
    gfx::gfxGetFormatInfo(srcTexture->getDesc()->format, &formatInfo);

    auto mipLevel = pTexture->getSubresourceMipLevel(subresourceIndex);
    pThis->mActualRowSize =
        uint32_t((pTexture->getWidth(mipLevel) + formatInfo.blockWidth - 1) / formatInfo.blockWidth * formatInfo.blockSizeInBytes);
    size_t rowAlignment = 1;
    pCtx->mpDevice->getGfxDevice()->getTextureRowAlignment(&rowAlignment);
    pThis->mRowSize = align_to(static_cast<uint32_t>(rowAlignment), pThis->mActualRowSize);
    uint64_t rowCount = (pTexture->getHeight(mipLevel) + formatInfo.blockHeight - 1) / formatInfo.blockHeight;
    uint64_t size = pTexture->getDepth(mipLevel) * rowCount * pThis->mRowSize;

    // Create buffer
    pThis->mpBuffer = pCtx->getDevice()->createBuffer(size, ResourceBindFlags::None, MemoryType::ReadBack, nullptr);

    // Copy from texture to buffer
    pCtx->resourceBarrier(pTexture, Resource::State::CopySource);
    auto encoder = pCtx->getLowLevelData()->getResourceCommandEncoder();
    gfx::SubresourceRange srcSubresource = {};
    srcSubresource.baseArrayLayer = pTexture->getSubresourceArraySlice(subresourceIndex);
    srcSubresource.mipLevel = mipLevel;
    srcSubresource.layerCount = 1;
    srcSubresource.mipLevelCount = 1;
    encoder->copyTextureToBuffer(
        pThis->mpBuffer->getGfxBufferResource(),
        0,
        size,
        pThis->mRowSize,
        srcTexture,
        gfx::ResourceState::CopySource,
        srcSubresource,
        gfx::ITextureResource::Offset3D(0, 0, 0),
        gfx::ITextureResource::Extents{
            static_cast<gfx::GfxIndex>(pTexture->getWidth(mipLevel)),
            static_cast<gfx::GfxIndex>(pTexture->getHeight(mipLevel)),
            static_cast<gfx::GfxIndex>(pTexture->getDepth(mipLevel))}
    );
    pCtx->setPendingCommands(true);

    // Create a fence and signal
    pThis->mpFence = pCtx->getDevice()->createFence();
    pThis->mpFence->breakStrongReferenceToDevice();
    pCtx->submit(false);
    pCtx->signal(pThis->mpFence.get());
    pThis->mRowCount = (uint32_t)rowCount;
    pThis->mDepth = pTexture->getDepth(mipLevel);
    return pThis;
}

void CopyContext::ReadTextureTask::getData(void* pData, size_t size) const
{
    FALCOR_ASSERT(size == size_t(mRowCount) * mActualRowSize * mDepth);

    mpFence->wait();

    uint8_t* pDst = reinterpret_cast<uint8_t*>(pData);
    const uint8_t* pSrc = reinterpret_cast<const uint8_t*>(mpBuffer->map(Buffer::MapType::Read));

    for (uint32_t z = 0; z < mDepth; z++)
    {
        const uint8_t* pSrcZ = pSrc + z * (size_t)mRowSize * mRowCount;
        uint8_t* pDstZ = pDst + z * (size_t)mActualRowSize * mRowCount;
        for (uint32_t y = 0; y < mRowCount; y++)
        {
            const uint8_t* pSrcY = pSrcZ + y * (size_t)mRowSize;
            uint8_t* pDstY = pDstZ + y * (size_t)mActualRowSize;
            std::memcpy(pDstY, pSrcY, mActualRowSize);
        }
    }

    mpBuffer->unmap();
}

std::vector<uint8_t> CopyContext::ReadTextureTask::getData() const
{
    std::vector<uint8_t> result(size_t(mRowCount) * mActualRowSize * mDepth);
    getData(result.data(), result.size());
    return result;
}

bool CopyContext::textureBarrier(const Texture* pTexture, Resource::State newState)
{
    auto resourceEncoder = getLowLevelData()->getResourceCommandEncoder();
    bool recorded = false;
    if (pTexture->getGlobalState() != newState)
    {
        gfx::ITextureResource* textureResource = pTexture->getGfxTextureResource();
        resourceEncoder->textureBarrier(
            1, &textureResource, getGFXResourceState(pTexture->getGlobalState()), getGFXResourceState(newState)
        );
        mCommandsPending = true;
        recorded = true;
    }
    pTexture->setGlobalState(newState);
    return recorded;
}

bool CopyContext::bufferBarrier(const Buffer* pBuffer, Resource::State newState)
{
    FALCOR_ASSERT(pBuffer);
    if (pBuffer->getMemoryType() != MemoryType::DeviceLocal)
        return false;
    bool recorded = false;
    if (pBuffer->getGlobalState() != newState)
    {
        auto resourceEncoder = getLowLevelData()->getResourceCommandEncoder();
        gfx::IBufferResource* bufferResource = pBuffer->getGfxBufferResource();
        resourceEncoder->bufferBarrier(1, &bufferResource, getGFXResourceState(pBuffer->getGlobalState()), getGFXResourceState(newState));
        pBuffer->setGlobalState(newState);
        mCommandsPending = true;
        recorded = true;
    }
    return recorded;
}

void CopyContext::apiSubresourceBarrier(
    const Texture* pTexture,
    Resource::State newState,
    Resource::State oldState,
    uint32_t arraySlice,
    uint32_t mipLevel
)
{
    auto resourceEncoder = getLowLevelData()->getResourceCommandEncoder();
    auto subresourceState = pTexture->getSubresourceState(arraySlice, mipLevel);
    if (subresourceState != newState)
    {
        gfx::ITextureResource* textureResource = pTexture->getGfxTextureResource();
        gfx::SubresourceRange subresourceRange = {};
        subresourceRange.baseArrayLayer = arraySlice;
        subresourceRange.mipLevel = mipLevel;
        subresourceRange.layerCount = 1;
        subresourceRange.mipLevelCount = 1;
        resourceEncoder->textureSubresourceBarrier(
            textureResource, subresourceRange, getGFXResourceState(subresourceState), getGFXResourceState(newState)
        );
        mCommandsPending = true;
    }
}

void CopyContext::uavBarrier(const Resource* pResource)
{
    auto resourceEncoder = getLowLevelData()->getResourceCommandEncoder();

    if (pResource->getType() == Resource::Type::Buffer)
    {
        gfx::IBufferResource* bufferResource = static_cast<gfx::IBufferResource*>(pResource->getGfxResource());
        resourceEncoder->bufferBarrier(1, &bufferResource, gfx::ResourceState::UnorderedAccess, gfx::ResourceState::UnorderedAccess);
    }
    else
    {
        gfx::ITextureResource* textureResource = static_cast<gfx::ITextureResource*>(pResource->getGfxResource());
        resourceEncoder->textureBarrier(1, &textureResource, gfx::ResourceState::UnorderedAccess, gfx::ResourceState::UnorderedAccess);
    }
    mCommandsPending = true;
}

void CopyContext::copyResource(const Resource* pDst, const Resource* pSrc)
{
    // Copy from texture to texture or from buffer to buffer.
    FALCOR_ASSERT(pDst->getType() == pSrc->getType());

    resourceBarrier(pDst, Resource::State::CopyDest);
    resourceBarrier(pSrc, Resource::State::CopySource);

    auto resourceEncoder = getLowLevelData()->getResourceCommandEncoder();

    if (pDst->getType() == Resource::Type::Buffer)
    {
        const Buffer* pSrcBuffer = static_cast<const Buffer*>(pSrc);
        const Buffer* pDstBuffer = static_cast<const Buffer*>(pDst);

        FALCOR_ASSERT(pSrcBuffer->getSize() <= pDstBuffer->getSize());

        resourceEncoder->copyBuffer(pDstBuffer->getGfxBufferResource(), 0, pSrcBuffer->getGfxBufferResource(), 0, pSrcBuffer->getSize());
    }
    else
    {
        const Texture* pSrcTexture = static_cast<const Texture*>(pSrc);
        const Texture* pDstTexture = static_cast<const Texture*>(pDst);
        gfx::SubresourceRange subresourceRange = {};
        resourceEncoder->copyTexture(
            pDstTexture->getGfxTextureResource(),
            gfx::ResourceState::CopyDestination,
            subresourceRange,
            gfx::ITextureResource::Offset3D(0, 0, 0),
            pSrcTexture->getGfxTextureResource(),
            gfx::ResourceState::CopySource,
            subresourceRange,
            gfx::ITextureResource::Offset3D(0, 0, 0),
            gfx::ITextureResource::Extents{0, 0, 0}
        );
    }
    mCommandsPending = true;
}

void CopyContext::copySubresource(const Texture* pDst, uint32_t dstSubresourceIdx, const Texture* pSrc, uint32_t srcSubresourceIdx)
{
    copySubresourceRegion(pDst, dstSubresourceIdx, pSrc, srcSubresourceIdx, uint3(0), uint3(0), uint3(-1));
}

void CopyContext::updateBuffer(const Buffer* pBuffer, const void* pData, size_t offset, size_t numBytes)
{
    if (numBytes == 0)
    {
        numBytes = pBuffer->getSize() - offset;
    }

    if (pBuffer->adjustSizeOffsetParams(numBytes, offset) == false)
    {
        logWarning("CopyContext::updateBuffer() - size and offset are invalid. Nothing to update.");
        return;
    }

    bufferBarrier(pBuffer, Resource::State::CopyDest);
    auto resourceEncoder = getLowLevelData()->getResourceCommandEncoder();
    resourceEncoder->uploadBufferData(pBuffer->getGfxBufferResource(), offset, numBytes, (void*)pData);

    mCommandsPending = true;
}

void CopyContext::readBuffer(const Buffer* pBuffer, void* pData, size_t offset, size_t numBytes)
{
    if (numBytes == 0)
        numBytes = pBuffer->getSize() - offset;

    if (pBuffer->adjustSizeOffsetParams(numBytes, offset) == false)
    {
        logWarning("CopyContext::readBuffer() - size and offset are invalid. Nothing to read.");
        return;
    }

    const auto& pReadBackHeap = mpDevice->getReadBackHeap();

    auto allocation = pReadBackHeap->allocate(numBytes);

    bufferBarrier(pBuffer, Resource::State::CopySource);

    auto resourceEncoder = getLowLevelData()->getResourceCommandEncoder();
    resourceEncoder->copyBuffer(allocation.gfxBufferResource, allocation.offset, pBuffer->getGfxBufferResource(), offset, numBytes);
    mCommandsPending = true;
    submit(true);

    std::memcpy(pData, allocation.pData, numBytes);

    pReadBackHeap->release(allocation);
}

void CopyContext::copyBufferRegion(const Buffer* pDst, uint64_t dstOffset, const Buffer* pSrc, uint64_t srcOffset, uint64_t numBytes)
{
    resourceBarrier(pDst, Resource::State::CopyDest);
    resourceBarrier(pSrc, Resource::State::CopySource);

    auto resourceEncoder = getLowLevelData()->getResourceCommandEncoder();
    resourceEncoder->copyBuffer(pDst->getGfxBufferResource(), dstOffset, pSrc->getGfxBufferResource(), srcOffset, numBytes);
    mCommandsPending = true;
}

void CopyContext::copySubresourceRegion(
    const Texture* pDst,
    uint32_t dstSubresourceIdx,
    const Texture* pSrc,
    uint32_t srcSubresourceIdx,
    const uint3& dstOffset,
    const uint3& srcOffset,
    const uint3& size
)
{
    resourceBarrier(pDst, Resource::State::CopyDest);
    resourceBarrier(pSrc, Resource::State::CopySource);

    gfx::SubresourceRange dstSubresource = {};
    dstSubresource.baseArrayLayer = pDst->getSubresourceArraySlice(dstSubresourceIdx);
    dstSubresource.layerCount = 1;
    dstSubresource.mipLevel = pDst->getSubresourceMipLevel(dstSubresourceIdx);
    dstSubresource.mipLevelCount = 1;

    gfx::SubresourceRange srcSubresource = {};
    srcSubresource.baseArrayLayer = pSrc->getSubresourceArraySlice(srcSubresourceIdx);
    srcSubresource.layerCount = 1;
    srcSubresource.mipLevel = pSrc->getSubresourceMipLevel(srcSubresourceIdx);
    srcSubresource.mipLevelCount = 1;

    gfx::ITextureResource::Extents copySize = {(int)size.x, (int)size.y, (int)size.z};

    if (size.x == uint(-1))
    {
        copySize.width = pSrc->getWidth(srcSubresource.mipLevel) - srcOffset.x;
        copySize.height = pSrc->getHeight(srcSubresource.mipLevel) - srcOffset.y;
        copySize.depth = pSrc->getDepth(srcSubresource.mipLevel) - srcOffset.z;
    }

    auto resourceEncoder = getLowLevelData()->getResourceCommandEncoder();
    resourceEncoder->copyTexture(
        pDst->getGfxTextureResource(),
        gfx::ResourceState::CopyDestination,
        dstSubresource,
        gfx::ITextureResource::Offset3D(dstOffset.x, dstOffset.y, dstOffset.z),
        pSrc->getGfxTextureResource(),
        gfx::ResourceState::CopySource,
        srcSubresource,
        gfx::ITextureResource::Offset3D(srcOffset.x, srcOffset.y, srcOffset.z),
        copySize
    );
    mCommandsPending = true;
}

void CopyContext::addAftermathMarker(std::string_view name)
{
#if FALCOR_HAS_AFTERMATH
    if (AftermathContext* pAftermathContext = mpDevice->getAftermathContext())
        pAftermathContext->addMarker(mpLowLevelData.get(), name);
#endif
};

} // namespace Falcor
