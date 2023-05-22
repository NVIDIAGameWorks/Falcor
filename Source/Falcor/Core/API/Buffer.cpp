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
#include "Buffer.h"
#include "Device.h"
#include "GFXAPI.h"
#include "NativeHandleTraits.h"
#include "Core/Assert.h"
#include "Core/Errors.h"
#include "Core/ObjectPython.h"
#include "Core/Program/Program.h"
#include "Core/Program/ShaderVar.h"
#include "Utils/Logger.h"
#include "Utils/Scripting/ScriptBindings.h"

#define GFX_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT (256)
#define GFX_TEXTURE_DATA_PLACEMENT_ALIGNMENT (512)

namespace Falcor
{
// TODO: Replace with include?
void getGFXResourceState(Resource::BindFlags flags, gfx::ResourceState& defaultState, gfx::ResourceStateSet& allowedStates);

static ref<Buffer> createStructuredFromType(
    ref<Device> pDevice,
    const ReflectionType* pType,
    const std::string& varName,
    uint32_t elementCount,
    ResourceBindFlags bindFlags,
    Buffer::CpuAccess cpuAccess,
    const void* pInitData,
    bool createCounter
)
{
    const ReflectionResourceType* pResourceType = pType->unwrapArray()->asResourceType();
    if (!pResourceType || pResourceType->getType() != ReflectionResourceType::Type::StructuredBuffer)
    {
        throw RuntimeError("Can't create a structured buffer from the variable '{}'. The variable is not a structured buffer.", varName);
    }

    FALCOR_ASSERT(pResourceType->getSize() <= std::numeric_limits<uint32_t>::max());
    return Buffer::createStructured(
        pDevice, (uint32_t)pResourceType->getSize(), elementCount, bindFlags, cpuAccess, pInitData, createCounter
    );
}

static void prepareGFXBufferDesc(
    gfx::IBufferResource::Desc& bufDesc,
    size_t size,
    Resource::BindFlags bindFlags,
    Buffer::CpuAccess cpuAccess
)
{
    bufDesc.sizeInBytes = size;
    switch (cpuAccess)
    {
    case Buffer::CpuAccess::None:
        bufDesc.memoryType = gfx::MemoryType::DeviceLocal;
        break;
    case Buffer::CpuAccess::Read:
        bufDesc.memoryType = gfx::MemoryType::ReadBack;
        break;
    case Buffer::CpuAccess::Write:
        bufDesc.memoryType = gfx::MemoryType::Upload;
        break;
    default:
        FALCOR_UNREACHABLE();
        break;
    }
    getGFXResourceState(bindFlags, bufDesc.defaultState, bufDesc.allowedStates);
    bufDesc.isShared = is_set(bindFlags, Buffer::BindFlags::Shared);
}

// TODO: This is also used in Device
Slang::ComPtr<gfx::IBufferResource> createBuffer(
    ref<Device> pDevice,
    Buffer::State initState,
    size_t size,
    Buffer::BindFlags bindFlags,
    Buffer::CpuAccess cpuAccess
)
{
    FALCOR_ASSERT(pDevice);

    // Create the buffer
    gfx::IBufferResource::Desc bufDesc = {};
    prepareGFXBufferDesc(bufDesc, size, bindFlags, cpuAccess);

    Slang::ComPtr<gfx::IBufferResource> pApiHandle;
    FALCOR_GFX_CALL(pDevice->getGfxDevice()->createBufferResource(bufDesc, nullptr, pApiHandle.writeRef()));
    FALCOR_ASSERT(pApiHandle);

    return pApiHandle;
}

static size_t getBufferDataAlignment(const Buffer* pBuffer)
{
    // This in order of the alignment size
    const auto& bindFlags = pBuffer->getBindFlags();
    if (is_set(bindFlags, Buffer::BindFlags::Constant))
        return GFX_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT;
    if (is_set(bindFlags, Buffer::BindFlags::Index))
        return sizeof(uint32_t); // This actually depends on the size of the index, but we can handle losing 2 bytes

    return GFX_TEXTURE_DATA_PLACEMENT_ALIGNMENT;
}

Buffer::Buffer(ref<Device> pDevice, size_t size, BindFlags bindFlags, CpuAccess cpuAccess, const void* pInitData)
    : Resource(pDevice, Type::Buffer, bindFlags, size), mCpuAccess(cpuAccess)
{
    checkArgument(size > 0, "Can't create GPU buffer of size zero");

    // Check that buffer size is within 4GB limit. Larger buffers are currently not well supported in D3D12.
    // TODO: Revisit this check in the future.
    if (size > (1ull << 32))
    {
        logWarning("Creating GPU buffer of size {} bytes. Buffers above 4GB are not currently well supported.", size);
    }

    if (mCpuAccess != CpuAccess::None && is_set(mBindFlags, BindFlags::Shared))
    {
        throw RuntimeError("Can't create shared resource with CPU access other than 'None'.");
    }

    if (mBindFlags == BindFlags::Constant)
    {
        mSize = align_to((size_t)GFX_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT, mSize);
    }

    if (mCpuAccess == CpuAccess::Write)
    {
        mState.global = Resource::State::GenericRead;
        if (pInitData) // Else the allocation will happen when updating the data
        {
            mDynamicData = mpDevice->getUploadHeap()->allocate(mSize, getBufferDataAlignment(this));
            mGfxBufferResource = mDynamicData.gfxBufferResource;
            mGpuVaOffset = mDynamicData.offset;
        }
    }
    else if (mCpuAccess == CpuAccess::Read && mBindFlags == BindFlags::None)
    {
        mState.global = Resource::State::CopyDest;
        mGfxBufferResource = createBuffer(mpDevice, mState.global, mSize, mBindFlags, mCpuAccess);
    }
    else
    {
        mState.global = Resource::State::Common;
        if (is_set(mBindFlags, BindFlags::AccelerationStructure))
            mState.global = Resource::State::AccelerationStructure;
        mGfxBufferResource = createBuffer(mpDevice, mState.global, mSize, mBindFlags, mCpuAccess);
    }

    if (pInitData)
        setBlob(pInitData, 0, size);
    mElementCount = uint32_t(size);
}

ref<Buffer> Buffer::create(ref<Device> pDevice, size_t size, BindFlags bindFlags, CpuAccess cpuAccess, const void* pInitData)
{
    return ref<Buffer>(new Buffer(pDevice, size, bindFlags, cpuAccess, pInitData));
}

ref<Buffer> Buffer::createTyped(
    ref<Device> pDevice,
    ResourceFormat format,
    uint32_t elementCount,
    BindFlags bindFlags,
    CpuAccess cpuAccess,
    const void* pInitData
)
{
    size_t size = (size_t)elementCount * getFormatBytesPerBlock(format);
    ref<Buffer> pBuffer = create(pDevice, size, bindFlags, cpuAccess, pInitData);
    FALCOR_ASSERT(pBuffer);

    pBuffer->mFormat = format;
    pBuffer->mElementCount = elementCount;
    return pBuffer;
}

ref<Buffer> Buffer::createStructured(
    ref<Device> pDevice,
    uint32_t structSize,
    uint32_t elementCount,
    ResourceBindFlags bindFlags,
    CpuAccess cpuAccess,
    const void* pInitData,
    bool createCounter
)
{
    size_t size = (size_t)structSize * elementCount;
    ref<Buffer> pBuffer = create(pDevice, size, bindFlags, cpuAccess, pInitData);
    FALCOR_ASSERT(pBuffer);

    pBuffer->mElementCount = elementCount;
    pBuffer->mStructSize = structSize;
    static const uint32_t zero = 0;
    if (createCounter)
    {
        pBuffer->mpUAVCounter =
            Buffer::create(pDevice, sizeof(uint32_t), Resource::BindFlags::UnorderedAccess, Buffer::CpuAccess::None, &zero);
    }
    return pBuffer;
}

ref<Buffer> Buffer::createStructured(
    ref<Device> pDevice,
    const ShaderVar& shaderVar,
    uint32_t elementCount,
    ResourceBindFlags bindFlags,
    CpuAccess cpuAccess,
    const void* pInitData,
    bool createCounter
)
{
    return createStructuredFromType(
        pDevice, shaderVar.getType().get(), "<Unknown ShaderVar>", elementCount, bindFlags, cpuAccess, pInitData, createCounter
    );
}

ref<Buffer> Buffer::createStructured(
    ref<Device> pDevice,
    const Program* pProgram,
    const std::string& name,
    uint32_t elementCount,
    ResourceBindFlags bindFlags,
    CpuAccess cpuAccess,
    const void* pInitData,
    bool createCounter
)
{
    const auto& pDefaultBlock = pProgram->getReflector()->getDefaultParameterBlock();
    const ReflectionVar* pVar = pDefaultBlock ? pDefaultBlock->getResource(name).get() : nullptr;
    if (pVar == nullptr)
    {
        throw RuntimeError("Can't find a structured buffer named '{}' in the program", name);
    }
    return createStructuredFromType(pDevice, pVar->getType().get(), name, elementCount, bindFlags, cpuAccess, pInitData, createCounter);
}

ref<Buffer> Buffer::aliasResource(
    ref<Device> pDevice,
    ref<Buffer> pBaseResource,
    GpuAddress offset,
    size_t size,
    Resource::BindFlags bindFlags
)
{
    FALCOR_ASSERT(pBaseResource);
    CpuAccess cpuAccess = pBaseResource->asBuffer() ? pBaseResource->asBuffer()->getCpuAccess() : CpuAccess::None;
    checkArgument(
        cpuAccess != CpuAccess::None, "'pBaseResource' has CpuAccess:{} which is illegal. Aliased resources must have CpuAccess::None.",
        to_string(cpuAccess)
    );
    checkArgument(
        (pBaseResource->getBindFlags() & bindFlags) != bindFlags, "'bindFlags' ({}) don't match aliased resource bind flags {}.",
        to_string(bindFlags), to_string(pBaseResource->getBindFlags())
    );
    if (offset >= pBaseResource->getSize() || (offset + size) >= pBaseResource->getSize())
    {
        throw ArgumentError(
            "'offset' ({}) and 'size' ({}) don't fit inside the aliased resource size {}.", offset, size, pBaseResource->getSize()
        );
    }

    ref<Buffer> pBuffer = create(pDevice, size, bindFlags, CpuAccess::None);
    pBuffer->mpAliasedResource = pBaseResource;
    pBuffer->mGfxBufferResource = pBaseResource->mGfxBufferResource;
    pBuffer->mGpuVaOffset = offset;
    return pBuffer;
}

ref<Buffer> Buffer::createFromResource(
    ref<Device> pDevice,
    gfx::IBufferResource* pResource,
    size_t size,
    Resource::BindFlags bindFlags,
    CpuAccess cpuAccess
)
{
    FALCOR_ASSERT(pResource);
    ref<Buffer> pBuffer = create(pDevice, size, bindFlags, cpuAccess);
    pBuffer->mGfxBufferResource = pResource;
    return pBuffer;
}

ref<Buffer> Buffer::createFromNativeHandle(
    ref<Device> pDevice,
    NativeHandle handle,
    size_t size,
    Resource::BindFlags bindFlags,
    CpuAccess cpuAccess
)
{
    gfx::IBufferResource::Desc bufDesc = {};
    prepareGFXBufferDesc(bufDesc, size, bindFlags, cpuAccess);

    gfx::InteropHandle gfxNativeHandle = {};
#if FALCOR_HAS_D3D12
    if (pDevice->getType() == Device::Type::D3D12 && handle.getType() == NativeHandleType::ID3D12Resource)
    {
        gfxNativeHandle.api = gfx::InteropHandleAPI::D3D12;
        gfxNativeHandle.handleValue = reinterpret_cast<uint64_t>(handle.as<ID3D12Resource*>());
    }
#endif
#if FALCOR_HAS_VULKAN
    if (pDevice->getType() == Device::Type::Vulkan && handle.getType() == NativeHandleType::VkBuffer)
    {
        gfxNativeHandle.api = gfx::InteropHandleAPI::Vulkan;
        gfxNativeHandle.handleValue = reinterpret_cast<uint64_t>(handle.as<VkBuffer>());
    }
#endif

    if (gfxNativeHandle.api == gfx::InteropHandleAPI::Unknown)
    {
        // TODO: throw error
    }

    Slang::ComPtr<gfx::IBufferResource> gfxBuffer;
    FALCOR_GFX_CALL(pDevice->getGfxDevice()->createBufferFromNativeHandle(gfxNativeHandle, bufDesc, gfxBuffer.writeRef()));

    return Buffer::createFromResource(pDevice, gfxBuffer, size, bindFlags, cpuAccess);
}

Buffer::~Buffer()
{
    if (mpAliasedResource)
        return;

    if (mDynamicData.gfxBufferResource)
    {
        mpDevice->getUploadHeap()->release(mDynamicData);
    }
    else
    {
        mpDevice->releaseResource(mGfxBufferResource);
    }
}

gfx::IResource* Buffer::getGfxResource() const
{
    return mGfxBufferResource;
}

ref<ShaderResourceView> Buffer::getSRV(uint32_t firstElement, uint32_t elementCount)
{
    ResourceViewInfo view = ResourceViewInfo(firstElement, elementCount);

    if (mSrvs.find(view) == mSrvs.end())
        mSrvs[view] = ShaderResourceView::create(getDevice().get(), this, firstElement, elementCount);

    return mSrvs[view];
}

ref<ShaderResourceView> Buffer::getSRV()
{
    return getSRV(0);
}

ref<UnorderedAccessView> Buffer::getUAV(uint32_t firstElement, uint32_t elementCount)
{
    ResourceViewInfo view = ResourceViewInfo(firstElement, elementCount);

    if (mUavs.find(view) == mUavs.end())
        mUavs[view] = UnorderedAccessView::create(getDevice().get(), this, firstElement, elementCount);

    return mUavs[view];
}

ref<UnorderedAccessView> Buffer::getUAV()
{
    return getUAV(0);
}

void Buffer::setBlob(const void* pData, size_t offset, size_t size)
{
    if (offset + size > mSize)
    {
        throw ArgumentError("'offset' ({}) and 'size' ({}) don't fit the buffer size {}.", offset, size, mSize);
    }

    if (mCpuAccess == CpuAccess::Write)
    {
        uint8_t* pDst = (uint8_t*)map(MapType::WriteDiscard) + offset;
        std::memcpy(pDst, pData, size);
    }
    else
    {
        mpDevice->getRenderContext()->updateBuffer(this, pData, offset, size);
    }
}

void* Buffer::map(MapType type)
{
    if (type == MapType::Write)
    {
        checkArgument(
            mCpuAccess == CpuAccess::Write, "Trying to map a buffer for write, but it wasn't created with the write permissions."
        );
        return mDynamicData.pData;
    }
    else if (type == MapType::WriteDiscard)
    {
        checkArgument(
            mCpuAccess == CpuAccess::Write, "Trying to map a buffer for write, but it wasn't created with the write permissions."
        );

        // Allocate a new buffer
        if (mDynamicData.gfxBufferResource)
        {
            mpDevice->getUploadHeap()->release(mDynamicData);
        }
        mDynamicData = mpDevice->getUploadHeap()->allocate(mSize, getBufferDataAlignment(this));
        mGfxBufferResource = mDynamicData.gfxBufferResource;
        mGpuVaOffset = mDynamicData.offset;
        invalidateViews();
        return mDynamicData.pData;
    }
    else
    {
        FALCOR_ASSERT(type == MapType::Read);

        if (mCpuAccess == CpuAccess::Write)
        {
            // Buffers on the upload heap are already mapped, just return the ptr.
            FALCOR_ASSERT(mDynamicData.gfxBufferResource);
            FALCOR_ASSERT(mDynamicData.pData);
            return mDynamicData.pData;
        }
        else if (mCpuAccess == CpuAccess::Read)
        {
            FALCOR_ASSERT(mBindFlags == BindFlags::None);
            void* pData = nullptr;
            FALCOR_GFX_CALL(mGfxBufferResource->map(nullptr, &pData));
            return pData;
        }
        else
        {
            // For buffers without CPU access we must copy the contents to a staging buffer.
            logWarning(
                "Buffer::map() performance warning - using staging resource which require us to flush the pipeline and wait for the GPU to "
                "finish its work"
            );
            if (mpStagingResource == nullptr)
            {
                mpStagingResource = Buffer::create(mpDevice, mSize, Buffer::BindFlags::None, Buffer::CpuAccess::Read, nullptr);
            }

            // Copy the buffer and flush the pipeline
            RenderContext* pContext = mpDevice->getRenderContext();
            FALCOR_ASSERT(mGpuVaOffset == 0);
            pContext->copyResource(mpStagingResource.get(), this);
            pContext->flush(true);
            return mpStagingResource->map(MapType::Read);
        }
    }
}

void Buffer::unmap()
{
    // Only unmap read buffers, write buffers are persistently mapped.
    if (mpStagingResource)
    {
        FALCOR_GFX_CALL(mpStagingResource->mGfxBufferResource->unmap(nullptr));
    }
    else if (mCpuAccess == CpuAccess::Read)
    {
        FALCOR_GFX_CALL(mGfxBufferResource->unmap(nullptr));
    }
}

uint32_t Buffer::getElementSize() const
{
    if (mStructSize != 0)
        return mStructSize;
    if (mFormat == ResourceFormat::Unknown)
        return 1;

    throw RuntimeError("Inferring element size from resource format is not implemented");
}

bool Buffer::adjustSizeOffsetParams(size_t& size, size_t& offset) const
{
    if (offset >= mSize)
    {
        logWarning("Buffer::adjustSizeOffsetParams() - offset is larger than the buffer size.");
        return false;
    }

    if (offset + size > mSize)
    {
        logWarning("Buffer::adjustSizeOffsetParams() - offset + size will cause an OOB access. Clamping size");
        size = mSize - offset;
    }
    return true;
}

uint64_t Buffer::getGpuAddress() const
{
    // slang-gfx backend does not includ the mGpuVaOffset.
    return mGpuVaOffset + mGfxBufferResource->getDeviceAddress();
}

FALCOR_SCRIPT_BINDING(Buffer)
{
    FALCOR_SCRIPT_BINDING_DEPENDENCY(Resource)

    pybind11::class_<Buffer, Resource, ref<Buffer>> buffer(m, "Buffer");

    pybind11::enum_<Buffer::CpuAccess> cpuAccess(buffer, "CpuAccess");
    cpuAccess.value("None_", Buffer::CpuAccess::None);
    cpuAccess.value("Read", Buffer::CpuAccess::Read);
    cpuAccess.value("Write", Buffer::CpuAccess::Write);
}
} // namespace Falcor
