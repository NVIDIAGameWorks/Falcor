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
#include "PythonHelpers.h"
#include "Core/Error.h"
#include "Core/ObjectPython.h"
#include "Core/Program/Program.h"
#include "Core/Program/ShaderVar.h"
#include "Utils/Logger.h"
#include "Utils/Scripting/ScriptBindings.h"
#include "Utils/Scripting/ndarray.h"

#if FALCOR_HAS_CUDA
#include "Utils/CudaUtils.h"
#endif

namespace Falcor
{
// TODO: Replace with include?
void getGFXResourceState(ResourceBindFlags flags, gfx::ResourceState& defaultState, gfx::ResourceStateSet& allowedStates);

static void prepareGFXBufferDesc(gfx::IBufferResource::Desc& bufDesc, size_t size, ResourceBindFlags bindFlags, MemoryType memoryType)
{
    bufDesc.sizeInBytes = size;
    switch (memoryType)
    {
    case MemoryType::DeviceLocal:
        bufDesc.memoryType = gfx::MemoryType::DeviceLocal;
        break;
    case MemoryType::ReadBack:
        bufDesc.memoryType = gfx::MemoryType::ReadBack;
        break;
    case MemoryType::Upload:
        bufDesc.memoryType = gfx::MemoryType::Upload;
        break;
    default:
        FALCOR_UNREACHABLE();
        break;
    }
    getGFXResourceState(bindFlags, bufDesc.defaultState, bufDesc.allowedStates);
    bufDesc.isShared = is_set(bindFlags, ResourceBindFlags::Shared);
}

// TODO: This is also used in GpuMemoryHeap
Slang::ComPtr<gfx::IBufferResource> createBufferResource(
    ref<Device> pDevice,
    Buffer::State initState,
    size_t size,
    ResourceBindFlags bindFlags,
    MemoryType memoryType
)
{
    FALCOR_ASSERT(pDevice);

    // Create the buffer
    gfx::IBufferResource::Desc bufDesc = {};
    prepareGFXBufferDesc(bufDesc, size, bindFlags, memoryType);

    Slang::ComPtr<gfx::IBufferResource> pApiHandle;
    FALCOR_GFX_CALL(pDevice->getGfxDevice()->createBufferResource(bufDesc, nullptr, pApiHandle.writeRef()));
    FALCOR_ASSERT(pApiHandle);

    return pApiHandle;
}

Buffer::Buffer(ref<Device> pDevice, size_t size, ResourceBindFlags bindFlags, MemoryType memoryType, const void* pInitData)
    : Resource(pDevice, Type::Buffer, bindFlags, size), mMemoryType(memoryType)
{
    FALCOR_CHECK(size > 0, "Can't create GPU buffer of size zero");

    // Check that buffer size is within 4GB limit. Larger buffers are currently not well supported in D3D12.
    // TODO: Revisit this check in the future.
    if (size > (1ull << 32))
    {
        logWarning("Creating GPU buffer of size {} bytes. Buffers above 4GB are not currently well supported.", size);
    }

    if (mMemoryType != MemoryType::DeviceLocal && is_set(mBindFlags, ResourceBindFlags::Shared))
    {
        FALCOR_THROW("Can't create shared resource with CPU access other than 'None'.");
    }

    mSize = align_to(mpDevice->getBufferDataAlignment(bindFlags), mSize);

    if (mMemoryType == MemoryType::DeviceLocal)
    {
        mState.global = Resource::State::Common;
        if (is_set(mBindFlags, ResourceBindFlags::AccelerationStructure))
            mState.global = Resource::State::AccelerationStructure;
    }
    else if (mMemoryType == MemoryType::Upload)
    {
        mState.global = Resource::State::GenericRead;
    }
    else if (mMemoryType == MemoryType::ReadBack)
    {
        mState.global = Resource::State::CopyDest;
    }

    mGfxBufferResource = createBufferResource(mpDevice, mState.global, mSize, mBindFlags, mMemoryType);

    if (pInitData)
        setBlob(pInitData, 0, size);

    mElementCount = uint32_t(size);
}

Buffer::Buffer(
    ref<Device> pDevice,
    ResourceFormat format,
    uint32_t elementCount,
    ResourceBindFlags bindFlags,
    MemoryType memoryType,
    const void* pInitData
)
    : Buffer(pDevice, (size_t)getFormatBytesPerBlock(format) * elementCount, bindFlags, memoryType, pInitData)
{
    mFormat = format;
    mElementCount = elementCount;
}

Buffer::Buffer(
    ref<Device> pDevice,
    uint32_t structSize,
    uint32_t elementCount,
    ResourceBindFlags bindFlags,
    MemoryType memoryType,
    const void* pInitData,
    bool createCounter
)
    : Buffer(pDevice, (size_t)structSize * elementCount, bindFlags, memoryType, pInitData)
{
    mElementCount = elementCount;
    mStructSize = structSize;
    static const uint32_t zero = 0;
    if (createCounter)
    {
        mpUAVCounter = make_ref<Buffer>(mpDevice, sizeof(uint32_t), ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, &zero);
    }
}

// TODO: Its wasteful to create a buffer just to replace it afterwards with the supplied one!
Buffer::Buffer(ref<Device> pDevice, gfx::IBufferResource* pResource, size_t size, ResourceBindFlags bindFlags, MemoryType memoryType)
    : Buffer(pDevice, size, bindFlags, memoryType, nullptr)
{
    FALCOR_ASSERT(pResource);
    mGfxBufferResource = pResource;
}

inline Slang::ComPtr<gfx::IBufferResource> gfxResourceFromNativeHandle(
    Device* pDevice,
    NativeHandle handle,
    size_t size,
    ResourceBindFlags bindFlags,
    MemoryType memoryType
)
{
    gfx::IBufferResource::Desc bufDesc = {};
    prepareGFXBufferDesc(bufDesc, size, bindFlags, memoryType);

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
        FALCOR_THROW("Unknown native handle type");

    Slang::ComPtr<gfx::IBufferResource> gfxBuffer;
    FALCOR_GFX_CALL(pDevice->getGfxDevice()->createBufferFromNativeHandle(gfxNativeHandle, bufDesc, gfxBuffer.writeRef()));

    return gfxBuffer;
}

Buffer::Buffer(ref<Device> pDevice, NativeHandle handle, size_t size, ResourceBindFlags bindFlags, MemoryType memoryType)
    : Buffer(pDevice, gfxResourceFromNativeHandle(pDevice.get(), handle, size, bindFlags, memoryType), size, bindFlags, memoryType)
{}

Buffer::~Buffer()
{
    mpDevice->releaseResource(mGfxBufferResource);
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
    FALCOR_CHECK(offset + size <= mSize, "'offset' ({}) and 'size' ({}) don't fit the buffer size {}.", offset, size, mSize);

    if (mMemoryType == MemoryType::Upload)
    {
        bool wasMapped = mMappedPtr != nullptr;
        uint8_t* pDst = (uint8_t*)map(MapType::Write) + offset;
        std::memcpy(pDst, pData, size);
        if (!wasMapped)
            unmap();
        // TODO we should probably use a barrier instead
        invalidateViews();
    }
    else if (mMemoryType == MemoryType::DeviceLocal)
    {
        mpDevice->getRenderContext()->updateBuffer(this, pData, offset, size);
    }
    else if (mMemoryType == MemoryType::ReadBack)
    {
        FALCOR_THROW("Cannot set data to a buffer that was created with MemoryType::ReadBack.");
    }
}

void Buffer::getBlob(void* pData, size_t offset, size_t size) const
{
    FALCOR_CHECK(offset + size <= mSize, "'offset' ({}) and 'size' ({}) don't fit the buffer size {}.", offset, size, mSize);

    if (mMemoryType == MemoryType::ReadBack)
    {
        bool wasMapped = mMappedPtr != nullptr;
        const uint8_t* pSrc = (const uint8_t*)map(MapType::Read) + offset;
        std::memcpy(pData, pSrc, size);
        if (!wasMapped)
            unmap();
    }
    else if (mMemoryType == MemoryType::DeviceLocal)
    {
        mpDevice->getRenderContext()->readBuffer(this, pData, offset, size);
    }
    else if (mMemoryType == MemoryType::Upload)
    {
        FALCOR_THROW("Cannot get data from a buffer that was created with MemoryType::Upload.");
    }
}

void* Buffer::map(MapType type) const
{
    if (type == MapType::WriteDiscard)
        FALCOR_THROW("MapType::WriteDiscard not supported anymore");

    if (type == MapType::Write && mMemoryType != MemoryType::Upload)
        FALCOR_THROW("Trying to map a buffer for writing, but it wasn't created with the write permissions.");

    if (type == MapType::Read && mMemoryType != MemoryType::ReadBack)
        FALCOR_THROW("Trying to map a buffer for reading, but it wasn't created with the read permissions.");

    if (!mMappedPtr)
        FALCOR_GFX_CALL(mGfxBufferResource->map(nullptr, &mMappedPtr));

    return mMappedPtr;
}

void Buffer::unmap() const
{
    if (mMappedPtr)
    {
        FALCOR_GFX_CALL(mGfxBufferResource->unmap(nullptr));
        mMappedPtr = nullptr;
    }
}

uint32_t Buffer::getElementSize() const
{
    if (mStructSize != 0)
        return mStructSize;
    if (mFormat == ResourceFormat::Unknown)
        return 1;

    FALCOR_THROW("Inferring element size from resource format is not implemented");
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
    return mGfxBufferResource->getDeviceAddress();
}

#if FALCOR_HAS_CUDA
cuda_utils::ExternalMemory* Buffer::getCudaMemory() const
{
    if (!mCudaMemory)
        mCudaMemory = make_ref<cuda_utils::ExternalMemory>(ref<Resource>(const_cast<Buffer*>(this)));
    return mCudaMemory.get();
}
#endif

inline pybind11::ndarray<pybind11::numpy> buffer_to_numpy(const Buffer& self)
{
    size_t bufferSize = self.getSize();
    void* cpuData = new uint8_t[bufferSize];
    self.getBlob(cpuData, 0, bufferSize);

    pybind11::capsule owner(cpuData, [](void* p) noexcept { delete[] reinterpret_cast<uint8_t*>(p); });

    if (auto dtype = resourceFormatToDtype(self.getFormat()))
    {
        uint32_t channelCount = getFormatChannelCount(self.getFormat());
        if (channelCount == 1)
        {
            pybind11::size_t shape[1] = {self.getElementCount()};
            return pybind11::ndarray<pybind11::numpy>(cpuData, 1, shape, owner, nullptr, *dtype, pybind11::device::cpu::value);
        }
        else
        {
            pybind11::size_t shape[2] = {self.getElementCount(), channelCount};
            return pybind11::ndarray<pybind11::numpy>(cpuData, 2, shape, owner, nullptr, *dtype, pybind11::device::cpu::value);
        }
    }
    else
    {
        pybind11::size_t shape[1] = {bufferSize};
        return pybind11::ndarray<pybind11::numpy>(
            cpuData, 1, shape, owner, nullptr, pybind11::dtype<uint8_t>(), pybind11::device::cpu::value
        );
    }
}

inline void buffer_from_numpy(Buffer& self, pybind11::ndarray<pybind11::numpy> data)
{
    FALCOR_CHECK(isNdarrayContiguous(data), "numpy array is not contiguous");

    size_t bufferSize = self.getSize();
    size_t dataSize = getNdarrayByteSize(data);
    FALCOR_CHECK(dataSize <= bufferSize, "numpy array is larger than the buffer ({} > {})", dataSize, bufferSize);

    self.setBlob(data.data(), 0, dataSize);
}

#if FALCOR_HAS_CUDA
inline pybind11::ndarray<pybind11::pytorch> buffer_to_torch(const Buffer& self, std::vector<size_t> shape, DataType dtype)
{
    cuda_utils::ExternalMemory* cudaMemory = self.getCudaMemory();

    return pybind11::ndarray<pybind11::pytorch>(
        cudaMemory->getMappedData(), shape.size(), shape.data(), nullptr, nullptr, dataTypeToDtype(dtype), pybind11::device::cuda::value
    );
}

inline void buffer_from_torch(Buffer& self, pybind11::ndarray<pybind11::pytorch> data)
{
    FALCOR_CHECK(isNdarrayContiguous(data), "torch tensor is not contiguous");
    FALCOR_CHECK(data.device_type() == pybind11::device::cuda::value, "torch tensor is not on the device");

    cuda_utils::ExternalMemory* cudaMemory = self.getCudaMemory();
    size_t dataSize = getNdarrayByteSize(data);
    FALCOR_CHECK(dataSize <= cudaMemory->getSize(), "torch tensor is larger than the buffer ({} > {})", dataSize, cudaMemory->getSize());

    cuda_utils::memcpyDeviceToDevice(cudaMemory->getMappedData(), data.data(), dataSize);
}

inline void buffer_copy_to_torch(Buffer& self, pybind11::ndarray<pybind11::pytorch>& data)
{
    FALCOR_CHECK(isNdarrayContiguous(data), "torch tensor is not contiguous");
    FALCOR_CHECK(data.device_type() == pybind11::device::cuda::value, "torch tensor is not on the device");

    cuda_utils::ExternalMemory* cudaMemory = self.getCudaMemory();
    size_t dataSize = getNdarrayByteSize(data);
    FALCOR_CHECK(dataSize >= cudaMemory->getSize(), "torch tensor is smaller than the buffer ({} < {})", dataSize, cudaMemory->getSize());

    cuda_utils::memcpyDeviceToDevice(data.data(), cudaMemory->getMappedData(), dataSize);
}
#endif

FALCOR_SCRIPT_BINDING(Buffer)
{
    using namespace pybind11::literals;

    FALCOR_SCRIPT_BINDING_DEPENDENCY(Types)
    FALCOR_SCRIPT_BINDING_DEPENDENCY(Resource)

    pybind11::falcor_enum<MemoryType>(m, "MemoryType");

    pybind11::class_<Buffer, Resource, ref<Buffer>> buffer(m, "Buffer");
    buffer.def_property_readonly("memory_type", &Buffer::getMemoryType);
    buffer.def_property_readonly("size", &Buffer::getSize);
    buffer.def_property_readonly("is_typed", &Buffer::isTyped);
    buffer.def_property_readonly("is_structured", &Buffer::isStructured);
    buffer.def_property_readonly("format", &Buffer::getFormat);
    buffer.def_property_readonly("element_size", &Buffer::getElementSize);
    buffer.def_property_readonly("element_count", &Buffer::getElementCount);
    buffer.def_property_readonly("struct_size", &Buffer::getStructSize);

    buffer.def("to_numpy", buffer_to_numpy);
    buffer.def("from_numpy", buffer_from_numpy, "data"_a);
#if FALCOR_HAS_CUDA
    buffer.def("to_torch", buffer_to_torch, "shape"_a, "dtype"_a = DataType::float32);
    buffer.def("from_torch", buffer_from_torch, "data"_a);
    buffer.def("copy_to_torch", buffer_copy_to_torch, "data"_a);
#endif
}
} // namespace Falcor
