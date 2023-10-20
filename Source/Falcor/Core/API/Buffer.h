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
#include "Core/Macros.h"
#include "Core/Enum.h"
#include "Resource.h"
#include "ResourceViews.h"

namespace Falcor
{
class Program;
struct ShaderVar;

namespace cuda_utils
{
class ExternalMemory;
};

namespace detail
{
/**
 * Helper for converting host type to resource format for typed buffers.
 * See list of supported formats for typed UAV loads:
 * https://docs.microsoft.com/en-us/windows/win32/direct3d12/typed-unordered-access-view-loads
 */
template<typename T>
struct FormatForElementType
{};

#define CASE(TYPE, FORMAT)                                \
    template<>                                            \
    struct FormatForElementType<TYPE>                     \
    {                                                     \
        static constexpr ResourceFormat kFormat = FORMAT; \
    }

// Guaranteed supported formats on D3D12.
CASE(float, ResourceFormat::R32Float);
CASE(uint32_t, ResourceFormat::R32Uint);
CASE(int32_t, ResourceFormat::R32Int);

// Optionally supported formats as a set on D3D12. If one is supported all are supported.
CASE(float4, ResourceFormat::RGBA32Float);
CASE(uint4, ResourceFormat::RGBA32Uint);
CASE(int4, ResourceFormat::RGBA32Int);
// R16G16B16A16_FLOAT
// R16G16B16A16_UINT
// R16G16B16A16_SINT
// R8G8B8A8_UNORM
// R8G8B8A8_UINT
// R8G8B8A8_SINT
// R16_FLOAT
CASE(uint16_t, ResourceFormat::R16Uint);
CASE(int16_t, ResourceFormat::R16Int);
// R8_UNORM
CASE(uint8_t, ResourceFormat::R8Uint);
CASE(int8_t, ResourceFormat::R8Int);

// Optionally and individually supported formats on D3D12. Query for support individually.
// R16G16B16A16_UNORM
// R16G16B16A16_SNORM
CASE(float2, ResourceFormat::RG32Float);
CASE(uint2, ResourceFormat::RG32Uint);
CASE(int2, ResourceFormat::RG32Int);
// R10G10B10A2_UNORM
// R10G10B10A2_UINT
// R11G11B10_FLOAT
// R8G8B8A8_SNORM
// R16G16_FLOAT
// R16G16_UNORM
// R16G16_UINT
// R16G16_SNORM
// R16G16_SINT
// R8G8_UNORM
// R8G8_UINT
// R8G8_SNORM
// 8G8_SINT
// R16_UNORM
// R16_SNORM
// R8_SNORM
// A8_UNORM
// B5G6R5_UNORM
// B5G5R5A1_UNORM
// B4G4R4A4_UNORM

// Additional formats that may be supported on some hardware.
CASE(float3, ResourceFormat::RGB32Float);

#undef CASE
} // namespace detail

/// Buffer memory types.
enum class MemoryType
{
    DeviceLocal, ///< Device local memory. The buffer can be updated using Buffer::setBlob().
    Upload,      ///< Upload memory. The buffer can be mapped for CPU writes.
    ReadBack,    ///< Read-back memory. The buffer can be mapped for CPU reads.

    // NOTE: In older version of Falcor this enum used to be Buffer::CpuAccess.
    // Use the following mapping to update your code:
    // - CpuAccess::None -> MemoryType::DeviceLocal
    // - CpuAccess::Write -> MemoryType::Upload
    // - CpuAccess::Read -> MemoryType::ReadBack
};
FALCOR_ENUM_INFO(
    MemoryType,
    {
        {MemoryType::DeviceLocal, "DeviceLocal"},
        {MemoryType::Upload, "Upload"},
        {MemoryType::ReadBack, "ReadBack"},
    }
);
FALCOR_ENUM_REGISTER(MemoryType);

/**
 * Low-level buffer object
 * This class abstracts the API's buffer creation and management
 */
class FALCOR_API Buffer : public Resource
{
    FALCOR_OBJECT(Buffer)
public:
    enum class MapType
    {
        Read,         ///< Map the buffer for read access.
        Write,        ///< Map the buffer for write access.
        WriteDiscard, ///< Deprecated and not supported.
    };

    /// Constructor for raw buffer.
    Buffer(ref<Device> pDevice, size_t size, ResourceBindFlags bindFlags, MemoryType memoryType, const void* pInitData);

    /// Constructor for typed buffer.
    Buffer(
        ref<Device> pDevice,
        ResourceFormat format,
        uint32_t elementCount,
        ResourceBindFlags bindFlags,
        MemoryType memoryType,
        const void* pInitData
    );

    /// Constructor for structured buffer.
    Buffer(
        ref<Device> pDevice,
        uint32_t structSize,
        uint32_t elementCount,
        ResourceBindFlags bindFlags,
        MemoryType memoryType,
        const void* pInitData,
        bool createCounter
    );

    /// Constructor with existing resource.
    Buffer(ref<Device> pDevice, gfx::IBufferResource* pResource, size_t size, ResourceBindFlags bindFlags, MemoryType memoryType);

    /// Constructor with native handle.
    Buffer(ref<Device> pDevice, NativeHandle handle, size_t size, ResourceBindFlags bindFlags, MemoryType memoryType);

    /// Destructor.
    ~Buffer();

    gfx::IBufferResource* getGfxBufferResource() const { return mGfxBufferResource; }

    virtual gfx::IResource* getGfxResource() const override;

    /**
     * Get a shader-resource view.
     * @param[in] firstElement The first element of the view. For raw buffers, an element is a single float
     * @param[in] elementCount The number of elements to bind
     */
    ref<ShaderResourceView> getSRV(uint32_t firstElement, uint32_t elementCount = kMaxPossible);

    /**
     * Get an unordered access view.
     * @param[in] firstElement The first element of the view. For raw buffers, an element is a single float
     * @param[in] elementCount The number of elements to bind
     */
    ref<UnorderedAccessView> getUAV(uint32_t firstElement, uint32_t elementCount = kMaxPossible);

    /**
     * Get a shader-resource view for the entire resource
     */
    virtual ref<ShaderResourceView> getSRV() override;

    /**
     * Get an unordered access view for the entire resource
     */
    virtual ref<UnorderedAccessView> getUAV() override;

    /**
     * Get the size of each element in this buffer.
     *
     * For a typed buffer, this will be the size of the format.
     * For a structured buffer, this will be the same value as `getStructSize()`.
     * For a raw buffer, this will be the number of bytes.
     */
    uint32_t getElementSize() const;

    /**
     * Update the buffer's data
     * @param[in] pData Pointer to the source data.
     * @param[in] offset Byte offset into the destination buffer, indicating where to start copy into.
     * @param[in] size Number of bytes to copy.
     * Throws an exception if data causes out-of-bound access to the buffer.
     */
    virtual void setBlob(const void* pData, size_t offset, size_t size);

    /**
     * Read the buffer's data
     * @param pData Pointer to the destination data.
     * @param offset Byte offset into the source buffer, indicating where to start copy from.
     * @param size Number of bytes to copy.
     */
    void getBlob(void* pData, size_t offset, size_t size) const;

    /**
     * Get the GPU address (this includes the offset)
     */
    uint64_t getGpuAddress() const;

    /**
     * Get the size of the buffer in bytes.
     */
    size_t getSize() const { return mSize; }

    /**
     * Get the element count. For structured-buffers, this is the number of structs. For typed-buffers, this is the number of elements. For
     * other buffer, will return the size in bytes.
     */
    uint32_t getElementCount() const { return mElementCount; }

    /**
     * Get the size of a single struct. This call is only valid for structured-buffer. For other buffer types, it will return 0
     */
    uint32_t getStructSize() const { return mStructSize; }

    /**
     * Get the buffer format. This call is only valid for typed-buffers, for other buffer types it will return ResourceFormat::Unknown
     */
    ResourceFormat getFormat() const { return mFormat; }

    /**
     * Get the UAV counter buffer
     */
    const ref<Buffer>& getUAVCounter() const { return mpUAVCounter; }

    /**
     * Map the buffer.
     * Only buffers with MemoryType::Upload or MemoryType::ReadBack can be mapped.
     * To map a buffer with MemoryType::Upload, use MapType::Write.
     * To map a buffer with MemoryType::ReadBack, use MapType::Read.
     */
    void* map(MapType Type) const;

    /**
     * Unmap the buffer
     */
    void unmap() const;

    /**
     * Get safe offset and size values
     */
    bool adjustSizeOffsetParams(size_t& size, size_t& offset) const;

    /**
     * Get the memory type
     */
    MemoryType getMemoryType() const { return mMemoryType; }

    /**
     * Check if this is a typed buffer
     */
    bool isTyped() const { return mFormat != ResourceFormat::Unknown; }

    /**
     * Check if this is a structured-buffer
     */
    bool isStructured() const { return mStructSize != 0; }

    template<typename T>
    void setElement(uint32_t index, const T& value)
    {
        setBlob(&value, sizeof(T) * index, sizeof(T));
    }

    template<typename T>
    std::vector<T> getElements(uint32_t firstElement = 0, uint32_t elementCount = 0) const
    {
        if (elementCount == 0)
            elementCount = (mSize / sizeof(T)) - firstElement;

        std::vector<T> data(elementCount);
        getBlob(data.data(), firstElement * sizeof(T), elementCount * sizeof(T));
        return data;
    }

    template<typename T>
    T getElement(uint32_t index) const
    {
        T data;
        getBlob(&data, index * sizeof(T), sizeof(T));
        return data;
    }

#if FALCOR_HAS_CUDA
    cuda_utils::ExternalMemory* getCudaMemory() const;
#endif

protected:
    Slang::ComPtr<gfx::IBufferResource> mGfxBufferResource;

    MemoryType mMemoryType;
    uint32_t mElementCount = 0;
    ResourceFormat mFormat = ResourceFormat::Unknown;
    uint32_t mStructSize = 0;
    ref<Buffer> mpUAVCounter; // For structured-buffers
    mutable void* mMappedPtr = nullptr;

#if FALCOR_HAS_CUDA
    mutable ref<cuda_utils::ExternalMemory> mCudaMemory;
#endif
};

inline std::string to_string(MemoryType c)
{
#define a2s(ca_)          \
    case MemoryType::ca_: \
        return #ca_;
    switch (c)
    {
        a2s(DeviceLocal);
        a2s(Upload);
        a2s(ReadBack);
    default:
        FALCOR_UNREACHABLE();
        return "";
    }
#undef a2s
}

inline std::string to_string(Buffer::MapType mt)
{
#define t2s(t_)               \
    case Buffer::MapType::t_: \
        return #t_;
    switch (mt)
    {
        t2s(Read);
        t2s(Write);
        t2s(WriteDiscard);
    default:
        FALCOR_UNREACHABLE();
        return "";
    }
#undef t2s
}
} // namespace Falcor
