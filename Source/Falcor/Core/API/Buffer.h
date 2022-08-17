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
#pragma once
#include "GpuMemoryHeap.h"
#include "Core/Macros.h"
#include "Resource.h"
#include "ResourceViews.h"
#include <memory>

namespace Falcor
{
    class Program;
    struct ShaderVar;

    /** Low-level buffer object
        This class abstracts the API's buffer creation and management
    */
    class FALCOR_API Buffer : public Resource
    {
    public:
        using SharedPtr = std::shared_ptr<Buffer>;
        using WeakPtr = std::weak_ptr<Buffer>;
        using SharedConstPtr = std::shared_ptr<const Buffer>;

        /** Buffer access flags.
            These flags are hints the driver how the buffer will be used.
        */
        enum class CpuAccess
        {
            None,    ///< The CPU can't access the buffer's content. The buffer can be updated using Buffer#updateData()
            Write,   ///< The buffer can be mapped for CPU writes
            Read,    ///< The buffer can be mapped for CPU reads
        };

        enum class MapType
        {
            Read,           ///< Map the buffer for read access.
            Write,          ///< Map the buffer for write access. Buffer had to be created with CpuAccess::Write flag.
            WriteDiscard,   ///< Map the buffer for write access, discarding the previous content of the entire buffer. Buffer had to be created with CpuAccess::Write flag.
        };

        ~Buffer();

        /** Create a new buffer.
            \param[in] size Size of the buffer in bytes.
            \param[in] bindFlags Buffer bind flags.
            \param[in] cpuAccess Flags indicating how the buffer can be updated.
            \param[in] pInitData Optional parameter. Initial buffer data. Pointed buffer size should be at least 'size' bytes.
            \return A pointer to a new buffer object, or throws an exception if creation failed.
        */
        static SharedPtr create(
            size_t size,
            Resource::BindFlags bindFlags = Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess,
            CpuAccess cpuAccess = Buffer::CpuAccess::None,
            const void* pInitData = nullptr);

        /** Create a new typed buffer.
            \param[in] format Typed buffer format.
            \param[in] elementCount Number of elements.
            \param[in] bindFlags Buffer bind flags.
            \param[in] cpuAccess Flags indicating how the buffer can be updated.
            \param[in] pInitData Optional parameter. Initial buffer data. Pointed buffer should hold at least 'elementCount' elements.
            \return A pointer to a new buffer object, or throws an exception if creation failed.
        */
        static SharedPtr createTyped(
            ResourceFormat format,
            uint32_t elementCount,
            Resource::BindFlags bindFlags = Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess,
            CpuAccess cpuAccess = Buffer::CpuAccess::None,
            const void* pInitData = nullptr);

        /** Create a new typed buffer. The format is deduced from the template parameter.
            \param[in] elementCount Number of elements.
            \param[in] bindFlags Buffer bind flags.
            \param[in] cpuAccess Flags indicating how the buffer can be updated.
            \param[in] pInitData Optional parameter. Initial buffer data. Pointed buffer should hold at least 'elementCount' elements.
            \return A pointer to a new buffer object, or throws an exception if creation failed.
        */
        template<typename T>
        static SharedPtr createTyped(
            uint32_t elementCount,
            Resource::BindFlags bindFlags = Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess,
            CpuAccess cpuAccess = Buffer::CpuAccess::None,
            const T* pInitData = nullptr)
        {
            return createTyped(FormatForElementType<T>::kFormat, elementCount, bindFlags, cpuAccess, pInitData);
        }

        /** Create a new structured buffer.
            \param[in] structSize Size of the struct in bytes.
            \param[in] elementCount Number of elements.
            \param[in] bindFlags Buffer bind flags.
            \param[in] cpuAccess Flags indicating how the buffer can be updated.
            \param[in] pInitData Optional parameter. Initial buffer data. Pointed buffer should hold at least 'elementCount' elements.
            \param[in] createCounter True if the associated UAV counter should be created.
            \return A pointer to a new buffer object, or throws an exception if creation failed.
        */
        static SharedPtr createStructured(
            uint32_t structSize,
            uint32_t elementCount,
            ResourceBindFlags bindFlags = Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess,
            CpuAccess cpuAccess = Buffer::CpuAccess::None,
            const void* pInitData = nullptr,
            bool createCounter = true);

        /** Create a new structured buffer.
            \param[in] pProgram Program declaring the buffer.
            \param[in] name Variable name in the program.
            \param[in] elementCount Number of elements.
            \param[in] bindFlags Buffer bind flags.
            \param[in] cpuAccess Flags indicating how the buffer can be updated.
            \param[in] pInitData Optional parameter. Initial buffer data. Pointed buffer should hold at least 'elementCount' elements.
            \param[in] createCounter True if the associated UAV counter should be created.
            \return A pointer to a new buffer object, or throws an exception if creation failed.
        */
        static SharedPtr createStructured(
            const Program* pProgram,
            const std::string& name,
            uint32_t elementCount,
            ResourceBindFlags bindFlags = Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess,
            CpuAccess cpuAccess = Buffer::CpuAccess::None,
            const void* pInitData = nullptr,
            bool createCounter = true);

        /** Create a new structured buffer.
            \param[in] shaderVar ShaderVar pointing to the buffer variable.
            \param[in] elementCount Number of elements.
            \param[in] bindFlags Buffer bind flags.
            \param[in] cpuAccess Flags indicating how the buffer can be updated.
            \param[in] pInitData Optional parameter. Initial buffer data. Pointed buffer should hold at least 'elementCount' elements.
            \param[in] createCounter True if the associated UAV counter should be created.
            \return A pointer to a new buffer object, or throws an exception if creation failed.
        */
        static SharedPtr createStructured(
            const ShaderVar& shaderVar,
            uint32_t elementCount,
            ResourceBindFlags bindFlags = Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess,
            CpuAccess cpuAccess = Buffer::CpuAccess::None,
            const void* pInitData = nullptr,
            bool createCounter = true);

        static SharedPtr aliasResource(Resource::SharedPtr pBaseResource, GpuAddress offset, size_t size, Resource::BindFlags bindFlags);

        /** Create a new buffer from an existing API handle.
            \param[in] handle Handle of already allocated resource.
            \param[in] size The size of the buffer in bytes.
            \param[in] bindFlags Buffer bind flags. Flags must match the bind flags of the original resource.
            \param[in] cpuAccess Flags indicating how the buffer can be updated. Flags must match those of the heap the original resource is allocated on.
            \return A pointer to a new buffer object, or throws an exception if creation failed.
        */
        static SharedPtr createFromApiHandle(ApiHandle handle, size_t size, Resource::BindFlags bindFlags, CpuAccess cpuAccess);

        /** Create a new buffer from an existing D3D12 handle. Available only when D3D12 is the underlying graphics API.
            \param[in] handle Handle of already allocated resource.
            \param[in] size The size of the buffer in bytes.
            \param[in] bindFlags Buffer bind flags. Flags must match the bind flags of the original resource.
            \param[in] cpuAccess Flags indicating how the buffer can be updated. Flags must match those of the heap the original resource is allocated on.
            \return A pointer to a new buffer object, or throws an exception if creation failed.
        */
        static SharedPtr createFromD3D12Handle(D3D12ResourceHandle handle, size_t size, Resource::BindFlags bindFlags, CpuAccess cpuAccess);

        /** Get a shader-resource view.
            \param[in] firstElement The first element of the view. For raw buffers, an element is a single float
            \param[in] elementCount The number of elements to bind
        */
        ShaderResourceView::SharedPtr getSRV(uint32_t firstElement, uint32_t elementCount = kMaxPossible);

        /** Get an unordered access view.
            \param[in] firstElement The first element of the view. For raw buffers, an element is a single float
            \param[in] elementCount The number of elements to bind
        */
        UnorderedAccessView::SharedPtr getUAV(uint32_t firstElement, uint32_t elementCount = kMaxPossible);

        /** Get a shader-resource view for the entire resource
        */
        virtual ShaderResourceView::SharedPtr getSRV() override;

        /** Get an unordered access view for the entire resource
        */
        virtual UnorderedAccessView::SharedPtr getUAV() override;

#if FALCOR_HAS_CUDA
        /** Get the CUDA device address for this resource.
            \return CUDA device address.
            Throws an exception if the buffer is not shared.
        */
        virtual void* getCUDADeviceAddress() const override;

        /** Get the CUDA device address for a view of this resource.
        */
        virtual void* getCUDADeviceAddress(ResourceViewInfo const& viewInfo) const override;
#endif

        /** Get the size of each element in this buffer.

            For a typed buffer, this will be the size of the format.
            For a structured buffer, this will be the same value as `getStructSize()`.
            For a raw buffer, this will be the number of bytes.
        */
        uint32_t getElementSize() const;

        /** Get a constant buffer view
        */
        ConstantBufferView::SharedPtr getCBV();

        /** Update the buffer's data
            \param[in] pData Pointer to the source data.
            \param[in] offset Byte offset into the destination buffer, indicating where to start copy into.
            \param[in] size Number of bytes to copy.
            Throws an exception if data causes out-of-bound access to the buffer.
        */
        virtual void setBlob(const void* pData, size_t offset, size_t size);

        /** Get the offset from the beginning of the GPU resource
        */
        uint64_t getGpuAddressOffset() const { return mGpuVaOffset; };

        /** Get the GPU address (this includes the offset)
        */
        uint64_t getGpuAddress() const;

        /** Get the size of the buffer in bytes.
        */
        size_t getSize() const { return mSize; }

        /** Get the element count. For structured-buffers, this is the number of structs. For typed-buffers, this is the number of elements. For other buffer, will return the size in bytes.
        */
        uint32_t getElementCount() const { return mElementCount; }

        /** Get the size of a single struct. This call is only valid for structured-buffer. For other buffer types, it will return 0
        */
        uint32_t getStructSize() const { return mStructSize; }

        /** Get the buffer format. This call is only valid for typed-buffers, for other buffer types it will return ResourceFormat::Unknown
        */
        ResourceFormat getFormat() const { return mFormat; }

        /** Get the UAV counter buffer
        */
        const Buffer::SharedPtr& getUAVCounter() const { return mpUAVCounter; }

        /** Map the buffer.

            The low-level behavior depends on MapType and the CpuAccess flags of the buffer:
            - For CPU accessible buffers, the caller should ensure CPU/GPU memory accesses do not conflict.
            - For GPU-only buffers, map for read will create an internal staging buffer that is safe to read.
            - Mapping a CPU write buffer for WriteDiscard will cause the buffer to be internally re-allocated,
              causing its address range to change and invalidating all previous views to the buffer.
        */
        void* map(MapType Type);

        /** Unmap the buffer
        */
        void unmap();

        /** Get safe offset and size values
        */
        bool adjustSizeOffsetParams(size_t& size, size_t& offset) const;

        /** Get the CPU access flags
        */
        CpuAccess getCpuAccess() const { return mCpuAccess; }

        /** Check if this is a typed buffer
        */
        bool isTyped() const { return mFormat != ResourceFormat::Unknown; }

        /** Check if this is a structured-buffer
        */
        bool isStructured() const { return mStructSize != 0; }

        template<typename T>
        void setElement(uint32_t index, T const& value)
        {
            setBlob(&value, sizeof(T)*index, sizeof(T));
        }

    protected:
        Buffer(size_t size, BindFlags bindFlags, CpuAccess cpuAccess);
        void apiInit(bool hasInitData);

        CpuAccess mCpuAccess;
        GpuMemoryHeap::Allocation mDynamicData;
        Buffer::SharedPtr mpStagingResource; // For buffers that have both CPU read flag and can be used by the GPU
        Resource::SharedPtr mpAliasedResource;
        uint32_t mElementCount = 0;
        ResourceFormat mFormat = ResourceFormat::Unknown;
        uint32_t mStructSize = 0;
        ConstantBufferView::SharedPtr mpCBV; // For constant-buffers
        Buffer::SharedPtr mpUAVCounter; // For structured-buffers

        mutable void* mCUDAExternalMemory = nullptr;
        mutable void* mCUDADeviceAddress = nullptr;

        /** Helper for converting host type to resource format for typed buffers.
            See list of supported formats for typed UAV loads:
            https://docs.microsoft.com/en-us/windows/win32/direct3d12/typed-unordered-access-view-loads
        */
        template<typename T>
        struct FormatForElementType {};

#define CASE(TYPE, FORMAT) \
        template<> struct FormatForElementType<TYPE> { static const ResourceFormat kFormat = FORMAT; }

        // Guaranteed supported formats on D3D12.
        CASE(float,     ResourceFormat::R32Float);
        CASE(uint32_t,  ResourceFormat::R32Uint);
        CASE(int32_t,   ResourceFormat::R32Int);

        // Optionally supported formats as a set on D3D12. If one is supported all are supported.
        CASE(float4,    ResourceFormat::RGBA32Float);
        CASE(uint4,     ResourceFormat::RGBA32Uint);
        CASE(int4,      ResourceFormat::RGBA32Int);
        //R16G16B16A16_FLOAT
        //R16G16B16A16_UINT
        //R16G16B16A16_SINT
        //R8G8B8A8_UNORM
        //R8G8B8A8_UINT
        //R8G8B8A8_SINT
        //R16_FLOAT
        CASE(uint16_t,  ResourceFormat::R16Uint);
        CASE(int16_t,   ResourceFormat::R16Int);
        //R8_UNORM
        CASE(uint8_t,   ResourceFormat::R8Uint);
        CASE(int8_t,    ResourceFormat::R8Int);

        // Optionally and individually supported formats on D3D12. Query for support individually.
        //R16G16B16A16_UNORM
        //R16G16B16A16_SNORM
        CASE(float2,    ResourceFormat::RG32Float);
        CASE(uint2,     ResourceFormat::RG32Uint);
        CASE(int2,      ResourceFormat::RG32Int);
        //R10G10B10A2_UNORM
        //R10G10B10A2_UINT
        //R11G11B10_FLOAT
        //R8G8B8A8_SNORM
        //R16G16_FLOAT
        //R16G16_UNORM
        //R16G16_UINT
        //R16G16_SNORM
        //R16G16_SINT
        //R8G8_UNORM
        //R8G8_UINT
        //R8G8_SNORM
        //8G8_SINT
        //R16_UNORM
        //R16_SNORM
        //R8_SNORM
        //A8_UNORM
        //B5G6R5_UNORM
        //B5G5R5A1_UNORM
        //B4G4R4A4_UNORM

        // Additional formats that may be supported on some hardware.
        CASE(float3,    ResourceFormat::RGB32Float);

#undef CASE
    };

    inline std::string to_string(Buffer::CpuAccess c)
    {
#define a2s(ca_) case Buffer::CpuAccess::ca_: return #ca_;
        switch (c)
        {
            a2s(None);
            a2s(Write);
            a2s(Read);
        default:
            FALCOR_UNREACHABLE();
            return "";
        }
#undef a2s
    }

    inline std::string to_string(Buffer::MapType mt)
    {
#define t2s(t_) case Buffer::MapType::t_: return #t_;
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
}
