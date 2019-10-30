/***************************************************************************
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
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
***************************************************************************/
#pragma once
#include "Resource.h"
#include "GpuMemoryHeap.h"

namespace Falcor
{
    /** Low-level buffer object
        This class abstracts the API's buffer creation and management
    */
    class dlldecl Buffer : public Resource, public inherit_shared_from_this<Resource, Buffer>
    {
    public:
        using SharedPtr = std::shared_ptr<Buffer>;
        using WeakPtr = std::weak_ptr<Buffer>;
        using SharedConstPtr = std::shared_ptr<const Buffer>;
        using inherit_shared_from_this<Resource, Buffer>::shared_from_this;

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
            Read,           ///< Map the buffer for read access. Buffer had to be created with AccessFlags#MapWrite flag.
            WriteDiscard,   ///< Map the buffer for write access, discarding the previous content of the entire buffer. Buffer had to be created with AccessFlags#MapWrite flag.
        };
        
        ~Buffer();

        /** Create a new buffer
            \param[in] size Size of the buffer in bytes
            \param[in] bind Buffer bind flags
            \param[in] cpuAccess Flags indicating how the buffer can be updated
            \param[in] pInitData Optional parameter. Initial buffer data. Pointed buffer size should be at least 'size' bytes.
            \return A pointer to a new buffer object, or nullptr if creation failed.
        */
        static SharedPtr create(size_t size, Resource::BindFlags bind, CpuAccess cpuAccess, const void* pInitData = nullptr);
        static SharedPtr aliasResource(Resource::SharedPtr pBaseResource, GpuAddress offset, size_t size, Resource::BindFlags bindFlags);

        /** Create a new buffer from an existing API handle
            \param[in] ApiHandle Handle of already-allocated resource
            \param[in] size The size of the buffer in bytes.
            \param[in] bind Buffer bind flags. Flags must match the bind flags of the original resource.
            \param[in] cpuAccess Flags indicating how the buffer can be updated. Flags must match those of the heap the original resource is allocated on.
            \return A pointer to a new buffer object, or nullptr if creation failed.
        */
        static SharedPtr createFromApiHandle(ApiHandle handle, size_t size, Resource::BindFlags bind, CpuAccess cpuAccess);

        /** Get a shader-resource view.
            \param[in] firstElement The first element of the view. For raw buffers, an element is a single float
            \param[in] elementCount The number of elements to bind
        */
        ShaderResourceView::SharedPtr getSRV(uint32_t firstElement = 0, uint32_t elementCount = kMaxPossible);

        /** Get an unordered access view.
            \param[in] firstElement The first element of the view. For raw buffers, an element is a single float
            \param[in] elementCount The number of elements to bind
        */
        UnorderedAccessView::SharedPtr getUAV(uint32_t firstElement = 0, uint32_t elementCount = kMaxPossible);

        /** Update the buffer's data
            \param[in] pData Pointer to the source data.
            \param[in] offset Byte offset into the destination buffer, indicating where to start copy into.
            \param[in] size Number of bytes to copy.
            If offset and size will cause an out-of-bound access to the buffer, an error will be logged and the update will fail.
        */
        virtual bool setBlob(const void* pData, size_t offset, size_t size);

        deprecate("4.0", "Use setBlob() instead")
        void updateData(const void* pData, size_t offset, size_t size);

        /** Get the offset from the beginning of the GPU resource
        */
        uint64_t getGpuAddressOffset() const { return mGpuVaOffset; };

        /** Get the GPU address (this includes the offset)
        */
        uint64_t getGpuAddress() const;

        /** Get the size of the buffer
        */
        size_t getSize() const { return mSize; }

        /** Map the buffer
        */
        void* map(MapType Type);

        /** Unmap the buffer
        */
        void unmap();

        /** Get safe offset and size values
        */
        bool adjustSizeOffsetParams(size_t& size, size_t& offset) const
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

        /** Get the CPU access flags
        */
        CpuAccess getCpuAccess() const { return mCpuAccess; }

    protected:
        bool apiInit(bool hasInitData);
        Buffer(size_t size, BindFlags bind, CpuAccess update) : Resource(Type::Buffer, bind, size), mCpuAccess(update){}

        CpuAccess mCpuAccess;
        GpuMemoryHeap::Allocation mDynamicData;
        Buffer::SharedPtr mpStagingResource; // For buffers that have both CPU read flag and can be used by the GPU
        Resource::SharedPtr mpAliasedResource;
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
            should_not_get_here();
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
            t2s(WriteDiscard);
        default:
            should_not_get_here();
            return "";
        }
#undef t2s
    }
}
