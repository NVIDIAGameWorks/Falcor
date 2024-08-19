/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
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
#include "Utils/HostDeviceShared.slangh"
import Utils.Attributes;

struct S32B
{
    float4 f0, f1;
};

struct S4B
{
    float f;
};

#ifdef USE_4B_SIZE
typedef S4B TElementType;
#else
typedef S32B TElementType;
#endif

#ifndef SPLIT_BYTE_BUFFER_BUFFER_COUNT
#define SPLIT_BYTE_BUFFER_BUFFER_COUNT 1 // here for the benefit of the IntelliSense
#define SPLIT_BYTE_BUFFER_BUFFER_INDEX_BITS 1
#endif // SPLIT_BYTE_BUFFER_BUFFER_COUNT

struct SplitByteBuffer
{
    typedef uint ElementType;
    static constexpr uint kBufferIndexBits = SPLIT_BYTE_BUFFER_BUFFER_INDEX_BITS;
    static constexpr uint kBufferIndexOffset = 32 - kBufferIndexBits;
    static constexpr uint kElementAddressMask = (1u << kBufferIndexOffset) - 1;
    static constexpr uint kBufferCount = SPLIT_BYTE_BUFFER_BUFFER_COUNT;

#if SPLIT_BYTE_BUFFER_BUFFER_COUNT > 0
    ByteAddressBuffer data[kBufferCount];

    /// baseOffset - offset to the start of the mesh
    /// byteIndex - bytes to read from offset
    /// Returns bytes starting baseOffset*4 + byteIndex, handling the uint overflow
    /// for large buffers.
    /// expected usage is LoadUint16_t3(baseOffset, baseOffset*4 + index)
    uint16_t3 Load16b(uint baseOffset, uint triangleIndex)
    {
        if (kBufferCount == 1)
            return data[0].Load<uint16_t3>(baseOffset * 4 + triangleIndex * 6);
        uint bufferIndex = baseOffset >> kBufferIndexOffset;
        uint byteOffset = (baseOffset & kElementAddressMask) * 4;
        return data[bufferIndex].Load<uint16_t3>(byteOffset + triangleIndex * 6);
    }

    uint3 Load32b(uint baseOffset, uint triangleIndex)
    {
        if (kBufferCount == 1)
            return data[0].Load3(baseOffset * 4 + triangleIndex * 12);
        uint bufferIndex = baseOffset >> kBufferIndexOffset;
        uint byteOffset = (baseOffset & kElementAddressMask) * 4;
        return data[bufferIndex].Load3(byteOffset + triangleIndex * 12);
    }
#endif // SCENE_INDEX_BUFFER_COUNT > 0
};

#ifndef SPLIT_STRUCT_BUFFER_BUFFER_COUNT
#define SPLIT_STRUCT_BUFFER_BUFFER_COUNT 1 // here for the benefit of the IntelliSense
#define SPLIT_STRUCT_BUFFER_BUFFER_INDEX_BITS 1
#endif // SPLIT_STRUCT_BUFFER_BUFFER_COUNT

struct SplitStructBuffer
{
    typedef TElementType ElementType;
    static constexpr uint kBufferIndexBits = SPLIT_STRUCT_BUFFER_BUFFER_INDEX_BITS;
    static constexpr uint kBufferIndexOffset = 32 - kBufferIndexBits;
    static constexpr uint kElementIndexMask = (1u << kBufferIndexOffset) - 1;
    static constexpr uint kBufferCount = SPLIT_STRUCT_BUFFER_BUFFER_COUNT;

    StructuredBuffer<ElementType> data[kBufferCount];

    __subscript(uint index)->ElementType
    {
        get {
            if (kBufferCount == 1)
                return data[0][index];
            uint bufferIndex = index >> kBufferIndexOffset;
            uint elementIndex = index & kElementIndexMask;
            return data[bufferIndex][elementIndex];
        }
    }
};

RWStructuredBuffer<uint16_t> gByteBufferUint16_t3[SPLIT_BYTE_BUFFER_BUFFER_COUNT];
RWStructuredBuffer<uint32_t> gByteBufferUint32_t3[SPLIT_BYTE_BUFFER_BUFFER_COUNT];
SplitByteBuffer gSplitByteBuffer;

RWStructuredBuffer<TElementType> gStructuredBuffer[SPLIT_STRUCT_BUFFER_BUFFER_COUNT];
SplitStructBuffer gSplitStructBuffer;

struct RangeDesc
{
    // The tested concept
    uint offset;
    uint count;
    // The debugging info
    uint bufferIndex;
    uint bufferOffset;
};

uniform uint gRangeCount;
StructuredBuffer<RangeDesc> gRangeDescs;

[numthreads(256, 1, 1)]
void testSplitByteBuffer48b(uint3 threadId: SV_DispatchThreadID)
{
    if (threadId.x >= gRangeCount)
        return;

    RangeDesc desc = gRangeDescs[threadId.x];
    for (uint i = 0; i < desc.count; ++i)
    {
        uint16_t3 item = gSplitByteBuffer.Load16b(desc.offset, i);
        gByteBufferUint16_t3[desc.bufferIndex][desc.bufferOffset * 2 + i * 3 + 0] = item[0];
        gByteBufferUint16_t3[desc.bufferIndex][desc.bufferOffset * 2 + i * 3 + 1] = item[1];
        gByteBufferUint16_t3[desc.bufferIndex][desc.bufferOffset * 2 + i * 3 + 2] = item[2];
    }
}

[numthreads(256, 1, 1)]
void testSplitByteBuffer96b(uint3 threadId: SV_DispatchThreadID)
{
    if (threadId.x >= gRangeCount)
        return;

    RangeDesc desc = gRangeDescs[threadId.x];
    for (uint i = 0; i < desc.count; ++i)
    {
        uint32_t3 item = gSplitByteBuffer.Load32b(desc.offset, i);
        gByteBufferUint32_t3[desc.bufferIndex][desc.bufferOffset + i * 3 + 0] = item[0];
        gByteBufferUint32_t3[desc.bufferIndex][desc.bufferOffset + i * 3 + 1] = item[1];
        gByteBufferUint32_t3[desc.bufferIndex][desc.bufferOffset + i * 3 + 2] = item[2];
    }
}

[numthreads(256, 1, 1)]
void testSplitStructuredBuffer(uint3 threadId: SV_DispatchThreadID)
{
    if (threadId.x >= gRangeCount)
        return;

    RangeDesc desc = gRangeDescs[threadId.x];
    for (uint i = 0; i < desc.count; ++i)
    {
        TElementType item = gSplitStructBuffer[desc.offset + i];
        gStructuredBuffer[desc.bufferIndex][desc.bufferOffset + i] = item;
    }
}
