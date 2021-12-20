/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
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
#include "Utils/NVAPI.slangh" // We need this to get interlocked compare exchange for uint64.

import Scene.SDFs.SDFVoxelCommon;
import Utils.Math.HashUtils;

/** SDF voxel hash table for writing and reading.
    Voxels are stored as:

    struct Voxel
    {
        uint64 locationCode;    ///< The voxel location code.
        uint32 svoOffset;       ///< The offset of this voxel in the SVO.
        uint32 validMask;       ///< The valid mask of the voxel, encoding what children are valid, padded so voxel is multiple of 4 bytes in size.
        uint8 packedValues[8];  ///< The eight corner values of the SDFVoxel, packed as 8-bit snorms.
    };

*/
struct SDFSVOHashTable
{
    static const uint2 kEmptyLocationCode = uint2(0);

    static const uint kVoxelByteSize = 24;
    static const uint kSVOOffsetOffset = 8;
    static const uint kValuesOffset = 16;

    RWByteAddressBuffer buffer;                 ///< The buffer storing all voxels.
    uint capacity;                              ///< The capacity of the table, size in number of entries of "voxelsBuffer", must be power of 2.

    static void write(RWByteAddressBuffer buffer, uint byteAddress, uint2 locationCode, uint validMask, uint2 packedValues)
    {
        uint4 data = uint4(locationCode.x, locationCode.y, 0, validMask & 0xff);
        buffer.Store4(byteAddress, data);
        buffer.Store2(byteAddress + kValuesOffset, packedValues);
    }

    static void writeValues(RWByteAddressBuffer buffer, uint byteAddress, uint validMask, uint2 packedValues)
    {
        uint4 data = uint4(0, validMask & 0xff, packedValues.x, packedValues.y);
        buffer.Store4(byteAddress + kSVOOffsetOffset, data);
    }

    static void writeSVOOffset(RWByteAddressBuffer buffer, uint byteAddress, uint svoOffset)
    {
        buffer.Store(byteAddress + kSVOOffsetOffset, svoOffset);
    }

    static uint2 readLocationCode(RWByteAddressBuffer buffer, uint byteAddress)
    {
        return buffer.Load2(byteAddress);
    }

    static void readValues(RWByteAddressBuffer buffer, uint byteAddress, out uint svoOffset, out uint validMask, out uint2 packedValues)
    {
        uint4 data = buffer.Load4(byteAddress + kSVOOffsetOffset);
        svoOffset = data.x;
        validMask = data.y & 0xff;
        packedValues = data.zw;
    }

    /** Hashes the location code to a 32-bit uint.
    */
    static uint hashLocationCode(uint2 locationCode)
    {
        uint hashX = jenkinsHash(locationCode.x);
        uint hashY = jenkinsHash(locationCode.y);
        return hashY + 0x9e3779b9 + (hashX << 6) + (hashX >> 2);
    }

    /** Insert a voxel into the hash table.
        \param[in] locationCode The location code of the voxel.
        \param[in] validMask The SDF corner values to be inserted.
        \param[in] values The SDF corner values.
        \return true if the insertion succeded, false otherwise. A return value of false means that the table is full.
    */
    bool insert(uint2 locationCode, uint validMask, float4 values0xx, float4 values1xx, out uint slot)
    {
        uint hash = hashLocationCode(locationCode);
        slot = (hash & (capacity - 1));

        for (uint i = 0; i < capacity; i++)
        {
            uint byteAddress = slot * kVoxelByteSize;

            uint2 origValue = NvInterlockedCompareExchangeUint64(buffer, byteAddress, kEmptyLocationCode, locationCode);
            if ((origValue.y >> 31) == 0)
            {
                uint2 packedValues = SDFVoxelCommon::packValues(values0xx, values1xx);
                write(buffer, byteAddress, locationCode, validMask, packedValues);
                return true;
            }
            else if (all(locationCode == origValue))
            {
                uint2 packedValues = SDFVoxelCommon::packValues(values0xx, values1xx);
                writeValues(buffer, byteAddress, validMask, packedValues);
                return true;
            }

            slot = (slot + 1) & (capacity - 1);
        }

        return false;
    }

    /** Set the SVO offset of a voxel in the hash table.
        \param[in] locationCode The location code of the voxel.
        \param[in] svoOffset The address of the voxel in the final SVO.
        \return true if the insertion succeded, false otherwise. A return value of false means that the table is full.
    */
    bool setSVOOffset(uint2 locationCode, uint svoOffset)
    {
        uint hash = hashLocationCode(locationCode);
        uint slot = (hash & (capacity - 1));

        for (uint i = 0; i < capacity; i++)
        {
            uint byteAddress = slot * kVoxelByteSize;
            uint2 slottedLocationCode = readLocationCode(buffer, byteAddress);

            if (all(slottedLocationCode == locationCode))
            {
                writeSVOOffset(buffer, byteAddress, svoOffset);
                return true;
            }

            if ((slottedLocationCode.y >> 31) == 0)
            {
                return false;
            }

            slot = (slot + 1) & (capacity - 1);
        }

        return false;
    }

    /** Lookup a voxel from the hash table.
        \param[in] locationCode The location code of the voxel.
        \param[out] svoOffset The address of the voxel in the final SVO.
        \param[out] validMask The SDF corner values to be inserted.
        \param[out] packedValues The packed SDF corner values.
        \return true if the lookup succeded, false otherwise.
    */
    bool lookup(uint2 locationCode, out uint svoOffset, out uint validMask, out uint2 packedValues)
    {
        uint hash = hashLocationCode(locationCode);
        uint slot = (hash & (capacity - 1));

        for (uint i = 0; i < capacity; i++)
        {
            uint byteAddress = slot * kVoxelByteSize;
            uint2 slottedLocationCode = readLocationCode(buffer, byteAddress);

            if (all(slottedLocationCode == locationCode))
            {
                readValues(buffer, byteAddress, svoOffset, validMask, packedValues);
                return true;
            }

            if ((slottedLocationCode.y >> 31) == 0)
            {
                return false;
            }

            slot = (slot + 1) & (capacity - 1);
        }

        return false;
    }

    /** Check if a voxel is contained in the hash table without retrieving it.
        \param[in] locationCode The location code of the voxel.
        \return true if the voxel is contained in the hash table, false otherwise.
    */
    bool contains(uint2 locationCode)
    {
        uint hash = hashLocationCode(locationCode);
        uint slot = (hash & (capacity - 1));

        for (uint i = 0; i < capacity; i++)
        {
            uint byteAddress = slot * kVoxelByteSize;
            uint2 slottedLocationCode = readLocationCode(buffer, byteAddress);

            if (all(slottedLocationCode == locationCode))
            {
                return true;
            }

            if ((slottedLocationCode.y >> 31) == 0)
            {
                return false;
            }

            slot = (slot + 1) & (capacity - 1);
        }

        return false;
    }
};

