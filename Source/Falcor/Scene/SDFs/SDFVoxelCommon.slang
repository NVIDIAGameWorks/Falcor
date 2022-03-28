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
#include "Utils/Math/MathConstants.slangh"

import Utils.Math.FormatConversion;

struct SDFVoxelCommon
{
    // This is the maximum level of an SDF grid. A grid cannot have a width larger than 2^kMaxLevel.
    static const uint kMaxLevel = 19;

    static const uint kLocationCodeVoxelCoordMask = (1 << kMaxLevel) - 1;
    static const uint64_t kLocationCodeVoxelCoordsMask = (1ull << (3 * kMaxLevel)) - 1;

    static const uint kLocationCodeLevelOffset = 3 * kMaxLevel;
    static const uint kLocationCodeLevelMask = 0x1f; // Equal to ceilPow2(kMaxLevel) - 1

    /** Encodes the local coordinates and level into a 64-bit location code.
        The location code is encoded as | 2 unused bits | 5 bits encoding the level | 57 bits for coordinates |.
        Note that the level is not the same as LOD, a level always represents a grid with width 2^level.
        \param[in] levelLocalVoxelCoords Voxel coordinates local to level.
        \param[in] level Which level the local coordinates are local to.
        \return A 64-bit location code stored in two 32-bit uints.
    */
    static uint2 encodeLocation(const uint3 levelLocalVoxelCoords, const uint level)
    {
        uint3 globalCoords = levelLocalToGlobalCoords(levelLocalVoxelCoords, level);
        uint64_t shiftedX = shiftCoord(globalCoords.x);
        uint64_t shiftedY = shiftCoord(globalCoords.y);
        uint64_t shiftedZ = shiftCoord(globalCoords.z);

        uint64_t locationCode = (uint64_t(level) << kLocationCodeLevelOffset) | (((shiftedX << 2) | (shiftedY << 1) | shiftedZ) & kLocationCodeVoxelCoordsMask);
        return u64to2x32(locationCode);
    }

    /** Decodes a 64-bit location code into local coordinates and a level.
        Note that the level is not the same as LOD, a level always represents a grid with width 2^level.
        \param[in] locationCode 64-bit location code.
        \param[out] levelLocalVoxelCoords Voxel coordinates local to level.
        \param[out] level Which level the local coordinates are local to.
    */
    static void decodeLocation(const uint2 locationCode, out uint3 levelLocalVoxelCoords, out uint level)
    {
        level = ((locationCode.y >> (kLocationCodeLevelOffset - 32)) & kLocationCodeLevelMask);

        uint64_t coordBits = u2x32to64(locationCode) & kLocationCodeVoxelCoordsMask;
        uint3 globalCoords = uint3(unshiftCoord(coordBits >> 2), unshiftCoord(coordBits >> 1), unshiftCoord(coordBits));
        levelLocalVoxelCoords = globalToLevelLocalCoords(globalCoords, level);
    }

    /** Creates the location code that corresponds to any of the eight children of a voxel encoded as a location code.
        \param[in] locationCode The location code of the parent voxel.
        \param[in] childID A 3-bit ID of the child for which a location code should be created, must be in the interval [0, 7] where the bit representation of childID can be interpreted as the local xyz offsets of the child.
        \param[out] childLocationCode The location code of the child.
        \return True if the child location code is valid, false otherwise.
    */
    static bool createChildLocationCode(uint2 locationCode, uint childID, out uint2 childLocationCode)
    {
        uint level = 1 + ((locationCode.y >> (kLocationCodeLevelOffset - 32)) & kLocationCodeLevelMask);
        bool valid = level < kMaxLevel;
        level = min(level, kMaxLevel);

        uint64_t bits = u2x32to64(locationCode);
        bits |= uint64_t(childID & 0x7) << (kLocationCodeLevelOffset - 3 * level);
        bits &= kLocationCodeVoxelCoordsMask;
        bits |= (uint64_t(level) << kLocationCodeLevelOffset);

        childLocationCode = u64to2x32(bits);
        return valid;
    }


    /** Shifts and dilates the low 21 bits of the value to make room for interleaving two other values of 21 bits each in the 64-bit return value.
        \param[in] x The value that should be shifted, only the low 21 bits will be kept.
        \return A 64 bit value with the highest bit unused and every third bit after that containing a bit from x.
    */
    static uint64_t shiftCoord(const uint x)
    {
        uint64_t y = uint64_t(x);
        y = (y | y << 32) & 0x1f00000000ffffull;
        y = (y | y << 16) & 0x1f0000ff0000ffull;
        y = (y | y << 8) & 0x100f00f00f00f00full;
        y = (y | y << 4) & 0x10c30c30c30c30c3ull;
        y = (y | y << 2) & 0x1249249249249249ull;
        return y;
    }

    /** Unshifts and compacts every third bits of the value to decode the 21 bit value starting at second highest bit and encompassing every third bit after that.
        \param[in] x The value that should be unshifted.
        \return A 32 bit value with the low 21 bits containing the unshifted and compacted value.
    */
    static uint unshiftCoord(const uint64_t x)
    {
        uint64_t y = x & 0x9249249249249249ull;
        y = (y | (y >> 2)) & 0x30c30c30c30c30c3ull;
        y = (y | (y >> 4)) & 0xf00f00f00f00f00full;
        y = (y | (y >> 8)) & 0x00ff0000ff0000ffull;
        y = (y | (y >> 16)) & 0xffff00000000ffffull;
        y = (y | (y >> 32)) & 0x00000000ffffffffull;
        return uint(y);
    }

    /** Converts local voxel coordinates (i.e., voxel coordinates [0, 2^level - 1]) to global voxel coordinates (i.e., [0, 2^kMaxLevel - 1]).
        \param[in] levelLocalVoxelCoords Voxel coordinates local to level.
        \param[in] level Which level the local coordinates are local to.
        \return Global voxel coordinates.
    */
    static uint3 levelLocalToGlobalCoords(const uint3 levelLocalVoxelCoords, const uint level)
    {
        return (levelLocalVoxelCoords & kLocationCodeVoxelCoordMask) << (kMaxLevel - level);
    }

    /** Converts the global voxel coordinates (i.e., voxel coordinates [0, 2^kMaxLevel - 1]) to local voxel coordinates (i.e., [0, 2^level - 1]).
        \param[in] globalCoords Global voxel coordinates.
        \param[in] level Which level the local coordinates should be local to.
        \return Local voxel coordinates, located on level.
    */
    static uint3 globalToLevelLocalCoords(const uint3 globalCoords, uint level)
    {
        return (globalCoords >> (kMaxLevel - level));
    }


    /** Packs the eight distance values into 8 bytes using an snorm format.
    */
    static uint2 packValues(float4 values0xx, float4 values1xx)
    {
        uint2 packedValues;
        packedValues.x = packSnorm8(values0xx[0]);
        packedValues.x |= (packSnorm8(values0xx[1]) << 8);
        packedValues.x |= (packSnorm8(values0xx[2]) << 16);
        packedValues.x |= (packSnorm8(values0xx[3]) << 24);
        packedValues.y = packSnorm8(values1xx[0]);
        packedValues.y |= (packSnorm8(values1xx[1]) << 8);
        packedValues.y |= (packSnorm8(values1xx[2]) << 16);
        packedValues.y |= (packSnorm8(values1xx[3]) << 24);
        return packedValues;
    }

    /** Unpacks eight distance values from an 8 byte snorm format.
    */
    static void unpackValues(const uint2 packedValues, out float4 values0xx, out float4 values1xx)
    {
        values0xx[0] = unpackSnorm8(packedValues.x & 0x000000ff);
        values0xx[1] = unpackSnorm8((packedValues.x >> 8) & 0x000000ff);
        values0xx[2] = unpackSnorm8((packedValues.x >> 16) & 0x000000ff);
        values0xx[3] = unpackSnorm8((packedValues.x >> 24) & 0x000000ff);
        values1xx[0] = unpackSnorm8(packedValues.y & 0x000000ff);
        values1xx[1] = unpackSnorm8((packedValues.y >> 8) & 0x000000ff);
        values1xx[2] = unpackSnorm8((packedValues.y >> 16) & 0x000000ff);
        values1xx[3] = unpackSnorm8((packedValues.y >> 24) & 0x000000ff);
    }

    /** Checks if the voxel defined by the eight corner values conservatively contains part of the implicit surface.
    */
    static bool containsSurface(const float4 values0xx, const float4 values1xx)
    {
        return  (any(values0xx <= 0.0f) || any(values1xx <= 0.0f)) &&
                (any(values0xx >= 0.0f) || any(values1xx >= 0.0f));
    }

    /** Trilinearly interpolates the eight corner values of a voxel using voxel unit coords.
    */
    static float sdfVoxelTrilin(const float4 values0xx, const float4 values1xx, const float3 voxelUnitCoords)
    {
        float4 cXs = lerp(values0xx, values1xx, voxelUnitCoords.x);
        float2 cYs = lerp(cXs.xy, cXs.zw, voxelUnitCoords.y);
        return lerp(cYs.x, cYs.y, voxelUnitCoords.z);
    }
};
