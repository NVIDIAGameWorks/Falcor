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
#include "BrickedGrid.h"
#include "BC4Encode.h"
#include "Core/API/Formats.h"
#include "Utils/Logger.h"
#include "Utils/HostDeviceShared.slangh"
#include "Utils/NumericRange.h"
#include "Utils/Math/Vector.h"
#include "Utils/Timing/CpuTimer.h"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4244 4267)
#endif
#include <nanovdb/NanoVDB.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <algorithm>
#include <atomic>
#include <execution>
#include <vector>

namespace Falcor
{
    template <typename TexelType, unsigned int kBitsPerTexel> struct NanoVDBToBricksConverter;
    using NanoVDBConverterBC4 = NanoVDBToBricksConverter<uint64_t, 4>;
    using NanoVDBConverterUNORM8 = NanoVDBToBricksConverter<uint8_t, 8>;
    using NanoVDBConverterUNORM16 = NanoVDBToBricksConverter<uint16_t, 16>;

    template <typename TexelType, unsigned int kBitsPerTexel>
    struct NanoVDBToBricksConverter
    {
    public:
        NanoVDBToBricksConverter(const nanovdb::FloatGrid* grid);
        NanoVDBToBricksConverter(const NanoVDBToBricksConverter& rhs) = delete;

        BrickedGrid convert();

    private:
        const static uint32_t kBrickSize = 8; // Must be 8, to match both NanoVDB leaf size.
        const static int32_t kBC4Compress = kBitsPerTexel == 4;

        void convertSlice(int z);
        void computeMip(int mip);

        inline uint3 getAtlasSizeBricks() const { return mAtlasSizeBricks; }
        inline uint3 getAtlasSizePixels() const { return mAtlasSizeBricks * kBrickSize; }
        inline uint32_t getAtlasMaxBrick() const { return mAtlasSizeBricks.x * mAtlasSizeBricks.y * mAtlasSizeBricks.z; }

        inline ResourceFormat getAtlasFormat() {
            switch (kBitsPerTexel) {
            case 4: return ResourceFormat::BC4Unorm;
            case 8: return ResourceFormat::R8Unorm;
            case 16: return ResourceFormat::R16Unorm;
            default: throw RuntimeError("Unsupported bitdepth in NanoVDBToBricksConverter");
            }
        }

        inline float2 combineMajMin(float2 a, float2 b)
        {
            return float2(std::max(a.x, b.x), std::min(a.y, b.y));
        }

        inline float2 unpackMajMin(const uint32_t* data)
        {
            const uint16_t* data16 = (const uint16_t*)data;
            return float2(f16tof32(data16[0]), f16tof32(data16[1]));
        }

        inline void expandMinorantMajorant(float value, float& min_inout, float& maj_inout)
        {
            if (value < min_inout) min_inout = value;
            if (value > maj_inout) maj_inout = value;
        }

        const nanovdb::FloatGrid* mpFloatGrid;
        uint3 mAtlasSizeBricks;
        int3 mLeafDim[4];
        int3 mBBMin, mBBMax, mPixDim;
        uint32_t mLeafCount[4];
        std::vector<uint32_t> mRangeData;
        std::vector<uint32_t> mPtrData;
        std::vector<TexelType> mAtlasData;
        std::atomic_uint32_t mNonEmptyCount;
    };

    template <typename TexelType, unsigned int kBitsPerTexel>
    NanoVDBToBricksConverter<TexelType, kBitsPerTexel>::NanoVDBToBricksConverter(const nanovdb::FloatGrid* grid)
    {
        mNonEmptyCount.store(0);
        mpFloatGrid = grid;
        auto& voxelbox = mpFloatGrid->indexBBox();
        mBBMin = (int3(voxelbox.min().x(), voxelbox.min().y(), voxelbox.min().z())) & (~7);
        mBBMax = (int3(voxelbox.max().x(), voxelbox.max().y(), voxelbox.max().z()) + 7) & (~7);
        mPixDim = mBBMax - mBBMin;
        mPixDim = (mPixDim + 63) & ~63; // Since we compute 4 mipmaps, the coarsest mip corresponds to 8*2^3 = 64 leaf voxels.
        for (uint i = 0; i < 4; ++i)
        {
            mLeafDim[i] = mPixDim / (8 << i);
            mLeafCount[i] = (mLeafDim[i].x * mLeafDim[i].y * mLeafDim[i].z) + (i ? mLeafCount[i - 1] : 0); // Cumulative leaf count up the mips.
        }
        uint leafCount = grid->tree().nodeCount(0);
        uint approxdim = 1u << uint(log2f((float)leafCount + 1.f) / 3.f); // Choose the first 2 dimensions to be powers of 2.
        uint lastdim = (leafCount + approxdim * approxdim - 1) / (approxdim * approxdim);
        mAtlasSizeBricks = uint3(approxdim, approxdim, lastdim);
        uint3 atlasSizePixels = getAtlasSizePixels();
        uint leafTexelCount = atlasSizePixels.x * atlasSizePixels.y * atlasSizePixels.z;
        mRangeData.resize(mLeafCount[3]);
        mPtrData.resize(mLeafCount[0]);
        mAtlasData.resize(kBC4Compress ? (leafTexelCount / 16) : leafTexelCount);
    }

    template <typename TexelType, unsigned int kBitsPerTexel>
    void NanoVDBToBricksConverter<TexelType, kBitsPerTexel>::convertSlice(int z)
    {
        uint3 atlasSizePixels = getAtlasSizePixels();
        uint brickMax = getAtlasMaxBrick();
        uint bricksPerSlice = mAtlasSizeBricks.x * mAtlasSizeBricks.y;
        uint pixelsPerSlice = atlasSizePixels.x * atlasSizePixels.y;

        size_t offset = z * mLeafDim[0].x * mLeafDim[0].y;
        uint32_t* rangedst = mRangeData.data() + offset;
        uint32_t* ptrdst = mPtrData.data() + offset;
        auto a = mpFloatGrid->getAccessor();
        for (int y = 0; y < mLeafDim[0].y; ++y)
        {
            for (int x = 0; x < mLeafDim[0].x; ++x)
            {
                nanovdb::Coord ijk = { x * 8 + mBBMin.x, y * 8 + mBBMin.y, z * 8 + mBBMin.z };
                auto val = a.getValue(ijk);
                auto leaf = a.probeLeaf(ijk);
                float minorant = val, majorant = val;
                uint myleaf = 0;
                if (leaf)
                {
                    // Nanovdb only stores minorant/majorant for active voxels, but we need all of them... Grab the central 8x8x8 first the quick way.
                    const float* data = leaf->data()->mValues;
                    for (int i = 0; i < kBrickSize * kBrickSize * kBrickSize; ++i) expandMinorantMajorant(data[i], minorant, majorant);
                    // We also need the 1-halo from neighbouring bricks. Fetch them in an order that maximises nanovdb's internal cache reuse.
                    for (int j = -1; j <= kBrickSize; ++j) for (int i = 0; i < kBrickSize; ++i) expandMinorantMajorant(a.getValue(ijk + nanovdb::Coord(i, j, -1)), minorant, majorant);
                    for (int j = -1; j <= kBrickSize; ++j) for (int i = 0; i < kBrickSize; ++i) expandMinorantMajorant(a.getValue(ijk + nanovdb::Coord(i, j, kBrickSize)), minorant, majorant);
                    for (int j = 0; j < kBrickSize; ++j) for (int i = 0; i < kBrickSize; ++i) expandMinorantMajorant(a.getValue(ijk + nanovdb::Coord(i, -1, j)), minorant, majorant);
                    for (int j = 0; j < kBrickSize; ++j) for (int i = 0; i < kBrickSize; ++i) expandMinorantMajorant(a.getValue(ijk + nanovdb::Coord(i, kBrickSize, j)), minorant, majorant);
                    for (int j = -1; j <= kBrickSize; ++j) for (int i = 0; i < kBrickSize; ++i) expandMinorantMajorant(a.getValue(ijk + nanovdb::Coord(-1, j, i)), minorant, majorant);
                    for (int j = -1; j <= kBrickSize; ++j) for (int i = 0; i < kBrickSize; ++i) expandMinorantMajorant(a.getValue(ijk + nanovdb::Coord(kBrickSize, j, i)), minorant, majorant);
                    for (int j = -1; j <= kBrickSize; ++j) expandMinorantMajorant(a.getValue(ijk + nanovdb::Coord(-1, j, -1)), minorant, majorant);
                    for (int j = -1; j <= kBrickSize; ++j) expandMinorantMajorant(a.getValue(ijk + nanovdb::Coord(kBrickSize, j, -1)), minorant, majorant);
                    for (int j = -1; j <= kBrickSize; ++j) expandMinorantMajorant(a.getValue(ijk + nanovdb::Coord(-1, j, kBrickSize)), minorant, majorant);
                    for (int j = -1; j <= kBrickSize; ++j) expandMinorantMajorant(a.getValue(ijk + nanovdb::Coord(kBrickSize, j, kBrickSize)), minorant, majorant);

                    if (minorant != majorant) myleaf = mNonEmptyCount.fetch_add(1);
                }
                if (majorant == minorant || myleaf >= brickMax || leaf == nullptr)
                {
                    *rangedst++ = f32tof16(majorant) + (f32tof16(majorant) << 16); // force identical major and minor
                    *ptrdst++ = 0;
                }
                else
                {
                    const float* data = leaf->data()->mValues;
                    majorant = f16tof32(f32tof16(majorant) + 1);
                    minorant = f16tof32(f32tof16(minorant));
                    *rangedst++ = f32tof16(majorant) + (f32tof16(minorant) << 16);
                    uint32_t atlasx = myleaf % mAtlasSizeBricks.x;
                    uint32_t atlasy = (myleaf / mAtlasSizeBricks.x) % mAtlasSizeBricks.y;
                    uint32_t atlasz = myleaf / bricksPerSlice;
                    *ptrdst++ = (atlasx + (atlasy << 8) + (atlasz << 16));

                    if (!kBC4Compress) {
                        float invRange = ((1 << kBitsPerTexel) - 1.f) / (majorant - minorant);
                        TexelType* atlasdst = (TexelType*)mAtlasData.data() + atlasx * kBrickSize + atlasy * (atlasSizePixels.x * kBrickSize) + atlasz * (pixelsPerSlice * kBrickSize);
                        for (int pixz = 0; pixz < kBrickSize; ++pixz)
                        {
                            for (int pixy = 0; pixy < kBrickSize; ++pixy)
                            {
                                for (int pixx = 0; pixx < kBrickSize; ++pixx)
                                {
                                    float f = data[pixx * kBrickSize * kBrickSize + pixy * kBrickSize + pixz];
                                    *atlasdst++ = TexelType((f - minorant) * invRange);
                                }
                                atlasdst += (atlasSizePixels.x - kBrickSize); // next scanline
                            }
                            atlasdst += (pixelsPerSlice - (atlasSizePixels.x * kBrickSize)); // next slice
                        }
                    }
                    else {
                        // BC4 compression:
                        float invRange = (255.f) / (majorant - minorant);
                        uint64_t* atlasdst = ((uint64_t*)mAtlasData.data() + atlasx * (kBrickSize / 4) + atlasy * ((atlasSizePixels.x / 4) * kBrickSize / 4) + atlasz * (pixelsPerSlice / 16 * kBrickSize));
                        for (int pixz = 0; pixz < kBrickSize; ++pixz)
                        {
                            for (int tiley = 0; tiley < kBrickSize; tiley += 4)
                            {
                                for (int tilex = 0; tilex < kBrickSize; tilex += 4) {
                                    uint8_t tilevals[4][4];
                                    uint8_t tileminorant = 255, tilemajorant = 0;
                                    for (int pixy = 0; pixy < 4; ++pixy)
                                    {
                                        for (int pixx = 0; pixx < 4; ++pixx)
                                        {
                                            float f = data[(pixx + tilex) * (kBrickSize * kBrickSize) + (pixy + tiley) * kBrickSize + pixz];
                                            uint8_t voxel = uint8_t((f - minorant) * invRange);
                                            tileminorant = std::min(tileminorant, voxel);
                                            tilemajorant = std::max(tilemajorant, voxel);
                                            tilevals[pixy][pixx] = voxel;
                                        }
                                    }
                                    CompressAlphaDxt5((uint8_t*)&tilevals[0][0], atlasdst);
                                    atlasdst++;
                                }
                                atlasdst += (atlasSizePixels.x / 4 - kBrickSize / 4); // next scanline
                            }
                            atlasdst += (pixelsPerSlice / 16 - (atlasSizePixels.x / 4 * kBrickSize / 4)); // next slice
                        } // z slice loop
                    } // bc4 compress?
                } // non empty brick?
            } // x brick loop
        } // y brick loop
    }

    template <typename TexelType, unsigned int kBitsPerTexel>
    void NanoVDBToBricksConverter<TexelType, kBitsPerTexel>::computeMip(int mip)
    {
        uint32_t* rangedst = mRangeData.data() + mLeafCount[mip - 1];
        uint32_t* rangesrc = mRangeData.data() + ((mip > 1) ? mLeafCount[mip - 2] : 0);
        int3 leafdim_src = mLeafDim[mip - 1];
        uint32_t rowstride_src = leafdim_src.x;
        uint32_t slicestride_src = leafdim_src.y * rowstride_src;

        int3 leafdim_tgt = mLeafDim[mip];
        uint32_t rowstride_tgt = leafdim_tgt.x;
        uint32_t slicestride_tgt = leafdim_tgt.y * rowstride_tgt;

        for (int z = 0; z < leafdim_tgt.z; ++z, rangesrc += slicestride_src)
        {
            for (int y = 0; y < leafdim_tgt.y; ++y, rangesrc += rowstride_src)
            {
                for (int x = 0; x < leafdim_tgt.x; ++x, rangesrc += 2)
                {
                    float2 majmin_dst = combineMajMin(
                        combineMajMin(
                            combineMajMin(unpackMajMin(rangesrc), unpackMajMin(rangesrc + 1)),
                            combineMajMin(unpackMajMin(rangesrc + rowstride_src), unpackMajMin(rangesrc + 1 + rowstride_src))
                        ),
                        combineMajMin(
                            combineMajMin(unpackMajMin(rangesrc + slicestride_src), unpackMajMin(rangesrc + slicestride_src + 1)),
                            combineMajMin(unpackMajMin(rangesrc + slicestride_src + rowstride_src), unpackMajMin(rangesrc + slicestride_src + 1 + rowstride_src))
                        )
                    );
                    *rangedst++ = f32tof16(majmin_dst.x) + (f32tof16(majmin_dst.y) << 16);
                } // x
            } // y
        } // z
    }

    template <typename TexelType, unsigned int kBitsPerTexel>
    BrickedGrid NanoVDBToBricksConverter<TexelType, kBitsPerTexel>::convert()
    {
        auto t0 = CpuTimer::getCurrentTimePoint();
        auto range = NumericRange<int>(0, mLeafDim[0].z);
        std::for_each(std::execution::par, range.begin(), range.end(), [&](int z) { convertSlice(z); });
        for (int mip = 1; mip < 4; ++mip) computeMip(mip);
        double dt = CpuTimer::calcDuration(t0, CpuTimer::getCurrentTimePoint());
        logInfo("converted in {}ms: mNonEmptyCount {} vs max {}", dt, mNonEmptyCount, getAtlasMaxBrick());

        BrickedGrid bricks;
        bricks.range = Texture::create3D(mLeafDim[0].x, mLeafDim[0].y, mLeafDim[0].z, ResourceFormat::RG16Float, 4, mRangeData.data(), ResourceBindFlags::ShaderResource, false);
        bricks.indirection = Texture::create3D(mLeafDim[0].x, mLeafDim[0].y, mLeafDim[0].z, ResourceFormat::RGBA8Uint, 1, mPtrData.data(), ResourceBindFlags::ShaderResource, false);
        bricks.atlas = Texture::create3D(getAtlasSizePixels().x, getAtlasSizePixels().y, getAtlasSizePixels().z, getAtlasFormat(), 1, mAtlasData.data(), ResourceBindFlags::ShaderResource, false);
        return bricks;
    }
}
