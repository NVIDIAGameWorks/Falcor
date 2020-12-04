/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#define PNANOVDB_HLSL
#include "nanovdb/PNanoVDB.h"

/** Voxel grid based on NanoVDB.
*/
struct Grid
{
    typedef pnanovdb_readaccessor_t Accessor;

    StructuredBuffer<uint> buf;

    /** Get the minimum index stored in the grid.
        \return Returns minimum index stored in the grid.
    */
    int3 getMinIndex()
    {
        pnanovdb_root_handle_t root = pnanovdb_tree_get_root(buf, pnanovdb_grid_get_tree(buf, { pnanovdb_address_null() }));
        return pnanovdb_root_get_bbox_min(buf, root);
    }

    /** Get the maximum index stored in the grid.
        \return Returns maximum index stored in the grid.
    */
    int3 getMaxIndex()
    {
        pnanovdb_root_handle_t root = pnanovdb_tree_get_root(buf, pnanovdb_grid_get_tree(buf, { pnanovdb_address_null() }));
        return pnanovdb_root_get_bbox_max(buf, root);
    }

    /** Get the minimum value stored in the grid.
        \return Returns minimum value stored in the grid.
    */
    float getMinValue()
    {
        pnanovdb_root_handle_t root = pnanovdb_tree_get_root(buf, pnanovdb_grid_get_tree(buf, { pnanovdb_address_null() }));
        return pnanovdb_read_float(buf, pnanovdb_root_get_min_address(PNANOVDB_GRID_TYPE_FLOAT, buf, root));
    }

    /** Get the maximum value stored in the grid.
        \return Returns maximum value stored in the grid.
    */
    float getMaxValue()
    {
        pnanovdb_root_handle_t root = pnanovdb_tree_get_root(buf, pnanovdb_grid_get_tree(buf, { pnanovdb_address_null() }));
        return pnanovdb_read_float(buf, pnanovdb_root_get_max_address(PNANOVDB_GRID_TYPE_FLOAT, buf, root));
    }

    /** Transform position from world- to index-space.
        \param[in] pos Position in world-space.
        \return Returns position in index-space.
    */
    float3 worldToIndexPos(float3 pos)
    {
        return pnanovdb_grid_world_to_indexf(buf, { pnanovdb_address_null() }, pos);
    }

    /** Transform direction from world- to index-space.
        \param[in] dir Direction in world-space.
        \return Returns direction in index-space.
    */
    float3 worldToIndexDir(float3 dir)
    {
        return normalize(pnanovdb_grid_world_to_index_dirf(buf, { pnanovdb_address_null() }, dir));
    }

    /** Transform position from index- to world-space.
        \param[in] pos Position in index-space.
        \return Returns position in world-space.
    */
    float3 indexToWorldPos(float3 pos)
    {
        return pnanovdb_grid_index_to_worldf(buf, { pnanovdb_address_null() }, pos);
    }

    /** Transform direction from index- to world-space.
        \param[in] dir Direction in index-space.
        \return Returns direction in world-space.
    */
    float3 indexToWorldDir(float3 dir)
    {
        return normalize(pnanovdb_grid_index_to_world_dirf(buf, { pnanovdb_address_null() }, dir));
    }

    /** Create an grid accessor.
        \return Returns the new grid accessor.
    */
    Accessor createAccessor()
    {
        Accessor accessor;
        pnanovdb_root_handle_t root = pnanovdb_tree_get_root(buf, pnanovdb_grid_get_tree(buf, { pnanovdb_address_null() }));
        pnanovdb_readaccessor_init(accessor, root);
        return accessor;
    }

    /** Lookup the grid using nearest-neighbor sampling.
        \param[in] pos Position in world-space.
        \param[in,out] accessor Grid accessor.
        \return Returns the value in the grid.
    */
    float lookupWorld(const float3 pos, inout Accessor accessor)
    {
        return lookupIndex(worldToIndexPos(pos), accessor);
    }

    /** Lookup the grid using nearest-neighbor sampling.
        \param[in] index Fractional voxel index.
        \param[in,out] accessor Grid accessor.
        \return Returns the value in the grid.
    */
    float lookupIndex(const int3 index, inout Accessor accessor)
    {
        pnanovdb_address_t address = pnanovdb_readaccessor_get_value_address(PNANOVDB_GRID_TYPE_FLOAT, buf, accessor, index);
        return pnanovdb_read_float(buf, address);
    }

    /** Lookup the grid using tri-linear sampling.
        \param[in] pos Position in world-space.
        \param[in,out] accessor Grid accessor.
        \return Returns the interpolated value in the grid.
    */
    float lookupLinearWorld(const float3 pos, inout Accessor accessor)
    {
        return lookupLinearIndex(worldToIndexPos(pos), accessor);
    }

    /** Lookup the grid using tri-linear sampling.
        \param[in] index Fractional voxel index.
        \param[in,out] accessor Grid accessor.
        \return Returns the interpolated value in the grid.
    */
    float lookupLinearIndex(const float3 index, inout Accessor accessor)
    {
        const float3 indexOffset = index - 0.5f;
        const int3 i = floor(indexOffset);
        const float3 f = indexOffset - i;
        const float x0z0 = lerp(lookupIndex(i + int3(0, 0, 0), accessor), lookupIndex(i + int3(1, 0, 0), accessor), f.x);
        const float x1z0 = lerp(lookupIndex(i + int3(0, 1, 0), accessor), lookupIndex(i + int3(1, 1, 0), accessor), f.x);
        const float y0 = lerp(x0z0, x1z0, f.y);
        const float x0z1 = lerp(lookupIndex(i + int3(0, 0, 1), accessor), lookupIndex(i + int3(1, 0, 1), accessor), f.x);
        const float x1z1 = lerp(lookupIndex(i + int3(0, 1, 1), accessor), lookupIndex(i + int3(1, 1, 1), accessor), f.x);
        const float y1 = lerp(x0z1, x1z1, f.y);
        return lerp(y0, y1, f.z);
    }

    /** Lookup the grid using stochastic tri-linear sampling.
        \param[in] pos Position in world-space.
        \param[in] u Uniform random number in [0..1).
        \param[in,out] accessor Grid accessor.
        \return Returns the sampled value in the grid.
    */
    float lookupStochasticWorld(const float3 pos, const float3 u, inout Accessor accessor)
    {
        return lookupStochasticIndex(worldToIndexPos(pos), u, accessor);
    }

    /** Lookup the grid using stochastic tri-linear sampling.
        \param[in] index Fractional voxel index.
        \param[in] u Uniform random number in [0..1).
        \param[in,out] accessor Grid accessor.
        \return Returns the sampled value in the grid.
    */
    float lookupStochasticIndex(const float3 index, const float3 u, inout Accessor accessor)
    {
        const float3 dist = frac(index) - 0.5f;
        const int3 offset = u < abs(dist) ? sign(dist) : int3(0);
        return lookupIndex(index + offset, accessor);
    }
};
