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
#define PNANOVDB_HLSL
#include "nanovdb/PNanoVDB.h"

/** Voxel grid based on NanoVDB.
*/
struct Grid
{
    // Grid data.
    int3 minIndex;
    float minValue;
    int3 maxIndex;
    float maxValue;
    // NanoVDB data.
    typedef pnanovdb_readaccessor_t Accessor;
    StructuredBuffer<uint> buf;
    // Brick atlas data.
    Texture3D<float2> rangeTex;
    Texture3D<uint4> indirectionTex;
    Texture3D<float> atlasTex;

    /** Get the minimum index stored in the grid.
        \return Returns minimum index stored in the grid.
    */
    int3 getMinIndex()
    {
        return minIndex;
    }

    /** Get the maximum index stored in the grid.
        \return Returns maximum index stored in the grid.
    */
    int3 getMaxIndex()
    {
        return maxIndex;
    }

    /** Get the minimum value stored in the grid.
        \return Returns minimum value stored in the grid.
    */
    float getMinValue()
    {
        return minValue;
    }

    /** Get the maximum value stored in the grid.
        \return Returns maximum value stored in the grid.
    */
    float getMaxValue()
    {
        return maxValue;
    }

    /** Get the mean value stored in the grid.
    \return Returns the mean value stored in the grid.
    */
    float getMeanValue()
    {
        pnanovdb_root_handle_t root = pnanovdb_tree_get_root(buf, pnanovdb_grid_get_tree(buf, { pnanovdb_address_null() }));
        return pnanovdb_read_float(buf, pnanovdb_root_get_ave_address(PNANOVDB_GRID_TYPE_FLOAT, buf, root));
    }

    /** Get the stdandard deviation value stored in the grid.
    \return Returns the standard deviation value stored in the grid.
    */
    float getStdDevValue()
    {
        pnanovdb_root_handle_t root = pnanovdb_tree_get_root(buf, pnanovdb_grid_get_tree(buf, { pnanovdb_address_null() }));
        return pnanovdb_read_float(buf, pnanovdb_root_get_stddev_address(PNANOVDB_GRID_TYPE_FLOAT, buf, root));
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
        return normalize(worldToIndexDirUnnormalized(dir));
    }

    /** Transform direction from world- to index-space without normalizing it.
        \param[in] dir Direction in world-space.
        \return Returns unnormalized direction in index-space.
    */
    float3 worldToIndexDirUnnormalized(float3 dir)
    {
        return pnanovdb_grid_world_to_index_dirf(buf, { pnanovdb_address_null() }, dir);
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
        return normalize(indexToWorldDirUnnormalized(dir));
    }

    /** Transform direction from index- to world-space without normalizing it.
        \param[in] dir Direction in index-space.
        \return Returns direction in world-space.
    */
    float3 indexToWorldDirUnnormalized(float3 dir)
    {
        return pnanovdb_grid_index_to_world_dirf(buf, { pnanovdb_address_null() }, dir);
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

    /** Lookup the grid using nearest-neighbor sampling.
        \param[in] index Fractional voxel index.
        \return Returns the value in the grid.
    */
    float lookupIndexTex(const int3 index)
    {
        const int3 brick = index >> 3;
        const float2 range = rangeTex[brick];
        const uint3 ptr = indirectionTex[brick].xyz;
        return atlasTex[(ptr.xyz << 3) + (index & 7)].x * (range.x - range.y) + range.y;
    }

    /** Lookup the local majorant of the grid on a given mipmap level.
        \param[in] index Fractional voxel index.
        \return Returns the value in the grid.
    */
    float lookupIndexLocalMajorantTex(int3 index, const int mip)
    {
        index >>= (3 + mip);
        return rangeTex.Load(int4(index, mip)).x;
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

    /** Lookup the grid using tri-linear sampling.
        \param[in] index Fractional voxel index.
        \return Returns the interpolated value in the grid.
    */
    float lookupLinearIndexTex(const float3 index)
    {
        const float3 indexOffset = index - 0.5f;
        const int3 i = floor(indexOffset);
        const float3 f = indexOffset - i;
        const float x0z0 = lerp(lookupIndexTex(i + int3(0, 0, 0)), lookupIndexTex(i + int3(1, 0, 0)), f.x);
        const float x1z0 = lerp(lookupIndexTex(i + int3(0, 1, 0)), lookupIndexTex(i + int3(1, 1, 0)), f.x);
        const float y0 = lerp(x0z0, x1z0, f.y);
        const float x0z1 = lerp(lookupIndexTex(i + int3(0, 0, 1)), lookupIndexTex(i + int3(1, 0, 1)), f.x);
        const float x1z1 = lerp(lookupIndexTex(i + int3(0, 1, 1)), lookupIndexTex(i + int3(1, 1, 1)), f.x);
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
        return lookupIndex(floor(index + (u - 0.5f)), accessor);
    }

    /** Lookup the grid using stochastic tri-linear sampling.
        \param[in] index Fractional voxel index.
        \param[in] u Uniform random number in [0..1).
        \return Returns the sampled value in the grid.
    */
    float lookupStochasticIndexTex(const float3 index, float3 u)
    {
        return lookupIndexTex(index + (u - 0.5f));
    }

    /** Wrapper for NanoVDB DDA.
        The main feature of the wrapper is that it allows traversing the VDB tree
        at the leaf level (as well as on the voxel level).
    */
    struct DDA
    {
        enum class Level
        {
            // The values correspond to the dim values in PNanoVDB.
            Voxel = 1,
            Leaf = 8
        };

        static const uint   gridType = PNANOVDB_GRID_TYPE_FLOAT;

        float3 iorigin;
        float3 idir;
        float scale;
        Level lowestLevel;
        uint stepCount;

        pnanovdb_hdda_t hdda;
        pnanovdb_address_t cachedAddress;
        uint cachedAddressLevel;

        /** Initialize the DDA
            \param[in] grid VDB grid.
            \param[in] accessor Grid accessor.
            \param[in] pos Position in world space of the grid.
            \param[in] dir Direction in world space fo the grid.
            \param[in] tmin Minimum ray parameter in world space of the grid to start the traversal.
            \param[in] tmax Maximum ray parameter in world space of the grid.
            \param[in] lev VDB level to perform the traversal on (either voxel or leaf).
        */
        [mutating] void initialize(const Grid grid, inout Accessor accessor, const float3 pos, const float3 dir, float tmin, float tmax, DDA::Level level)
        {
            iorigin = grid.worldToIndexPos(pos);
            idir = grid.worldToIndexDirUnnormalized(dir);
            const float length = sqrt(dot(idir, idir));
            tmin *= length;
            tmax *= length;
            scale = 1.f / length;
            idir *= scale;
            lowestLevel = level;

            // Get the level (represented as dim) under the current position.
            const pnanovdb_vec3_t tpos = pnanovdb_hdda_ray_start(iorigin, tmin - 0.0001f, idir);
            int dim = pnanovdb_uint32_as_int32(pnanovdb_readaccessor_get_dim(gridType, grid.buf, accessor, pnanovdb_hdda_pos_to_ijk(tpos)));
            dim = int(lowestLevel);

            // Setup HDDA to run over voxels initially and apply small offset to avoid div by zero.
            pnanovdb_hdda_init(hdda, iorigin, tmin - 0.0001f, idir, tmax + 0.0001f, dim);
            stepCount = 0;
        }


        /** Check if the desired element (voxel or interior node) in the VDB grid is active.
            If we traverse at the leaf level, this function checks whether any of the interior nodes
            above the indexed voxel is active and caches the address of that node.
            If we traverse at the voxel level, simply check if the indexed voxel is active.
            \param[in] grid VDB grid.
            \param[in] accessor Grid accessor.
            \param[in] ijk Voxel coordinate in index space.
        */
        [mutating] bool isActive(const Grid grid, inout Accessor accessor, const pnanovdb_coord_t ijk)
        {
            if (lowestLevel == DDA::Level::Leaf)
            {
                // Check if nodes above the voxel are active and cache the address
                // of the lowest existing node to fetch data from later.
                // TODO: Check if the down traversal can be avoided.
                pnanovdb_root_tile_handle_t tile = pnanovdb_root_find_tile(gridType, grid.buf, accessor.root, ijk);
                if (!pnanovdb_int64_is_zero(pnanovdb_root_tile_get_child(grid.buf, tile)))
                {
                    pnanovdb_upper_handle_t node2 = pnanovdb_root_get_child(gridType, grid.buf, accessor.root, tile);
                    pnanovdb_uint32_t n2 = pnanovdb_upper_coord_to_offset(ijk);
                    if (pnanovdb_upper_get_child_mask(grid.buf, node2, n2))
                    {
                        pnanovdb_lower_handle_t node1 = pnanovdb_upper_get_child(gridType, grid.buf, node2, n2);
                        pnanovdb_uint32_t n1 = pnanovdb_lower_coord_to_offset(ijk);
                        if (pnanovdb_lower_get_child_mask(grid.buf, node1, n1))
                        {
                            pnanovdb_leaf_handle_t node0 = pnanovdb_lower_get_child(gridType, grid.buf, node1, n1);
                            cachedAddress = node0.address;
                            cachedAddressLevel = 0;
                            return true;
                        }
                        else
                        {
                            cachedAddress = node1.address;
                            cachedAddressLevel = 1;
                            return pnanovdb_lower_get_value_mask(grid.buf, node1, n1);
                        }
                    }
                    else
                    {
                        cachedAddress = node2.address;
                        cachedAddressLevel = 2;
                        return pnanovdb_upper_get_value_mask(grid.buf, node2, n2);
                    }
                }
                else
                {
                    cachedAddress = tile.address;
                    cachedAddressLevel = 3;
                    return pnanovdb_root_tile_get_state(grid.buf, tile) != 0u;
                }
            }
            else
            {
                // Voxel traversal.
                return pnanovdb_readaccessor_is_active(gridType, grid.buf, accessor, ijk);
            }
        }

        /** Make a DDA step.
            NanoVDB HDDA skips over the first interval to the first intersected element (voxel or node) along the ray.
            This is undesired in volume rendering, hence we need to handle the initial interval explicitly before
            we start stepping using the HDDA.
            \param[in] grid VDB grid.
            \param[in] accessor Grid accessor.
            \param[in,out] hasData Flag whether there is valid data in the grid in the step that the function performs.
            \param[in,out] dt Step length.
        */
        [mutating] bool step(const Grid grid, inout Accessor accessor, inout bool hasData, inout float dt)
        {
            // Only make a step if this is not the first call.
            if (stepCount > 0)
            {
                if (!pnanovdb_hdda_step(hdda)) return false;

                // Update the level of traversal.
                int dim = pnanovdb_uint32_as_int32(pnanovdb_readaccessor_get_dim(gridType, grid.buf, accessor, hdda.voxel));
                dim = max(dim, int(lowestLevel));
                pnanovdb_hdda_update(hdda, iorigin, idir, dim);
            }
            // Data is present only if the element at the lowest traversed level
            // (or any node above) are marked as active.
            hasData = isActive(grid, accessor, hdda.voxel);
            dt = (min(min(hdda.next.x, hdda.next.y), hdda.next.z) - hdda.tmin) * scale;

            stepCount++;
            return true;
        }

        /** Returns grid data along the last DDA step.
            \param[in] grid VDB grid.
            \param[in] accessor Grid accessor.
        */
        float getStepData(const Grid grid, inout Accessor accessor)
        {
            pnanovdb_address_t address;
            if (lowestLevel == DDA::Level::Leaf)
            {
                // TODO: add configurability for specifying what leaf value is returned (max, min, ave, stddev).
                if (cachedAddressLevel == 0)
                    address = pnanovdb_leaf_get_max_address(gridType, grid.buf, { cachedAddress } );
                else if (cachedAddressLevel == 1)
                    address = pnanovdb_lower_get_max_address(gridType, grid.buf, { cachedAddress } );
                else if (cachedAddressLevel == 2)
                    address = pnanovdb_upper_get_max_address(gridType, grid.buf, { cachedAddress } );
                else
                    address = pnanovdb_root_get_max_address(gridType, grid.buf, { cachedAddress } );
            }
            else
            {
                // Voxel
                address = pnanovdb_readaccessor_get_value_address(gridType, grid.buf, accessor, hdda.voxel);
            }
            return pnanovdb_read_float(grid.buf, address);
        }
    };
};
