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
#include "Scene/SDFs/SDFGridNoDefines.slangh"

__exported import Scene.SDFs.SDFGridHitData;

 /** The scene sets the SCENE_SDF_GRID_IMPLEMENTATION define to select what SDF grid implementation should be used.
     This code typedefs the chosen type to the type 'SDFGrid'.
     All SDF grid implementation adhere to the same interface.
 */

#if SCENE_SDF_GRID_IMPLEMENTATION == SCENE_SDF_GRID_IMPLEMENTATION_NDSDF
import Scene.SDFs.NormalizedDenseSDFGrid.NDSDFGrid;
typedef NDSDFGrid SDFGrid;
#elif SCENE_SDF_GRID_IMPLEMENTATION == SCENE_SDF_GRID_IMPLEMENTATION_SVS
import Scene.SDFs.SparseVoxelSet.SDFSVS;
typedef SDFSVS SDFGrid;
#elif SCENE_SDF_GRID_IMPLEMENTATION == SCENE_SDF_GRID_IMPLEMENTATION_SBS
import Scene.SDFs.SparseBrickSet.SDFSBS;
typedef SDFSBS SDFGrid;
#elif SCENE_SDF_GRID_IMPLEMENTATION == SCENE_SDF_GRID_IMPLEMENTATION_SVO
import Scene.SDFs.SparseVoxelOctree.SDFSVO;
typedef SDFSVO SDFGrid;
#else
// If no SDFs are present, define an SDFGrid dummy so that compilation doesn't fail.
import Scene.SDFs.SDFGridBase;
struct SDFGridDummy : SDFGridBase, ISDFGrid
{
    bool intersectSDF(const float3 rayOrigLocal, const float3 rayDirLocal, const float tMin, const float tMax, const uint primitiveID, out float t, out SDFGridHitData hitData) { t = -1.0f; hitData = {}; return false; }
    bool intersectSDFAny(const float3 rayOrigLocal, const float3 rayDirLocal, const float tMin, const float tMax, const uint primitiveID) { return false; }

    void decodeHit(const SDFGridHitData hitData, const float3 hitPosition, out float3 gradient) { gradient = float3(0.0f); };
};
typedef SDFGridDummy SDFGrid;
#endif
