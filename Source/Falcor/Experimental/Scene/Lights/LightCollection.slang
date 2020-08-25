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
#include "Scene/Material/MaterialDefines.slangh"

import Utils.Attributes;
import Scene.ShadingData;
__exported import Experimental.Scene.Lights.LightCollectionShared;
__exported import Experimental.Scene.Lights.MeshLightData;

/** This struct represents all emissive geometry in the scene.

    All triangle vertices are pre-transformed to world space and important
    quantities such as emitted flux are pre-computed.

    The user code instantiates LightCollection in e.g. a parameter block and
    calls member functions on the CPU-side class to update/bind it to their program.
*/
struct LightCollection
{
    uint                                        triangleCount;          ///< Total number of emissive triangles in all mesh lights.
    uint                                        activeTriangleCount;    ///< Total number of active (non-culled) emissive triangles in all mesh lights.
    uint                                        meshCount;              ///< Total number of mesh lights.
    uint                                        _pad;

    // These buffers are only valid if triangleCount > 0.
    [root] StructuredBuffer<PackedEmissiveTriangle> triangleData;       ///< Per-triangle geometry data for emissive triangles.

    StructuredBuffer<uint>                      activeTriangles;        ///< Indices of all active (non-culled) emissive triangles. This is only valid if activeTriangleCount > 0.
    StructuredBuffer<EmissiveFlux>              fluxData;               ///< Per-triangle flux data for emissive triangles.
    StructuredBuffer<MeshLightData>             meshData;               ///< Per-mesh data for emissive meshes.
    StructuredBuffer<uint>                      perMeshInstanceOffset;  ///< Per-mesh instance offset into emissive triangles array, or kInvalidIndex if mesh has no emissive triangles.

    static const uint kInvalidIndex = 0xffffffff;

    /** Returns the total number of emissive triangles.
        \return Emissive triangle count.
    */
    uint getTriangleCount() { return triangleCount; }

    /** Returns the number of active (non-culled) emissive triangles.
        \return Active emissive triangle count.
    */
    uint getActiveTriangleCount() { return activeTriangleCount; }

    /** Checks if the light collection is empty.
        \return True if there are no emissive triangles.
    */
    bool isEmpty() { return triangleCount == 0; }

    /** Returns the data for a given triangle.
        \param[in] triIdx Emissive triangle index.
        \return Emissive triangle data.
    */
    EmissiveTriangle getTriangle(uint triIdx)
    {
        return triangleData[triIdx].unpack();
    }

    /** Returns a triangle index given the mesh instance ID and primitive index.
        \param[in] meshInstanceID Global mesh instance ID.
        \param[in] primitiveIndex Primitive index in the given mesh.
        \return Emissive triangle index, or kInvalidIndex if not an emissive triangle.
    */
    uint getTriangleIndex(uint meshInstanceID, uint primitiveIndex)
    {
        uint offset = perMeshInstanceOffset[meshInstanceID];
        return offset != kInvalidIndex ? offset + primitiveIndex : kInvalidIndex;
    }

    /** Returns the mesh light index for a given triangle.
        \param[in] triIdx Emissive triangle index.
        \return Mesh light index.
    */
    uint getLightIdx(uint triIdx)
    {
        return getTriangle(triIdx).lightIdx;
    }

    /** Returns the mesh light data for a given triangle.
        \param[in] triIdx Emissive triangle index.
        \return Mesh light data.
    */
    MeshLightData getMeshData(uint triIdx)
    {
        return meshData[getLightIdx(triIdx)];
    }
};
