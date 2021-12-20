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
#include "Utils/Math/MathConstants.slangh"

import Scene.Scene;

cbuffer CB
{
    uint gTriangleCount;                    ///< Total number of triangles.
}

StructuredBuffer<MeshLightData> gMeshData;                  ///< Per-mesh data for emissive meshes.
RWStructuredBuffer<PackedEmissiveTriangle> gTriangleData;   ///< Per-triangle geometry data for emissive triangles.

/** Kernel updating the emissive triangles for all mesh lights.
    Single dispatch with one thread per triangle.
*/
[numthreads(256, 1, 1)]
void updateTriangleVertices(uint3 DTid : SV_DispatchThreadID)
{
    const uint triIdx = DTid.x; // Global emissive triangle index.
    const uint vtxIdx = triIdx * 3; // The emissive triangle list is non-indexed.

    if (triIdx >= gTriangleCount) return;

    // Get the data for the mesh that this triangle belongs to.
    uint lightIdx = gTriangleData[triIdx].lightIdx;
    const MeshLightData meshData = gMeshData[lightIdx];

    GeometryInstanceID instanceID = { meshData.instanceID };
    uint triangleIndex = triIdx - meshData.triangleOffset; // Local triangle index in the mesh

    // Load triangle data.
    EmissiveTriangle tri = gTriangleData[triIdx].unpack();

    // Update triangle data.
    gScene.getVertexPositionsW(instanceID, triangleIndex, tri.posW);
    tri.normal = gScene.computeFaceNormalAndAreaW(instanceID, tri.posW, tri.area);
    gTriangleData[triIdx].pack(tri);
}
