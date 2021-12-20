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
import Scene.SceneTypes;
import Utils.Math.AABB;

/** Compute pass for updating AABBs of animated vertices.

    The dispatch size is one thread per AABB.
*/

struct CurveAABBUpdater
{
    uint AABBCount;
    uint dimX;

    // Curve data
    ByteAddressBuffer curveIndexData;
    StructuredBuffer<StaticCurveVertexData> curveVertices;
    RWStructuredBuffer<AABB> curveAABBs;

    void updateAABB(uint3 dispatchThreadID)
    {
        uint indexID = dispatchThreadID.y * dimX + dispatchThreadID.x;
        if (indexID >= AABBCount) return;

        uint vertexID = curveIndexData.Load(indexID * 4);

        AABB bbox;
        bbox.invalidate();
        for (uint i = 0; i < 2; i++)
        {
            const StaticCurveVertexData vertex = curveVertices[vertexID + i];
            bbox.include(vertex.position - vertex.radius);
            bbox.include(vertex.position + vertex.radius);
        }

        curveAABBs[indexID] = bbox;
    }
};

ParameterBlock<CurveAABBUpdater> gCurveAABBUpdater;

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    gCurveAABBUpdater.updateAABB(dispatchThreadID);
}
