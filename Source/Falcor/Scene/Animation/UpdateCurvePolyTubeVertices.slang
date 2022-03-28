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
import Scene.SceneTypes;
import Utils.Math.Quaternion;
import SharedTypes;

#ifndef CURVE_KEYFRAME_COUNT
// CURVE_KEYFRAME_COUNT must be defined in order to use this module.
#error "CURVE_KEYFRAME_COUNT not defined!"
#endif

struct CurvePerKeyframe
{
    StructuredBuffer<DynamicCurveVertexData> vertexData;
};

struct CurvePolyTubeVertexUpdater
{
    uint2 keyframeIndices;
    float t;
    bool copyPrev;

    uint vertexCount;
    uint indexCount;

    // Curve vertex caches at all keyframes
#if CURVE_KEYFRAME_COUNT > 0
    CurvePerKeyframe curvePerKeyframe[CURVE_KEYFRAME_COUNT];
#else
    CurvePerKeyframe curvePerKeyframe[1];
#endif

    StructuredBuffer<PerMeshMetadata> perMeshData;
    StructuredBuffer<PerCurveMetadata> perCurveData;
    ByteAddressBuffer curveStrandIndexData;

    // Output
    RWStructuredBuffer<PackedStaticVertexData> sceneVertexData;
    RWStructuredBuffer<PrevVertexData> prevVertexData;

    // Accessors
    DynamicCurveVertexData interpolateCurveVertex(uint vertexID)
    {
        DynamicCurveVertexData v0, v1;
        v0 = curvePerKeyframe[keyframeIndices.x].vertexData[vertexID];
        v1 = curvePerKeyframe[keyframeIndices.y].vertexData[vertexID];

        DynamicCurveVertexData result;
        result.position = lerp(v0.position, v1.position, t);
        return result;
    }

    void updateVertex(uint threadID, uint curveID)
    {
        const uint curveVertexIndex = threadID / 4;
        const uint meshVertexOffset = threadID % 4;

        // Curve and mesh metadata is placed at the same place in respecive buffers.
        const uint meshID = curveID;
        PerCurveMetadata curveMeta = perCurveData[curveID];
        PerMeshMetadata meshMeta = perMeshData[meshID];

        if (curveVertexIndex >= curveMeta.vertexCount) return;

        if (copyPrev)
        {
            uint meshVertexIndex = 4 * curveVertexIndex + meshVertexOffset;
            if (meshVertexIndex >= meshMeta.vertexCount) return;

            uint globalMeshVertexIndex = meshMeta.sceneVbOffset + meshVertexIndex;
            uint globalPrevVertexIndex = meshMeta.prevVbOffset + meshVertexIndex;

            StaticVertexData meshVertex = sceneVertexData[globalMeshVertexIndex].unpack();

            prevVertexData[globalPrevVertexIndex].position = meshVertex.position;
        }
        else
        {
            uint globalCurveVertexIndex = curveMeta.vertexOffset + curveVertexIndex;
            float3 curCurvePos = interpolateCurveVertex(globalCurveVertexIndex).position;
            float3 curCurveTangent;

            // 0 - first index of the strand, 0xFFFFFFFF - last index of the strand
            uint curveStrandIndex = curveStrandIndexData.Load(curveMeta.vertexOffset * 4 + curveVertexIndex * 4);
            if (curveStrandIndex != 0xffffffff)
            {
                float3 curCurvePosNext = interpolateCurveVertex(globalCurveVertexIndex + 1).position;
                curCurveTangent = normalize(curCurvePosNext - curCurvePos);
            }
            else
            {
                float3 curCurvePosPrev = interpolateCurveVertex(globalCurveVertexIndex - 1).position;
                curCurveTangent = normalize(curCurvePos - curCurvePosPrev);
            }

            uint meshVertexIndex = 4 * curveVertexIndex + meshVertexOffset;
            if (meshVertexIndex >= meshMeta.vertexCount) return;

            uint globalMeshVertexIndex = meshMeta.sceneVbOffset + meshVertexIndex;
            uint globalPrevVertexIndex = meshMeta.prevVbOffset + meshVertexIndex;

            StaticVertexData meshVertex = sceneVertexData[globalMeshVertexIndex].unpack();

            prevVertexData[globalPrevVertexIndex].position = meshVertex.position;

            float3 prevMeshTangent = normalize(meshVertex.tangent.xyz);
            float4 quat = from_to_rotation(prevMeshTangent, curCurveTangent);
            float3 curNormal = rotate_vector(meshVertex.normal, quat);

            meshVertex.tangent = float4(curCurveTangent, 1.f);
            meshVertex.normal = curNormal;
            meshVertex.position = curCurvePos + meshVertex.curveRadius * meshVertex.normal;

            if (any(isnan(meshVertex.position)) || any(isinf(meshVertex.position)))
            {
                meshVertex.position = curCurvePos;
            }

            sceneVertexData[globalMeshVertexIndex].pack(meshVertex);
        }
    }
};

ParameterBlock<CurvePolyTubeVertexUpdater> gCurvePolyTubeVertexUpdater;

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    gCurvePolyTubeVertexUpdater.updateVertex(dispatchThreadID.x, dispatchThreadID.y);
}
