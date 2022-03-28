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

#ifndef CURVE_KEYFRAME_COUNT
// CURVE_KEYFRAME_COUNT must be defined in order to use this module.
#error "CURVE_KEYFRAME_COUNT not defined!"
#endif

/** Compute pass for updating animated vertices.

    The dispatch size is one thread per dynamic vertex.
*/

struct CurvePerKeyframe
{
    StructuredBuffer<DynamicCurveVertexData> vertexData;
};

struct CurveVertexUpdater
{
    uint2 keyframeIndices;
    float t;
    bool copyPrev;

    uint dimX;
    uint vertexCount;

    // Curve vertex caches at all keyframes
#if CURVE_KEYFRAME_COUNT > 0
    CurvePerKeyframe curvePerKeyframe[CURVE_KEYFRAME_COUNT];
#else
    CurvePerKeyframe curvePerKeyframe[1];
#endif

    // Output
    RWStructuredBuffer<DynamicCurveVertexData> prevCurveVertices;
    RWStructuredBuffer<StaticCurveVertexData> curveVertices;

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

    void updateVertex(uint3 dispatchThreadID)
    {
        uint vertexID = dispatchThreadID.y * dimX + dispatchThreadID.x;
        if (vertexID >= vertexCount) return;

        if (copyPrev)
        {
            prevCurveVertices[vertexID].position = curveVertices[vertexID].position;
        }
        else
        {
            float3 curPos = interpolateCurveVertex(vertexID).position;
            float3 prevPos = curveVertices[vertexID].position;

            prevCurveVertices[vertexID].position = prevPos;
            curveVertices[vertexID].position = curPos;
        }
    }
};

ParameterBlock<CurveVertexUpdater> gCurveVertexUpdater;

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    gCurveVertexUpdater.updateVertex(dispatchThreadID);
}
