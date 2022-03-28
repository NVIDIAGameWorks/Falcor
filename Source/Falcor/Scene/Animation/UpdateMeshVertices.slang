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
import SharedTypes;

struct MeshPerKeyframe
{
    StructuredBuffer<PackedStaticVertexData> vertexData;
};

struct MeshVertexUpdater
{
    bool copyPrev;

    // Per Mesh
    StructuredBuffer<InterpolationInfo> perMeshInterp;
    StructuredBuffer<PerMeshMetadata> perMeshData;

    // All keyframes for all meshes in a linear array
#if MESH_KEYFRAME_COUNT > 0
    MeshPerKeyframe meshPerKeyframe[MESH_KEYFRAME_COUNT];
#else
    MeshPerKeyframe meshPerKeyframe[1];
#endif

    // Output
    RWStructuredBuffer<PackedStaticVertexData> sceneVertexData;
    RWStructuredBuffer<PrevVertexData> prevVertexData;

    StaticVertexData interpolateVertex(StaticVertexData v0, StaticVertexData v1, float t)
    {
        StaticVertexData out;
        out.position = lerp(v0.position, v1.position, t);
        out.normal = normalize(lerp(v0.normal, v1.normal, t));
        out.tangent = lerp(v0.tangent, v1.tangent, t);
        out.tangent.xyz = normalize(out.tangent.xyz);
        return out;
    }

    void updateVertex(uint meshVertexID, uint meshID)
    {
        PerMeshMetadata meta = perMeshData[meshID];
        if (meshVertexID >= meta.vertexCount) return;

        uint outVertexID = meta.sceneVbOffset + meshVertexID;
        StaticVertexData orig = sceneVertexData[outVertexID].unpack();
        uint prevVertexID = meta.prevVbOffset + meshVertexID;

        if (copyPrev)
        {
            prevVertexData[prevVertexID].position = orig.position;
            return;
        }

        InterpolationInfo interp = perMeshInterp[meshID];
        MeshPerKeyframe meshKeyframeA = meshPerKeyframe[meta.keyframeBufferOffset + interp.keyframeIndices.x];
        MeshPerKeyframe meshKeyframeB = meshPerKeyframe[meta.keyframeBufferOffset + interp.keyframeIndices.y];
        StaticVertexData v0 = meshKeyframeA.vertexData[meshVertexID].unpack();
        StaticVertexData v1 = meshKeyframeB.vertexData[meshVertexID].unpack();
        StaticVertexData result = interpolateVertex(v0, v1, interp.t);

        prevVertexData[prevVertexID].position = orig.position;

        result.texCrd = orig.texCrd; // Same
        sceneVertexData[outVertexID].pack(result);
    }
};

ParameterBlock<MeshVertexUpdater> gMeshVertexUpdater;

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    gMeshVertexUpdater.updateVertex(dispatchThreadID.x, dispatchThreadID.y);
}
