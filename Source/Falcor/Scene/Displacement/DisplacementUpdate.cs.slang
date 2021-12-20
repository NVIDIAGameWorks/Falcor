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
import Scene.Scene;
import Scene.Displacement.DisplacementMapping;
import Utils.Math.AABB;
import DisplacementUpdateTask;

static const bool kUsePreciseShellBounds = true;

cbuffer CB
{
    uint gTaskCount;

    StructuredBuffer<DisplacementUpdateTask> gTasks;
    RWStructuredBuffer<AABB> gAABBs;
};

/** This kernel is used for computing AABBs for displaced triangles.
    Work is organized in tasks (described by DisplacementUpdateTask).
    Each tasks computes AABBs for a range of triangles from a single mesh.
    A fixed number of threads (DisplacementUpdateTask::kThreadCount) is launched for each task,
    processing triangles in a fixed stride of kThreadCount.
*/
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint threadIndex = dispatchThreadId.x;
    const uint taskIndex = dispatchThreadId.y;

    if (threadIndex >= DisplacementUpdateTask::kThreadCount || taskIndex >= gTaskCount) return;

    const DisplacementUpdateTask task = gTasks[taskIndex];

    const uint materialID = gScene.meshes[task.meshID].materialID;

    DisplacementData displacementData;
    gScene.materials.loadDisplacementData(materialID, displacementData);

    const float globalExpansion = displacementData.getConservativeGlobalExpansion();
    const uint iterationCount = (task.count + DisplacementUpdateTask::kThreadCount - 1) / DisplacementUpdateTask::kThreadCount;

    for (uint iteration = 0; iteration < iterationCount; ++iteration)
    {
        const uint index = iteration * DisplacementUpdateTask::kThreadCount + threadIndex;
        if (index >= task.count) return;

        const uint triangleIndex = task.triangleIndex + index;
        const uint AABBIndex = task.AABBIndex + index;

        const uint3 indices = gScene.getIndices(task.meshID, triangleIndex);
        StaticVertexData vertices[3] = { gScene.getVertex(indices[0]), gScene.getVertex(indices[1]), gScene.getVertex(indices[2]) };

        AABB aabb;
        aabb.invalidate();

        if (!kUsePreciseShellBounds)
        {
            for (uint i = 0; i < 3; ++i) aabb.include(vertices[i].position);

            aabb.minPoint -= globalExpansion;
            aabb.maxPoint += globalExpansion;
        }
        else
        {
            const float2 shellMinMax = displacementData.getShellMinMax(vertices[0].texCrd, vertices[1].texCrd, vertices[2].texCrd);

            for (uint i = 0; i < 3; ++i)
            {
                aabb.include(vertices[i].position + (vertices[i].normal) * shellMinMax.x);
                aabb.include(vertices[i].position + (vertices[i].normal) * shellMinMax.y);

            }
        }

        gAABBs[AABBIndex] = aabb;
    }
}
