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
import ParticleData;

static const uint numThreads = 256;

struct DrawArguments
{
    uint vertexCountPerInstance;
    uint instanceCount;
    uint startVertexLocation;
    uint startInstanceLocation;
};

cbuffer PerFrame
{
#ifdef _SORT
    SimulateWithSortPerFrame perFrame;
#else
    SimulatePerFrame perFrame;
#endif
};

AppendStructuredBuffer<uint> deadList;
#ifdef _SORT
AppendStructuredBuffer<SortData> aliveList;
RWStructuredBuffer<uint> sortIterationCounter;
#else
AppendStructuredBuffer<uint> aliveList;
#endif
RWStructuredBuffer<Particle> particlePool;
RWByteAddressBuffer numDead;
RWStructuredBuffer<DrawArguments> drawArgs;

uint getNextPow2(uint n)
{
    if (n == 0)
        return 0;

    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;

    return n;
}

[numthreads(numThreads, 1, 1)]
void main(uint3 groupID : SV_GroupID, uint groupIndex : SV_GroupIndex)
{
    uint index = getParticleIndex(groupID.x, numThreads, groupIndex);
    //check if the particle is alive
    if (particlePool[index].life > 0)
    {
        particlePool[index].life -= perFrame.dt;
        //check if the particle died this frame
        if (particlePool[index].life <= 0)
        {
            deadList.Append(index);
        }
        else
        {
            particlePool[index].pos += particlePool[index].vel * perFrame.dt;
            particlePool[index].vel += particlePool[index].accel * perFrame.dt;
            particlePool[index].scale = max(particlePool[index].scale + particlePool[index].growth * perFrame.dt, 0);            
            particlePool[index].rot += particlePool[index].rotVel * perFrame.dt;
        #ifdef _SORT
            SortData data;
            data.index = index;
            data.depth = mul(float4(particlePool[index].pos, 1.f), perFrame.view).z;
            aliveList.Append(data);
        #else
            aliveList.Append(index);
        #endif
        }
    }

    //Set up data for sort/draw
    DeviceMemoryBarrierWithGroupSync();
    if (index == 0)
    {
        uint numDeadParticles = (uint)(numDead.Load(0));
        uint numAliveParticles = perFrame.maxParticles - numDeadParticles;
#ifdef _SORT
        sortIterationCounter[0] = max(kParticleSortThreads, getNextPow2(numAliveParticles));
        sortIterationCounter[1] = sortIterationCounter[0] / kParticleSortThreads;
#endif
        drawArgs[0].instanceCount = numAliveParticles;
    }
}
