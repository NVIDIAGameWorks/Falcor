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

cbuffer PerEmit
{
    EmitData emitData;
};

ConsumeStructuredBuffer<uint> deadList;
RWStructuredBuffer<Particle> particlePool;
StructuredBuffer<Particle> emitList;
ByteAddressBuffer numAlive;

[numthreads(kParticleEmitThreads, 1, 1)]
void main(int3 groupID : SV_GroupID, int3 threadID : SV_GroupThreadID)
{
    uint index = kParticleEmitThreads * groupID.y + threadID.x;
    uint numAliveParticles = (uint)(numAlive.Load(0));
    //make sure this corresponds to an emitted particle, and isnt a redundant thread
    if (index < emitData.numEmit)
    {
        //make sure there's actually room for this particle
        if (index < emitData.maxParticles - (numAliveParticles + emitData.numEmit))
        {
            uint deadIndex = deadList.Consume();
            particlePool[deadIndex] = emitList[index];
        }
    }
}
