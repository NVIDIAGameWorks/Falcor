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
import ParticleData;

RWStructuredBuffer<SortData> sortList;
//[0] is total num particles to sort, [1] is [0] / 1024 (required iterations per pass)
StructuredBuffer<uint> iterationCounter;

void Swap(uint index, uint compareIndex)
{
    SortData temp = sortList[index];
    sortList[index] = sortList[compareIndex];
    sortList[compareIndex] = temp;
}

[numthreads(kParticleSortThreads, 1, 1)]
void main(uint3 groupID : SV_GroupID, uint groupIndex : SV_GroupIndex)
{
    int threadIndex = (int)getParticleIndex(groupID.x, kParticleSortThreads, groupIndex);
    //set size used to determine whether a subset of the data should be ascending or descending
    for (uint setSize = 2; setSize <= iterationCounter[0]; setSize *= 2)
    {
        for (uint compareDist = setSize / 2; compareDist > 0; compareDist /= 2)
        {
            //shader can only touch 1024 sets of values (2048 total values) at a time, if the sort needs to affect 
            //more than 2048 paticles, the thread index is offset here in this loop to do each pass in multiple 2048 sized chunks 
            for (uint i = 0; i < iterationCounter[1]; ++i)
            {
                uint effectiveThreadIndex = threadIndex + i * kParticleSortThreads;
                uint index = 2 * compareDist * (effectiveThreadIndex / compareDist) + effectiveThreadIndex % compareDist;
                uint compareIndex = index + compareDist;
                uint descending = (index / setSize) % 2;

                if (descending)
                {
                    //if this is less than other, not descending
                    if (sortList[index].depth < sortList[compareIndex].depth)
                    {
                        Swap(index, compareIndex);
                    }
                }
                else
                {
                    //if this is greater than other, not ascending
                    if (sortList[index].depth > sortList[compareIndex].depth)
                    {
                        Swap(index, compareIndex);
                    }
                }

                DeviceMemoryBarrierWithGroupSync();
            }
        }
    }
}
