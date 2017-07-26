/***************************************************************************
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
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
***************************************************************************/

Texture2D gInput;
RWTexture2D<float4> gOutput;

groupshared float4 colors[16][16];
groupshared float4 pixelated;

[numthreads(16, 16, 1)]
void main(uint3 groupId : SV_GroupID, uint3 groupThreadId : SV_GroupThreadId)
{
    // Calculate the start position of the block    
    uint3 resDim;
    gOutput.GetDimensions(resDim.x, resDim.y);

    uint2 posStart = groupId.xy * 16;
    uint2 crd = posStart + groupThreadId.xy;

    // Fetch all of the data into the shared local memory
    colors[groupThreadId.x][groupThreadId.y] = gInput[crd];

#ifdef _PIXELATE
    GroupMemoryBarrierWithGroupSync();
    if(any(groupThreadId) == false)
    {
        pixelated = 0;
        for(int i = 0 ; i < 16 ; i++)
        {
            for(int j = 0 ; j < 16 ; j++)
            {
                pixelated += colors[i][j];
            }
        }
        pixelated /= 16*16;
    }

    GroupMemoryBarrierWithGroupSync();
    gOutput[crd] = pixelated.bgra;
#else
    gOutput[crd] = colors[groupThreadId.x][groupThreadId.y].bgra;
#endif
}