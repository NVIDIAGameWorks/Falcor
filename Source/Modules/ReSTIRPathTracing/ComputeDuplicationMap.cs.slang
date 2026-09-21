/***************************************************************************
 # Copyright (c) 2015-26, NVIDIA CORPORATION. All rights reserved.
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
import PathReservoir;
import ReSTIRPT;
import Utils.Debug.PixelDebug;
import Rendering.Utils.PixelStats;

groupshared uint sampleIDs[32][32];

struct Pass
{
    ReSTIRPathTracingParams params; 
    Texture2D<uint> sampleIDTexture;
    RWTexture2D<uint> duplicationMap;
    void execute(const uint2 pixel, const uint2 blockPixel, const uint2 blockId, const uint blockIndex)
    {
        if (any(pixel >= params.frameDim)) return;

        // local memory
        int2 topLeftPixel = 16 * blockId.xy;

        // [-8, 23] x [-8, 23] window around top Left Pixel
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
            {
                int2 readPixel = int2(topLeftPixel.x - 8 + 2 * blockPixel.x + j, topLeftPixel.y - 8 + 2 * blockPixel.y + i);
                sampleIDs[2 * blockPixel.y + i][2 * blockPixel.x + j] =
                (any(readPixel >= params.frameDim || readPixel < 0)) ? 0 : sampleIDTexture[readPixel];
            }

        GroupMemoryBarrierWithGroupSync();

        // check 17x17 neighborhood

        uint ownSample = sampleIDs[blockPixel.y + 8][blockPixel.x + 8];
        uint dupCount = 0;

        if (ownSample != 0)
        {
            for (int i = -8; i <= 8; i++)
                for (int j = -8; j <= 8; j++)
                {
                    if (any(pixel + int2(j, i) >= params.frameDim || pixel + int2(j, i) < int2(0, 0))) continue;
                    uint neighborSample = sampleIDs[blockPixel.y + i + 8][blockPixel.x + j + 8];
                    if (ownSample == neighborSample && (i != 0 || j != 0))
                    {
                        dupCount++;
                    }
                }
        }

        duplicationMap[pixel] = dupCount;
    }
};

cbuffer CB
{
    Pass gPass;
}

[numthreads(16, 16, 1)]
void main( uint3 dispatchThreadId: SV_DispatchThreadID, uint3 groupThreadID: SV_GroupThreadID, uint3 groupID: SV_GroupID, uint groupIndex: SV_GroupIndex)
{
    uint2 pixel = dispatchThreadId.xy;
    if (any(pixel >= gPass.params.frameDim)) return;
    printSetPixel(pixel);
    logSetPixel(pixel);

    gPass.execute(dispatchThreadId.xy, groupThreadID.xy, groupID.xy, groupIndex);
}
