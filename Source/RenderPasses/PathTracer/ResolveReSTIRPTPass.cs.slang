/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
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
import ColorType;
import Params;
import ReSTIRPathTracing.PathReservoir;
import Utils.Debug.PixelDebug;
import Rendering.Utils.PixelStats;
import Utils.Math.HashUtils;
import Utils.Color.ColorMap;
import LoadShadingData;

struct ResolveReSTIRPTPass
{
    PathTracerParams params;
    StructuredBuffer<PathReservoir> pathReservoirs;
    Texture2D<uint> sampleOffset;
    RWTexture2D<float4> outputColor;

    Texture2D<PackedHitInfo> vbuffer;
    Texture2D<float2> motionVectors;

    Texture2D<uint> duplicationMap;
    Texture2D<uint> sampleSurvivedFrames;

    RWTexture2D<float4> outDuplicationMap;

    bool useDecoupledShading;
    bool useAreaReservoirs;

    float3 pseudocolor(uint value)
    {
        uint h = jenkinsHash(value);
        return (uint3(h, h >> 8, h >> 16) & 0xff) / 255.f;
    }

    void execute(const uint2 pixel)
    {
        if (!useAreaReservoirs)
        {
            HitInfo hit;
            hit = HitInfo(vbuffer[pixel]);
            if (!hit.isValid())
            {
                outDuplicationMap[pixel] = 0.f;
                return;
            }
        }
        uint reservoirIdx = params.getReservoirOffset(pixel);
        float3 color = pathReservoirs[reservoirIdx].integrand * pathReservoirs[reservoirIdx].weight;

        if (any(isnan(color) || isinf(color) || color < 0.f)) color = 0.f;

        float temporalDuplicationValue = min(1.f, sampleSurvivedFrames[pixel] / 40.f);


        if (temporalDuplicationValue < 0.f || isinf(temporalDuplicationValue) || isnan(temporalDuplicationValue)) temporalDuplicationValue = 0.f;

        float spatialDuplicationValue = min(1.f, duplicationMap[pixel] / 288.f);

        outDuplicationMap[pixel] = float4(spatialDuplicationValue, temporalDuplicationValue, 0.f, 0.f);


        if (!useDecoupledShading)
        {
            if (kSamplesPerPixel == 1)
            {
                outputColor[pixel].rgb += color;
            }
            else
            {
                outputColor[pixel].rgb = color;
            }
        }
    }
}

cbuffer CB
{
    ResolveReSTIRPTPass gResolveReSTIRPTPass;
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    printSetPixel(dispatchThreadId.xy);
    logSetPixel(dispatchThreadId.xy);
    gResolveReSTIRPTPass.execute(dispatchThreadId.xy);
}
