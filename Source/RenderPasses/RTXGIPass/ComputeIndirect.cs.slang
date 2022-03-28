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

/** Fullscreen pass for computing indirect illumination using RTXGI.

    For each pixel, a lookup in the RTXGI probe volume is performed,
    and the irradiance scaled by the perfectly diffuse albedo is returned.
*/
#include "Utils/Math/MathConstants.slangh"

import Rendering.RTXGI.RTXGIVolume;
import LoadShadingData;

cbuffer CB
{
    uint2 gFrameDim;
    uint gFrameCount;
}

RWTexture2D<float4> gOutput;


[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadId.xy;
    if (any(pixel >= gFrameDim)) return;

    float3 Lindirect = float3(0);
    let lod = createTextureSampler(pixel);

    ShadingData sd;
    if (loadShadingData(pixel, gFrameDim, gScene.camera, lod, sd))
    {
        // Create BSDF instance.
        let bsdf = gScene.materials.getBSDF(sd, lod);

        float3 albedo = bsdf.getProperties(sd).diffuseReflectionAlbedo;
        Lindirect = gRTXGIVolume.evalRadiance(sd.posW, sd.N, sd.V, albedo);
    }

    gOutput[pixel] = float4(Lindirect, 1.f);
}
