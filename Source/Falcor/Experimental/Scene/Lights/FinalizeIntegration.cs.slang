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
#include "Utils/Math/MathConstants.slangh"

import Scene.Scene;
import Utils.Color.ColorHelpers;
import Experimental.Scene.Lights.LightCollection;

cbuffer CB
{
    uint gTriangleCount;                    ///< Total number of triangles.
}

ByteAddressBuffer gTexelSum;                            ///< Sum over texels (RGB) + number of texels (A) in RGBA32Float format.
StructuredBuffer<PackedEmissiveTriangle> gTriangleData; ///< Per-triangle geometry data for emissive triangles.
RWStructuredBuffer<EmissiveFlux> gFluxData;             ///< Per-triangle flux data for emissive triangles.


/** Kernel computing the final pre-integrated triangle average radiance and flux.
    One dispatch with one thread per triangle (the dispatch is arranged as Y blocks of 256x1 threads).
*/
[numthreads(256, 1, 1)]
void finalizeIntegration(uint3 DTid : SV_DispatchThreadID)
{
    const uint triIdx = DTid.y * 256 + DTid.x;
    if (triIdx >= gTriangleCount) return;

    // Compute the triangle's average emitted radiance (RGB).
    EmissiveTriangle tri = gTriangleData[triIdx].unpack();
    const MaterialData materialData = gScene.getMaterial(tri.materialID);
    float3 averageEmissiveColor = materialData.emissive;

    if (gScene.isEmissiveTextured(tri.materialID))
    {
        // Compute the triangle's average textured emissive color based on the pre-integration results.
        // The alpha channel stores the texel count. If it is zero, the triangle didn't cover any samples
        // and we don't know anything about its radiance. In that case, assign a default value (1.0) to avoid bias.
        // TODO: This fallback can be removed when we use conservative rasterization in the pre-integration pass.
        float4 texelSum = asfloat(gTexelSum.Load4(triIdx * 16u));
        float weight = texelSum.a;
        averageEmissiveColor = weight > 0.f ? texelSum.rgb / weight : float3(1.0f);
    }
    float3 averageRadiance = averageEmissiveColor * materialData.emissiveFactor;

    // Pre-compute the luminous flux emitted, which is what we use during sampling to set probabilities.
    // We assume diffuse emitters and integrate per side (hemisphere) => the scale factor is pi.
    // Triangle area in m^2 (the scene units are assumed to be in meters).
    float flux = luminance(averageRadiance) * tri.area * (float) M_PI;  // Flux in lumens.

    gFluxData[triIdx].flux = flux;
    gFluxData[triIdx].averageRadiance = averageRadiance;
}
