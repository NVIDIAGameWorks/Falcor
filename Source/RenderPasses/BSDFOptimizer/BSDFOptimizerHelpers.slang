/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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

import Scene.Shading;

float2 getViewportCoord(uint2 pixel, float2 viewportOffset, float2 viewportScale)
{
    float2 p = pixel + float2(0.5f);
    return (p - viewportOffset) * viewportScale;
}

float3 calculateSliceGeometry(float2 uv, out VertexData v, out float3 viewDir)
{
    // Setup local surface frame as T,B,N.
    v.posW = float3(0, 0, 0);
    v.normalW = float3(0, 0, 1);
    v.tangentW = float4(1, 0, 0, 1);
    v.texC = float2(0, 0); // Assume no textures.
    v.faceNormalW = v.normalW;

    // Compute dot products.
    // These are based on the axes in the 2D slice (theta_h, theta_d) with origin in lower-left corner.
    // This is the same format as the slices in Burley et al. 2012, 2015.
    float theta_h = uv.x * (M_PI / 2);
    float theta_d = (1.f - uv.y) * (M_PI / 2);

    float NdotH = cos(theta_h);
    float HdotL = cos(theta_d); // Note: HdotL = HdotV

    // Place the H vector at (0,0,1) to start.
    // Compute L, V that are mirrored about the yz-plane.
    float3 L = float3(sqrt(1.f - HdotL * HdotL), 0, HdotL);
    float3 V = float3(-L.x, 0.f, L.z);

    // Rotate L, V about the x-axis by an angle theta_h.
    float cos_h = NdotH;
    float sin_h = sqrt(1 - NdotH * NdotH);
    L = float3(L.x, cos_h * L.y - sin_h * L.z, sin_h * L.y + cos_h * L.z);
    V = float3(V.x, cos_h * V.y - sin_h * V.z, sin_h * V.y + cos_h * V.z);

    // Return vectors.
    viewDir = V;
    L = normalize(L);
    return L;
}
