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

// Debug modes
#define ShowPos         1
#define ShowNormals     2
#define ShowAlbedo      3
#define ShowLighting    4

#ifdef FALCOR_HLSL
// Some macros to help us call this function from HLSL
#define vec3 float3
#define vec4 float4
#endif

vec3 shade(vec3 posW, vec3 normalW, float linearRoughness, vec4 albedo)
{
    // Discard empty pixels
    if (albedo.a <= 0)
    {
        discard;
    }

    /* Reconstruct the hit-point */
    HitPoint hitPt = initHitPoint();
    hitPt.posW = posW;
    hitPt.V = normalize(gCamera.posW - posW);
    hitPt.N = normalW;
    hitPt.NdotV = abs(dot(hitPt.V, hitPt.N));
    hitPt.linearRoughness = linearRoughness;

    /* Reconstruct layers (one diffuse layer) */
    hitPt.diffuse = albedo.rgb;
    hitPt.opacity = 0;

    /* Do lighting */
    ShadingResult dirResult = evalMaterial(hitPt, gDirLight, 1);
    ShadingResult pointResult = evalMaterial(hitPt, gPointLight, 1);

    float3 result;
    // Debug vis
    if (gDebugMode == ShowPos)
        result = posW;
    else if (gDebugMode == ShowNormals)
        result = 0.5 * normalW + 0.5f;
    else if (gDebugMode == ShowAlbedo)
        result = albedo.rgb;
    else if (gDebugMode == ShowLighting)
        result = (dirResult.diffuseBrdf + pointResult.diffuseBrdf) / hitPt.diffuse.rgb;
    else
        result = dirResult.diffuse + pointResult.diffuse;

    return result;
}