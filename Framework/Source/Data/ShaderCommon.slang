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

/*******************************************************************
                    Common OGL bindings
*******************************************************************/

#ifndef _FALCOR_SHADER_COMMON_H_
#define _FALCOR_SHADER_COMMON_H_

#include "HostDeviceData.h"

cbuffer InternalPerFrameCB : register(b10)
{
    CameraData gCam;
    float3 gAmbientLighting;
    uint32_t gLightsCount;
    LightData gLights[MAX_LIGHT_SOURCES];
};

cbuffer InternalPerMeshCB : register(b11)
{
    float4x4 gWorldMat[64]; // If the mesh has bones, these are the bones matrices
    float4x4 gPrevWorldMat[64]; // previous world mat
    float3x3 gWorldInvTransposeMat[64]; // Per-instance matrices for transforming normals
    uint32_t gDrawId[64]; // Zero-based order/ID of Mesh Instances drawn per SceneRenderer::renderScene call.
    uint32_t gMeshId;
};

#ifdef _VERTEX_BLENDING
float4x4 getBlendedWorldMat(float4 weights, uint4 ids)
{
    float4x4 worldMat = gWorldMat[ids.x] * weights.x;
    worldMat += gWorldMat[ids.y] * weights.y;
    worldMat += gWorldMat[ids.z] * weights.z;
    worldMat += gWorldMat[ids.w] * weights.w;

    return worldMat;
}

float3x3 getBlendedInvTransposeWorldMat(float4 weights, uint4 ids)
{
    float3x3 mat = gWorldInvTransposeMat[ids.x] * weights.x;
    mat += gWorldInvTransposeMat[ids.y] * weights.y;
    mat += gWorldInvTransposeMat[ids.z] * weights.z;
    mat += gWorldInvTransposeMat[ids.w] * weights.w;

    return mat;
}

#endif

cbuffer InternalPerMaterialCB : register(b12)
{
    MaterialData gMaterial;
    MaterialData gTemporalMaterial;
    float gTemporalLODThreshold;
    bool gEnableTemporalNormalMaps;
    bool gDebugTemporalMaterial;
};

float2 calcMotionVector(float2 pixelCrd, float4 prevPosH, float2 renderTargetDim)
{
    float2 prevCrd = prevPosH.xy / prevPosH.w;
#ifdef FALCOR_GLSL
    prevCrd *= float2(0.5, 0.5);
#else
    prevCrd *= float2(0.5, -0.5);
#endif
    prevCrd += 0.5f;
    float2 normalizedCrd = pixelCrd / renderTargetDim;
    return prevCrd - normalizedCrd;
}

/*******************************************************************
                    GLSL Evaluation routines
*******************************************************************/
#if 0
bool isSamplerBound(in sampler2D sampler)
{
    return any(ufloat2(sampler) != 0);
}

float4 fetchTextureIfFound(in sampler2D sampler, in float2 uv, in float2 duvdx, in float2 duvdy)
{
    float4 ret = float4(1.0f);
    if(isSamplerBound(sampler)) 
    {
        ret = textureGrad(sampler, uv, duvdx, duvdy);
    }
    return ret;
}
#endif // defined(FALCOR_GLSL)
#endif  // _FALCOR_SHADER_COMMON_H_