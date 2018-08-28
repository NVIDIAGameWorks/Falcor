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
#ifndef _FALCOR_SHADER_COMMON_H_
#define _FALCOR_SHADER_COMMON_H_

#include "HostDeviceData.h"

shared cbuffer InternalPerFrameCB : register(b10)
{
    CameraData gCamera;
    uint32_t gLightsCount;
    float3 internalPerFrameCBPad;
    LightData gLights[MAX_LIGHT_SOURCES];
    LightProbeData gLightProbe;
    LightProbeSharedResources gProbeShared;
};

cbuffer InternalPerMeshCB
{
    float4x4 gWorldMat[MAX_INSTANCES];              // Per-instance world transforms
    float4x4 gPrevWorldMat[MAX_INSTANCES];          // Previous frame world transforms
    float3x4 gWorldInvTransposeMat[MAX_INSTANCES];  // Per-instance matrices for transforming normals
    uint32_t gDrawId[MAX_INSTANCES];                // Zero-based order/ID of Mesh Instances drawn per SceneRenderer::renderScene call.
    uint32_t gMeshId;
};

cbuffer InternalBoneCB
{
    float4x4 gBoneMat[MAX_BONES];               // Per-model bone matrices
    float4x4 gInvTransposeBoneMat[MAX_BONES];   // Per-model bone inverse transpose matrices
};

#ifdef _VERTEX_BLENDING
float4x4 getBlendedBoneMat(float4 weights, uint4 ids)
{
    float4x4 boneMat = gBoneMat[ids.x] * weights.x;
    boneMat += gBoneMat[ids.y] * weights.y;
    boneMat += gBoneMat[ids.z] * weights.z;
    boneMat += gBoneMat[ids.w] * weights.w;

    return boneMat;
}

float3x3 getBlendedInvTransposeBoneMat(float4 weights, uint4 ids)
{
    float3x3 mat = (float3x3)gInvTransposeBoneMat[ids.x] * weights.x;
    mat += (float3x3)gInvTransposeBoneMat[ids.y] * weights.y;
    mat += (float3x3)gInvTransposeBoneMat[ids.z] * weights.z;
    mat += (float3x3)gInvTransposeBoneMat[ids.w] * weights.w;

    return mat;
}
#endif

ParameterBlock<MaterialData> gMaterial;

float2 calcMotionVector(float2 pixelCrd, float4 prevPosH, float2 renderTargetDim)
{
    float2 prevCrd = prevPosH.xy / prevPosH.w;
#ifdef FALCOR_VK
    prevCrd *= float2(0.5, 0.5);
#else
    prevCrd *= float2(0.5, -0.5);
#endif
    prevCrd += 0.5f;
    float2 normalizedCrd = pixelCrd / renderTargetDim;
    return prevCrd - normalizedCrd;
}

/** This struct describes the geometric data for a specific hit point used for lighting calculations 
*/
struct ShadingData
{
    float3    posW;         ///< Shading hit position in world space
    float3    V;            ///< Direction to the eye at shading hit
    float3    N;            ///< Shading normal at shading hit
    float3    T;            ///< Shading tangent at shading hit
    float3    B;            ///< Shading bitangent at shading hit
    float2    uv;           ///< Texture mapping coordinates
    float     NdotV;        // Unclamped, can be negative

    // Pre-loaded texture data
    float3 diffuse;
    float  opacity;
    float3 specular;
    float  linearRoughness; // This is the original roughness, before re-mapping. It is required for the Disney diffuse term
    float  roughness;       // This is the re-mapped roughness value, which should be used for GGX computations
    float3 emissive;
    float4 occlusion;
    float3 lightMap;
    float2 height;
    float IoR;
    bool doubleSidedMaterial;
};

#endif  // _FALCOR_SHADER_COMMON_H_