/***************************************************************************
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
#include "HostDeviceData.h"
import Helpers;

/** Calculate screen-space motion vector.
    \param[in] pixelCrd Sample in current frame expressed in pixel coordinates with origin in the top-left corner.
    \param[in] prevPosH Sample in previous frame expressed in homogeneous clip space coordinates. Note that the definition differs between D3D12 and Vulkan.
    \param[in] renderTargetDim Render target dimension in pixels.
    \return Motion vector pointing from current to previous position expressed in sceen space [0,1] with origin in the top-left corner.
*/
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
    float     NdotV;        // Unclamped, can be negative.

    // Primitive data
    float3    faceN;        ///< Face normal in world space, always on the front-facing side.
    bool      frontFacing;  ///< True if primitive seen from the front-facing side.
    bool      doubleSided;  ///< Material double-sided flag, if false only shade front face.

    // Pre-loaded texture data
    float3 diffuse;
    float  opacity;
    float3 specular;
    float  linearRoughness;  // This is the original roughness, before re-mapping. It is required for the Disney diffuse term
    float  ggxAlpha;   // This is the re-mapped roughness value, which should be used for GGX computations. It equals `roughness^2`
    float3 emissive;
    float4 occlusion;
    float2 height;
    float IoR;
    float metallic; // If the shading model is metal/rough, this is the metallic paramter. Otherwise, it will be 0

    // Utility functions

    /** Computes new ray origin based on the hit point to avoid self-intersection.
        The method is described in Ray Tracing Gems, Chapter 6, "A Fast and Robust
        Method for Avoiding Self-Intersection" by Carsten Wächter and Nikolaus Binder.
        \param[in] viewside True if the origin should be on the view side (reflection) or false otherwise (transmission).
        \return Ray origin of the new ray.
    */
    float3 computeNewRayOrigin(bool viewside = true)
    {
        return computeRayOrigin(posW, (frontFacing == viewside) ? faceN : -faceN);
    }
};

#define interfacify(a) typedef a I##a
interfacify(Texture1D);
interfacify(Texture1DArray);
interfacify(Texture2D);
interfacify(Texture2DArray);
interfacify(Texture3D);
interfacify(TextureCube);
interfacify(TextureCubeArray);

interfacify(RWTexture1D);
interfacify(RWTexture1DArray);
interfacify(RWTexture2D);
interfacify(RWTexture2DArray);
interfacify(RWTexture3D);
#undef interfacify
