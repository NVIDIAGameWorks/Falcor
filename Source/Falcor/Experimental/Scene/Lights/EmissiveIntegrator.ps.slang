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

/** Raster program that integrates each textured emissive triangle.
    The host is expected to set _VIEWPORT_DIMENSION.
*/
import Scene.Scene;
import Experimental.Scene.Lights.LightCollection;

// Setup NvApi. We need this for fp32 atomics.
#define NV_SHADER_EXTN_SLOT u63
#define NV_SHADER_EXTN_REGISTER_SPACE space0
#include "NVAPI/nvHLSLExtns.h"

// Check that defines are set.
#ifndef _VIEWPORT_DIM
#error _VIEWPORT_DIM is not defined
#endif

ParameterBlock<LightCollection> gLightCollection;

RWByteAddressBuffer gTexelSum;          ///< Sum over texels (RGB) + number of texels (A) in RGBA32Float format. Using raw buffer for fp32 atomics compatibility.
SamplerState        gPointSampler;      ///< Sampler for fetching individual texels without filtering.

struct VsOut
{
    float2 uv                       : TexCoord;
    nointerpolation uint materialID : MaterialID;
    float4 posH                     : SV_Position;
};


/** Vertex shader.
    We place textured emissive triangles in texture space scaled so that we get
    one pixel shader execution per texel.
    Non-textured emissive triangles are culled.
*/
VsOut vsMain(uint vtxIdx : SV_VertexID)
{
    // Fetch light index
    const uint triIdx = vtxIdx / 3;
    const EmissiveTriangle tri = gLightCollection.getTriangle(triIdx);

    VsOut vsOut;
    vsOut.uv = float2(0, 0);
    vsOut.posH = float4(0, 0, 0, 1);
    vsOut.materialID = tri.materialID;

    // Check if triangle is textured.
    if (gScene.isEmissiveTextured(tri.materialID))
    {
        // TODO: We could do these computations in a geometry shader once per triangle,
        // instead of duplicated per-vertex. Which is faster? This code is not performance critical.

        // Get triangle's bounding box in texture coordinates.
        const float2 uvMin = min(min(tri.texCoords[0], tri.texCoords[1]), tri.texCoords[2]);
        const float2 uvMax = max(max(tri.texCoords[0], tri.texCoords[1]), tri.texCoords[2]);

        // Pass on texture coordinate unmodified for pixel to use.
        vsOut.uv = tri.texCoords[vtxIdx % 3];

        // Query texture dimensions.
        float width = 0, height = 0, mips = 0;
        gScene.materialResources[tri.materialID].emissive.GetDimensions(0, width, height, mips);

        // Compute raster position in texture space.
        // We do this by offsetting UV so that it's always positive, scaled by texture dimension.
        float2 offset = floor(uvMin);
        float2 texelPos = (vsOut.uv - offset) * float2(width, height);  // Vertex pos in texels

        // TODO: Check if triangle is too large for viewport.
        // We need to detect that and handle it with a fallback mechanism (print error for now).

        // Convert to 2DH normalized device coordinates.
        float2 ndcPos = texelPos * (2.f / _VIEWPORT_DIM) - 1.f;

        vsOut.posH = float4(ndcPos, 0.f, 1.f);      // Place triangle at w = 1.
    }

    // If triangle isn't textured, all three vertices are at (0,0) => culled degenerate triangle.
    return vsOut;
}

/** Pixel shader.
    This is executed once per texel that the emissive triangle covers.
*/
void psMain(VsOut vsIn, uint triIdx : SV_PrimitiveID)
{
    // TODO: Use conservative rasterization and compute analytic coverage in pixel shader.
    // With standard rasterization, we only get approximate integrals and it's possible to entire
    // miss small triangles. This leads to bias unless we explicitly set a small non-zero probability elsewhere.
    // We could alternatively use MSAA to reduce errors, but we still have the problem with missed primitives.

    // Fetch texel using nearest-neighbor sampling.
    float3 color = gScene.materialResources[vsIn.materialID].emissive.SampleLevel(gPointSampler, vsIn.uv, 0.f).rgb;  // Sampler at mip 0

    // Compute weighted color. TODO: For now assume full coverage (= 1.0 in weight).
    float weight = 1.f;
    color *= weight;

    // Atomically accumulate texel color and sample count.
    // TODO: Coalesce atomics across warp before global atomic to increase perf.
    uint address = triIdx * 16;
    NvInterlockedAddFp32(gTexelSum, address + 0, color.r);  // Sum colors in fp32 RGB
    NvInterlockedAddFp32(gTexelSum, address + 4, color.g);
    NvInterlockedAddFp32(gTexelSum, address + 8, color.b);
    NvInterlockedAddFp32(gTexelSum, address + 12, weight);  // Sum weights in fp32
}
