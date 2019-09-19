/***************************************************************************
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

/** Helper functions for generating the G-buffer.
*/

#include "GBufferParams.h"

import Utils.Math.MathHelpers;
__exported import Experimental.Scene.Material.MaterialHelpers;

struct GBuffer
{
    float4 posW;
    float4 normW;
    float4 bitangentW;
    float4 texC;
    float4 diffuseOpacity;
    float4 specRough;
    float4 emissive;
    float4 matlExtra;
};

shared cbuffer PerFrameCB
{
    GBufferParams   gParams;
};

static const uint kInvalidIndex = 0xffffffff;

/** Helper function to store G-buffer channels.
*/
GBuffer storeGBufferOutput(ShadingData sd)
{
    GBuffer gbuf;

    // Check that tangent space exists, otherwise create one based on the normal.
    // Note that this check also catches NaNs, should they occur.
    float3 bitangent = dot(sd.B, sd.B) > 0.f ? sd.B : perp_stark(sd.N);

    gbuf.posW           = float4(sd.posW, 1.f);
    gbuf.normW          = float4(sd.N, 0.f);
    gbuf.bitangentW     = float4(bitangent, 0.f);
    gbuf.texC           = float4(sd.uv, 0.f, 0.f);

    MaterialParams matParams = getMaterialParams(sd);
    gbuf.diffuseOpacity = matParams.diffuseOpacity;
    gbuf.specRough      = matParams.specularRoughness;
    gbuf.emissive       = matParams.emissive;
    gbuf.matlExtra      = matParams.extraParams;

    return gbuf;
}
