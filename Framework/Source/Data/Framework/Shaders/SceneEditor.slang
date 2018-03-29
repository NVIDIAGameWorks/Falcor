/***************************************************************************
# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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
__import ShaderCommon;
__import DefaultVS;

struct EditorVSOut
{
    VertexOut vOut;

#ifdef PICKING
    uint drawID : DRAW_ID;
#endif

#ifdef CULL_REAR_SECTION
    float3 toVertex : VERTEX_DIR;
#endif
};

struct DebugDrawVSIn
{
    float3 position : POSITION;
    float3 color : COLOR;
};

struct DebugDrawVSOut
{
    float4 position : SV_POSITION;
    float3 color : COLOR;
};

DebugDrawVSOut debugDrawVs(DebugDrawVSIn vIn)
{
    DebugDrawVSOut vOut;

    vOut.position =  mul(float4(vIn.position, 1), gCamera.viewProjMat);
    vOut.color = vIn.color;

    return vOut;
}

EditorVSOut editorVs(VertexIn vIn)
{
    EditorVSOut vOut;
    vOut.vOut = defaultVS(vIn);

#ifdef PICKING
    vOut.drawID = gDrawId[vIn.instanceID];
#endif

#ifdef CULL_REAR_SECTION
    // Get instance origin
    float4x4 worldMtx = getWorldMat(vIn);
    float3 instancePos = float3(worldMtx[3][0], worldMtx[3][1], worldMtx[3][2]);

    // Direction to vertex
    vOut.toVertex = normalize(vOut.vOut.posW - instancePos);
#endif

    return vOut;
}

float4 debugDrawPs(DebugDrawVSOut vOut) : SV_TARGET
{
    return float4(vOut.color, 1);
}

#ifndef PICKING
cbuffer ConstColorCB : register(b0)
{
    float3 gColor;
};
#endif

// PS Output
#ifdef PICKING
#define PSOut uint
#else
#define PSOut float4
#endif

PSOut editorPs(EditorVSOut vOut) : SV_TARGET
{
    float3 toCamera = normalize(gCamera.posW - vOut.vOut.posW);

#ifdef CULL_REAR_SECTION
if (dot(toCamera, vOut.toVertex) < -0.1)
{
    discard;
}
#endif

#ifdef PICKING
return vOut.drawID;
#else
#ifdef SHADING
float shading = lerp(0.2f, 1.0f, dot(toCamera, normalize(vOut.vOut.normalW)));
return float4(gColor * shading, 1);
#else
return float4(gColor, 1);
#endif
#endif
}

