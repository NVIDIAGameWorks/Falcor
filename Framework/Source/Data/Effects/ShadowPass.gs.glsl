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
#include "VertexAttrib.h"
__import ShaderCommon;
__import Effects.CascadedShadowMap;

layout(set = 0, binding = 3) uniform PerLightCB
{
    CsmData gCsmData;
};

layout(location = 0) out vec2 outputData_texC;

layout(location = 0) in vec2 input_texC[3];
layout(location = 1) in vec4 input_pos[3];


layout(invocations = _CASCADE_COUNT) in;
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

void main()
{
    int InstanceID = gl_InvocationID;

    // void main(triangle ShadowPassVSOut input[3], uint InstanceID : SV_GSInstanceID, inout TriangleStream<ShadowPassPSIn> outStream)

    for(int i = 0 ; i < 3 ; i++)
    {
        gl_Position = gCsmData.globalMat * input_pos[i];
        gl_Position.xyz /= input_pos[i].w;
        gl_Position.xyz *= gCsmData.cascadeScale[InstanceID].xyz;
        gl_Position.xyz += gCsmData.cascadeOffset[InstanceID].xyz;

        outputData_texC = input_texC[i];
        gl_Layer = InstanceID;

        EmitVertex();
    }

    EndPrimitive();
}
