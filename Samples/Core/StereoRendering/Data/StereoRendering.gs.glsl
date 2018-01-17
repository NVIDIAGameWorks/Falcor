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
__import DefaultVS;
__import ShaderCommon;

layout(triangles) in;
layout(triangle_strip, max_vertices = 6) out;

in VertexOut vsOut[];
out VertexOut gsOut;

void main()
{
    // Left Eye
    gl_Layer = 0;
    for (int i = 0; i < 3; i++)
    {
        gsOut.normalW = vsOut[i].normalW;
        gsOut.bitangentW = vsOut[i].bitangentW;
        gsOut.texC = vsOut[i].texC;
        gsOut.posW = vsOut[i].posW;
        gsOut.colorV = vsOut[i].colorV;

        vec4 posW = vec4(vsOut[i].posW, 1.0f);
        gsOut.prevPosH = gCam.prevViewProjMat * posW;
        gl_Position = gCam.viewProjMat * posW;

        EmitVertex();
    }
    EndPrimitive();

    // Right Eye
    gl_Layer = 1;
    for (int i = 0; i < 3; i++)
    {
        gsOut.normalW = vsOut[i].normalW;
        gsOut.bitangentW = vsOut[i].bitangentW;
        gsOut.texC = vsOut[i].texC;
        gsOut.posW = vsOut[i].posW;
        gsOut.colorV = vsOut[i].colorV;

        vec4 posW = vec4(vsOut[i].posW, 1.0f);
        gsOut.prevPosH = gCam.rightEyePrevViewProjMat * posW;
        gl_Position = gCam.rightEyeViewProjMat * posW;

        EmitVertex();
    }
    EndPrimitive();
}
