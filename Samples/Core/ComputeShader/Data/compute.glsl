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
#version 450

layout(set = 0, binding = 0) uniform texture2D gInput;
layout(set = 0, binding = 1, rgba32f) uniform image2D gOutput;
layout(set = 0, binding = 2) uniform sampler gSampler;
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

shared vec4 colors[16][16];
shared vec4 pixelated;

void main()
{
    // Calculate the start position of the block
    ivec2 resDim = imageSize(gOutput);

    uvec2 posStart = gl_WorkGroupID.xy * 16;
    uvec2 crd = posStart + gl_LocalInvocationID.xy;

    // Fetch all of the data into the shared local memory
    colors[gl_LocalInvocationID.x][gl_LocalInvocationID.y] = texelFetch(sampler2D(gInput, gSampler), ivec2(crd), 0);

#ifdef _PIXELATE
    groupMemoryBarrier();
    barrier();

    if(gl_LocalInvocationID == uvec3(0,0,0))
    {
        pixelated = vec4(0,0,0,0);
        for(int i = 0 ; i < 16 ; i++)
        {
            for(int j = 0 ; j < 16 ; j++)
            {
                pixelated += colors[i][j];
            }
        }
        pixelated /= 16*16;
    }

    groupMemoryBarrier();
    barrier();

    imageStore(gOutput, ivec2(crd), pixelated.bgra);
#else
    imageStore(gOutput, ivec2(crd), colors[gl_LocalInvocationID.x][gl_LocalInvocationID.y].bgra);
#endif
}