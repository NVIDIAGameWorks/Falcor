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

//  Simple Pixel Shader.
struct VsOut
{
    float4 sv_position_out : SV_POSITION;
};

cbuffer PerFrameCB : register(b0)
{

    float4 color_value0;

#ifdef _USE_COLOR_INPUT1
    float4 color_value1;
#endif

#ifdef _USE_COLOR_INPUT2
    float4 color_value2;
#endif

#ifdef _USE_COLOR_INPUT3
    float4 color_value3;
#endif

#ifdef _USE_COLOR_INPUT4
    float4 color_value4;
#endif

#ifdef _USE_COLOR_INPUT5
    float4 color_value5;
#endif

#ifdef _USE_COLOR_INPUT6
    float4 color_value6;
#endif

#ifdef _USE_COLOR_INPUT7
    float4 color_value7;
#endif

};

struct PsOut
{

    float4 color_out0 : SV_TARGET0;

#ifdef _USE_SV_1
    float4 color_out1 : SV_TARGET1;
#endif

#ifdef _USE_SV_2
    float4 color_out2 : SV_TARGET2;
#endif

#ifdef _USE_SV_3
    float4 color_out3 : SV_TARGET3;
#endif

#ifdef _USE_SV_4
    float4 color_out4 : SV_TARGET4;
#endif

#ifdef _USE_SV_5
    float4 color_out5 : SV_TARGET5;
#endif

#ifdef _USE_SV_6
    float4 color_out6 : SV_TARGET6;
#endif

#ifdef _USE_SV_7
    float4 color_out7 : SV_TARGET7;
#endif

#ifdef _USE_DEPTH
    float depth_value : SV_DEPTH;
#endif

};

//  
PsOut main(VsOut vOut)
{
    PsOut psOut;

    float4 colorVal = color_value0;

#ifdef _USE_COLOR_INPUT1
    colorVal = colorVal + color_value1;
#endif

#ifdef _USE_COLOR_INPUT2
    colorVal = colorVal + color_value2;
#endif

#ifdef _USE_COLOR_INPUT3
    colorVal = colorVal + color_value3;
#endif

#ifdef _USE_COLOR_INPUT4
    colorVal = colorVal + color_value4;
#endif

#ifdef _USE_COLOR_INPUT5
    colorVal = colorVal + color_value5;
#endif

#ifdef _USE_COLOR_INPUT6
    colorVal = colorVal + color_value6;
#endif

#ifdef _USE_COLOR_INPUT7
    colorVal = colorVal + color_value7;
#endif

    psOut.color_out0 = colorVal;

//  Default Outputs.

#ifdef _USE_SV_1
    psOut.color_out1 = color_value1;
#endif

#ifdef _USE_SV_2
    psOut.color_out2 = color_value2;
#endif

#ifdef _USE_SV_3
    psOut.color_out3 = color_value3;
#endif

#ifdef _USE_SV_4
    psOut.color_out4 = color_value4;
#endif

#ifdef _USE_SV_5
    psOut.color_out5 = color_value5;
#endif

#ifdef _USE_SV_6
    psOut.color_out6 = color_value6;
#endif

#ifdef _USE_SV_7
    psOut.color_out7 = color_value7;
#endif


#ifdef _USE_DEPTH
    psOut.depth_value = depth_value;
#endif


    return psOut;
}
