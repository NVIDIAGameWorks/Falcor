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

/** Helpers for mapping scalar values to RGB color for visualization purposes.

    The input is clamped to [0,1] and mapped to a continuous color range.
    The colormapJet() function matches the output of Matlab's 'jet' color map.
*/

/** Maps scalar value to grayscale RGB value.
    Values outside the [0,1] range are clamped.
    \param[in] x Scalar value.
    \return float3 Continuous RGB color in range [0,1].
*/
float3 colormapGray(float x)
{
    float v = saturate(x);
    return float3(v, v, v);
}

/** Maps scalar value to the commonly used 'jet' color map in Matlab.
    Values outside the [0,1] range are clamped to the end points.
    \param[in] x Scalar value.
    \return float3 Continuous RGB color in range [0,1].
*/
float3 colormapJet(float x)
{
    // Code written in Matlab to match jet.m output:
    //x = max(0, min(1, x));
    //R = 1.5 - abs(x - 0.75) * 4;
    //G = 1.5 - abs(x - 0.50) * 4;
    //B = 1.5 - abs(x - 0.25) * 4;
    //y = [R G B];
    //y = max(0, min(1, y));
    return saturate(1.5 - abs(4 * clamp(x, 0, 1) - float3(3, 2, 1)));
}
