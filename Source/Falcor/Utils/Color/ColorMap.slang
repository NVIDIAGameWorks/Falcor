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

/** Helpers for mapping scalar values to RGB color for visualization purposes.

    The input is clamped to [0,1] and mapped to a continuous color range.
    The colormapJet() function matches the output of Matlab's 'jet' color map.

    The colormapViridis(), colormapPlasma(), colormapMagma() and colormapInferno() are
    based on https://www.shadertoy.com/view/WlfXRN. They are fitted from data found on
    https://github.com/BIDS/colormap/blob/master/colormaps.py.
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
    return saturate(1.5 - abs(4 * saturate(x) - float3(3, 2, 1)));
}

/** Maps scalar value to the 'viridis' color map.
    Values outside the [0,1] range are clamped to the end points.
    \param[in] x Scalar value.
    \return float3 Continuous RGB color in range [0,1].
*/
float3 colormapViridis(float x)
{
    const float3 c0 = float3(0.2777273272234177, 0.005407344544966578, 0.3340998053353061);
    const float3 c1 = float3(0.1050930431085774, 1.404613529898575, 1.384590162594685);
    const float3 c2 = float3(-0.3308618287255563, 0.214847559468213, 0.09509516302823659);
    const float3 c3 = float3(-4.634230498983486, -5.799100973351585, -19.33244095627987);
    const float3 c4 = float3(6.228269936347081, 14.17993336680509, 56.69055260068105);
    const float3 c5 = float3(4.776384997670288, -13.74514537774601, -65.35303263337234);
    const float3 c6 = float3(-5.435455855934631, 4.645852612178535, 26.3124352495832);
    x = saturate(x);
    return saturate(c0+x*(c1+x*(c2+x*(c3+x*(c4+x*(c5+x*c6))))));
}

/** Maps scalar value to the 'plasma' color map.
    Values outside the [0,1] range are clamped to the end points.
    \param[in] x Scalar value.
    \return float3 Continuous RGB color in range [0,1].
*/
float3 colormapPlasma(float x)
{
    const float3 c0 = float3(0.05873234392399702, 0.02333670892565664, 0.5433401826748754);
    const float3 c1 = float3(2.176514634195958, 0.2383834171260182, 0.7539604599784036);
    const float3 c2 = float3(-2.689460476458034, -7.455851135738909, 3.110799939717086);
    const float3 c3 = float3(6.130348345893603, 42.3461881477227, -28.51885465332158);
    const float3 c4 = float3(-11.10743619062271, -82.66631109428045, 60.13984767418263);
    const float3 c5 = float3(10.02306557647065, 71.41361770095349, -54.07218655560067);
    const float3 c6 = float3(-3.658713842777788, -22.93153465461149, 18.19190778539828);
    x = saturate(x);
    return saturate(c0+x*(c1+x*(c2+x*(c3+x*(c4+x*(c5+x*c6))))));
}

/** Maps scalar value to the 'magma' color map.
    Values outside the [0,1] range are clamped to the end points.
    \param[in] x Scalar value.
    \return float3 Continuous RGB color in range [0,1].
*/
float3 colormapMagma(float x)
{
    const float3 c0 = float3(-0.002136485053939582, -0.000749655052795221, -0.005386127855323933);
    const float3 c1 = float3(0.2516605407371642, 0.6775232436837668, 2.494026599312351);
    const float3 c2 = float3(8.353717279216625, -3.577719514958484, 0.3144679030132573);
    const float3 c3 = float3(-27.66873308576866, 14.26473078096533, -13.64921318813922);
    const float3 c4 = float3(52.17613981234068, -27.94360607168351, 12.94416944238394);
    const float3 c5 = float3(-50.76852536473588, 29.04658282127291, 4.23415299384598);
    const float3 c6 = float3(18.65570506591883, -11.48977351997711, -5.601961508734096);
    x = saturate(x);
    return saturate(c0+x*(c1+x*(c2+x*(c3+x*(c4+x*(c5+x*c6))))));
}

/** Maps scalar value to the 'inferno' color map.
    Values outside the [0,1] range are clamped to the end points.
    \param[in] x Scalar value.
    \return float3 Continuous RGB color in range [0,1].
*/
float3 colormapInferno(float x)
{
    const float3 c0 = float3(0.0002189403691192265, 0.001651004631001012, -0.01948089843709184);
    const float3 c1 = float3(0.1065134194856116, 0.5639564367884091, 3.932712388889277);
    const float3 c2 = float3(11.60249308247187, -3.972853965665698, -15.9423941062914);
    const float3 c3 = float3(-41.70399613139459, 17.43639888205313, 44.35414519872813);
    const float3 c4 = float3(77.162935699427, -33.40235894210092, -81.80730925738993);
    const float3 c5 = float3(-71.31942824499214, 32.62606426397723, 73.20951985803202);
    const float3 c6 = float3(25.13112622477341, -12.24266895238567, -23.07032500287172);
    x = saturate(x);
    return saturate(c0+x*(c1+x*(c2+x*(c3+x*(c4+x*(c5+x*c6))))));
}
