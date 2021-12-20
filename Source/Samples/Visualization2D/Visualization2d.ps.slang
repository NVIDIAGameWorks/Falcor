/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
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
import Utils.SDF.SDF2DDraw;
import Utils.SDF.SDF2DPrimitives;
import Utils.SDF.SDFOperations;

cbuffer Visual2DCB
{
    float2  iResolution;
    float2  iMousePosition;
    float   iGlobalTime;
};

static const float side = 0.1;
static const float4 gray = float4(0.5, 0.5, 0.5, 1);
static const float4 red = float4(0.9, 0, 0, 1);
static const float4 cyan = float4(0.0, 0.7, 1.0, 1);
static const float4 green = float4(0, 0.7, 0, 1);
static const float4 greenAlpha = float4(0, 0.5, 0, 0.8);
static const float4 blue = float4(0, 0, 1, 1);
static const float4 blueAlpha = float4(0, 0, 1, 0.5);
static const float4 orange = float4(1.0, 0.5, 0.0, 1);
static const float4 black = float4(0.0, 0.0, 0.0, 1.0);


float4 main(float2 texC : TEXCOORD) : SV_TARGET
{
    const float blendRadius = 1.0 / min(iResolution.x, iResolution.y);
    const float circleSize = 0.01;
    const float lineSize = 0.0005;

    float4 color = float4(0.95, 0.95, 0.95, 1.0);                       // Init with background color.
    float2 pixelCoords = float2(texC.x, 1.0 - texC.y) * iResolution;    // X is right, Y is up.

    float2 mouseCoords = float2(iMousePosition.x, iResolution.y - 1 - iMousePosition.y) / iResolution.x;
    float2 coords = pixelCoords / iResolution.x;

    float angle = iGlobalTime ;

    // Test all the simple primitives.
    color = sdfDraw(SDF2DCircle(0.025), coords, float2(1.0), 0.0, float2(0.1, 0.4), black, color, blendRadius);
    color = sdfDraw(SDF2DSquare(), coords, float2(0.05), 22.5 * 3.1415 / 180, float2(0.2, 0.4), black, color, blendRadius);
    color = sdfDraw(SDF2DDiamond(), coords, float2(0.05), 0.0, float2(0.3, 0.4), black, color, blendRadius);
    color = sdfDraw(SDF2DHeart(), coords, float2(0.05), angle, float2(0.4, 0.4), red, color, blendRadius);
    color = sdfDraw(SDF2DChevron(), coords, float2(0.05), 0.0, float2(0.5, 0.4), black, color, blendRadius);
    color = sdfDraw(SDF2DRing(), coords, float2(0.05), 0.0, float2(0.6, 0.4), black, color, blendRadius);
    color = sdfDraw(SDF2DTag(), coords, float2(0.05), angle, float2(0.7, 0.4), black, color, blendRadius);
    color = sdfDraw(SDF2DCross(), coords, float2(0.05), 0.0, float2(0.8, 0.4), black, color, blendRadius);
    color = sdfDraw(SDF2DAsterisk(), coords, float2(0.05), 0.0, float2(0.9, 0.4), black, color, blendRadius);
    color = sdfDraw(SDF2DInfinity(), coords, float2(0.05), 0.0, float2(0.1, 0.3), black, color, blendRadius);
    color = sdfDraw(SDF2DPin(), coords, float2(0.05), 3.1415, float2(0.2, 0.3), black, color, blendRadius);
    color = sdfDraw(SDF2DArrow(), coords, float2(0.05), 0.0, float2(0.3, 0.3), black, color, blendRadius);

    // Render a vector (line with triangle at tip) and a rounded line + a disc where the mouse was clicked.
    color = sdfDraw(SDF2DVector(mouseCoords, mouseCoords + float2(0.1, 0.2), 0.005, 0.03), coords, cyan, color, blendRadius);
    color = sdfDraw(SDF2DRoundedLine(mouseCoords, mouseCoords + float2(-0.1, 0.2), 0.005), coords, blue, color, blendRadius);
    color = sdfDraw(SDF2DCircle(0.008), coords, float2(1.0), 0.0, mouseCoords, black, color, blendRadius);

    // Render a fat rounded line.
    color = sdfDraw(SDF2DRoundedLine(float2(0.8, 0.275), float2(0.9, 0.325), 0.05), coords, greenAlpha, color, blendRadius);

    // Render to boxes with rounded corners.
    color = sdfDraw(SDF2DRoundedBox(float2(0.1, 0.05), 0.025), coords, float2(1.0), 0.0, float2(0.15, 0.2), green, color, blendRadius);
    color = sdfDraw(SDF2DRoundedBox(float2(0.05, 0.03), 0.01), coords, float2(1.0), 0.3, float2(0.15, 0.17), blueAlpha, color, blendRadius);

    // Render an arbitrary triangle with animated vertices.
    float2 center = float2(0.5, 0.25);
    float radius = 0.05;
    float2 p0 = center + float2(0.0, radius) + 0.1 * radius * float2(cos(4.0 * iGlobalTime), sin(2.0 * iGlobalTime));
    float2 p1 = center + float2(+radius * 0.5, -radius * 0.25) + 0.1 * radius * float2(cos(8 * iGlobalTime), sin(6 * iGlobalTime));
    float2 p2 = center + float2(-radius * 0.5, -radius * 0.25) + 0.1 * radius * float2(cos(5 * iGlobalTime), sin(3 * iGlobalTime));
    color = sdfDraw(SDF2DTriangle(p0, p1, p2), coords, gray, color, blendRadius);

    // Do it yourself blending. Smooth union of two discs, then a smooth subtraction of a smaller disc.
    center = float2(0.65, 0.2);
    float d1 = sdfCircle(sdfTransform(coords, float2(1.0), 0.0, center), 0.05);
    float d2 = sdfCircle(sdfTransform(coords, float2(1.0), 0.0, center + 0.09 * float2(cos(0.8 * iGlobalTime), sin(0.6 * iGlobalTime))), 0.02);
    float d3 = sdfCircle(sdfTransform(coords, float2(1.0), 0.0, center + 0.075 * float2(cos(0.5 * iGlobalTime), sin(1.1 * iGlobalTime))), 0.0125);
    float d = sdfSmoothSubtraction(sdfSmoothUnion(d1, d2, 0.1), d3, 0.1f);
    float4 blendcolor = sdfFilled(d, 0.0f, blendRadius * 2.0, orange);
    color =  (1 - blendcolor.a) * color + blendcolor.a * blendcolor;

    return color;
}
