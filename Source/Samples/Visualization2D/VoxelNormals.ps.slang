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
import Utils.Math.MathHelpers;
import Utils.SDF.SDF2DDraw;
import Utils.SDF.SDF2DPrimitives;

cbuffer Visual2DCB
{
    float2  iResolution;
    float2  iMousePosition;
    float   iGlobalTime;
};

cbuffer VoxelNormalsCB
{
    int     iShowNormalField;
    int     iShowBoxes;
    int     iShowBoxAroundPoint;
    int     iShowBoxDiagonals;
    int     iShowBorderLines;
};

static const float side = 0.1;
static const float4 lightgray = float4(0.7, 0.7, 0.7, 1);
static const float4 red = float4(0.9, 0, 0, 1);
static const float4 cyan = float4(0.0, 0.7, 1.0, 1);
static const float4 green = float4(0, 0.7, 0, 1);
static const float4 blue = float4(0, 0, 1, 1);
static const float4 black = float4(0.0, 0.0, 0.0, 1.0);
static const float2 start = float2(0.025, 0.025);
static const int2 numVoxels = int2(4, 4);

static const float sdf[5][5] = {       // Could not use numVoxels.x+1, numVoxels.y+1 here, for some reason...
    {+0.50, +0.30, +0.10, +0.10, +0.20},
    {-0.80, +0.50, -0.50, -0.10, +0.20},
    {-0.50, +0.20, +0.40, +0.20, +1.00},
    {+0.70, -0.50, -0.40, +0.50, +1.00},
    {-1.00, +0.45, -0.50, -0.10, -1.00} };

static float2 P[5][5];
static float2 X[5][5];  // P warped into X points

void evalBilinear(float2 barys, float s00, float s10, float s01, float s11, out float sdf)
{
    sdf = lerp(lerp(s00, s10, barys.x), lerp(s01, s11, barys.x), barys.y);
}

void evalBilinear(float2 barys, float2 p00, float2 p10, float2 p01, float2 p11, out float2 point)
{
    point.x = lerp(lerp(p00.x, p10.x, barys.x), lerp(p01.x, p11.x, barys.x), barys.y);
    point.y = lerp(lerp(p00.y, p10.y, barys.x), lerp(p01.y, p11.y, barys.x), barys.y);
}


bool isInsideTri(float2 coords, float2 _p00, float2 _p10, float2 _p01, out float2 barys)
{
    float2 e1 = _p10 - _p00;
    float2 e2 = _p01 - _p00;
    coords -= _p00;
    float denom = cross(float3(e1.x, e1.y, 0), float3(e2.x, e2.y, 0)).z;
    barys.x = cross(float3(coords.x, coords.y, 0.0), float3(e2.x, e2.y, 0.0)).z / denom;
    barys.y = cross(float3(e1.x, e1.y, 0.0), float3(coords.x, coords.y, 0.0)).z / denom;
    return barys.x >= 0.0 && barys.y >= 0.0 && barys.x + barys.y <= 1.0;
}

float2 normalVector(float2 uv, float _s00, float _s10, float _s01, float _s11)
{
    float x = _s10 - _s00 + uv.y * (_s11 - _s10 - (_s01 - _s00));
    float y = _s01 - _s00 + uv.x * (_s11 - _s01 - (_s10 - _s00));
    return normalize(float2(x, y));
}

float2 computeSimpleNormal(float2 pos)
{
    float2 c = (pos - P[0][0]) / side;
    int2 indexS = int2(floor(c));
    int2 indexE = indexS + 1;
    indexS = clamp(indexS, int2(0), int2(numVoxels.x));
    indexE = clamp(indexE, int2(0), int2(numVoxels.x));

    c = (pos - P[indexS.x][indexS.y]) / side; // in [0,1]
    float2 n = normalVector(c, sdf[indexS.x][indexS.y], sdf[indexE.x][indexS.y], sdf[indexS.x][indexE.y], sdf[indexE.x][indexE.y]);
    return n;
}

float2 convertToDualVoxelSpaceIndex(float2 pos, out int2 voxelIndex)    // voxelIndex can go outside the grid -- it is up to the user to clamp later, when needed.
{
    float2 centerFirstVoxel = P[0][0] + float2(side * 0.5);
    float2 delta = pos - centerFirstVoxel;
    voxelIndex = int2(floor(delta / side));

    float2 voxelMin = centerFirstVoxel + float2(voxelIndex) * side;
    float2 barys = (pos - voxelMin) / side; // In [0, 1]^3.
    return barys;
}

void computeStandardPositionsAndNormals(float2 position, out float2 normal, out float2 warpedPosition, out float2 warpedNormal)
{
    float2 barys;
    bool inside = false;
    normal = float2(0, 0);
    warpedPosition = float2(0, 0);
    warpedNormal = float2(0, 0);

    int x = int((position.x - P[0][0].x) / side);
    int y = int((position.y - P[0][0].y) / side);
    if (isInsideTri(position, P[x][y], P[x + 1][y], P[x][y + 1], barys))
    {
        inside = true;
        warpedPosition = X[x][y] + barys.x * (X[x + 1][y] - X[x][y]) + barys.y * (X[x][y + 1] - X[x][y]);
        normal = computeSimpleNormal(position);
        warpedNormal = normal.x * (X[x + 1][y] - X[x][y]) + normal.y * (X[x][y + 1] - X[x][y]);
    }
    else if (isInsideTri(position, P[x + 1][y + 1], P[x][y + 1], P[x + 1][y], barys))
    {
        inside = true;
        warpedPosition = X[x + 1][y + 1] + barys.x * (X[x][y + 1] - X[x + 1][y + 1]) + barys.y * (X[x + 1][y] - X[x + 1][y + 1]);
        normal = computeSimpleNormal(position);
        warpedNormal = normal.x * (X[x + 1][y + 1] - X[x][y + 1]) + normal.y * (X[x + 1][y + 1] - X[x + 1][y]);
    }
}


void drawBox(float2 coords, int voxelX, int voxelY, float2 _p00, float2 _p10, float2 _p01, float2 _p11, float _s00, float _s10, float _s01, float _s11, float w, float blendRadius, bool isWarped, inout float4 color)
{
    float sdfValue;
    float2 barys;
    bool inside = false;
    float2 point;

    if (isInsideTri(coords, _p00, _p10, _p01, barys))
    {
        inside = true;
        point = P[voxelX][voxelY] + barys.x * (P[voxelX + 1][voxelY] - P[voxelX][voxelY]) + barys.y * (P[voxelX][voxelY + 1] - P[voxelX][voxelY]);
    }
    else if (isInsideTri(coords, _p11, _p01, _p10, barys))
    {
        inside = true;
        point = P[voxelX + 1][voxelY + 1] + barys.x * (P[voxelX][voxelY + 1] - P[voxelX + 1][voxelY + 1]) + barys.y * (P[voxelX + 1][voxelY] - P[voxelX + 1][voxelY + 1]);
        barys.x = 1.0 - barys.x;
        barys.y = 1.0 - barys.y;
    }

    float2 normal = float2(0, 0), wp, warpedNormal = float2(0, 0);
    computeStandardPositionsAndNormals(point, normal, wp, warpedNormal);
    normal = (normalize(normal) + 1.0) * 0.5;
    warpedNormal = (normalize(warpedNormal) + 1.0) * 0.5;

    if (inside)
    {
        if (iShowNormalField != 0)
        {
            color.xy = isWarped ? warpedNormal.xy : normal.xy;
            color.z = 0;
        }
        evalBilinear(barys, _s00, _s10, _s01, _s11, sdfValue);
        if (sdfValue < 0.0)
        {
            float alpha = 0.6;
            color = (1 - alpha) * color + alpha * float4(0.2, 0.2, 0.9, 1.0);
        }
    }
    if (iShowBoxDiagonals != 0)
    {
        color = sdfDraw(SDF2DRoundedLine(_p10, _p01, w * 0.5), coords, float4(0, 0, 0, 0.25), color, blendRadius);
    }
    if (iShowBoxes != 0)
    {
        color = sdfDraw(SDF2DRoundedLine(_p00, _p01, w ), coords, black, color, blendRadius);
        color = sdfDraw(SDF2DRoundedLine(_p10, _p11, w), coords, black, color, blendRadius);
        color = sdfDraw(SDF2DRoundedLine(_p00, _p10, w), coords, black, color, blendRadius);
        color = sdfDraw(SDF2DRoundedLine(_p01, _p11, w), coords, black, color, blendRadius);
    }
}

float4 main(float2 texC : TEXCOORD) : SV_TARGET
{
    const float blendRadius = 1.0 / min(iResolution.x, iResolution.y);
    const float circleRadius = 0.005;
    const float lineSize = 0.0005;

    float4 color = float4(0.95, 0.95, 0.95, 1.0);                       // Init with background color
    float2 pixelCoords = float2(texC.x, 1.0 - texC.y) * iResolution;    // X is right, Y is up.

    float2 mouseCoords = float2(iMousePosition.x, iResolution.y - 1 - iMousePosition.y) / iResolution.x;
    float2 coords = pixelCoords / iResolution.x;

    for (int y = 0; y < numVoxels.y + 1; y++)
    {
        for (int x = 0; x < numVoxels.x + 1; x++)
        {
            P[x][y] = start + float2(side * x, side * y);
        }
    }
    float r = side * 0.1;
    float t = 5.0 * iGlobalTime;
    for (int y = 0; y < numVoxels.y + 1; y++)
    {
        for (int x = 0; x < numVoxels.x + 1; x++)
        {
            float d1 = ((x + 2) * (y + 3));
            float d2 = ((x + 4) * (y + 5));
            float f1 = fmod(d1, 0.71);
            float f2 = fmod(d2, 0.67);
            X[x][y] = P[x][y] + float2(numVoxels.x * side + 0.1, 0.0) + r * float2(sin(t * f1 + d1), cos(t * f2 + d2));
        }
    }

    for (int y = 0; y < numVoxels.y; y++)
    {
        for (int x = 0; x < numVoxels.x; x++)
        {
            drawBox(coords, x, y, P[x][y], P[x + 1][y], P[x][y + 1], P[x + 1][y + 1], sdf[x][y], sdf[x + 1][y], sdf[x][y + 1], sdf[x + 1][y + 1], lineSize * 2, blendRadius, false, color);
            drawBox(coords, x, y, X[x][y], X[x + 1][y], X[x][y + 1], X[x + 1][y + 1], sdf[x][y], sdf[x + 1][y], sdf[x][y + 1], sdf[x + 1][y + 1], lineSize * 2, blendRadius, true, color);
        }
    }

    // Compute normals at the mouse position.
    float2 warpedPosition, warpedStandardNormal, standardNormal;
    computeStandardPositionsAndNormals(mouseCoords, standardNormal, warpedPosition, warpedStandardNormal);
    color = sdfDraw(SDF2DVector(mouseCoords, mouseCoords + standardNormal * side, lineSize * 2.0, side * 0.075), coords, red, color, blendRadius);
    if (warpedStandardNormal.x != 0.0 && warpedStandardNormal.y != 0.0)
        color = sdfDraw(SDF2DVector(warpedPosition, warpedPosition + normalize(warpedStandardNormal) * side, lineSize * 2.0, side * 0.075), coords, red, color, blendRadius);
    if (iShowBorderLines != 0)
    {
        color = sdfDraw(SDF2DRoundedLine(P[0][0] + float2(side * 0.5, 0.0), P[0][numVoxels.y] + float2(side * 0.5, 0.0), lineSize * 0.25), coords, lightgray, color, blendRadius);
        color = sdfDraw(SDF2DRoundedLine(P[numVoxels.x - 1][0] + float2(side * 0.5, 0.0), P[numVoxels.x - 1][numVoxels.y] + float2(side * 0.5, 0.0), lineSize * 0.25), coords, lightgray, color, blendRadius);
        color = sdfDraw(SDF2DRoundedLine(P[0][0] + float2(0.0, side * 0.5), P[numVoxels.x][0] + float2(0.0, side * 0.5), lineSize * 0.25), coords, lightgray, color, blendRadius);
        color = sdfDraw(SDF2DRoundedLine(P[0][numVoxels.y - 1] + float2(0.0, side * 0.5), P[numVoxels.x][numVoxels.y - 1] + float2(0.0, side * 0.5), lineSize * 0.25), coords, lightgray, color, blendRadius);
    }
    if (iShowBoxAroundPoint != 0)
    {
        const float2 d = float2(-side, +side) * 0.5;
        color = sdfDraw(SDF2DRoundedLine(mouseCoords + d.xx, mouseCoords + d.yx, lineSize * 0.25), coords, black, color, blendRadius * 0.5);
        color = sdfDraw(SDF2DRoundedLine(mouseCoords + d.xy, mouseCoords + d.yy, lineSize * 0.25), coords, black, color, blendRadius * 0.5);
        color = sdfDraw(SDF2DRoundedLine(mouseCoords + d.xx, mouseCoords + d.xy, lineSize * 0.25), coords, black, color, blendRadius * 0.5);
        color = sdfDraw(SDF2DRoundedLine(mouseCoords + d.yx, mouseCoords + d.yy, lineSize * 0.25), coords, black, color, blendRadius * 0.5);
    }

    color = sdfDraw(SDF2DCircle(circleRadius), coords, float2(1.0), 0.0, mouseCoords, black, color, blendRadius);
    color = sdfDraw(SDF2DCircle(circleRadius), coords, float2(1.0), 0.0, warpedPosition, black, color, blendRadius);

    return color;
}
