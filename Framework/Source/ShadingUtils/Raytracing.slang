/************************************************************************************************************************************\
|*                                                                                                                                    *|
|*     Copyright © 2017 NVIDIA Corporation.  All rights reserved.                                                                     *|
|*                                                                                                                                    *|
|*  NOTICE TO USER:                                                                                                                   *|
|*                                                                                                                                    *|
|*  This software is subject to NVIDIA ownership rights under U.S. and international Copyright laws.                                  *|
|*                                                                                                                                    *|
|*  This software and the information contained herein are PROPRIETARY and CONFIDENTIAL to NVIDIA                                     *|
|*  and are being provided solely under the terms and conditions of an NVIDIA software license agreement                              *|
|*  and / or non-disclosure agreement.  Otherwise, you have no rights to use or access this software in any manner.                   *|
|*                                                                                                                                    *|
|*  If not covered by the applicable NVIDIA software license agreement:                                                               *|
|*  NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOFTWARE FOR ANY PURPOSE.                                            *|
|*  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.                                                           *|
|*  NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,                                                                     *|
|*  INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.                       *|
|*  IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES,                               *|
|*  OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT,                         *|
|*  NEGLIGENCE OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOURCE CODE.            *|
|*                                                                                                                                    *|
|*  U.S. Government End Users.                                                                                                        *|
|*  This software is a "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT 1995),                                       *|
|*  consisting  of "commercial computer  software"  and "commercial computer software documentation"                                  *|
|*  as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government only as a commercial end item.     *|
|*  Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995),                                          *|
|*  all U.S. Government End Users acquire the software with only those rights set forth herein.                                       *|
|*                                                                                                                                    *|
|*  Any use of this software in individual and commercial software must include,                                                      *|
|*  in the user documentation and internal comments to the code,                                                                      *|
|*  the above Disclaimer (as applicable) and U.S. Government End Users Notice.                                                        *|
|*                                                                                                                                    *|
 \************************************************************************************************************************************/
#ifndef __DXR_COMMON_H__
#define __DXR_COMMON_H__

__import Shading;
__import DefaultVS;

ByteAddressBuffer gIndices      : register(t50);
ByteAddressBuffer gTexCrds      : register(t51);
ByteAddressBuffer gNormals      : register(t52);
ByteAddressBuffer gBitangents   : register(t53);
ByteAddressBuffer gPositions    : register(t54);
ByteAddressBuffer gLightMapUVs  : register(t55);
RaytracingAccelerationStructure gRtScene : register(t56);

struct BuiltinIntersectionAttribs
{
    float2 baryCrd;
};

cbuffer DxrPerFrame : register(b13)
{
    uint hitProgramCount;
};

uint3 getIndices(uint triangleIndex)
{
    uint baseIndex = triangleIndex * 3;
    int address = baseIndex * 4;
    return gIndices.Load3(address);
}

VertexOut getVertexAttributes(uint triangleIndex, float3 barycentrics)
{
    uint3 indices = getIndices(triangleIndex);
    VertexOut v;
    v.texC = 0;
    v.normalW = 0;
    v.bitangentW = 0;
    v.posW = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    v.colorV = 0;
    v.prevPosH = 0;
    v.lightmapC = 0;

    [unroll]
    for (int i = 0; i < 3; i++)
    {
        int address = (indices[i] * 3) * 4;
        v.texC       += asfloat(gTexCrds.Load2(address))     * barycentrics[i];
        v.normalW    += asfloat(gNormals.Load3(address))     * barycentrics[i];
        v.bitangentW += asfloat(gBitangents.Load3(address))  * barycentrics[i];
        v.lightmapC  += asfloat(gLightMapUVs.Load2(address)) * barycentrics[i];
    }
    v.normalW = normalize(v.normalW);
    v.bitangentW = normalize(v.bitangentW);
    return v;
}

VertexOut getVertexAttributes(uint triangleIndex, BuiltinIntersectionAttribs attribs)
{
    float3 barycentrics = float3(1.0 - attribs.baryCrd.x - attribs.baryCrd.y, attribs.baryCrd.x, attribs.baryCrd.y);
    return getVertexAttributes(triangleIndex, barycentrics);
}

float3 getGeoNormal(const float3 e[2])
{
    return normalize(cross(e[0], e[1]));
}

void getTriNormalsAndEdges(uint triangleIndex, out float3 n[3], out float3 e[2])
{
    uint3 indices = getIndices(triangleIndex);

    float3 p[3];
    p[0] = asfloat(gPositions.Load3((indices[0] * 3) * 4));
    p[1] = asfloat(gPositions.Load3((indices[1] * 3) * 4));
    p[2] = asfloat(gPositions.Load3((indices[2] * 3) * 4));

    e[0] = p[1] - p[0];
    e[1] = p[2] - p[0];

    n[0] = asfloat(gNormals.Load3((indices[0] * 3) * 4));
    n[1] = asfloat(gNormals.Load3((indices[1] * 3) * 4));
    n[2] = asfloat(gNormals.Load3((indices[2] * 3) * 4));
}
#endif // __DXR_COMMON_H__