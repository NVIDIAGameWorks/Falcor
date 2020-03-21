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
 **************************************************************************/

/** Helper functions for the texture level-of-detail (LOD) system.
    
    Supports texture LOD both for ray differentials (Igehy, SIGGRAPH 1999) and a method based on ray cones

    TODO: This file needs to be updated to the new Scene representation.
    See issue #494

*/

#define CLASSIC_IGEHY

// ----------------------------------------------------------------------------
// ray cone helpers
// ----------------------------------------------------------------------------

/** Struct representing a ray cone based on width and spread angle. Has both FP32 and FP16 support.
*/
struct RayCone
{
#ifndef USE_RAYCONES_WITH_FP16_IN_RAYPAYLOAD
    float width;
    float spreadAngle;
    float getWidth()
    {
        return width;
    }
    float getSpreadAngle()
    {
        return spreadAngle;
    }
#else
    uint width_spreadAngle_FP16;
    float getWidth()                    { return f16tof32(width_spreadAngle_FP16 >> 16); }
    float getSpreadAngle()              { return f16tof32(width_spreadAngle_FP16); } 
#endif

    /** propagate the raycone to the next hit point (hitT distance away) and with the current spreadAngle + update ray cone angle with the surfaceSpreadAngle
    */
    RayCone propagate(in float surfaceSpreadAngle, in float hitT)    // surfaceSpreadAngle = beta in our texLOD paper
    {
        float angle = getSpreadAngle();
        return createRayCone(angle * hitT + getWidth(), angle + surfaceSpreadAngle);
    }

    /** compute texture level of details based on ray cone
    */
    float computeLOD(in float triLODConstant, in float hitT, in float3 rayDir, in float3 normal, in float textureWidth, in float textureHeight)     // Note: call propagate() before computeLOD()
    {
        float lambda = triLODConstant; // constant per triangle
        float filterWidth = getWidth();
#if 0
        float distTerm = abs(filterWidth);                              // keeping this implementation since it is easier to read
        float normalTerm = abs(dot(rayDir, normal));
        lambda += 0.5 * log2(textureWidth * textureHeight);             // texture size term
        lambda += log2(distTerm);                                       // distance term
        lambda -= log2(normalTerm);                                     // surface orientation term
#else                                                                   // this implementation is a tiny bit faster
        float distTerm = filterWidth * filterWidth;
        float normalTerm = dot(rayDir, normal);
        lambda += 0.5 * log2(textureWidth * textureHeight * distTerm / (normalTerm * normalTerm));
#endif
        return lambda;
    }
};
    
/** create a ray cone struct
*/
RayCone createRayCone(in float w, in float a)
{
    RayCone rc;
#ifndef USE_RAYCONES_WITH_FP16_IN_RAYPAYLOAD
    rc.width = w;
    rc.spreadAngle = a;
#else
    rc.width_spreadAngle_FP16 = (f32tof16(w) << 16) | f32tof16(a);
#endif
    return rc;
}

/** based on ddx and ddy of normal and position, compute an estimate of the surface spread angle
*/
float computeScreenSpaceSurfaceSpreadAngle(in float3 positionW, in float3 normalW, in float3 eyeVectorW, in float betaFactorK1, in float betaFactorK2)
{
    float beta;
    float3 dNdx = ddx(normalW);
    float3 dNdy = ddy(normalW);
    float3 dPdx = ddx(positionW);
    float3 dPdy = ddy(positionW);
    
    beta = sqrt(dot(dNdx, dNdx) + dot(dNdy, dNdy)) * sign(dot(dNdx, dPdx) + dot(dNdy, dPdy));

    return 2.0 * beta * betaFactorK1 + betaFactorK2;
}

VSOut getVertexAttributesAndTriangleTexLODConstant(in uint triangleIndex, in BuiltInTriangleIntersectionAttributes attribs, out float txLODconstant)
{
    uint3 indices = getIndices(triangleIndex);
    float3 barycentrics = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y, attribs.barycentrics.x, attribs.barycentrics.y);

    VSOut v;
    v.texC = 0;
    v.normalW = 0;
    v.bitangentW = 0;
    v.posW = 0;
    v.colorV = 0;
    v.prevPosH = 0;

    uint idx[3];
    idx[0] = indices[0] * 3 * 4;
    idx[1] = indices[1] * 3 * 4;
    idx[2] = indices[2] * 3 * 4;

    // texLOD: compute area of triangle and area in texture coords
    float3 vertices[3];
    float2 txcoords[3];
    vertices[0] = asfloat(gPositions.Load3(idx[0]));
    vertices[1] = asfloat(gPositions.Load3(idx[1]));
    vertices[2] = asfloat(gPositions.Load3(idx[2]));

    vertices[0] = mul(float4(vertices[0], 1.0), gWorldMat[0]).xyz;
    vertices[1] = mul(float4(vertices[1], 1.0), gWorldMat[0]).xyz;
    vertices[2] = mul(float4(vertices[2], 1.0), gWorldMat[0]).xyz;
    float3 e10 = vertices[1] - vertices[0];
    float3 e20 = vertices[2] - vertices[0];

    float3 triangleNormal = cross(e10, e20);
    float oneDivPa = 1.0 / length(triangleNormal); // hopefully no triangles with zero area 

    txcoords[0] = asfloat(gTexCrds.Load2(idx[0]));
    txcoords[1] = asfloat(gTexCrds.Load2(idx[1]));
    txcoords[2] = asfloat(gTexCrds.Load2(idx[2]));
    v.texC += txcoords[0] * barycentrics[0];
    v.texC += txcoords[1] * barycentrics[1];
    v.texC += txcoords[2] * barycentrics[2];
    float2 tx1 = txcoords[1] - txcoords[0];
    float2 tx2 = txcoords[2] - txcoords[0];
    float Ta = abs(tx1.x * tx2.y - tx2.x * tx1.y);

    txLODconstant = 0.5 * log2(Ta * oneDivPa);

    [unroll]
    for (int i = 0; i < 3; i++)
    {
        int address = idx[i];
        v.normalW += asfloat(gNormals.Load3(address))     * barycentrics[i];
        v.bitangentW += asfloat(gBitangents.Load3(address))  * barycentrics[i];
        v.posW += vertices[i] * barycentrics[i];
    }
    v.posW = mul(float4(v.posW, 1.f), gWorldMat[0]).xyz;
    // Transform normal/bitangent to world space
    v.normalW = mul(v.normalW, (float3x3)gWorldInvTransposeMat[0]).xyz;
    v.bitangentW = mul(v.bitangentW, (float3x3)gWorldMat[0]).xyz;
    v.normalW = normalize(v.normalW);
    v.bitangentW = normalize(v.bitangentW);
    return v;
}

// ----------------------------------------------------------------------------
// ray differentials helpers
// ----------------------------------------------------------------------------

/** Struct representing a ray differential
*/
struct RayDiff
{
    float3 dOdx;
    float3 dOdy;
    float3 dDdx;
    float3 dDdy;

    float3 getdOdx()
    {
        return dOdx;
    }
    float3 getdOdy()
    {
        return dOdy;
    }
    float3 getdDdx()
    {
        return dDdx;
    }
    float3 getdDdy()
    {
        return dDdy;
    }

    /** propagate the ray differential t distances away
    */
    RayDiff propagate(in float3 O, in float3 D, in float t, in float3 N)
    {
        float3 dOdx = getdOdx() + t * getdDdx();    // part of Igehy Equation 10
        float3 dOdy = getdOdy() + t * getdDdy();

        float rcpDN = 1.0 / dot(D, N);              // Igehy Equation 10 and 12
        float dtdx = -dot(dOdx, N) * rcpDN;
        float dtdy = -dot(dOdy, N) * rcpDN;
        dOdx += D * dtdx;
        dOdy += D * dtdy;

        return createRayDiff(dOdx, dOdy, getdDdx(), getdDdy()); // dDdxy remain the same when transferring
    }

    /** compute texture level of details based on ray differential
    */
    float computeLOD(in float2 dUVdx, in float2 dUVdy, in float txWidth, in float txHeight)     // does not need to be a member function, but looks nicer (similar to RayCone)
    {
#if 0
        float filterWidth = 2.0 * max(txWidth * max(abs(dUVdx.x), abs(dUVdy.x)), txHeight * max(abs(dUVdx.y), abs(dUVdy.y))); // same as in PBRT
        return log2(filterWidth);
#else
        dUVdx *= txWidth;
        dUVdy *= txHeight;
        return 0.5 * log2(max(dUVdx.x * dUVdx.x + dUVdx.y * dUVdx.y, dUVdy.x * dUVdy.x + dUVdy.y * dUVdy.y));   // OpenGL style
#endif
    }
};

/** create a ray differential struct
*/
RayDiff createRayDiff(in float3 dOdx, in float3 dOdy, in float3 dDdx, in float3 dDdy)
{
    RayDiff ray;
    ray.dOdx = dOdx;
    ray.dOdy = dOdy;
    ray.dDdx = dDdx;
    ray.dDdy = dDdy;
    return ray;
}

VSOut getVertexAttributesAndBaryDifferentials(in uint triangleIndex, in BuiltInTriangleIntersectionAttributes attribs, in float3 rayOrg, in float3 rayDir,
    in float hitT, in float3 normalizedGeometricNormal, inout RayDiff raydiff, out float2 dUVdx, out float2 dUVdy)
{
    float3 barycentrics = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y, attribs.barycentrics.x, attribs.barycentrics.y);
    uint3 indices = getIndices(triangleIndex);
    VSOut v;
    v.texC = 0;
    v.normalW = 0;
    v.bitangentW = 0;
    v.posW = 0;
    v.colorV = 0;
    v.prevPosH = 0;

    uint idx[3];
    idx[0] = (indices[0] * 3) * 4;
    idx[1] = (indices[1] * 3) * 4;
    idx[2] = (indices[2] * 3) * 4;

    float3 unormalizedN;
    float3 vertices[3];
    float3 normals[3];
    float2 txcoords[3];

    vertices[0] = asfloat(gPositions.Load3(idx[0]));
    vertices[1] = asfloat(gPositions.Load3(idx[1]));
    vertices[2] = asfloat(gPositions.Load3(idx[2]));

    normals[0] = asfloat(gNormals.Load3(idx[0]));
    normals[1] = asfloat(gNormals.Load3(idx[1]));
    normals[2] = asfloat(gNormals.Load3(idx[2]));
    v.normalW += normals[0] * barycentrics[0];
    v.normalW += normals[1] * barycentrics[1];
    v.normalW += normals[2] * barycentrics[2];
    unormalizedN = v.normalW;
    v.normalW = normalize(v.normalW);

    txcoords[0] = asfloat(gTexCrds.Load2(idx[0]));
    txcoords[1] = asfloat(gTexCrds.Load2(idx[1]));
    txcoords[2] = asfloat(gTexCrds.Load2(idx[2]));
    v.texC += txcoords[0] * barycentrics[0];
    v.texC += txcoords[1] * barycentrics[1];
    v.texC += txcoords[2] * barycentrics[2];

    [unroll]
    for (int i = 0; i < 3; i++)
    {
        int address = idx[i];
        v.bitangentW += asfloat(gBitangents.Load3(address))  * barycentrics[i];
        v.posW += vertices[i] * barycentrics[i];
    }
    v.posW = mul(float4(v.posW, 1.f), gWorldMat[0]).xyz;
    // Transform normal/bitangent to world space
    v.normalW = mul(v.normalW, (float3x3)gWorldInvTransposeMat[0]).xyz;
    v.bitangentW = mul(v.bitangentW, (float3x3)gWorldMat[0]).xyz;
    v.normalW = normalize(v.normalW);
    v.bitangentW = normalize(v.bitangentW);

    vertices[0] = mul(float4(vertices[0], 1.0), gWorldMat[0]).xyz;
    vertices[1] = mul(float4(vertices[1], 1.0), gWorldMat[0]).xyz;
    vertices[2] = mul(float4(vertices[2], 1.0), gWorldMat[0]).xyz;
    normals[0] = mul(normals[0], (float3x3) gWorldInvTransposeMat[0]).xyz;
    normals[1] = mul(normals[1], (float3x3) gWorldInvTransposeMat[0]).xyz;
    normals[2] = mul(normals[2], (float3x3) gWorldInvTransposeMat[0]).xyz;

    normals[0] = normalize(normals[0]);
    normals[1] = normalize(normals[1]);
    normals[2] = normalize(normals[2]);

    // compute differential barycentric coordinates
    float3 dBarydx, dBarydy;
//#ifdef CLASSIC_IGEHY        // CLASSIC IGEHY IMPLEMENTATION
#if 1
    float3 Nu = cross(vertices[2] - vertices[1], normalizedGeometricNormal);    // Igehy "Normal-Interpolated Triangles", page 182 SIGGRAPH 1999
    float3 Nv = cross(vertices[0] - vertices[2], normalizedGeometricNormal);

    float4 Lu = float4(Nu, -dot(Nu, vertices[1]));                                // plane equations for the triangle edges
    float4 Lv = float4(Nv, -dot(Nv, vertices[2]));

    Lu /= dot(Lu, float4(vertices[0], 1.0));                                    // planes scaled in order to make the dot with the opposive vertex = 1
    Lv /= dot(Lv, float4(vertices[1], 1.0));

    dBarydx.x = dot(Lu.xyz, raydiff.getdOdx());        // du / dx
    dBarydx.y = dot(Lv.xyz, raydiff.getdOdx());        // dv / dx
    dBarydx.z = -dBarydx.x - dBarydx.y;                // dw / dx  (w = 1 - u - w  --> dw/dx = - du/dx - dv/dx )

    dBarydy.x = dot(Lu.xyz, raydiff.getdOdy());        // du / dy
    dBarydy.y = dot(Lv.xyz, raydiff.getdOdy());        // dv / dy
    dBarydy.z = -dBarydy.x - dBarydy.y;                // dw / dy

#else        // OPTIMIZED IMPLEMENTATION
    float3 e1 = vertices[1] - vertices[0];
    float3 e2 = vertices[2] - vertices[0];
    float3 Cu = cross(e2, rayDir);
    float3 Cv = cross(rayDir, e1);
    float  one_div_k = 1.0 / dot(Cu, e1);
    float3 qx = raydiff.getdOdx() + hitT * raydiff.getdDdx();
    float3 qy = raydiff.getdOdy() + hitT * raydiff.getdDdy();

    dBarydx.y = dot(Cu, qx) * one_div_k;            // du / dx
    dBarydx.z = dot(Cv, qx) * one_div_k;            // dv / dx
    dBarydx.x = -dBarydx.y - dBarydx.z;                // dw / dx

    dBarydy.y = dot(Cu, qy) * one_div_k;            // du / dy
    dBarydy.z = dot(Cv, qy) * one_div_k;            // dv / dy
    dBarydy.x = -dBarydy.y - dBarydy.z;                // dw / dy
#endif

    // compute dUdx, dVdx, dUdy, dVdy for texture lookup
    dUVdx = dBarydx.x * txcoords[0] + dBarydx.y * txcoords[1] + dBarydx.z * txcoords[2];
    dUVdy = dBarydy.x * txcoords[0] + dBarydy.y * txcoords[1] + dBarydy.z * txcoords[2];

    // differential normal (see "Normal-Interpolated Triangles" in Igehy's paper)
    float NN = dot(unormalizedN, unormalizedN); // normal must be unnormalized! (otherwise NN would be 1)
    float rcpNN = 1.0 / (NN * sqrt(NN));
    float3 dndx = dBarydx.x * normals[0] + dBarydx.y * normals[1] + dBarydx.z * normals[2];
    float3 dndy = dBarydy.x * normals[0] + dBarydy.y * normals[1] + dBarydy.z * normals[2];
    float3 dNdx = (dndx * NN - unormalizedN * dot(unormalizedN, dndx)) * rcpNN;
    float3 dNdy = (dndy * NN - unormalizedN * dot(unormalizedN, dndy)) * rcpNN;

    // differential of reflected ray direction (perfect specular reflection) -- Equation 14 and 15 in Igehy's paper
    float dDNdx = dot(raydiff.getdDdx(), v.normalW) + dot(rayDir, dNdx);
    float dDNdy = dot(raydiff.getdDdy(), v.normalW) + dot(rayDir, dNdy);
    float DN = dot(rayDir, v.normalW);
//#ifdef CLASSIC_IGEHY
#if 1
    float3 dOdx = raydiff.getdOdx();
    float3 dOdy = raydiff.getdOdy();
#else
    float3 dOdx = e1 * dBarydx.y + e2 * dBarydx.z;
    float3 dOdy = e1 * dBarydy.y + e2 * dBarydy.z;
#endif
    float3 dDdx = raydiff.getdDdx() - 2.0 * (dNdx * DN + v.normalW * dDNdx);
    float3 dDdy = raydiff.getdDdy() - 2.0 * (dNdy * DN + v.normalW * dDNdy);
    raydiff = createRayDiff(dOdx, dOdy, dDdx, dDdy);

    return v;
}

// the getRayDirectionDifferentials() function differentiates normalize(getCameraRayDir()), where getCameraRayDir() is:
//float3 getCameraRayDir(uint2 pixel, uint2 frameDim)
//{
//    float2 p = (pixel.xy + float2(0.5f, 0.5f)) / frameDim.xy; // Pixel center on image plane in [0,1] where (0,0) is top-left
//    float2 ndc = float2(2, -2) * p + float2(-1, 1);
//    return ndc.x * gCamera.cameraU + ndc.y * gCamera.cameraV + gCamera.cameraW; // rayDir = world-space direction to point on image plane (unnormalized)
//}
void getRayDirectionDifferentials(in float3 cameraRaydir, in float3 cameraRight, in float3 cameraUp, in float2 viewportDims, out float3 dDdx, out float3 dDdy)
{
    // Igehy Equation 8, adapted to getRayDirection() above
    float dd = dot(cameraRaydir, cameraRaydir);
    float divd = 2.0f / (dd * sqrt(dd));
    float dr = dot(cameraRaydir, cameraRight);
    float du = dot(cameraRaydir, cameraUp);
    dDdx = ((dd * cameraRight) - (dr * cameraRaydir)) * divd / viewportDims.x;
    dDdy = -((dd * cameraUp) - (du * cameraRaydir)) * divd / viewportDims.y;
}

float3 getTransformedGeometricNormal(in uint triangleIndex)
{
    uint3 indices = getIndices(triangleIndex);
    float3 vertices[3];
    vertices[0] = asfloat(gPositions.Load3(indices[0] * 3 * 4));
    vertices[1] = asfloat(gPositions.Load3(indices[1] * 3 * 4));
    vertices[2] = asfloat(gPositions.Load3(indices[2] * 3 * 4));  
        
    float3 N = cross(vertices[2] - vertices[0], vertices[1] - vertices[0]);
    N = mul(N, (float3x3) gWorldInvTransposeMat[0]);
    return normalize(N);
}

static const uint numRainbowColors = 7;
static const float4 rainbowColors[7] =
{
    float4(1.0, 0.0, 0.0, 1.0),
    float4(1.0, 1.0, 0.0, 1.0),
    float4(0.0, 1.0, 0.0, 1.0),
    float4(0.0, 1.0, 1.0, 1.0),
    float4(0.0, 0.0, 1.0, 1.0),
    float4(1.0, 0.0, 1.0, 1.0),
    float4(1.0, 1.0, 1.0, 1.0),
};

float4 lambdaToRainbowColor(in float lambda)
{
    uint idx = uint(clamp(lambda, 0, numRainbowColors - 1));
    float t = lambda - idx;
    return (1.0 - t) * rainbowColors[idx] + t * rainbowColors[idx + 1];
}
