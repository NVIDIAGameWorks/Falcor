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
    
    Supports texture LOD both for ray differentials (Igehy, SIGGRAPH 1999) and a method based on ray cones,
    described in "Strategies for Texture Level-of-Detail for Real-Time Ray Tracing," by
    Tomas Akenine-Moller et al., Ray Tracing Gems, 2019.

    Note that the actual texture lookups are baked into the TextureSampler interfaces.

    See WhittedRayTracer.* for an example using these functions.
*/

import Experimental.Scene.Material.TexLODTypes;

#define CLASSIC_IGEHY   // if not defined, we use the ray diff method described in Section 20.3.2.2 in the Ray Tracing Gems

// ----------------------------------------------------------------------------
// ray cone helpers
// ----------------------------------------------------------------------------

/** Describes a ray cone for texture level-of-detail

    Representing a ray cone based on width and spread angle. Has both FP32 and FP16 support.
    Use #define USE_RAYCONES_WITH_FP16_IN_RAYPAYLOAD to use FP16
*/
struct RayCone
{
#ifndef USE_RAYCONES_WITH_FP16_IN_RAYPAYLOAD
    float width;
    float spreadAngle;
    float getWidth()            { return width; }
    float getSpreadAngle()      { return spreadAngle; }
#else
    uint width_spreadAngle_FP16;
    float getWidth()            { return f16tof32(width_spreadAngle_FP16 >> 16); }
    float getSpreadAngle()      { return f16tof32(width_spreadAngle_FP16); } 
#endif

    /** propagate the raycone to the next hit point (hitT distance away) and with the current spreadAngle + update ray cone angle with the surfaceSpreadAngle

        \param[in] surfaceSpreadAngle Surface spread angle, computed using computeScreenSpaceSurfaceSpreadAngle().
        \param[in] hitT Distance to the hit point.
    */
    RayCone propagate(in float surfaceSpreadAngle, in float hitT)    // surfaceSpreadAngle = beta in our texLOD paper
    {
        float angle = getSpreadAngle();
        return RayCone.create(angle * hitT + getWidth(), angle + surfaceSpreadAngle);
    }

    /** compute texture level of details based on ray cone
    */
    // should really remove this one, but may be good to have for other users, or code readers?!!!
    float computeLOD(in float triLODConstant, in float hitT, in float3 rayDir, in float3 normal, in float textureWidth, in float textureHeight)     // Note: call propagate() before computeLOD()
    {
        float lambda = triLODConstant; // constant per triangle
        float filterWidth = getWidth();
        // keeping this implementation since it is easier to read, but the following three lines that are not commented out are a bit faster
        //float distTerm = abs(filterWidth);
        //float normalTerm = abs(dot(rayDir, normal));
        //lambda += 0.5 * log2(textureWidth * textureHeight);             // texture size term
        //lambda += log2(distTerm);                                       // distance term
        //lambda -= log2(normalTerm);                                     // surface orientation term
        float distTerm = filterWidth * filterWidth;
        float normalTerm = dot(rayDir, normal);
        lambda += 0.5 * log2(textureWidth * textureHeight * distTerm / (normalTerm * normalTerm));
        return lambda;
    }

    /** Compute texture level of details based on ray cone
        Note that this versions excludes texture dimension dependency, which is instead added back in
        using the ExplicitRayConesLodTextureSampler:ITextureSampler in order to support baseColor, specular, etc per surfaces

        \param[in] triLODConstant Value computed by computeRayConeTriangleLODValue().
        \param[in] hitT Distance to the hit point.
        \param[in] rayDir Ray direction.
        \param[in] normal Normal at the hit point.
    */
    float computeLOD(in float triLODConstant, in float hitT, in float3 rayDir, in float3 normal)
    {
        float lambda = triLODConstant; // constant per triangle
        float filterWidth = getWidth();

        // keeping this implementation since it is easier to read, but the following three lines that are not commented out are a bit faster
        //float distTerm = abs(filterWidth);
        //float normalTerm = abs(dot(rayDir, normal));
        //lambda += log2(distTerm);                                       // distance term
        //lambda -= log2(normalTerm);                                     // surface orientation term

        float distTerm = filterWidth * filterWidth;
        float normalTerm = dot(rayDir, normal);
        lambda += 0.5 * log2(distTerm / (normalTerm * normalTerm));
        return lambda;
    }


    /** Create a ray cone struct
        \param[in] width The width of the ray cone.
        \param[in] angle The angle of the ray cone.
    */
    static RayCone create(in float width, in float angle)
    {
        RayCone rc;
#ifndef USE_RAYCONES_WITH_FP16_IN_RAYPAYLOAD
        rc.width = width;
        rc.spreadAngle = angle;
#else
        rc.width_spreadAngle_FP16 = (f32tof16(width) << 16) | f32tof16(angle);
#endif
        return rc;
    }

};
    
/** Compute the triangle LOD value based on triangle vertices and texture coordinates, used by ray cones, and
    and as a side effect, return the normalized normal as well
    \param[in] vertices Triangle vertices
    \param[in] txcoords Texture coordinates at triangle vertices.
    \param[in] worldMat 3x3 world matrix.
    \param[out] normalizedNormalW Normalized normal in world space.
    \return Triangle LOD value.
*/
float computeRayConeTriangleLODValue(in float3 vertices[3], in float2 txcoords[3], in float3x3 worldMat, out float3 normalizedNormalW)
{
    float2 tx10 = txcoords[1] - txcoords[0];
    float2 tx20 = txcoords[2] - txcoords[0];
    float Ta = abs(tx10.x * tx20.y - tx20.x * tx10.y);

    // we need the area of the triangle, which is length(triangleNormal) in worldspace, and I
    // could not figure out a way with fewer than two 3x3 mtx multiplies for ray cones.
    float3 edge10 = mul(vertices[1] - vertices[0], worldMat);
    float3 edge20 = mul(vertices[2] - vertices[0], worldMat);

    float3 triangleNormal = cross(edge10, edge20);              // in world space, by design
    float oneDivPa = 1.0 / length(triangleNormal);
    normalizedNormalW = triangleNormal * oneDivPa;              // normalize face normal with oneDivPa
    return 0.5 * log2(Ta * oneDivPa);                           // value used by texture LOD cones model               
}

/** Compute screen space spread angle at the first hit point based on ddx and ddy of normal and position.
    \param[in] positionW Position of the hit point in world space.
    \param[in] normalW Normal of the hit point in world space.
    \param[in] betaFactorK1 Optional factor, default value = 1. See Ray Tracing Gems, chapter 20.
    \param[in] betaFactorK2 Optional factor, default value = 0. See Ray Tracing Gems, chapter 20.
    \return Spread angle at hit point.
*/
float computeScreenSpaceSurfaceSpreadAngle(in float3 positionW, in float3 normalW, in float betaFactorK1 = 1.0, in float betaFactorK2 = 0.0)
{
    float3 dNdx = ddx(normalW);
    float3 dNdy = ddy(normalW);
    float3 dPdx = ddx(positionW);
    float3 dPdy = ddy(positionW);
    
   float beta = sqrt(dot(dNdx, dNdx) + dot(dNdy, dNdy)) * sign(dot(dNdx, dPdx) + dot(dNdy, dPdy));

    return 2.0 * beta * betaFactorK1 + betaFactorK2;
}


// ----------------------------------------------------------------------------
// ray differentials helpers
// ----------------------------------------------------------------------------

/** Describes a ray differential for texture level-of-detail

    Representing a ray differential based dOdx, dOdy (for ray origin) and dDdx, dDdy (for ray direction)
*/

struct RayDiff
{
    float3 dOdx;
    float3 dOdy;
    float3 dDdx;
    float3 dDdy;

    float3 getdOdx() { return dOdx; }   // these are not super-useful right now, but TODO to add FP16 version later on
    float3 getdOdy() { return dOdy; }
    float3 getdDdx() { return dDdx; }
    float3 getdDdy() { return dDdy; }

    /** Propagate the ray differential t distances away
        \param[in] O Ray origin.
        \param[in] D Ray direction.
        \param[in] t The distance to the hit point.
        \param[in] N The normal at the hit point.
    */
    RayDiff propagate(in float3 O, in float3 D, in float t, in float3 N)
    {
#ifdef CLASSIC_IGEHY
        float3 dOdx = getdOdx() + t * getdDdx();    // part of Igehy Equation 10
        float3 dOdy = getdOdy() + t * getdDdy();

        float rcpDN = 1.0 / dot(D, N);              // Igehy Equation 10 and 12
        float dtdx = -dot(dOdx, N) * rcpDN;
        float dtdy = -dot(dOdy, N) * rcpDN;
        dOdx += D * dtdx;
        dOdy += D * dtdy;
#else
        float3 dOdx = getdOdx(); // when not using CLASSIC_IGEHY, there is no propagate step, so we leave the ray diff unchanged.
        float3 dOdy = getdOdy();
#endif
        return RayDiff.create(dOdx, dOdy, getdDdx(), getdDdy());
    }

    /** Create a ray differential struct
        \param[in] dOdx The differential ray origin in x.
        \param[in] dOdy The differential ray origin in y.
        \param[in] dDdx The differential ray direction in x.
        \param[in] dDdy The differential ray direction in y.
    */
    static RayDiff create(in float3 dOdx, in float3 dOdy, in float3 dDdx, in float3 dDdy)
    {
        RayDiff rd;
        rd.dOdx = dOdx;
        rd.dOdy = dOdy;
        rd.dDdx = dDdx;
        rd.dDdy = dDdy;
        return rd;
    }
};

/** Computes the ray direction differential under the assumption that getCameraRayDir() is as commented out just above.
    \param[in] nonNormalizedCameraRaydir Non-normalized camera ray direction.
    \param[in] cameraRight Camera right vector.
    \param[in] cameraUp Camera up vector.
    \param[in] viewportDims Dimensions of the viewport.
    \param[out] dDdx The differential ray direction in x.
    \param[out] dDdy The differential ray direction in y.

    the getRayDirectionDifferentials() function differentiates normalize(getCameraRayDir()), where getCameraRayDir() is:
    float3 getCameraRayDir(uint2 pixel, uint2 frameDim)
    {
        float2 p = (pixel.xy + float2(0.5f, 0.5f)) / frameDim.xy; // Pixel center on image plane in [0,1] where (0,0) is top-left
        float2 ndc = float2(2, -2) * p + float2(-1, 1);
        return ndc.x * gCamera.cameraU + ndc.y * gCamera.cameraV + gCamera.cameraW; // rayDir = world-space direction to point on image plane (unnormalized)
    }
*/
void getRayDirectionDifferentials(in float3 nonNormalizedCameraRaydir, in float3 cameraRight, in float3 cameraUp, in float2 viewportDims, out float3 dDdx, out float3 dDdy)
{
    // Igehy Equation 8, adapted to getRayDirection() above
    float dd = dot(nonNormalizedCameraRaydir, nonNormalizedCameraRaydir);
    float divd = 2.0f / (dd * sqrt(dd));
    float dr = dot(nonNormalizedCameraRaydir, cameraRight);
    float du = dot(nonNormalizedCameraRaydir, cameraUp);
    dDdx = ((dd * cameraRight) - (dr * nonNormalizedCameraRaydir)) * divd / viewportDims.x;
    dDdy = -((dd * cameraUp) - (du * nonNormalizedCameraRaydir)) * divd / viewportDims.y;
}

/** Computes the differential barycentric coordinates and differential texture coordinates
    \param[in] rayDiff RayDifferential to be used for these computations.
    \param[in] rayDir Ray direction.
    \param[in] verticesW Triangle vertices in world space.
    \param[in] txcoords Texture coordinates at the three vertices of the triangle.
    \param[in] triangleEdge10 Triangle vertex 1 minus triangle vertex 0.
    \param[in] triangleEdge20 Triangle vertex 2 minus triangle vertex 0.
    \param[in] faceNormalW Normal of the triangle in world space.
    \param[in] hitT Distance to the hit point.
    \param[out] dBarydx Differential barycentric coordinates in x. Note that we skip the third component, since w=1-u-v and thus dw/dx=-du/dx-dv/dx
    \param[out] dBarydy Differential barycentric coordinates in y. Note that we skip the third component, since w=1-u-v and thus dw/dy=-du/dy-dv/dy
    \param[out] dUVdx Differential texture coordinates in x.
    \param[out] dUVdy Differential texture coordinates in y.
*/
void computeDifferentialsBarysAndUVs(in RayDiff rayDiff, in float3 rayDir, in float3 verticesW[3], in float2 txcoords[3],
    in float3 triangleEdge10, in float3 triangleEdge20, in float3 faceNormalW, in float hitT,
    out float2 dBarydx, out float2 dBarydy, out float2 dUVdx, out float2 dUVdy)
{
#ifdef CLASSIC_IGEHY
    float3 Nu = cross(verticesW[2] - verticesW[1], faceNormalW);    // Igehy "Normal-Interpolated Triangles", page 182 SIGGRAPH 1999
    float3 Nv = cross(-triangleEdge20, faceNormalW);

    float4 Lu = float4(Nu, -dot(Nu, verticesW[1]));                  // plane equations for the triangle edges
    float4 Lv = float4(Nv, -dot(Nv, verticesW[2]));

    Lu /= dot(Lu, float4(verticesW[0], 1.0));                        // planes scaled in order to make the dot with the opposive vertex = 1
    Lv /= dot(Lv, float4(verticesW[1], 1.0));

    dBarydx.x = dot(Lu.xyz, rayDiff.getdOdx());                     // du / dx
    dBarydx.y = dot(Lv.xyz, rayDiff.getdOdx());                     // dv / dx
    dBarydy.x = dot(Lu.xyz, rayDiff.getdOdy());                     // du / dy
    dBarydy.y = dot(Lv.xyz, rayDiff.getdOdy());                     // dv / dy

#else        // optimized implementation from Section 20.3.2.2 from Ray Tracing Gems
    float3 Cu = cross(triangleEdge20, rayDir);
    float3 Cv = cross(rayDir, triangleEdge10);

    float k = dot(Cu, triangleEdge10)
    float oneDivK = abs(k) > 0.0001f ? rcp(k) : 0.0f;

    float3 qx = rayDiff.getdOdx() + hitT * rayDiff.getdDdx();
    float3 qy = rayDiff.getdOdy() + hitT * rayDiff.getdDdy();

    dBarydx.y = dot(Cu, qx) * oneDivK;                // du / dx
    dBarydx.z = dot(Cv, qx) * oneDivK;                // dv / dx
    dBarydx.x = -dBarydx.y - dBarydx.z;

    dBarydy.y = dot(Cu, qy) * oneDivK;                // du / dy
    dBarydy.z = dot(Cv, qy) * oneDivK;                // dv / dy
    dBarydy.x = -dBarydy.y - dBarydy.z;
#endif

    // compute dUdx, dVdx, dUdy, dVdy for texture lookup
    float2 tx02 = txcoords[0] - txcoords[2];
    float2 tx12 = txcoords[1] - txcoords[2];
    dUVdx = dBarydx.x * tx02 + dBarydx.y * tx12;
    dUVdy = dBarydy.x * tx02 + dBarydy.y * tx12;

}

/** Reflects a ray differential
    \param[in,out] rayDiff RayDifferential to be reflected, result is returned here as well.
    \param[in] rayDir Ray direction.
    \param[in] nonNormalizedInterpolatedNormalW Interpolated NON-normalized normal in world space.
    \param[in] normalizedInterpolatedNormalW Interpolated normalized normal in world space.
    \param[in] dBarydx Differential barycentric coordinates wrt x.
    \param[in] dBarydy Differential barycentric coordinates wrt y.
    \param[in] triangleEdge10 Triangle vertex 1 minus triangle vertex 0.
    \param[in] triangleEdge20 Triangle vertex 2 minus triangle vertex 0.
    \param[in] normals The triangle's three normalized normals in world space.
*/
void reflectRayDifferential(inout RayDiff rayDiff, in float3 rayDir, in float3 nonNormalizedInterpolatedNormalW,
    in float3 normalizedInterpolatedNormalW, in float2 dBarydx, in float2 dBarydy,
    in float3 triangleEdge10, in float3 triangleEdge20, in float3 normals[3])
{
    // differential normal (see "Normal-Interpolated Triangles" in Igehy's paper)
    float NN = dot(nonNormalizedInterpolatedNormalW, nonNormalizedInterpolatedNormalW); // normal must be unnormalized! (otherwise NN would be 1)
    float rcpNN = 1.0 / (NN * sqrt(NN));
    float3 n02 = normals[0] - normals[2];
    float3 n12 = normals[1] - normals[2];
    float3 dndx = dBarydx.x * n02 + dBarydx.y * n12;
    float3 dndy = dBarydy.x * n02 + dBarydy.y * n12;
    float3 dNdx = (dndx * NN - nonNormalizedInterpolatedNormalW * dot(nonNormalizedInterpolatedNormalW, dndx)) * rcpNN;
    float3 dNdy = (dndy * NN - nonNormalizedInterpolatedNormalW * dot(nonNormalizedInterpolatedNormalW, dndy)) * rcpNN;

    // differential of reflected ray direction (perfect specular reflection) -- Equation 14 and 15 in Igehy's paper
    float dDNdx = dot(rayDiff.getdDdx(), normalizedInterpolatedNormalW) + dot(rayDir, dNdx);
    float dDNdy = dot(rayDiff.getdDdy(), normalizedInterpolatedNormalW) + dot(rayDir, dNdy);
    float DN = dot(rayDir, normalizedInterpolatedNormalW);
#ifdef CLASSIC_IGEHY
    float3 dOdx = rayDiff.getdOdx();
    float3 dOdy = rayDiff.getdOdy();
#else   
    float dBarydx_z = -dBarydx.x - dBarydx.y;   // we usually only need u,v and not w=1-u-v, but here we need it
    float dBarydy_z = -dBarydy.x - dBarydy.y;
    // the reason why this following code looks different is that in Chapter 20, Ray Tracing Gems, equation 12
    // defines a point in the plane as: P = P_0 + u*e_1 + v*e_2, where e1 = P_1 - P_0 and e_2 = P_2 - P_0
    // but in the rest of the paper we used P = u * P_0 + v * P_1 + w * P_2,
    // which means that in Equation 22, du/dx is dBaryDx.y (that is dv/dx) and dv/dx is dBarydx_z (that dw/dx)
    float3 dOdx = triangleEdge10 * dBarydx.y + triangleEdge20 * dBarydx_z;
    float3 dOdy = triangleEdge10 * dBarydy.y + triangleEdge20 * dBarydy_z;
#endif
    float3 dDdx = rayDiff.getdDdx() - 2.0 * (dNdx * DN + normalizedInterpolatedNormalW * dDNdx);
    float3 dDdy = rayDiff.getdDdy() - 2.0 * (dNdy * DN + normalizedInterpolatedNormalW * dDNdy);
    rayDiff = RayDiff.create(dOdx, dOdy, dDdx, dDdy);
}

