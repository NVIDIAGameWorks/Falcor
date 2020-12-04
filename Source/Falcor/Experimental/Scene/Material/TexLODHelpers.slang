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

/** Helper functions for the texture level-of-detail (LOD) system.

    Supports texture LOD both for ray differentials (Igehy, SIGGRAPH 1999) and a method based on ray cones,
    described in "Strategies for Texture Level-of-Detail for Real-Time Ray Tracing," by
    Tomas Akenine-Moller et al., Ray Tracing Gems, 2019 and
    in "Improved Shader and Texture Level-of-Detail using Ray Cones" by Akenine-Moller et al. (submitted to JCGT)

    Note that the actual texture lookups are baked into the TextureSampler interfaces.

    See WhittedRayTracer.* for an example using these functions.
*/

#include "Utils/Math/MathConstants.slangh"
import Experimental.Scene.Material.TexLODTypes;
import Scene.SceneTypes;        // Needed for ray bounce helpers.


#define TEXLOD_SPREADANGLE_FROM_CURVATURE_MODE          2       // Spread angle from curvature computation mode. 0: Original approach derived from RTG. 1: New arc-length integration approach, unoptimized. 2: New arc-length integration approach, optimized.

// ----------------------------------------------------------------------------
// Ray cone helpers
// ----------------------------------------------------------------------------

/** Describes a ray cone for texture level-of-detail.

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
    uint widthSpreadAngleFP16;
    float getWidth()            { return f16tof32(widthSpreadAngleFP16 >> 16); }
    float getSpreadAngle()      { return f16tof32(widthSpreadAngleFP16); }
#endif

    /** Propagate the raycone to the next hit point (hitT distance away) and with the current spreadAngle + update ray cone angle with the surfaceSpreadAngle.

        \param[in] surfaceSpreadAngle Surface spread angle, computed using computeScreenSpaceSurfaceSpreadAngle(). This is called beta in the Ray Tracing Gems article (chapter 20).
        \param[in] hitT Distance to the hit point.
        \return The propagated ray cone.
    */
    RayCone propagate(float surfaceSpreadAngle, float hitT)
    {
        float angle = getSpreadAngle();
        return RayCone.create(angle * hitT + getWidth(), angle + surfaceSpreadAngle);
    }

    /** Compute texture level of details based on ray cone.
    */
    // TODO: Should really remove this one, but may be good to have for other users, or code readers?!!!
    float computeLOD(float triLODConstant, float3 rayDir, float3 normal, float textureWidth, float textureHeight)     // Note: call propagate() before computeLOD()
    {
        float lambda = triLODConstant;  // Constant per triangle.
        float filterWidth = getWidth();
        // Keeping this implementation since it is easier to read, but the following three lines that are not commented out are a bit faster
        //float distTerm = abs(filterWidth);
        //float normalTerm = abs(dot(rayDir, normal));
        //lambda += 0.5f * log2(textureWidth * textureHeight);             // Texture size term.
        //lambda += log2(distTerm);                                       // Distance term.
        //lambda -= log2(normalTerm);                                     // Surface orientation term.
        float distTerm = filterWidth * filterWidth;
        float normalTerm = dot(rayDir, normal);
        lambda += 0.5f * log2(textureWidth * textureHeight * distTerm / (normalTerm * normalTerm));
        return lambda;
    }

    /** Compute texture level of details based on ray cone.
        Note that this versions excludes texture dimension dependency, which is instead added back in
        using the ExplicitRayConesLodTextureSampler:ITextureSampler in order to support baseColor, specular, etc per surfaces.

        \param[in] triLODConstant Value computed by computeRayConeTriangleLODValue().
        \param[in] rayDir Ray direction.
        \param[in] normal Normal at the hit point.
        \return The level of detail, lambda.
    */
    float computeLOD(float triLODConstant, float3 rayDir, float3 normal)
    {
        float lambda = triLODConstant; // constant per triangle
        float filterWidth = getWidth();

        // Keeping this implementation since it is easier to read, but the following three lines that are not commented out are a bit faster
        //float distTerm = abs(filterWidth);
        //float normalTerm = abs(dot(rayDir, normal));
        //lambda += log2(distTerm);                                       // Distance term
        //lambda -= log2(normalTerm);                                     // Surface orientation term

        float distTerm = filterWidth * filterWidth;
        float normalTerm = dot(rayDir, normal);
        lambda += 0.5f * log2(distTerm / (normalTerm * normalTerm));
        return lambda;
    }


    /** Create a ray cone struct.
        \param[in] width The width of the ray cone.
        \param[in] angle The angle of the ray cone.
        \return The ray cone.
    */
    static RayCone create(float width, float angle)
    {
        RayCone rc;
#ifndef USE_RAYCONES_WITH_FP16_IN_RAYPAYLOAD
        rc.width = width;
        rc.spreadAngle = angle;
#else
        rc.widthSpreadAngleFP16 = (f32tof16(width) << 16) | f32tof16(angle);
#endif
        return rc;
    }

};

/** Compute the triangle LOD value based on triangle vertices and texture coordinates, used by ray cones.
    \param[in] vertices Triangle vertices.
    \param[in] txcoords Texture coordinates at triangle vertices.
    \param[in] worldMat 3x3 world matrix.
    \return Triangle LOD value.
*/
float computeRayConeTriangleLODValue(float3 vertices[3], float2 txcoords[3], float3x3 worldMat)
{
    float2 tx10 = txcoords[1] - txcoords[0];
    float2 tx20 = txcoords[2] - txcoords[0];
    float Ta = abs(tx10.x * tx20.y - tx20.x * tx10.y);

    // We need the area of the triangle, which is length(triangleNormal) in worldspace, and
    // could not figure out a way with fewer than two 3x3 mtx multiplies for ray cones.
    float3 edge01 = mul(vertices[1] - vertices[0], worldMat);
    float3 edge02 = mul(vertices[2] - vertices[0], worldMat);

    float3 triangleNormal = cross(edge01, edge02);              // In world space, by design.
    float Pa = length(triangleNormal);                          // Twice the area of the triangle.
    return 0.5f * log2(Ta / Pa);                                // Value used by texture LOD cones model.
}

/** Compute the triangle LOD value based on triangle vertices and texture coordinates, used by ray cones.
    \param[in] meshInstanceID The mesh instance ID.
    \param[in] triangleIndex Index of the triangle in the given mesh.
    \return Triangle LOD value for ray cone.
*/
float computeRayConeTriangleLODValue(StaticVertexData triangleVertices[3], uint meshInstanceID, float3x3 worldMat)
{
    float2 txcoords[3];
    float3 positions[3];
    txcoords[0] = triangleVertices[0].texCrd;
    txcoords[1] = triangleVertices[1].texCrd;
    txcoords[2] = triangleVertices[2].texCrd;
    positions[0] = triangleVertices[0].position;
    positions[1] = triangleVertices[1].position;
    positions[2] = triangleVertices[2].position;

    return computeRayConeTriangleLODValue(positions, txcoords, worldMat);
}

/** Compute screen space spread angle at the first hit point based on ddx and ddy of normal and position.
    \param[in] positionW Position of the hit point in world space.
    \param[in] normalW Normal of the hit point in world space.
    \param[in] betaFactorK1 Optional factor, default value = 1. See Ray Tracing Gems, chapter 20.
    \param[in] betaFactorK2 Optional factor, default value = 0. See Ray Tracing Gems, chapter 20.
    \return Spread angle at hit point.
*/
float computeScreenSpaceSurfaceSpreadAngle(float3 positionW, float3 normalW, float betaFactorK1 = 1.0, float betaFactorK2 = 0.0)
{
    float3 dNdx = ddx(normalW);
    float3 dNdy = ddy(normalW);
    float3 dPdx = ddx(positionW);
    float3 dPdy = ddy(positionW);

    float beta = sqrt(dot(dNdx, dNdx) + dot(dNdy, dNdy)) * sign(dot(dNdx, dPdx) + dot(dNdy, dPdy));

    return 2.0f * beta * betaFactorK1 + betaFactorK2;
}

/** Compute screen space spread angle at the first hit point based on ddx and ddy of normal and position.
    \param[in] rightVector The difference vector between normalized eye ray direction at (x + 1, y) and (x, y).
    \param[in] cameraUpVector The difference vector between normalized eye ray direction at (x, y + 1) and (x, y).
    \param[in] dNdx Differential normal in the x-direction.
    \param[in] dNdy Differential normal in the y-direction.
    \return Spread angle at hit point.
*/
float computeScreenSpaceSurfaceSpreadAngle(float3 rightVector, float3 upVector, float3 dNdx, float3 dNdy)
{
    float betaX = atan(length(dNdx));
    float betaY = atan(length(dNdy));
    float betaCurvature = 2.0f * sqrt(betaX * betaX + betaY * betaY) * (betaX >= betaY ? sign(dot(rightVector, dNdx)) : sign(dot(upVector, dNdy)));
    return betaCurvature;
}

/** Compute spread from estimated curvature from a triangle for ray cones.
    \param[in] curvature Curvature value.
    \param[in] rayConeWidth The width of the ray cone.
    \param[in] rayDir The ray direction.
    \param[in] normal The normal.
    \return Spread angle.
*/
float computeSpreadAngleFromCurvatureIso(float curvature, float rayConeWidth, float3 rayDir, float3 normal)
{
    float dn = -dot(rayDir, normal);
    dn = abs(dn) < 1.0e-5 ? sign(dn) * 1.0e-5 : dn;

#if TEXLOD_SPREADANGLE_FROM_CURVATURE_MODE == 0
    // Original approach.
    float s = sign(curvature);
    float curvatureScaled = curvature * rayConeWidth * 0.5 / dn;
    float surfaceSpreadAngle = 4.0 * atan(abs(curvatureScaled) / sqrt(2.0)) * s;
#elif TEXLOD_SPREADANGLE_FROM_CURVATURE_MODE == 1
    // New approach, unoptimized: https://www.math24.net/curvature-plane-curves/

    float r = 1.0 / (curvature);
    float chord = (rayConeWidth) / (dn);
    float arcLength = asin(chord / (2.0 * r)) * (2.0 * r);
    float deltaPhi = (curvature) * (arcLength);

    float surfaceSpreadAngle = deltaPhi * 2.0;
#else
    // New approach : Fast Approximation.
    float deltaPhi = (curvature * rayConeWidth / dn);
    float surfaceSpreadAngle = deltaPhi * 2.0;
#endif

    return surfaceSpreadAngle;
}

/** Exploit ray cone to compute an approximate anisotropic filter. The idea is to find the width (2*radius) of the ray cone at
    the intersection point, and approximate the ray cone as a cylinder at that point with that radius. Then intersect the
    cylinder with the triangle plane to find the ellipse of anisotropy. Finally, convert to gradients in texture coordinates.
    \param[in] intersectionPoint The intersection point.
    \param[in] faceNormal The normal of the triangle.
    \param[in] rayConeDir Direction of the ray cone.
    \param[in] rayConeRadiusAtIntersection Radius of the cone at the intersection point, typically computed as coneRadiusAtIntersection = t * tan(rayConeAngle);
    \param[in] positions Positions of the triangle.
    \param[in] txcoords Texture coordinates of the vertices of the triangle.
    \param[in] interpolatedTexCoordsAtIntersection Interpolated texture coordinates at the intersection point.
    \param[in] texGradientX First gradient of texture coordinates.
    \param[in] texGradientY Second gradient of texture coordinates.
*/
void computeAnisotropicEllipseAxes(float3 intersectionPoint, float3 faceNormal, float3 rayConeDir,
    float rayConeRadiusAtIntersection, float3 positions[3], float2 txcoords[3], float2 interpolatedTexCoordsAtIntersection,
    out float2 texGradientX, out float2 texGradientY)
{
    // Compute ellipse axes.
    float3 ellipseAxis0 = rayConeDir - dot(faceNormal, rayConeDir) * faceNormal;                // Project rayConeDir onto the plane.
    float3 rayDirPlaneProjection0 = ellipseAxis0 - dot(rayConeDir, ellipseAxis0) * rayConeDir;  // Project axis onto the plane defined by the ray cone dir.
    ellipseAxis0 *= rayConeRadiusAtIntersection / max(0.0001f, length(rayDirPlaneProjection0)); // Using uniform triangles to find the scale.

    float3 ellipseAxis1 = cross(faceNormal, ellipseAxis0);
    float3 rayDirPlaneProjection1 = ellipseAxis1 - dot(rayConeDir, ellipseAxis1) * rayConeDir;
    ellipseAxis1 *= rayConeRadiusAtIntersection / max(0.0001f, length(rayDirPlaneProjection1));

    // Compute texture coordinate gradients.
    float3 edgeP;
    float u, v, Atriangle, Au, Av;
    float3 d = intersectionPoint - positions[0];
    float3 edge01 = positions[1] - positions[0];
    float3 edge02 = positions[2] - positions[0];
    float oneOverAreaTriangle = 1.0f / dot(faceNormal, cross(edge01, edge02));

    // Compute barycentrics.
    edgeP = d + ellipseAxis0;
    u = dot(faceNormal, cross(edgeP, edge02)) * oneOverAreaTriangle;
    v = dot(faceNormal, cross(edge01, edgeP)) * oneOverAreaTriangle;
    texGradientX = (1.0f - u - v) * txcoords[0] + u * txcoords[1] + v * txcoords[2] - interpolatedTexCoordsAtIntersection;

    edgeP = d + ellipseAxis1;
    u = dot(faceNormal, cross(edgeP, edge02)) * oneOverAreaTriangle;
    v = dot(faceNormal, cross(edge01, edgeP)) * oneOverAreaTriangle;
    texGradientY = (1.0f - u - v) * txcoords[0] + u * txcoords[1] + v * txcoords[2] - interpolatedTexCoordsAtIntersection;
}


// ----------------------------------------------------------------------------
// Ray differentials helpers
// ----------------------------------------------------------------------------

/** Describes a ray differential for texture level-of-detail.

    Representing a ray differential based dOdx, dOdy (for ray origin) and dDdx, dDdy (for ray direction).
*/
struct RayDiff
{
    float3 dOdx;
    float3 dOdy;
    float3 dDdx;
    float3 dDdy;

    float3 getdOdx() { return dOdx; }   // These are not super-useful right now, but TODO to add FP16 version later on
    float3 getdOdy() { return dOdy; }
    float3 getdDdx() { return dDdx; }
    float3 getdDdy() { return dDdy; }

    /** Propagate the ray differential t distances away.
        \param[in] O Ray origin.
        \param[in] D Ray direction.
        \param[in] t The distance to the hit point.
        \param[in] N The normal at the hit point.
        \return The propagated ray differential.
    */
    RayDiff propagate(float3 O, float3 D, float t, float3 N)
    {
        float3 dodx = getdOdx() + t * getdDdx();    // Part of Igehy Equation 10
        float3 dody = getdOdy() + t * getdDdy();

        float rcpDN = 1.0f / dot(D, N);              // Igehy Equation 10 and 12
        float dtdx = -dot(dodx, N) * rcpDN;
        float dtdy = -dot(dody, N) * rcpDN;
        dodx += D * dtdx;
        dody += D * dtdy;

        return RayDiff.create(dodx, dody, getdDdx(), getdDdy());
    }

    /** Create a ray differential struct.
        \param[in] dOdx The differential ray origin in x.
        \param[in] dOdy The differential ray origin in y.
        \param[in] dDdx The differential ray direction in x.
        \param[in] dDdy The differential ray direction in y.
        \return The created ray differential.
    */
    static RayDiff create(float3 dOdx, float3 dOdy, float3 dDdx, float3 dDdy)
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

    The computeRayDirectionDifferentials() function differentiates normalize(getCameraRayDir()), where getCameraRayDir() is:
    float3 getCameraRayDir(uint2 pixel, uint2 frameDim)
    {
        float2 p = (pixel.xy + float2(0.5f, 0.5f)) / frameDim.xy; // Pixel center on image plane in [0,1] where (0,0) is top-left
        float2 ndc = float2(2, -2) * p + float2(-1, 1);
        return ndc.x * gCamera.cameraU + ndc.y * gCamera.cameraV + gCamera.cameraW; // rayDir = world-space direction to point on image plane (unnormalized)
    }
*/
void computeRayDirectionDifferentials(float3 nonNormalizedCameraRaydir, float3 cameraRight, float3 cameraUp, float2 viewportDims, out float3 dDdx, out float3 dDdy)
{
    // Igehy Equation 8, adapted to getRayDirection() above
    float dd = dot(nonNormalizedCameraRaydir, nonNormalizedCameraRaydir);
    float divd = 2.0f / (dd * sqrt(dd));
    float dr = dot(nonNormalizedCameraRaydir, cameraRight);
    float du = dot(nonNormalizedCameraRaydir, cameraUp);
    dDdx = ((dd * cameraRight) - (dr * nonNormalizedCameraRaydir)) * divd / viewportDims.x;
    dDdy = -((dd * cameraUp) - (du * nonNormalizedCameraRaydir)) * divd / viewportDims.y;
}

/** Computes the differential barycentric coordinates.
    \param[in] rayDiff RayDifferential to be used for these computations.
    \param[in] rayDir Ray direction.
    \param[in] edge01 Position 1 minus position 0.
    \param[in] edge02 Position 2 minus position 0.
    \param[in] faceNormalW Normal of the triangle in world space.
    \param[out] dBarydx Differential barycentric coordinates in x. Note that we skip the third component, since w=1-u-v and thus dw/dx=-du/dx-dv/dx.
    \param[out] dBarydy Differential barycentric coordinates in y. Note that we skip the third component, since w=1-u-v and thus dw/dy=-du/dy-dv/dy.
*/
void computeBarycentricDifferentials(RayDiff rayDiff, float3 rayDir, float3 edge01, float3 edge02,
    float3 faceNormalW, out float2 dBarydx, out float2 dBarydy)
{
    float3 Nu = cross(edge02, faceNormalW);      // Igehy "Normal-Interpolated Triangles", page 182 SIGGRAPH 1999
    float3 Nv = cross(edge01, faceNormalW);
    float3 Lu = Nu / (dot(Nu, edge01));          // Plane equations for the triangle edges, scaled in order to make the dot with the opposive vertex = 1
    float3 Lv = Nv / (dot(Nv, edge02));

    dBarydx.x = dot(Lu, rayDiff.getdOdx());     // du / dx
    dBarydx.y = dot(Lv, rayDiff.getdOdx());     // dv / dx
    dBarydy.x = dot(Lu, rayDiff.getdOdy());     // du / dy
    dBarydy.y = dot(Lv, rayDiff.getdOdy());     // dv / dy
}


/** Interpolates vertex values using differential barycentrics for a single float per vertex.
    \param[in] dBarydx Differential barycentric coordinates in x.
    \param[in] dBarydy Differential barycentric coordinates in y.
    \param[in] vertexValues The three values at the triangle vertices to be interpolated.
    \param[out] dx Interpolated vertex values using differential barycentric coordinates in x.
    \param[out] dy Interpolated vertex values using differential barycentric coordinates in y.
*/
void interpolateDifferentials(float2 dBarydx, float2 dBarydy, float vertexValues[3], out float dx, out float dy)
{
    float delta1 = vertexValues[1] - vertexValues[0];
    float delta2 = vertexValues[2] - vertexValues[0];
    dx = dBarydx.x * delta1 + dBarydx.y * delta2;
    dy = dBarydy.x * delta1 + dBarydy.y * delta2;
}

/** Interpolates vertex values using differential barycentrics for a single float2 per vertex.
    \param[in] dBarydx Differential barycentric coordinates in x.
    \param[in] dBarydy Differential barycentric coordinates in y.
    \param[in] vertexValues The three values at the triangle vertices to be interpolated
    \param[out] dx Interpolated vertex values using differential barycentric coordinates in x.
    \param[out] dy Interpolated vertex values using differential barycentric coordinates in y.
*/
void interpolateDifferentials(float2 dBarydx, float2 dBarydy, float2 vertexValues[3], out float2 dx, out float2 dy)
{
    float2 delta1 = vertexValues[1] - vertexValues[0];
    float2 delta2 = vertexValues[2] - vertexValues[0];
    dx = dBarydx.x * delta1 + dBarydx.y * delta2;
    dy = dBarydy.x * delta1 + dBarydy.y * delta2;
}

/** Interpolates vertex values using differential barycentrics for a single float3 per vertex.
    \param[in] dBarydx Differential barycentric coordinates in x.
    \param[in] dBarydy Differential barycentric coordinates in y.
    \param[in] vertexValues The three values at the triangle vertices to be interpolated.
    \param[out] dx Interpolated vertex values using differential barycentric coordinates in x.
    \param[out] dy Interpolated vertex values using differential barycentric coordinates in y.
*/
void interpolateDifferentials(float2 dBarydx, float2 dBarydy, float3 vertexValues[3], out float3 dx, out float3 dy)
{
    float3 delta1 = vertexValues[1] - vertexValues[0];
    float3 delta2 = vertexValues[2] - vertexValues[0];
    dx = dBarydx.x * delta1 + dBarydx.y * delta2;
    dy = dBarydy.x * delta1 + dBarydy.y * delta2;
}

/** Computes the normal differentials using differntial barycentric coordinates.
    \param[in] rayDiff RayDifferential to be used for these computations.
    \param[in] nonNormalizedInterpolatedNormalW Interpolated NON-normalized normal in world space.
    \param[in] dBarydx Differential barycentric coordinates in x.
    \param[in] dBarydy Differential barycentric coordinates in y.
    \param[in] normals Normalized normals in world space at the three triangle vertices.
    \param[out] dNdx Differential normal in the x-direction.
    \param[out] dNdy Differential normal in the y-direction.
*/
void computeNormalDifferentials(RayDiff rayDiff, float3 nonNormalizedInterpolatedNormalW,
    float2 dBarydx, float2 dBarydy, float3 normals[3], out float3 dNdx, out float3 dNdy)
{
    // Differential normal (see "Normal-Interpolated Triangles" in Igehy's paper)
    float NN = dot(nonNormalizedInterpolatedNormalW, nonNormalizedInterpolatedNormalW); // normal must be unnormalized! (otherwise NN would be 1)
    float rcpNN = 1.0f / (NN * sqrt(NN));

    float3 dndx, dndy;
    interpolateDifferentials(dBarydx, dBarydy, normals, dndx, dndy);

    dNdx = (dndx * NN - nonNormalizedInterpolatedNormalW * dot(nonNormalizedInterpolatedNormalW, dndx)) * rcpNN;
    dNdy = (dndy * NN - nonNormalizedInterpolatedNormalW * dot(nonNormalizedInterpolatedNormalW, dndy)) * rcpNN;
}

/** Reflects a ray differential.
    \param[in,out] rayDiff RayDifferential to be reflected, result is returned here as well.
    \param[in] rayDir Ray direction.
    \param[in] nonNormalizedInterpolatedNormalW Interpolated NON-normalized normal in world space.
    \param[in] normalizedInterpolatedNormalW Interpolated normalized normal in world space.
    \param[in] dBarydx Differential barycentric coordinates wrt x.
    \param[in] dBarydy Differential barycentric coordinates wrt y.
    \param[in] normalsW The triangle's three normalized normals in world space.
*/
void reflectRayDifferential(inout RayDiff rayDiff, float3 rayDir, float3 nonNormalizedInterpolatedNormalW,
    float3 normalizedInterpolatedNormalW, float2 dBarydx, float2 dBarydy, float3 normalsW[3])
{
    float3 dNdx, dNdy;
    computeNormalDifferentials(rayDiff, nonNormalizedInterpolatedNormalW, dBarydx, dBarydy, normalsW, dNdx, dNdy);

    // Differential of reflected ray direction (perfect specular reflection) -- Equation 14 and 15 in Igehy's paper
    float dDNdx = dot(rayDiff.getdDdx(), normalizedInterpolatedNormalW) + dot(rayDir, dNdx);
    float dDNdy = dot(rayDiff.getdDdy(), normalizedInterpolatedNormalW) + dot(rayDir, dNdy);

    float DN = dot(rayDir, normalizedInterpolatedNormalW);

    float3 dOdx = rayDiff.getdOdx();
    float3 dOdy = rayDiff.getdOdy();
    float3 dDdx = rayDiff.getdDdx() - 2.0f * (dNdx * DN + normalizedInterpolatedNormalW * dDNdx);
    float3 dDdy = rayDiff.getdDdy() - 2.0f * (dNdy * DN + normalizedInterpolatedNormalW * dDNdy);
    rayDiff = RayDiff.create(dOdx, dOdy, dDdx, dDdy);
}

/** Prepares vertices for ray differentials so that computations happen in world space.
    \param[in] rayDir The direction of the ray.
    \param[in] vertices The vertices of the triangle.
    \param[in] worldMat World matrix.
    \param[in] worldInvTransposeMat Inverse transpose of world matrix.
    \param[in] barycentrics Barycentric coordinates.
    \param[out] edge01 Position 1 minus position 0 in world space.
    \param[out] edge02 Position 2 minus position 0 in world space.
    \param[out] normals Normals in world space (not normalized).
    \param[out] unnormalizedN Interpolated, unnormalized normal.
    \param[out] txcoords Texture coordinates.
*/
void prepareVerticesForRayDiffs(float3 rayDir, StaticVertexData vertices[3], float4x4 worldMat, float3x3 worldInvTransposeMat, float3 barycentrics,
    out float3 edge01, out float3 edge02, out float3 normals[3], out float3 unnormalizedN, out float2 txcoords[3])
{
    // Transform relevant data to world space.
    edge01 = mul(vertices[1].position - vertices[0].position, (float3x3)worldMat);
    edge02 = mul(vertices[2].position - vertices[0].position, (float3x3)worldMat);
    normals[0] = mul(vertices[0].normal, worldInvTransposeMat).xyz;
    normals[1] = mul(vertices[1].normal, worldInvTransposeMat).xyz;
    normals[2] = mul(vertices[2].normal, worldInvTransposeMat).xyz;
    // Note that we do not need to normalize the individual normals[], since the derivation
    // by Igehy page 182 normalizes the normal after interpolation.
    unnormalizedN = normals[0] * barycentrics[0];
    unnormalizedN += normals[1] * barycentrics[1];
    unnormalizedN += normals[2] * barycentrics[2];
    if (dot(unnormalizedN, rayDir) > 0)
    {
        unnormalizedN = -unnormalizedN;
        normals[0] = -normals[0];
        normals[1] = -normals[1];
        normals[2] = -normals[2];
    }

    txcoords[0] = vertices[0].texCrd;
    txcoords[1] = vertices[1].texCrd;
    txcoords[2] = vertices[2].texCrd;
}

/** Computes ray differentials parameters at surface hit point.
*/
void prepareRayDiffAtHitPoint(VertexData v, StaticVertexData triangleVertices[3], float3 barycentrics, float3 rayDir, float hitT, float4x4 worldMat, float3x3 worldInvTransposeMat,
    RayDiff rayDiff, out float2 dUVdx, out float2 dUVdy)
{

    float3 edge01, edge02;
    float3 unnormalizedN, normals[3];
    float2 txcoords[3];

    prepareVerticesForRayDiffs(rayDir, triangleVertices, worldMat, worldInvTransposeMat, barycentrics,
        edge01, edge02, normals, unnormalizedN, txcoords);

    float2 dBarydx, dBarydy;
    computeBarycentricDifferentials(rayDiff, rayDir, edge01, edge02, v.faceNormalW, dBarydx, dBarydy);
    interpolateDifferentials(dBarydx, dBarydy, txcoords, dUVdx, dUVdy);
}

/** Reflects ray differentials using interpolated vertex attributes.
    \param[in] v The mesh vertex data at hit point.
    \param[in] triangleVertices The vertices of the triangle.
    \param[in] barycentrics Barycentric coordinates in the triangle.
    \param[in] rayDir Ray direction.
    \param[in] worldMat World transformation matrix.
    \param[in] worldInvTransposeMat Inverse transpose of world transformation matrix.
    \param[in,out] rayDiff The ray differential used as input and output.
    \param[out] dUVdx The differential of the texture coordinates in pixel coordinate x.
    \param[out] dUVdy The differential of the texture coordinates in pixel coordinate y.
*/
void reflectRayDiffUsingVertexData(VertexData v, StaticVertexData triangleVertices[3], float3 barycentrics, float3 rayDir, float4x4 worldMat, float3x3 worldInvTransposeMat,
    inout RayDiff rayDiff, out float2 dUVdx, out float2 dUVdy)
{

    float3 unnormalizedN;   // Non-normalized interpolated normal for ray differential scatter.
    float3 normals[3];      // Non-normalized normals for ray differential scatter.
    float2 txcoords[3];
    float2 dBarydx, dBarydy;

    float3 edge01 = mul(triangleVertices[1].position - triangleVertices[0].position, (float3x3)worldMat);
    float3 edge02 = mul(triangleVertices[2].position - triangleVertices[0].position, (float3x3)worldMat);
    // Note that we do not need to normalize the individual normals[], since the derivation
    // by Igehy on page 182 normalizes the normal after interpolation
    // Note also that we do not need to (possibly) flip the sign of the normals due to doublesidedness,
    // etc, because the way the computations are used in reflectRayDifferential(), those signs
    // will cancel due to (-A)*(-B) = AB.
    normals[0] = mul(triangleVertices[0].normal, worldInvTransposeMat);
    normals[1] = mul(triangleVertices[1].normal, worldInvTransposeMat);
    normals[2] = mul(triangleVertices[2].normal, worldInvTransposeMat);
    txcoords[0] = triangleVertices[0].texCrd;
    txcoords[1] = triangleVertices[1].texCrd;
    txcoords[2] = triangleVertices[2].texCrd;

    unnormalizedN = normals[0] * barycentrics[0];
    unnormalizedN += normals[1] * barycentrics[1];
    unnormalizedN += normals[2] * barycentrics[2];

    computeBarycentricDifferentials(rayDiff, rayDir, edge01, edge02, v.faceNormalW, dBarydx, dBarydy);
    interpolateDifferentials(dBarydx, dBarydy, txcoords, dUVdx, dUVdy);
    reflectRayDifferential(rayDiff, rayDir, unnormalizedN, v.normalW, dBarydx, dBarydy, normals);
}


// ----------------------------------------------------------------------------
// Environment map sampling helpers
// ----------------------------------------------------------------------------

/** Compute the LOD used when performing a lookup in an environment map when using ray cones
    and under the assumption of using a longitude-latitude environment map. See Chapter 21 in Ray Tracing Gems 1.
    \param[in] spreadAngle The spread angle of the ray cone.
    \param[in] environmentMap The environment map.
    \return The level of detail, lambda.
*/
float computeEnvironmentMapLOD(float spreadAngle, Texture2D environmentMap)
{
    uint txw, txh;
    environmentMap.GetDimensions(txw, txh);
    return log2(abs(spreadAngle) * txh * M_1_PI);                                // From chapter 21 in Ray Tracing Gems
}

/** Compute the LOD used when performing a lookup in an environment map when using ray differentials
    and under the assumption of using a longitude-latitude environment map. See Chapter 21 in Ray Tracing Gems 1.
    \param[in] spreadAngle The spread angle of the ray cone.
    \param[in] environmentMap The environment map.
    \return The level of detail, lambda.
*/
float computeEnvironmentMapLOD(float3 dDdx, float3 dDdy, Texture2D environmentMap)
{
    uint txw, txh;
    environmentMap.GetDimensions(txw, txh);
    return log2(length(dDdx + dDdy) * txh * M_1_PI);                             // From chapter 21 in Ray Tracing Gems
}


// ----------------------------------------------------------------------------
// Curvature estimation helpers
// ----------------------------------------------------------------------------

/** Generic interface for the triangle curvature estimators.
*/
interface ITriangleCurvatureEstimator
{
    /** Returns the estimated curvature from vertex attributes for ray tracing.
        \param[in] edge01..12 The 3 vector edges of the triangle.
        \param[in] curvature01..12 The 3 curvatures associated to the respective edges.
        \return Estimated curvature.
    */
    float eval(float3 edge01, float3 edge02, float3 edge12, float curvature01, float curvature02, float curvature12);
};


struct TriangleCurvature_Average : ITriangleCurvatureEstimator
{
    float eval(float3 edge01, float3 edge02, float3 edge12, float curvature01, float curvature02, float curvature12)
    {
        return ((curvature01 + curvature02 + curvature12) / 3.0f);      // Average triangle curvature.
    }
};


struct TriangleCurvature_Max : ITriangleCurvatureEstimator
{
    float eval(float3 edge01, float3 edge02, float3 edge12, float curvature01, float curvature02, float curvature12)
    {
        float minCurvature = min(curvature01, min(curvature02, curvature12));
        float maxCurvature = max(curvature01, max(curvature02, curvature12));
        return maxCurvature > abs(minCurvature) ? maxCurvature : minCurvature;    // Return maximum of magnitudes with sign to get a conservative estimate.
    }
};

struct TriangleCurvature_DirClosestDP : ITriangleCurvatureEstimator
{
    float3 rayDir;

    float eval(float3 edge01, float3 edge02, float3 edge12, float curvature01, float curvature02, float curvature12)
    {
        // Interpolate the two "closest" curvatures.
        float d01 = abs(dot(rayDir, normalize(edge01)));
        float d02 = abs(dot(rayDir, normalize(edge02)));
        float d12 = abs(dot(rayDir, normalize(edge12)));
        if (d01 < d02)
        {
            if (d01 < d12)
            {
                return (curvature02 * d02 + curvature12 * d12) / (d02 + d12);
            }
            else
            {
                return (curvature01 * d01 + curvature02 * d02) / (d01 + d02);
            }
        }
        else
        {
            if (d02 < d12)
            {
                return (curvature01 * d01 + curvature12 * d12) / (d01 + d12);
            }
            else
            {
                return (curvature01 * d01 + curvature02 * d02) / (d01 + d02);
            }
        }
    }
};

struct TriangleCurvature_Directional : ITriangleCurvatureEstimator
{
    float3 rayDir;

    float eval(float3 edge01, float3 edge02, float3 edge12, float curvature01, float curvature02, float curvature12)
    {
        float3 usedDir = rayDir;

#if 1   // Interpolate using 2 closest angles.
        float a01 = acos(abs(dot(usedDir, normalize(edge01))));
        float a02 = acos(abs(dot(usedDir, normalize(edge02))));
        float a12 = acos(abs(dot(usedDir, normalize(edge12))));

        if (a01 > a02)
        {
            if (a01 > a12)  return (curvature02 * a12 + curvature12 * a02) / (a02 + a12);
            else            return (curvature01 * a02 + curvature02 * a01) / (a01 + a02);
        }
        else
        {
            if (a02 > a12)  return (curvature01 * a12 + curvature12 * a01) / (a01 + a12);
            else            return (curvature01 * a02 + curvature02 * a01) / (a01 + a02);
        }

#elif 0  // Interpolate using dp with closest edges.
        float d01 = abs(dot(usedDir, normalize(edge01)));
        float d02 = abs(dot(usedDir, normalize(edge02)));
        float d12 = abs(dot(usedDir, normalize(edge12)));

        if (d01 < d02)
        {
            if (d01 < d12)  return (curvature02 * d02 + curvature12 * d12) / (d02 + d12);
            else            return (curvature01 * d01 + curvature02 * d02) / (d01 + d02);
        }
        else
        {
            if (d02 < d12)  return (curvature01 * d01 + curvature12 * d12) / (d01 + d12);
            else            return (curvature01 * d01 + curvature02 * d02) / (d01 + d02);
        }

#elif 0   // Interpolate using angles from min/max k.
        float3 minEdge;
        float3 maxEdge;

        if (curvature01 == minCurvature)            minEdge = edge01;
        else if (curvature02 == minCurvature)       minEdge = edge02;
        else                                        minEdge = edge12;

        if (curvature01 == maxCurvature)            maxEdge = edge01;
        else if (curvature02 == maxCurvature)       maxEdge = edge02;
        else                                        maxEdge = edge12;

#if 1
        float aMin = acos(abs(dot(usedDir, normalize(minEdge))));
        float aMax = acos(abs(dot(usedDir, normalize(maxEdge))));
        return (minCurvature * aMax + maxCurvature * aMin) / (aMin + aMax);
#else
        // Dot product approx
        float aMin = abs(dot(usedDir, normalize(minEdge)));
        float aMax = abs(dot(usedDir, normalize(maxEdge)));
        return (minCurvature * aMin + maxCurvature * aMax) / (aMin + aMax);
#endif
#endif
    }
};

struct TriangleCurvature_EllipseVis : ITriangleCurvatureEstimator
{
    float3 rayDir;
    float rayConeWidth;
    float rayConeAngle;

    float eval(float3 edge01, float3 edge02, float3 edge12, float curvature01, float curvature02, float curvature12)
    {
        float minCurvature = min(curvature01, min(curvature02, curvature12));
        float maxCurvature = max(curvature01, max(curvature02, curvature12));

        // Compute ellipse fron raydir.
        float3 geoNormal = normalize(cross(edge01, edge02));
        float3 a1 = (rayDir - geoNormal * dot(geoNormal, rayDir));
        float3 a2 = (cross(geoNormal, a1));

        // Correct length of ellipse axes (from paper).
        float r = rayConeWidth * 0.5;
        a1 *= r / length(a1 - dot(rayDir, a1) * rayDir);
        a2 *= r / length(a2 - dot(rayDir, a2) * rayDir);

        float l1 = length(a1);
        float l2 = length(a2);

        // Find min/max k edges.
        float3 minEdge;
        float3 maxEdge;
        if (curvature01 == minCurvature)        minEdge = edge01;
        else if (curvature02 == minCurvature)   minEdge = edge02;
        else                                    minEdge = edge12;

        if (curvature01 == maxCurvature)        maxEdge = edge01;
        else if (curvature02 == maxCurvature)   maxEdge = edge02;
        else                                    maxEdge = edge12;

        // Transform WS triangle edges to tangent space.
        float3x3 worldToTangent = (float3x3(a1 / l1, a2 / l2, geoNormal));
        float2 edge0TS = (mul(worldToTangent, minEdge)).xy;
        float2 edge1TS = (mul(worldToTangent, maxEdge)).xy;

        // Normalize edges in TS.
        edge0TS.xy = normalize(edge0TS.xy);
        edge1TS.xy = normalize(edge1TS.xy);

        float2 ellipseEq = float2(l1, l2);

        // Distance of the intersection of the edge vectors with the ellipse.
        float aaE0 = (ellipseEq.x * ellipseEq.y) / sqrt(ellipseEq.x * ellipseEq.x * edge0TS.y * edge0TS.y + ellipseEq.y * ellipseEq.y * edge0TS.x * edge0TS.x);
        float aaE1 = (ellipseEq.x * ellipseEq.y) / sqrt(ellipseEq.x * ellipseEq.x * edge1TS.y * edge1TS.y + ellipseEq.y * ellipseEq.y * edge1TS.x * edge1TS.x);
        float maxA = max(aaE0, aaE1);

        // Scale by clipped length and normalize by max.
        float k0 = minCurvature * aaE0 / maxA;
        float k1 = maxCurvature * aaE1 / maxA;

#if 0   // Keep only largest curvature. Doesn't support surfaces with both convex and concave curvatures.
        return abs(k0) > abs(k1) ? k0 : k1;
#elif 0 // Account for current ray spread in case there is a sign choice (like on the pababoloid).
        const float zeroStep = 1.0e-5;
        float sk0 = sign(k0 + zeroStep);
        float sk1 = sign(k1 + zeroStep);

        if (sk0 != sk1)
        {
            if (sign(rayConeAngle) == sk0)  return k0;
            else                            return k1;
        }
        else
        {
            return abs(k0) > abs(k1) ? k0 : k1;
        }
#else   // Enforce that curvature generating largest spread will be used (in case of positive + negative curvatures).
        float dn = -dot(rayDir, geoNormal);
        dn = abs(dn) < 1.0e-5 ? sign(dn) * 1.0e-5 : dn;
        float surfaceSpreadAngle0 = (k0 * rayConeWidth / dn) * 2.0;
        float surfaceSpreadAngle1 = (k1 * rayConeWidth / dn) * 2.0;

        return abs(rayConeAngle + surfaceSpreadAngle0) > abs(rayConeAngle + surfaceSpreadAngle1) ? k0 : k1;
#endif
    }
};
