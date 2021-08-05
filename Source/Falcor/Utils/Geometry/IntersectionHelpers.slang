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
#include "Utils/Math/MathConstants.slangh"

import Utils.Math.MathHelpers;

/** Ray-sphere intersection.
    This function implements the standard analytic test with improvements to floating-point precision
    and returns the closest hit.
    \param[in] rayOrigin Ray origin.
    \param[in] rayDir Ray direction (does not have to be normalized).
    \param[in] center Sphere center.
    \param[in] radius Sphere radius.
    \param[out] t Distance to the closest intersection.
    \return True if the ray intersects the sphere.
*/
bool intersectRaySphere(const float3 rayOrigin, const float3 rayDir, const float3 sphereCenter, const float sphereRadius, out float t)
{
    // Implementation is taken from Chapter 7 of Ray-Tracing Gems
    float3 f = rayOrigin - sphereCenter;
    float a = dot(rayDir, rayDir);
    float b = dot(-f, rayDir);
    float discriminant = sphereRadius * sphereRadius - dot(f + b / a * rayDir, f + b / a * rayDir);

    // Negative discriminant means ray missed sphere.
    if (discriminant < 0.f) return false;

    // If b and discriminant are both 0, then the ray's origin lies on the sphere
    if (b == 0 && discriminant == 0)
    {
        t = 0.f;
        return true;
    }

    // There are two solutions t0 and t1, but one or both may be negative.
    float c = dot(f, f) - sphereRadius * sphereRadius;
    float signB = (b < 0) ? -1 : 1;
    float q = b + signB * sqrt(a * discriminant);
    float t0 = c / q;
    float t1 = q / a;

    float tc = t0 < 0.f ? t1 : t0; // tc is the closest hit we care about
    if (tc < 0.f) return false;

    t = tc;
    return true;
}

/** Ray-AABB intersection.
    \param[in] rayOrigin Ray origin.
    \param[in] rayDir Ray direction (does not have to be normalized).
    \param[in] aabbMin AABB minimum.
    \param[in] aabbMax AABB maximum.
    \param[out] nearFar Returns intersection interval along ray.
    \return True if the ray intersects the AABB.
*/
bool intersectRayAABB(const float3 rayOrigin, const float3 rayDir, const float3 aabbMin, const float3 aabbMax, out float2 nearFar)
{
    const float3 invDir = 1.f / rayDir;
    const float3 lo = (aabbMin - rayOrigin) * invDir;
    const float3 hi = (aabbMax - rayOrigin) * invDir;
    const float3 tmin = min(lo, hi), tmax = max(lo, hi);
    nearFar.x = max(0.f, max(tmin.x, max(tmin.y, tmin.z)));
    nearFar.y = min(tmax.x, min(tmax.y, tmax.z));
    return nearFar.x <= nearFar.y;
}

/** Ray-Triangle intersection based on "Watertight Ray/Triangle Intersection"
    Paper link: http://jcgt.org/published/0002/01/05/paper.pdf
    \param[in] rayOrigin Ray origin.
    \param[in] rayDir Ray direction (does not have to be normalized).
    \param[in] vertices Triangle vertices.
    \param[out] barycentrics Barycentric coordinates.
    \param[out] t Intersection distance.
    \return True if the ray intersects the triangle.
*/
bool intersectRayTriangle(const float3 rayOrigin, const float3 rayDir, const float3 vertices[3], out float3 barycentrics, out float t)
{
    const float3 absDir = abs(rayDir);
    uint axis = 0;
    if (absDir.y > absDir.x && absDir.y > absDir.z) axis = 1;
    if (absDir.z > absDir.x && absDir.z > absDir.y) axis = 2;

    uint kz = axis;
    uint kx = (kz + 1) % 3;
    uint ky = (kx + 1) % 3;
    if (rayDir[kz] < 0.f)
    {
        uint swap = kx;
        kx = ky;
        ky = swap;
    }

    float Sx = rayDir[kx] / rayDir[kz];
    float Sy = rayDir[ky] / rayDir[kz];
    float Sz = 1.f / rayDir[kz];

    const float3 A = vertices[0] - rayOrigin;
    const float3 B = vertices[1] - rayOrigin;
    const float3 C = vertices[2] - rayOrigin;

    const float Ax = A[kx] - Sx * A[kz];
    const float Ay = A[ky] - Sy * A[kz];
    const float Bx = B[kx] - Sx * B[kz];
    const float By = B[ky] - Sy * B[kz];
    const float Cx = C[kx] - Sx * C[kz];
    const float Cy = C[ky] - Sy * C[kz];

    float U = Cx * By - Cy * Bx;
    float V = Ax * Cy - Ay * Cx;
    float W = Bx * Ay - By * Ax;

    if (U == 0.f || V == 0.f || W == 0.f)
    {
        double CxBy = (double)Cx * (double)By;
        double CyBx = (double)Cy * (double)Bx;
        U = (float)(CxBy - CyBx);
        double AxCy = (double)Ax * (double)Cy;
        double AyCx = (double)Ay * (double)Cx;
        V = (float)(AxCy - AyCx);
        double BxAy = (double)Bx * (double)Ay;
        double ByAx = (double)By * (double)Ax;
        W = (float)(BxAy - ByAx);
    }

    if ((U < 0.f || V < 0.f || W < 0.f) && (U > 0.f || V > 0.f || W > 0.f)) return false;

    float det = U + V + W;
    if (det == 0.f) return false;

    const float Az = Sz * A[kz];
    const float Bz = Sz * B[kz];
    const float Cz = Sz * C[kz];
    const float T = U * Az + V * Bz + W * Cz;
    const float rcpDet = 1.f / det;

    t = T * rcpDet;
    barycentrics = float3(U, V, W) * rcpDet;

    return true;
}

/** Ray intersection against linear swept sphere based on [Han et al. 2019], Ray Tracing Generalized Tube Primitives: Method and Applications.
    Paper link: http://www.sci.utah.edu/publications/Han2019a/tubes-final.pdf
    \param[in] rayOrigin Ray origin position.
    \param[in] rayDir Unit ray direction vector.
    \param[in] sphereA Sphere (3D position + radius) at one end point.
    \param[in] sphereB Sphere at the other end point.
    \param[in] useSphereJoints Indicating whether we test ray-sphere intersection at curve joints or not.
    \param[out] result The closest intersection distance t, and a parameter u for linear interpolation (between 0 and 1).
    \return True if the ray intersects the linear swept sphere segment.
*/
bool intersectLinearSweptSphereHan19(float3 rayOrigin, float3 rayDir, float4 sphereA, float4 sphereB, bool useSphereJoints, out float2 result)
{
    result = float2(FLT_MAX);

    bool reversed = false;
    if (sphereA.w > sphereB.w)
    {
        float4 tmp = sphereA;
        sphereA = sphereB;
        sphereB = tmp;
        reversed = true;
    }

    const float3 P1 = sphereA.xyz;
    const float3 P2 = sphereB.xyz;
    const float r1 = sphereA.w;
    const float r2 = sphereB.w;

    // Perpendicular distance to closest of (p0, p1) minus the max radius.
    float t0 = min(dot(rayDir, P1 - rayOrigin), dot(rayDir, P2 - rayOrigin)) - max(r1, r2);
    t0 = max(0.f, t0);
    // For better numerical stability, push the ray to be as close as possible to the curve.
    rayOrigin += t0 * rayDir;

    if (useSphereJoints)
    {
        // Intersecting two sphere endcaps.
        float t;
        if (intersectRaySphere(rayOrigin, rayDir, P1, r1, t))
        {
            if (t < result.x)
            {
                result.x = t;
                result.y = (reversed ? 1.f : 0.f);
            }
        }
        if (intersectRaySphere(rayOrigin, rayDir, P2, r2, t))
        {
            if (t < result.x)
            {
                result.x = t;
                result.y = (reversed ? 0.f : 1.f);
            }
        }
    }

    // Intersecting cone.
    float3 C = P2 - P1;
    const float lengthC = length(C);
    C /= lengthC;

    const float p1 = lengthC * r1 / (r2 - r1);
    const float p2 = p1 * r2 / r1;
    const float3 A = P1 - p1 * C;
    const float z1 = p1 - r1 * r1 / p1;
    const float z2 = p2 - r2 * r2 / p2;
    const float w = p2 * r2 / sqrt(p2 * p2 - r2 * r2);

    const float3 vz = C;
    const float3 vx = perp_stark(vz);
    const float3 vy = cross(vz, vx);

    const float tmp1 = 1.f / z2;
    const float tmp2 = p2 / w;

    // Row-major matrix.
    float4x4 M =
    {
        tmp1 * tmp2 * float4(vx, -dot(vx, A)),
        tmp1 * tmp2 * float4(vy, -dot(vy, A)),
        tmp1 * float4(vz, -dot(vz, A)),
        float4(0, 0, 0, 1)
    };

    const float zCap = z1 * tmp1;
    const float3 rayOriginXf = mul(M, float4(rayOrigin, 1.f)).xyz;
    const float3 rayDirXf = mul((float3x3)M, rayDir);

    const float a = rayDirXf.x * rayDirXf.x + rayDirXf.y * rayDirXf.y - rayDirXf.z * rayDirXf.z;
    const float b = 2.f * (rayOriginXf.x * rayDirXf.x + rayOriginXf.y * rayDirXf.y - rayOriginXf.z * rayDirXf.z);
    const float c = rayOriginXf.x * rayOriginXf.x + rayOriginXf.y * rayOriginXf.y - rayOriginXf.z * rayOriginXf.z;
    const float disc = b * b - 4 * a * c;
    if (disc >= 0)
    {
        const float sqrtDisc = sqrt(disc);
        const float inv2a = 0.5f / a;

        [unroll]
        for (int i = 0; i < 2; i++)
        {
            float t = (-b + (i * 2.f - 1.f) * sqrtDisc) * inv2a;
            if (t >= 0 && t < result.x)
            {
                // Check if z is in the valid range.
                const float z = rayOriginXf.z + t * rayDirXf.z;
                if (z >= zCap && z <= 1.f)
                {
                    const float u = (z - zCap) / (1.f - zCap);
                    result.x = t;
                    result.y = (reversed ? 1.f - u : u);
                }
            }
        }
    }

    result.x += t0;
    return (result.x < FLT_MAX);
}
