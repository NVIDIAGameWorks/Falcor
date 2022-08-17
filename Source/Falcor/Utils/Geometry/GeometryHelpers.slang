/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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

/*******************************************************************
    Ray tracing
*******************************************************************/

/** Computes new ray origin based on hit position to avoid self-intersections.
    The function assumes that the hit position has been computed by barycentric
    interpolation, and not from the ray t which is less accurate.

    The method is described in Ray Tracing Gems, Chapter 6, "A Fast and Robust
    Method for Avoiding Self-Intersection" by Carsten Wächter and Nikolaus Binder.

    \param[in] pos Ray hit position.
    \param[in] normal Face normal of hit surface (normalized). The offset will be in the positive direction.
    \return Ray origin of the new ray.
*/
float3 computeRayOrigin(float3 pos, float3 normal)
{

    const float origin = 1.f / 16.f;
    const float fScale = 3.f / 65536.f;
    const float iScale = 3 * 256.f;

    // Per-component integer offset to bit representation of fp32 position.
    int3 iOff = int3(normal * iScale);
    float3 iPos = asfloat(asint(pos) + (pos < 0.f ? -iOff : iOff));

    // Select per-component between small fixed offset or above variable offset depending on distance to origin.
    float3 fOff = normal * fScale;
    return abs(pos) < origin ? pos + fOff : iPos;
}


/*******************************************************************
    Bounding cones
*******************************************************************/

/** Computes the cosine of the half angle of the minimum bounding cone that encloses an AABB, as seen from a particular viewpoint.
    We use an optimized algorithm that exploits the symmetry around the plane perpendicular to the central direction.

    \param[in] origin Viewpoint origin.
    \param[in] aabbMin minimum corner of the AABB.
    \param[in] aabbMax maximum corner of the AABB.
    \param[out] coneDir normalized vector defining the cone's axis.
    \param[out] sinTheta Sine of the angle from the central direction to the cone edge. If the AABB can't be bounded we return 0.
    \param[out] cosTheta Cosine of the angle from the central direction to the cone edge. If the AABB can't be bounded we return -1 (max cone).
*/
void boundBoxSubtendedConeAngleCenter(const float3 origin, const float3 aabbMin, const float3 aabbMax,
                                      out float3 coneDir, out float sinTheta, out float cosTheta)
{
    const float3 center = (aabbMax + aabbMin) * 0.5f;
    const float3 extent = (aabbMax - aabbMin) * 0.5f;
    const float3 dir = center - origin;         // dir = Central cone direction (unnormalized)
    const float extSqr = dot(extent, extent);   // extSqr = squared maximum extent
    const float distSqr = dot(dir, dir);        // distSqr = squared distance to AABB center

    coneDir = normalize(dir);

    // AABB has eight corners, located at p = center +- e[i] for i=1..4.
    // We use the absolute value of the dot product below to avoid having to test all eight.
    float3 e[4];
    e[0] = float3(extent.x, extent.y, extent.z);
    e[1] = float3(extent.x, extent.y, -extent.z);
    e[2] = float3(extent.x, -extent.y, extent.z);
    e[3] = float3(extent.x, -extent.y, -extent.z);

    cosTheta = 1.f;
    sinTheta = 0.f;

    [unroll]
    for (uint i = 0; i < 4; i++)
    {
        // Compute distance x from origin to corner projected onto central axis.
        // Note that x is scaled by |dir| since we use unnormalized vectors.
        float d = abs(dot(dir, e[i]));
        float x = distSqr - d;

        // Check if distance is negative, in which case the AABB can't be bounded by a cone and we return a cone that covers the whole sphere (theta = pi).
        if (x < 1e-5)
        {
            cosTheta = -1.f;
            sinTheta = 0.f;
            return;                         // TODO: Look at numerical precision.
        }

        // Compute distance y from the corner to the projection on the central axis (also scaled by |dir|).
        float y = sqrt(max(0, distSqr * extSqr - d * d));   // TODO: Look at numerical precision. Clamp for now just to be extra safe.

        // Compute hypotenuse of the triangle.
        float z = sqrt(x * x + y * y);

        // The cosine of the cone angle cos(theta) = x / z. Track the minimum cosTheta, since we want
        // cos(theta) for the maximum theta.
        cosTheta = min(cosTheta, x / z);
        // And along similar lines for sin(theta).
        sinTheta = max(sinTheta, y / z);
    }
}

/** Computes the solid angle subtended by an AABB by first computing the
    average vector to all of its vertices and then finding the maximum
    angle between that and each of the vectors to its vertices.

    \param[in] origin point from which the solid angle is being comupted.
    \param[in] aabbMin minimum corner of the AABB.
    \param[in] aabbMax maximum corner of the AABB.
    \param[out] coneDir central cone direction (normalized) or null vector if origin is inside the AABB.
    \param[out] sinTheta sine of the angle.
    \param[out] cosTheta cosine of the angle.
*/
void boundBoxSubtendedConeAngleAverage(float3 origin, float3 aabbMin, float3 aabbMax,
                                       out float3 coneDir, out float sinTheta, out float cosTheta)
{
    if (all(origin >= aabbMin && origin <= aabbMax))
    {
        // |origin| is inside the AABB.
        coneDir = float3(0.f, 0.f, 0.f);
        sinTheta = 0.f;
        cosTheta = -1.f;
        return;
    }

    // Compute the average vector to each of the bounding box corners.
    float3 dirSum = float3(0.f, 0.f, 0.f);
    [unroll]
    for (int i = 0; i < 8; ++i)
    {
        const float3 corner = float3((i & 1) ? aabbMin.x : aabbMax.x,
                                     (i & 2) ? aabbMin.y : aabbMax.y,
                                     (i & 4) ? aabbMin.z : aabbMax.z);
        dirSum += normalize(corner - origin);
    }
    coneDir = normalize(dirSum);

    // Compute the cosine of the maximum angle between a corner and the
    // average vector.
    cosTheta = 1.f;
    [unroll]
    for (int i = 0; i < 8; ++i)
    {
        const float3 corner = float3((i & 1) ? aabbMin.x : aabbMax.x,
                                     (i & 2) ? aabbMin.y : aabbMax.y,
                                     (i & 4) ? aabbMin.z : aabbMax.z);
        cosTheta = min(cosTheta, dot(normalize(corner - origin), coneDir));
    }
    sinTheta = sqrt(max(0.f, 1.f - cosTheta * cosTheta));
}

/** Computes the sine and cosine of the angle of the cone that encompasses
    a sphere of given radius and center subtend as seen from a point at the origin.

    \param[in] center Sphere's center.
    \param[in] sqrRadius Square of sphere's radius.
    \param[out] sinTheta Sine of the angle between a vector from the origin to |center| and a
                vector from the origin that is tangent to the sphere.
    \param[out] cosTheta Cosine of that angle.
*/
void boundSphereSubtendedConeAngle(float3 center, float sqrRadius, out float sinTheta, out float cosTheta)
{
    // Is the point inside the bounding sphere?
    float centerDistance2 = dot(center, center);
    if (centerDistance2 < sqrRadius)
    {
        // We're inside. Return the full cone.
        sinTheta = 0.f;
        cosTheta = -1.f;
    }
    else
    {
        // Compute the sine and then the cosine of the spread angle of a
        // cone that bounds the sphere as seen from |center|.
        float sin2Theta = sqrRadius / centerDistance2;
        cosTheta = sqrt(1.f - sin2Theta);
        sinTheta = sqrt(sin2Theta);
    }
}


/*******************************************************************
    Triangle utilities
*******************************************************************/

/** Computes the squared minimum distance between a point and a triangle.
    This function is not sensitive to the handedness of the coordinate system.
    \param[in] vertices Positions of the three vertices.
    \param[in] p Coordinates of the point.
    \return Squared minimum distance between p and the triangle.
*/
float computeSquaredMinDistanceToTriangle(const float3 vertices[3], const float3 p)
{
    // Project p onto the plane of the triangle (the result is independent of triangle winding).
    const float3 n = normalize(cross(vertices[1] - vertices[0], vertices[2] - vertices[0]));
    const float projDistance = dot(n, (p - vertices[0]));
    const float3 pProj = p - projDistance * n;

    // Edge tests to compute signed distance to each edge.
    // Positive result means the projected point is "inside" the edge.
    // With flipped winding, the edges are flipped but n is also flipped so it still works.
    const float3 edges[3] = {
        normalize(vertices[1] - vertices[0]),
        normalize(vertices[2] - vertices[1]),
        normalize(vertices[0] - vertices[2])
    };
    float sqrPlanarDistance = FLT_MAX;
    uint insideMask = 0u;
    [unroll]
    for (uint i = 0u; i < 3u; ++i)
    {
        const float3 edgeN = cross(n, edges[i]);
        const float edgeProjDistance = dot(edgeN, pProj - vertices[i]);
        if (edgeProjDistance >= 0.0f)
        {
            insideMask |= 1u << i;
        }
        else
        {
            const float3 vec = pProj - vertices[i];
            sqrPlanarDistance = min(edgeProjDistance * edgeProjDistance, sqrPlanarDistance);
        }
    }
    if (insideMask == 0x7)
    {
        sqrPlanarDistance = 0.0f;
    }

    // If only one edge is considering the point as inside, then the projected point
    // is closest to a triangle corner (the vertex opposite of that edge).
    else if (insideMask == 1u << 0u)
    {
        sqrPlanarDistance = dot(pProj - vertices[2], pProj - vertices[2]);
    }
    else if (insideMask == 1u << 1u)
    {
        sqrPlanarDistance = dot(pProj - vertices[0], pProj - vertices[0]);
    }
    else if (insideMask == 1u << 2u)
    {
        sqrPlanarDistance = dot(pProj - vertices[1], pProj - vertices[1]);
    }

    return projDistance * projDistance + sqrPlanarDistance;
}

/** Utility function to classify a point with respect to an axis-aligned plane in 2D.
    \param[in] p Point.
    \param[in] axis Axis of clip plane (0=x, 1=y).
    \param[in] sign Direction of clip plane normal (+1 positive, -1 negative).
    \param[in] c Location of clip plane along the chosen axis.
    \param[in] planeThickness Optional plane thickness.
    \return 1 if point is in front, -1 if behind, or 0 if it lies on the plane.
*/
int classifyPointPlane2D(
    const float2 p,
    const uint axis,
    const float sign,
    const float c,
    const float planeThickness = 1e-6f)
{
    float d = sign * (p[axis] - c);
    if (d > planeThickness) return 1;
    else if (d < -planeThickness) return -1;
    else return 0;
}

/** Utility function to clip a polygon against an axis-aligned plane in 2D.
    The input can be at most a hexagon (n=6) as one new vertex may be added.
    \param[in,out] p Polygon vertices before/after clipping.
    \param[in,out] n Number of vertices before/after clipping.
    \param[in] axis Axis of clip plane (0=x, 1=y).
    \param[in] sign Direction of clip plane normal.
    \param[in] c Location of clip plane along the chosen axis.
*/
void clipPolygonPlane2D(
    inout float2 p[7],
    inout uint n,
    const uint axis,
    const float sign,
    const float c)
{
    if (n <= 1)
    {
        n = 0;
        return;
    }

    float2 q[7];
    uint k = 0;
    bool fullyOnPlane = true;

    float2 p1 = p[n - 1];
    int d1 = classifyPointPlane2D(p1, axis, sign, c);

    // Iterate over all polygon edges (p1,p2) in order.
    for (uint i = 0; i < n; i++)
    {
        float2 p2 = p[i];
        int d2 = classifyPointPlane2D(p2, axis, sign, c);

        if (d2 == 0) // p2 lies on the plane
        {
            if (d1 != 0) q[k++] = p2;
        }
        else // p2 is on either side
        {
            fullyOnPlane = false;

            if (d1 == 0) // p1 lies on the plane
            {
                if (k == 0 || any(q[k - 1] != p1)) q[k++] = p1;
            }
            else if (d1 != d2) // p1 and p2 are on opposite sides => clip
            {
                float alpha = (p2[axis] - c) / (p2[axis] - p1[axis]);
                q[k++] = lerp(p2, p1, alpha);
            }

            if (d2 > 0) q[k++] = p2;
        }

        p1 = p2;
        d1 = d2;
    }

    if (fullyOnPlane) return;

    n = k;
    for (uint i = 0; i < k; i++) p[i] = q[i];
}

/** Computes the signed area of a triangle clipped to an axis-aligned box in 2D.
    The area is positive for counter-clockwise winding in the xy-plane, assuming
    a right-handed Cartesian coordinate system with x-axis right and y-axis up.
    \param[in] pos Positions of the three vertices.
    \param[in] minPoint Minimum point of the box.
    \param[in] maxPoint Maximum point of the box. It is assumed that maxPoint > minPoint.
    \return Signed area of the triangle clipped to the box.
*/
float computeClippedTriangleArea2D(const float2 pos[3], const float2 minPoint, const float2 maxPoint)
{
    // Clip triangle to axis-aligned box.
    uint n = 3;
    float2 p[7] = {};

    p[0] = pos[0];
    p[1] = pos[1];
    p[2] = pos[2];

    clipPolygonPlane2D(p, n, 0, +1.f, minPoint.x);
    clipPolygonPlane2D(p, n, 0, -1.f, maxPoint.x);
    clipPolygonPlane2D(p, n, 1, +1.f, minPoint.y);
    clipPolygonPlane2D(p, n, 1, -1.f, maxPoint.y);

    if (n < 3) return 0.f;

    // Compute area of convex polygon.
    // The vertices must be specified in order. If counter-clockwise, area is positive.
    float area = 0.f;
    for (uint i = 0; i < n; i++)
    {
        uint j = i + 1 < n ? i + 1 : 0;
        area += p[i].x * p[j].y - p[i].y * p[j].x;
    }

    return 0.5f * area;
}

/*******************************************************************
    Curve uv parameterization
*******************************************************************/

/** Computes the uv parameters (from 0 to 1) from ray-curve hit position.

    \param[in] hitPos Position of the ray-curve hit point.
    \param[in] sphereA Sphere (3D position + radius) at one end point.
    \param[in] sphereB Sphere (3D position + radius) at the other end point.
    \return UV parameters.
*/
float2 curveHitPosToUV(const float3 hitPos, const float4 sphereA, const float4 sphereB)
{
    // Build a reference frame.
    float3 fwd = sphereB.xyz - sphereA.xyz;
    float fwdLength = length(fwd);
    fwd /= fwdLength;
    float3 s = perp_stark(fwd);
    float3 t = cross(fwd, s);

    // Projection.
    float3 v = hitPos - sphereA.xyz;
    float z = dot(fwd, v);
    float x = dot(s, v);
    float y = dot(t, v);

    float2 result;
    // Remapping from [-r1, r2 + fwdLength] to [0, 1].
    result.x = (z + sphereA.w) / (sphereA.w + sphereB.w + fwdLength);
    // Remapping from [-pi, pi] to [0, 1].
    result.y = atan2(y, x) / M_2PI + 0.5f;
    return result;
}

/** Computes the ray-curve hit position and the corresponding sphere center from uv parameters.
    It is based on this paper: http://www.sci.utah.edu/publications/Han2019a/tubes-final.pdf

    \param[in] uv UV parameters of the hit point.
    \param[in] sphereA Sphere (3D position + radius) at one end point.
    \param[in] sphereB Sphere (3D position + radius) at the other end point.
    \param[out] hitPos Position of the ray-curve hit point.
    \param[out] sphereCenter Position of the corresponding sphere center (for computing the normal direction).
*/
void uvToCurveHitPos(const float2 uv, const float4 sphereA, const float4 sphereB, out float3 hitPos, out float3 sphereCenter)
{
    // Build a reference frame.
    float3 fwd = sphereB.xyz - sphereA.xyz;
    float fwdLength = length(fwd);
    fwd /= fwdLength;
    float3 s = perp_stark(fwd);
    float3 t = cross(fwd, s);

    // Remapping.
    float z = uv.x * (sphereA.w + sphereB.w + fwdLength) - sphereA.w;
    float phi = (uv.y - 0.5f) * M_2PI;

    float p1, z1, p2, z2;
    float radius;
    float deltaZ;

    if (abs(sphereA.w - sphereB.w) < 1e-6f)
    {
        p1 = z1 = 0.f;
        p2 = z2 = fwdLength;
    }
    else
    {
        p1 = fwdLength * sphereA.w / (sphereB.w - sphereA.w);
        p2 = p1 + fwdLength;
        z1 = p1 - sphereA.w * sphereA.w / p1;
        z2 = p2 - sphereB.w * sphereB.w / p2;
    }

    if (p1 + z < z1)
    {
        // Case 1: hitPos is on the end cap of sphereA.
        deltaZ = -z;
        radius = sphereA.w;

    }
    else if (p1 + z > z2)
    {
        // Case 2: hitPos is on the end cap of sphereB.
        deltaZ = fwdLength - z;
        radius = sphereB.w;
    }
    else
    {
        // Case 3: hitPos is on the linear segment.
        if (abs(sphereA.w - sphereB.w) < 1e-6f)
        {
            radius = sphereA.w;
            deltaZ = 0.f;
        }
        else
        {
            float ratio = (p1 + z) / z1;
            radius = sphereA.w * ratio;
            deltaZ = p1 * ratio - (p1 + z);
        }
    }

    float width = sqrt(max(radius * radius - deltaZ * deltaZ, 0.f));

    float3 perpToFwd = width * (cos(phi) * s + sin(phi) * t);
    hitPos = sphereA.xyz + z * fwd + perpToFwd;
    sphereCenter = sphereA.xyz + (z + deltaZ) * fwd;
}

/** Extract scale from a 4x4 transformation matrix.
    We assume scaling is isotropic; i.e., it keeps the sphere shape after transformation.

    \param[in] xform Transformation matrix.
    \return Isotropic scaling factor.
*/
float extractScaleFromTransform(const float4x4 xform)
{
    return sqrt(xform[0][0] * xform[0][0] + xform[0][1] * xform[0][1] + xform[0][2] * xform[0][2]);
}


/** Transform a sphere at curve's end.

    \param[in] sphere Sphere (3D position + radius) at curve's end point.
    \param[in] worldMat Local-to-world transformation matrix.
    \return Transformed sphere.
*/
float4 transformSphere(const float4 sphere, const float4x4 worldMat)
{
    float scaleFactor = extractScaleFromTransform(worldMat);
    float3 worldPos = mul(worldMat, float4(sphere.xyz, 1.f)).xyz;
    return float4(worldPos, sphere.w * scaleFactor);
}
