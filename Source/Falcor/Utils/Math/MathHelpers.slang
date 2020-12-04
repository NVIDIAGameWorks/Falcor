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

/** This file contains various math utility helper functions.

    Included functionality (in order):

    - Sherical coordinates mapping functions
    - Octahedral mapping functions
    - Sampling functions (disk, sphere, triangle etc.)
    - Misc stuff (matrix inversion, bounding cones etc.)

*/

// Include math constants (M_PI etc.). These are for use in this file only,
// as macro definitions are not exported from a Slang module.
#include "Utils/Math/MathConstants.slangh"

/******************************************************************************

    Spherical coordinates

    Functions for converting Cartesian coordinates to spherical coordinates
    using standard mathematical notations.

    The latitude-longitude map uses (phi,theta) as positions in two dimensions.
    Its using using other conventions to orient and wrap the map the same way
    as in common 3D software (e.g. Autodesk Maya).

******************************************************************************/

/** Converts Cartesian coordinates to spherical coordinates (unsigned normalized).
    'theta' is the polar angle (inclination) between the +z axis and the vector from origin to p, normalized to [0,1].
    'phi' is the azimuthal angle from the +x axis in the xy-plane, normalized to [0,1].
    \param[in] p Cartesian coordinates (x,y,z).
    \return Spherical coordinates (theta,phi).
*/
float2 cartesian_to_spherical_unorm(float3 p)
{
    p = normalize(p);
    float2 sph;
    sph.x = acos(p.z) * M_1_PI;
    sph.y = atan2(-p.y, -p.x) * M_1_2PI + 0.5f;
    return sph;
}

/** Converts Cartesian coordinates to spherical coordinates (radians).
    'theta' is the polar angle (inclination) between the +z axis and the vector from origin to p, in the range [0,pi].
    'phi' is the azimuthal angle from the +x axis in the xy-plane, in the range [0,2pi].
    \param[in] p Cartesian coordinates (x,y,z).
    \return Spherical coordinates (theta,phi).
*/
float2 cartesian_to_spherical_rad(float3 p)
{
    p = normalize(p);
    float2 sph;
    sph.x = acos(p.z);
    sph.y = atan2(-p.y, -p.x) + M_PI;
    return sph;
}

/** Convert world space direction to (u,v) coord in latitude-longitude map (unsigned normalized).
    The map is centered around the -z axis and wrapping around in clockwise order (left to right).
    \param[in] dir World space direction (unnormalized).
    \return Position in latitude-longitude map in [0,1] for each component.
*/
float2 world_to_latlong_map(float3 dir)
{
    float3 p = normalize(dir);
    float2 uv;
    uv.x = atan2(p.x, -p.z) * M_1_2PI + 0.5f;
    uv.y = acos(p.y) * M_1_PI;
    return uv;
}

/** Convert a coordinate in latitude-longitude map (unsigned normalized) to a world space direction.
    The map is centered around the -z axis and wrapping around in clockwise order (left to right).
    \param[in] latlong Position in latitude-longitude map in [0,1] for each component.
    \return Normalized direction in world space.
*/
float3 latlong_map_to_world(float2 latlong)
{
    float phi = M_PI * (2.f * saturate(latlong.x) - 1.f);
    float theta = M_PI * saturate(latlong.y);
    float sinTheta = sin(theta);
    float cosTheta = cos(theta);
    float sinPhi = sin(phi);
    float cosPhi = cos(phi);
    return float3(sinTheta * sinPhi, cosTheta, -sinTheta * cosPhi);
}

/******************************************************************************

    Octahedral mapping

    The center of the map represents the +z axis and its corners -z.
    The rotated inner square is the xy-plane projected onto the upper hemi-
    sphere, the outer four triangles folds down over the lower hemisphere.
    There are versions for equal-area and non-equal area (slightly faster).

    For details refer to:
    - Clarberg 2008, "Fast Equal-Area Mapping of the (Hemi)Sphere using SIMD".
    - Cigolle et al. 2014, "Survey of Efficient Representations for Independent Unit Vectors".

******************************************************************************/

/** Helper function to reflect the folds of the lower hemisphere
    over the diagonals in the octahedral map.
*/
float2 oct_wrap(float2 v)
{
    return (1.f - abs(v.yx)) * (v.xy >= 0.f ? 1.f : -1.f);
}

/** Converts normalized direction to the octahedral map (non-equal area, signed normalized).
    \param[in] n Normalized direction.
    \return Position in octahedral map in [-1,1] for each component.
*/
float2 ndir_to_oct_snorm(float3 n)
{
    // Project the sphere onto the octahedron (|x|+|y|+|z| = 1) and then onto the xy-plane.
    float2 p = n.xy * (1.f / (abs(n.x) + abs(n.y) + abs(n.z)));
    p = (n.z < 0.f) ? oct_wrap(p) : p;
    return p;
}

/** Converts normalized direction to the octahedral map (non-equal area, unsigned normalized).
    \param[in] n Normalized direction.
    \return Position in octahedral map in [0,1] for each component.
*/
float2 ndir_to_oct_unorm(float3 n)
{
    return ndir_to_oct_snorm(n) * 0.5f + 0.5f;
}

/** Converts point in the octahedral map to normalized direction (non-equal area, signed normalized).
    \param[in] p Position in octahedral map in [-1,1] for each component.
    \return Normalized direction.
*/
float3 oct_to_ndir_snorm(float2 p)
{
    float3 n = float3(p.xy, 1.0 - abs(p.x) - abs(p.y));
    n.xy = (n.z < 0.0) ? oct_wrap(n.xy) : n.xy;
    return normalize(n);
}

/** Converts point in the octahedral map to normalized direction (non-equal area, unsigned normalized).
    \param[in] p Position in octahedral map in [0,1] for each component.
    \return Normalized direction.
*/
float3 oct_to_ndir_unorm(float2 p)
{
    return oct_to_ndir_snorm(p * 2.f - 1.f);
}

/** Converts normalized direction to the octahedral map (equal-area, unsigned normalized).
    \param[in] n Normalized direction.
    \return Position in octahedral map in [0,1] for each component.
*/
float2 ndir_to_oct_equal_area_unorm(float3 n)
{
    // Use atan2 to avoid explicit div-by-zero check in atan(y/x).
    float r = sqrt(1.f - abs(n.z));
    float phi = atan2(abs(n.y), abs(n.x));

    // Compute p = (u,v) in the first quadrant.
    float2 p;
    p.y = r * phi * M_2_PI;
    p.x = r - p.y;

    // Reflect p over the diagonals, and move to the correct quadrant.
    if (n.z < 0.f) p = 1.f - p.yx;
    p *= sign(n.xy);

    return p * 0.5f + 0.5f;
}

/** Converts point in the octahedral map to normalized direction (equal area, unsigned normalized).
    \param[in] p Position in octahedral map in [0,1] for each component.
    \return Normalized direction.
*/
float3 oct_to_ndir_equal_area_unorm(float2 p)
{
    p = p * 2.f - 1.f;

    // Compute radius r without branching. The radius r=0 at +z (center) and at -z (corners).
    float d = 1.f - (abs(p.x) + abs(p.y));
    float r = 1.f - abs(d);

    // Compute phi in [0,pi/2] (first quadrant) and sin/cos without branching.
    // TODO: Analyze fp32 precision, do we need a small epsilon instead of 0.0 here?
    float phi = (r > 0.f) ? ((abs(p.y) - abs(p.x)) / r + 1.f) * M_PI_4 : 0.f;

    // Convert to Cartesian coordinates. Note that sign(x)=0 for x=0, but that's fine here.
    float f = r * sqrt(2.f - r*r);
    float x = f * sign(p.x) * cos(phi);
    float y = f * sign(p.y) * sin(phi);
    float z = sign(d) * (1.f - r*r);

    return float3(x, y, z);
}

/******************************************************************************

    Sampling functions

******************************************************************************/

/** Uniform sampling of the unit disk using polar coordinates.
    \param[in] u Uniform random number in [0,1)^2.
    \return Sampled point on the unit disk.
*/
float2 sample_disk(float2 u)
{
    float2 p;
    float r = sqrt(u.x);
    float phi = M_2PI * u.y;
    p.x = r * cos(phi);
    p.y = r * sin(phi);
    return p;
}

/** Uniform sampling of direction within a cone
    \param[in] u Uniform random number in [0,1)^2.
    \param[in] cosTheta Cosine of the cone half-angle
    \return Sampled direction within the cone with (0,0,1) axis
*/
float3 sample_cone(float2 u, float cosTheta)
{
    float z = u.x * (1.f - cosTheta) + cosTheta;
    float r = sqrt(1.f - z*z);
    float phi = M_2PI * u.y;
    return float3(r * cos(phi), r * sin(phi), z);
}

/** Uniform sampling of the unit sphere using spherical coordinates.
    \param[in] u Uniform random numbers in [0,1)^2.
    \return Sampled point on the unit sphere.
*/
float3 sample_sphere(float2 u)
{
    float phi = M_2PI * u.y;
    float cosTheta = 1.0f - 2.0f * u.x;
    float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));
    return float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
}

/** Uniform sampling of the unit disk using Shirley's concentric mapping.
    \param[in] u Uniform random numbers in [0,1)^2.
    \return Sampled point on the unit disk.
*/
float2 sample_disk_concentric(float2 u)
{
    u = 2.f * u - 1.f;
    if (u.x == 0.f && u.y == 0.f) return u;
    float phi, r;
    if (abs(u.x) > abs(u.y))
    {
        r = u.x;
        phi = (u.y / u.x) * M_PI_4;
    }
    else
    {
        r = u.y;
        phi = M_PI_2 - (u.x / u.y) * M_PI_4;
    }
    return r * float2(cos(phi), sin(phi));
}

/** Cosine-weighted sampling of the hemisphere using Shirley's concentric mapping.
    \param[in] u Uniform random numbers in [0,1)^2.
    \param[out] pdf Probability density of the sampled direction (= cos(theta)/pi).
    \return Sampled direction in the local frame (+z axis up).
*/
float3 sample_cosine_hemisphere_concentric(float2 u, out float pdf)
{
    float2 d = sample_disk_concentric(u);
    float z = sqrt(max(0.f, 1.f - dot(d, d)));
    pdf = z * M_1_PI;
    return float3(d, z);
}

/** Cosine-weighted sampling of the hemisphere using a polar coordinates.
    \param[in] u Uniform random numbers in [0,1)^2.
    \param[out] pdf Probability density of the sampled direction (= cos(theta)/pi).
    \return Sampled direction in the local frame (+z axis up).
*/
float3 sample_cosine_hemisphere_polar(float2 u, out float pdf)
{
    float3 p;
    float r = sqrt(u.x);
    float phi = M_2PI * u.y;
    p.x = r * cos(phi);
    p.y = r * sin(phi);
    p.z = sqrt(1.f - u.x);
    pdf = p.z * M_1_PI;
    return p;
}

/** Cosine-weighted sampling of the hemisphere using a polar coordinates.
    This overload does not compute the pdf for the generated sample.
    \param[in] u Uniform random numbers in [0,1)^2.
    \return Sampled direction in the local frame (+z axis up).
*/
float3 sample_cosine_hemisphere_polar(float2 u)
{
    float pdf;
    return sample_cosine_hemisphere_polar(u, pdf);
}

/** Uniform sampling of a triangle.
    \param[in] u Uniform random numbers in [0,1)^2.
    \return Barycentric coordinates (1-u-v,u,v) of the sampled point.
*/
float3 sample_triangle(float2 u)
{
    float su = sqrt(u.x);
    float2 b = float2(1.f - su, u.y * su);
    return float3(1.f - b.x - b.y, b.x, b.y);
}

/******************************************************************************

    Miscellaneous functions

******************************************************************************/

/** Inverts a 3x3 matrix.
*/
float3x3 inverse(float3x3 M)
{
    float3x3 inv;
    float invdet = 1.0f / determinant(M);
    inv[0][0] = (M[1][1] * M[2][2] - M[2][1] * M[1][2]) * invdet;
    inv[0][1] = (M[0][2] * M[2][1] - M[0][1] * M[2][2]) * invdet;
    inv[0][2] = (M[0][1] * M[1][2] - M[0][2] * M[1][1]) * invdet;
    inv[1][0] = (M[1][2] * M[2][0] - M[1][0] * M[2][2]) * invdet;
    inv[1][1] = (M[0][0] * M[2][2] - M[0][2] * M[2][0]) * invdet;
    inv[1][2] = (M[1][0] * M[0][2] - M[0][0] * M[1][2]) * invdet;
    inv[2][0] = (M[1][0] * M[2][1] - M[2][0] * M[1][1]) * invdet;
    inv[2][1] = (M[2][0] * M[0][1] - M[0][0] * M[2][1]) * invdet;
    inv[2][2] = (M[0][0] * M[1][1] - M[1][0] * M[0][1]) * invdet;
    return inv;
}

/** Generate a vector that is orthogonal to the input vector.
    This can be used to invent a tangent frame for meshes that don't have real tangents/bitangents.
*/
float3 perp_stark(float3 u)
{
    // TODO: Validate this and look at numerical precision etc. Are there better ways to do it?
    float3 a = abs(u);
    uint uyx = (a.x - a.y) < 0 ? 1 : 0;
    uint uzx = (a.x - a.z) < 0 ? 1 : 0;
    uint uzy = (a.y - a.z) < 0 ? 1 : 0;
    uint xm = uyx & uzx;
    uint ym = (1 ^ xm) & uzy;
    uint zm = 1 ^ (xm | ym);  // 1 ^ (xm & ym)
    float3 v = cross(u, float3(xm, ym, zm));
    return v;
}

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

    // Workaround slang/fxc bug (https://github.com/NVIDIAGameWorks/Falcor/issues/164). This can go away when we always use dxc.
#if 0
    [unroll]
    for (uint i = 0; i < 4; i++)
#else
    $for(i in Range(0,4))
#endif
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
    // Workaround slang/fxc bug (https://github.com/NVIDIAGameWorks/Falcor/issues/164). This can go away when we always use dxc.
#if 0
    [unroll]
    for (int i = 0; i < 8; ++i)
#else
    $for(i in Range(0,8))
#endif
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
#if 0
    [unroll]
    for (int i = 0; i < 8; ++i)
#else
    $for(i in Range(0,8))
#endif
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
