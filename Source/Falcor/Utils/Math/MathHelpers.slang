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

    Motion vectors

******************************************************************************/

/** Calculate screen-space motion vector.
    \param[in] pixelCrd Sample in current frame expressed in pixel coordinates with origin in the top-left corner.
    \param[in] prevPosH Sample in previous frame expressed in homogeneous clip space coordinates. Note that the definition differs between D3D12 and Vulkan.
    \param[in] renderTargetDim Render target dimension in pixels.
    \return Motion vector pointing from current to previous position expressed in sceen space [0,1] with origin in the top-left corner.
*/
float2 calcMotionVector(float2 pixelCrd, float4 prevPosH, float2 renderTargetDim)
{
    float2 prevCrd = prevPosH.xy / prevPosH.w;
#ifdef FALCOR_FLIP_Y
    prevCrd *= float2(0.5, 0.5);
#else
    prevCrd *= float2(0.5, -0.5);
#endif
    prevCrd += 0.5f;
    float2 normalizedCrd = pixelCrd / renderTargetDim;
    return prevCrd - normalizedCrd;
}

/******************************************************************************

    Miscellaneous functions

******************************************************************************/

/** Inverts a 2x2 matrix.
*/
float2x2 inverse(float2x2 M)
{
    float2x2 inv;
    float invdet = 1.0f / determinant(M);
    inv[0][0] = M[1][1] * invdet;
    inv[1][1] = M[0][0] * invdet;
    inv[0][1] = -M[1][0] * invdet;
    inv[1][0] = -M[0][1] * invdet;
    return inv;
}

/** Inverts a 2x3 matrix.
*/
float2x3 inverse(float2x3 M)
{
    float2x2 N = float2x2(M._m00, M._m01, M._m10, M._m11);
    float2x2 Ni = inverse(N);
    float2 t = -mul(Ni, float2(M._m02, M._m12));
    float2x3 Mi = float2x3(Ni._m00, Ni._m01, t.x, Ni._m10, Ni._m11, t.y);
    return Mi;
}

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
    \param[in] u Unit vector.
    \return v Unit vector that is orthogonal to u.
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
    float3 v = normalize(cross(u, float3(xm, ym, zm)));
    return v;
}

float  sqr(float  v) { return v*v; }

float2 sqr(float2 v) { return v*v; }

float3 sqr(float3 v) { return v*v; }

float4 sqr(float4 v) { return v*v; }
