/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
import Scene.Material.MaterialData;
import Scene.HitInfo;
import Utils.Geometry.GeometryHelpers;

/** This struct holds information needed for shading a hit point.

    This includes:
    - Geometric properties of the surface.
    - Texture coordinates.
    - Material ID and header.
    - Opacity value for alpha testing.
    - Index of refraction of the surrounding medium.

    Based on a ShadingData struct, the material system can be queried
    for a material instance at the hit using `gScene.materials.getMaterialInstance()`.
    The material instance has interfaces for sampling and evaluation, as well as for
    querying its properties at the hit.
*/
struct ShadingData
{
    // Geometry data
    float3  posW;                   ///< Shading hit position in world space.
    float3  V;                      ///< View direction, -incident direction (-ray.dir)
    // N,T,B is an orthonormal basis representing the interpolated shading frame in world space at the shading point.
    // Handedness is determined by the MikkTSpace library and stored in tangentW.w on the mesh, where B = cross(N, T) * tangentW.w.
    float3  N;                      ///< Shading normal at shading hit. Normalized.
    float3  T;                      ///< Shading tangent at shading hit. Normalized.
    float3  B;                      ///< Shading bitangent at shading hit. Normalized.
    float2  uv;                     ///< Texture mapping coordinates.
    float3  faceN;                  ///< Face normal in world space, always on the front-facing side.
    float4  tangentW;               ///< Geometric tangent (xyz) and sign (w) in world space. This is used for orthogonalization. Not normalized, but it is guaranteed to be nonzero and sign (w) is +-1.0.
    bool    frontFacing;            ///< True if primitive seen from the front-facing side.
    float   curveRadius;            ///< Curve cross-sectional radius. Valid only for geometry generated from curves.

    // Material data
    MaterialHeader mtl;             ///< Material header data.
    uint    materialID;             ///< Material ID at shading location.
    float   opacity;                ///< Opacity value in [0,1]. This is used for alpha testing.
    float   IoR;                    ///< Index of refraction for the medium on the front-facing side (i.e. "outside" the material at the hit).


    // Utility functions

    /** Computes new ray origin based on the hit point to avoid self-intersection.
        The method is described in Ray Tracing Gems, Chapter 6, "A Fast and Robust
        Method for Avoiding Self-Intersection" by Carsten Wächter and Nikolaus Binder.
        \param[in] viewside True if the origin should be on the view side (reflection) or false otherwise (transmission).
        \return Ray origin of the new ray.
    */
    float3 computeNewRayOrigin(bool viewside = true)
    {
        return computeRayOrigin(posW, (frontFacing == viewside) ? faceN : -faceN);
    }

    /** Transform vector from the local surface frame to world space.
        \param[in] v Vector in local space.
        \return Vector in world space.
    */
    float3 fromLocal(float3 v)
    {
        return T * v.x + B * v.y + N * v.z;
    }

    /** Transform vector from world space to the local surface frame.
        \param[in] v Vector in world space.
        \return Vector in local space.
    */
    float3 toLocal(float3 v)
    {
        return float3(dot(v, T), dot(v, B), dot(v, N));
    }

    /** Returns sign that gets applied to such that B = cross(N, T) * getHandednessSign();
        \return +1 for right handed, and -1 for left handed frames.
    */
    float getHandednessSign()
    {
        return dot(cross(N, T), B) >= 0.f ? 1.f : -1.f;
    }
};
