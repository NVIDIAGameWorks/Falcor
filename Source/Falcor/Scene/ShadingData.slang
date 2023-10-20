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
import Utils.Math.ShadingFrame;

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
struct ShadingData : IDifferentiable
{
    // Geometry data
    float3  posW;                       ///< Shading hit position in world space.
    float3  V;                          ///< View direction, -incident direction (-ray.dir)
    float2  uv;                         ///< Texture mapping coordinates.

    ShadingFrame frame;                 ///< Smooth interpolated shading frame in world space at the shading point. The normal is *not* automatically flipped for backfacing hits.
    no_diff float3  faceN;              ///< Face normal in world space, always on the front-facing side.
    no_diff float4  tangentW;           ///< Geometric tangent (xyz) and sign (w) in world space. This is used for orthogonalization. Not normalized, but it is guaranteed to be nonzero and sign (w) is +-1.0.
    bool            frontFacing;        ///< True if primitive seen from the front-facing side, i.e., dot(V, faceN) >= 0.0.
    no_diff float   curveRadius;        ///< Curve cross-sectional radius. Valid only for geometry generated from curves.

    // Material data
    MaterialHeader mtl;                 ///< Material header data.
    uint    materialID;                 ///< Material ID at shading location.
    float   IoR;                        ///< Index of refraction for the medium on the front-facing side (i.e. "outside" the material at the hit).

    uint materialGradOffset;            ///< Offset to the material gradient in the gradient buffer.
    uint geometryGradOffset;            ///< Offset to the geometry gradient in the gradient buffer.
    uint threadID;                      ///< Thread ID for gradient aggregation with a hash grid.


    // Utility functions

    /** Computes new ray origin based on the hit point to avoid self-intersection.
        The method is described in Ray Tracing Gems, Chapter 6, "A Fast and Robust
        Method for Avoiding Self-Intersection" by Carsten Wächter and Nikolaus Binder.
        \param[in] viewside True if the origin should be on the view side (reflection) or false otherwise (transmission).
        \return Ray origin of the new ray.
    */
    float3 computeRayOrigin(bool viewside = true)
    {
        return computeRayOrigin(posW, (frontFacing == viewside) ? faceN : -faceN);
    }

    /** Returns the oriented face normal.
        \return Face normal flipped to the same side as the view vector.
    */
    float3 getOrientedFaceNormal()
    {
        return frontFacing ? faceN : -faceN;
    }
};
