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
import Scene.Scene;
import Scene.Displacement.DisplacementMapping;
import Utils.Geometry.IntersectionHelpers;
import Utils.Geometry.GeometryHelpers;
__exported import Utils.Math.Ray;

/** Helper for intersecting rays against displaced triangle meshes.
*/
struct DisplacedTriangleMeshIntersector
{
    /** Intersection attributes.
    */
    struct Attribs
    {
        float2 barycentrics;
        float displacement;
    };

    /** Intersects a ray with a displaced triangle.
        \param[in] ray Ray in world-space.
        \param[in] instanceID Geometry instance ID.
        \param[in] primitiveIndex Primitive index.
        \param[out] attribs Intersection attributes.
        \param[out] t Intersection t.
        \return True if the ray intersects the displaced triangle.
    */
    static bool intersect(const Ray ray, const GeometryInstanceID instanceID, const uint primitiveIndex, out Attribs attribs, out float t)
    {
        const uint materialID = gScene.getMaterialID(instanceID);
        const uint3 indices = gScene.getIndices(instanceID, primitiveIndex);
        const StaticVertexData vertices[3] = { gScene.getVertex(indices[0]), gScene.getVertex(indices[1]), gScene.getVertex(indices[2]) };
        const float4x4 worldMat = gScene.getWorldMatrix(instanceID);

        DisplacementData displacementData;
        loadDisplacementData(gScene.materials[materialID], gScene.materialResources[materialID], displacementData);

        DisplacementIntersection result;
        if (intersectDisplacedTriangle(ray, vertices, worldMat, displacementData, result))
        {
            attribs.barycentrics = result.barycentrics;
            attribs.displacement = result.displacement;
            t = result.t;
            return true;
        }

        return false;
    }
};

/** Helper for intersecting rays against curves.
*/
struct CurveIntersector
{
    static const bool kUseSphereJoints = false;

    /** Intersection attributes.
    */
    struct Attribs
    {
        float2 barycentrics;
    }

    /** Intersects a ray with a curve.
        \param[in] ray Ray in world-space.
        \param[in] instanceID Geometry instance ID.
        \param[in] primitiveIndex Primitive index.
        \param[out] attribs Intersection attributes.
        \param[out] t Intersection t.
        \return True if the ray intersects the curve.
    */
    static bool intersect(const Ray ray, const GeometryInstanceID instanceID, const uint primitiveIndex, out Attribs attribs, out float t)
    {
        attribs = {};
        t = {};

        uint v0Index = gScene.getFirstCurveVertexIndex(instanceID, primitiveIndex);
        const StaticCurveVertexData v0 = gScene.getCurveVertex(v0Index);
        const StaticCurveVertexData v1 = gScene.getCurveVertex(v0Index + 1);

        const float4 sphereA = float4(v0.position, v0.radius);
        const float4 sphereB = float4(v1.position, v1.radius);

        const float rayLength = length(ray.dir);
        const float invRayLength = 1.f / rayLength;
        float3 unitRayDir = ray.dir * invRayLength;

        float2 result;
        bool isHit = intersectLinearSweptSphereHan19(ray.origin, unitRayDir, sphereA, sphereB, kUseSphereJoints, result);

        if (isHit)
        {
            t = result.x * invRayLength;
            bool isInRayExtent = (t >= ray.tMin && t <= ray.tMax);
            if (isInRayExtent)
            {
                // Back-face culling.
                float4 sphere = sphereA + result.y * (sphereB - sphereA);
                float3 hitPos = ray.origin + result.x * ray.dir;
                float3 normal = normalize(hitPos - sphere.xyz);

                if (dot(ray.dir, normal) < -1e-6f)
                {
                    // To avoid numerical issues, reprojecting from hitPos = ray.origin + t * ray.dir.
                    hitPos = sphere.xyz + sphere.w * normal;
                    attribs.barycentrics = curveHitPosToUV(hitPos, sphereA, sphereB);
                    return true;
                }
            }
        }

        return false;
    }
};
