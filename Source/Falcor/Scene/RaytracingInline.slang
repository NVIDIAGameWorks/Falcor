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

/** Utilities for inline ray tracing using DXR 1.1.

    Import this module in your shader and call Scene::setRaytracingShaderData()
    on the host to bind the necessary resources.
*/
#include "Scene/ScenePrimitiveDefines.slangh"
import Utils.Attributes;
import Scene.Intersection;
import Scene.Shading;
__exported import Scene.Scene;
__exported import Scene.HitInfo;
__exported import Scene.SceneRayQueryInterface;

/** Return the geometry type for a committed hit.
*/
PrimitiveTypeFlags getCommittedGeometryType<let Flags : int>(RayQuery<Flags> rayQuery)
{
    GeometryInstanceID instanceID = GeometryInstanceID(rayQuery.CommittedInstanceID(), rayQuery.CommittedGeometryIndex());
    return gScene.getGeometryInstanceType(instanceID);
}

/** Return the geometry type for a candidate hit.
*/
PrimitiveTypeFlags getCandidateGeometryType<let Flags : int>(RayQuery<Flags> rayQuery)
{
    GeometryInstanceID instanceID = GeometryInstanceID(rayQuery.CandidateInstanceID(), rayQuery.CandidateGeometryIndex());
    return gScene.getGeometryInstanceType(instanceID);
}

/** Create a TriangleHit for a RayQuery committed hit on triangles.
    \param[in] rayQuery RayQuery object.
    \return Committed triangle hit.
*/
TriangleHit getCommittedTriangleHit<let Flags : int>(RayQuery<Flags> rayQuery)
{
    TriangleHit hit;
    hit.instanceID = GeometryInstanceID(rayQuery.CommittedInstanceID(), rayQuery.CommittedGeometryIndex());
    hit.primitiveIndex = rayQuery.CommittedPrimitiveIndex();
    hit.barycentrics = rayQuery.CommittedTriangleBarycentrics();
    return hit;
}

/** Create a TriangleHit for a RayQuery candidate hit on triangles.
    \param[in] rayQuery RayQuery object.
    \return Candidate triangle hit.
*/
TriangleHit getCandidateTriangleHit<let Flags : int>(RayQuery<Flags> rayQuery)
{
    TriangleHit hit;
    hit.instanceID = GeometryInstanceID(rayQuery.CandidateInstanceID(), rayQuery.CandidateGeometryIndex());
    hit.primitiveIndex = rayQuery.CandidatePrimitiveIndex();
    hit.barycentrics = rayQuery.CandidateTriangleBarycentrics();
    return hit;
}

// TODO: Pass UseAlphaTest as a template argument.
bool traceSceneRayImpl<let Flags : int>(const bool useAlphaTest, RayQuery<Flags> rayQuery, const RayDesc ray, out HitInfo hit, out float hitT, uint rayFlags, uint instanceInclusionMask)
{
    rayQuery.TraceRayInline(gScene.rtAccel, rayFlags, instanceInclusionMask, ray);

    CurveIntersector::Attribs curveCommittedAttribs;
    DisplacedTriangleMeshIntersector::Attribs displacedTriangleMeshCommittedAttribs;

    while (rayQuery.Proceed())
    {
#if IS_SET(SCENE_PRIMITIVE_TYPE_FLAGS, PRIMITIVE_TYPE_TRIANGLE_MESH)
        if (useAlphaTest && rayQuery.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
        {
            // Alpha test for non-opaque geometry.
            const TriangleHit hit = getCandidateTriangleHit(rayQuery);
            const VertexData v = gScene.getVertexData(hit);
            const uint materialID = gScene.getMaterialID(hit.instanceID);

            if (alphaTest(v, gScene.materials[materialID], gScene.materialResources[materialID], 0.f)) continue;

            rayQuery.CommitNonOpaqueTriangleHit();
        }
#endif
#if IS_SET(SCENE_PRIMITIVE_TYPE_FLAGS, PRIMITIVE_TYPE_PROCEDURAL)
        if (rayQuery.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE)
        {
            const GeometryInstanceID instanceID = GeometryInstanceID(rayQuery.CandidateInstanceID(), rayQuery.CandidateGeometryIndex());
            const uint primitiveIndex = rayQuery.CandidatePrimitiveIndex();
            const PrimitiveTypeFlags type = gScene.getGeometryInstanceType(instanceID);
            const Ray raySegment = Ray(ray.Origin, ray.Direction, rayQuery.RayTMin(), rayQuery.CommittedRayT());
            switch (type)
            {
#if IS_SET(SCENE_PRIMITIVE_TYPE_FLAGS, PRIMITIVE_TYPE_DISPLACED_TRIANGLE_MESH)
            case PrimitiveTypeFlags::DisplacedTriangleMesh:
                {
                    DisplacedTriangleMeshIntersector::Attribs attribs;
                    float t;
                    if (DisplacedTriangleMeshIntersector::intersect(raySegment, instanceID, primitiveIndex, attribs, t))
                    {
                        rayQuery.CommitProceduralPrimitiveHit(t);
                        displacedTriangleMeshCommittedAttribs = attribs;
                    }
                }
                break;
#endif
#if IS_SET(SCENE_PRIMITIVE_TYPE_FLAGS, PRIMITIVE_TYPE_CURVE)
            case PrimitiveTypeFlags::Curve:
                {
                    CurveIntersector::Attribs attribs;
                    float t;
                    if (CurveIntersector::intersect(raySegment, instanceID, primitiveIndex, attribs, t))
                    {
                        rayQuery.CommitProceduralPrimitiveHit(t);
                        curveCommittedAttribs = attribs;
                    }
                }
                break;
#endif
            default:
                // Ignore other types of geometry.
            }
        }
#endif
    }

    hit = {};
    hitT = 0.f;

#if IS_SET(SCENE_PRIMITIVE_TYPE_FLAGS, PRIMITIVE_TYPE_TRIANGLE_MESH)
    if (rayQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        hit = HitInfo(getCommittedTriangleHit(rayQuery));
        hitT = rayQuery.CommittedRayT();
        return true;
    }
#endif
#if IS_SET(SCENE_PRIMITIVE_TYPE_FLAGS, PRIMITIVE_TYPE_PROCEDURAL)
    if (rayQuery.CommittedStatus() == COMMITTED_PROCEDURAL_PRIMITIVE_HIT)
    {
        const GeometryInstanceID instanceID = GeometryInstanceID(rayQuery.CommittedInstanceID(), rayQuery.CommittedGeometryIndex());
        const uint primitiveIndex = rayQuery.CommittedPrimitiveIndex();
        const PrimitiveTypeFlags type = gScene.getGeometryInstanceType(instanceID);
        switch (type)
        {
#if IS_SET(SCENE_PRIMITIVE_TYPE_FLAGS, PRIMITIVE_TYPE_DISPLACED_TRIANGLE_MESH)
        case PrimitiveTypeFlags::DisplacedTriangleMesh:
            DisplacedTriangleHit displacedTriangleHit;
            displacedTriangleHit.instanceID = instanceID;
            displacedTriangleHit.primitiveIndex = primitiveIndex;
            displacedTriangleHit.barycentrics = displacedTriangleMeshCommittedAttribs.barycentrics;
            displacedTriangleHit.displacement = displacedTriangleMeshCommittedAttribs.displacement;
            hit = HitInfo(displacedTriangleHit);
            hitT = rayQuery.CommittedRayT();
            return true;
#endif
#if IS_SET(SCENE_PRIMITIVE_TYPE_FLAGS, PRIMITIVE_TYPE_CURVE)
        case PrimitiveTypeFlags::Curve:
            CurveHit curveHit;
            curveHit.instanceID = instanceID;
            curveHit.primitiveIndex = primitiveIndex;
            curveHit.barycentrics = curveCommittedAttribs.barycentrics;
            hit = HitInfo(curveHit);
            hitT = rayQuery.CommittedRayT();
            return true;
#endif
        default:
            // Should not happen.
            return false;
        }
    }
#endif

    return false;
}

bool traceSceneRay<let UseAlphaTest : int>(const RayDesc ray, out HitInfo hit, out float hitT, uint rayFlags, uint instanceInclusionMask)
{
    // TODO: Use a constant expression to derive static ray flags.

    if (UseAlphaTest)
    {
#if IS_SET(SCENE_PRIMITIVE_TYPE_FLAGS, PRIMITIVE_TYPE_PROCEDURAL)
        RayQuery<RAY_FLAG_NONE> rayQuery;
#else
        RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> rayQuery;
#endif
        return traceSceneRayImpl(true, rayQuery, ray, hit, hitT, rayFlags, instanceInclusionMask);
    }
    else
    {
#if IS_SET(SCENE_PRIMITIVE_TYPE_FLAGS, PRIMITIVE_TYPE_PROCEDURAL)
        RayQuery<RAY_FLAG_FORCE_OPAQUE> rayQuery;
#else
        RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_FORCE_OPAQUE> rayQuery;
#endif
        return traceSceneRayImpl(false, rayQuery, ray, hit, hitT, rayFlags, instanceInclusionMask);
    }
}

bool traceSceneVisibilityRayImpl<let Flags : int>(const bool useAlphaTest, RayQuery<Flags> rayQuery, const RayDesc ray, uint rayFlags, uint instanceInclusionMask)
{
    rayQuery.TraceRayInline(gScene.rtAccel, rayFlags, instanceInclusionMask, ray);

    while (rayQuery.Proceed())
    {
#if IS_SET(SCENE_PRIMITIVE_TYPE_FLAGS, PRIMITIVE_TYPE_TRIANGLE_MESH)
        if (useAlphaTest && rayQuery.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
        {
            // Alpha test for non-opaque geometry.
            const TriangleHit hit = getCandidateTriangleHit(rayQuery);
            const VertexData v = gScene.getVertexData(hit);
            const uint materialID = gScene.getMaterialID(hit.instanceID);

            if (alphaTest(v, gScene.materials[materialID], gScene.materialResources[materialID], 0.f)) continue;

            rayQuery.CommitNonOpaqueTriangleHit();
        }
#endif
#if IS_SET(SCENE_PRIMITIVE_TYPE_FLAGS, PRIMITIVE_TYPE_PROCEDURAL)
        if (rayQuery.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE)
        {
            const GeometryInstanceID instanceID = GeometryInstanceID(rayQuery.CandidateInstanceID(), rayQuery.CandidateGeometryIndex());
            const uint primitiveIndex = rayQuery.CandidatePrimitiveIndex();
            const PrimitiveTypeFlags type = gScene.getGeometryInstanceType(instanceID);
            const Ray raySegment = Ray(ray.Origin, ray.Direction, rayQuery.RayTMin(), rayQuery.CommittedRayT());
            switch (type)
            {
#if IS_SET(SCENE_PRIMITIVE_TYPE_FLAGS, PRIMITIVE_TYPE_DISPLACED_TRIANGLE_MESH)
            case PrimitiveTypeFlags::DisplacedTriangleMesh:
                {
                    DisplacedTriangleMeshIntersector::Attribs attribs;
                    float t;
                    if (DisplacedTriangleMeshIntersector::intersect(raySegment, instanceID, primitiveIndex, attribs, t))
                    {
                        rayQuery.CommitProceduralPrimitiveHit(t);
                    }
                }
                break;
#endif
#if IS_SET(SCENE_PRIMITIVE_TYPE_FLAGS, PRIMITIVE_TYPE_CURVE)
            case PrimitiveTypeFlags::Curve:
                {
                    CurveIntersector::Attribs attribs;
                    float t;
                    if (CurveIntersector::intersect(raySegment, instanceID, primitiveIndex, attribs, t))
                    {
                        rayQuery.CommitProceduralPrimitiveHit(t);
                    }
                }
                break;
#endif
            default:
                // Ignore other types of geometry.
            }
        }
#endif
    }

    return rayQuery.CommittedStatus() == COMMITTED_NOTHING;
}

bool traceSceneVisibilityRay<let UseAlphaTest : int>(const RayDesc ray, uint rayFlags, uint instanceInclusionMask)
{
    // TODO: Use a constant expression to derive static ray flags.

    if (UseAlphaTest)
    {
#if IS_SET(SCENE_PRIMITIVE_TYPE_FLAGS, PRIMITIVE_TYPE_PROCEDURAL)
        RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> rayQuery;
#else
        RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> rayQuery;
#endif
        return traceSceneVisibilityRayImpl(true, rayQuery, ray, rayFlags, instanceInclusionMask);
    }
    else
    {
#if IS_SET(SCENE_PRIMITIVE_TYPE_FLAGS, PRIMITIVE_TYPE_PROCEDURAL)
        RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> rayQuery;
#else
        RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> rayQuery;
#endif
        return traceSceneVisibilityRayImpl(false, rayQuery, ray, rayFlags, instanceInclusionMask);
    }
}

/** Scene ray query implementation.
*/
struct SceneRayQuery<let UseAlphaTest : int> : ISceneRayQuery
{
    bool traceRay(const RayDesc ray, out HitInfo hit, out float hitT, uint rayFlags, uint instanceInclusionMask)
    {
        return traceSceneRay<UseAlphaTest>(ray, hit, hitT, rayFlags, instanceInclusionMask);
    }

    bool traceVisibilityRay(const RayDesc ray, uint rayFlags, uint instanceInclusionMask)
    {
        return traceSceneVisibilityRay<UseAlphaTest>(ray, rayFlags, instanceInclusionMask);
    }
}
