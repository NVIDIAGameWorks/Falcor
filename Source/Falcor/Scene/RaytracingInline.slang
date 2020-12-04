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

/** Utilities for inline ray tracing using DXR 1.1.

    Import this module in your shader and call Scene::setRaytracingShaderData()
    on the host to bind the necessary resources.
*/
import Utils.Attributes;
import Utils.Helpers;
import Scene.Intersection;
__exported import Scene.Scene;
__exported import Scene.HitInfo;

[root] RaytracingAccelerationStructure gRtScene;

/** Returns the global mesh instance ID for a ray hit.
    \param[in] instanceID The instance ID of the TLAS instance.
    \param[in] geometryIndex The geometry index of the BLAS.
    \return Global mesh instance ID.
*/
uint getMeshInstanceID(uint instanceID, uint geometryIndex)
{
    return instanceID + geometryIndex;
}

/** Create a HitInfo for a RayQuery committed hit on triangles.
    \param[in] rayQuery RayQuery object.
    \return Candidate hit info.
*/
HitInfo getCommittedTriangleHit<let Flags : int>(RayQuery<Flags> rayQuery)
{
    HitInfo hit;
    hit.type = InstanceType::TriangleMesh;
    hit.instanceID = getMeshInstanceID(rayQuery.CommittedInstanceID(), rayQuery.CommittedGeometryIndex());
    hit.primitiveIndex = rayQuery.CommittedPrimitiveIndex();
    hit.barycentrics = rayQuery.CommittedTriangleBarycentrics();
    return hit;
}

/** Create a HitInfo for a RayQuery candidate hit on triangles.
    \param[in] rayQuery RayQuery object.
    \return Candidate hit info.
*/
HitInfo getCandidateTriangleHit<let Flags : int>(RayQuery<Flags> rayQuery)
{
    HitInfo hit;
    hit.type = InstanceType::TriangleMesh;
    hit.instanceID = getMeshInstanceID(rayQuery.CandidateInstanceID(), rayQuery.CandidateGeometryIndex());
    hit.primitiveIndex = rayQuery.CandidatePrimitiveIndex();
    hit.barycentrics = rayQuery.CandidateTriangleBarycentrics();
    return hit;
}

/** Create a HitInfo for a RayQuery committed hit on curves.
    \param[in] rayQuery RayQuery object.
    \param[in] curveCommittedAttribs Attribution object.
    \return Candidate hit info.
*/
HitInfo getCommittedCurveHit<let Flags : int>(RayQuery<Flags> rayQuery, float2 curveCommittedAttribs)
{
    HitInfo hit;
    hit.type = InstanceType::Curve;
    hit.instanceID = gScene.getCurveInstanceID(rayQuery.CommittedInstanceID(), rayQuery.CommittedGeometryIndex());
    hit.primitiveIndex = rayQuery.CommittedPrimitiveIndex();
    hit.barycentrics = curveCommittedAttribs;
    return hit;
}

/** Create a HitInfo for a RayQuery candidate hit on curves.
    \param[in] rayQuery RayQuery object.
    \param[in] ray Ray object.
    \param[out] hitT Distance value t at the hit point.
    \param[out] hit Candidate hit info.
    \return True if finding a valid candidate curve hit.
*/
bool getCandidateCurveHit<let Flags : int>(RayQuery<Flags> rayQuery, const RayDesc ray, out float hitT, out HitInfo hit)
{
    float3 rayDir = ray.Direction;
    const float rayLength = length(rayDir);
    const float invRayLength = 1.f / rayLength;
    rayDir *= invRayLength;

    hit.type = InstanceType::Curve;
    hit.instanceID = gScene.getCurveInstanceID(rayQuery.CandidateInstanceID(), rayQuery.CandidateGeometryIndex());
    hit.primitiveIndex = rayQuery.CandidatePrimitiveIndex();

    uint v0Index = gScene.getFirstCurveVertexIndex(hit.instanceID, hit.primitiveIndex);
    StaticCurveVertexData v0 = gScene.getCurveVertex(v0Index);
    StaticCurveVertexData v1 = gScene.getCurveVertex(v0Index + 1);

    float2 result;
    bool isect = intersectLinearSweptSphere(ray.Origin, rayDir, float4(v0.position, v0.radius), float4(v1.position, v1.radius), result);
    result.x *= invRayLength;

    hitT = result.x;
    hit.barycentrics = float2(result.y, 0.f);
    return (isect && hitT >= rayQuery.RayTMin() && hitT < rayQuery.CommittedRayT());
}
