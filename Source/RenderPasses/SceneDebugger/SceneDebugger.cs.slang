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
import Scene.RaytracingInline;
import Scene.Scene;
import Scene.Shading;
import Utils.Helpers;
import Utils.Math.HashUtils;
import Utils.Color.ColorMap;
import Utils.Color.ColorHelpers;
import SharedTypes;

struct SceneDebugger
{
    SceneDebuggerParams params;
    StructuredBuffer<uint> meshToBlasID;
    StructuredBuffer<InstanceInfo> meshInstanceInfo;

    RWTexture2D<float4> output;
    RWStructuredBuffer<PixelData> pixelData;

    /** Run scene debugger for the given pixel.
    */
    void execute(const uint2 pixel)
    {
        if (any(pixel >= params.frameDim)) return;

        // Initialize pixel data for the selected pixel.
        if (all(pixel == params.selectedPixel))
        {
            pixelData[0].meshInstanceID = PixelData::kInvalidID;
            pixelData[0].meshID = PixelData::kInvalidID;
            pixelData[0].materialID = PixelData::kInvalidID;
            pixelData[0].blasID = PixelData::kInvalidID;
            pixelData[0].curveInstanceID = PixelData::kInvalidID;
            pixelData[0].curveID = PixelData::kInvalidID;
        }

        // Trace primary ray.
        RayDesc ray = gScene.camera.computeRayPinhole(pixel, params.frameDim).toRayDesc();

        RayQuery<RAY_FLAG_NONE> rayQuery;
        //RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> rayQuery;

        rayQuery.TraceRayInline(
            gRtScene,
            RAY_FLAG_NONE, // OR'd with template flags above
            0xff, // InstanceInclusionMask
            ray);

        float2 curveCommittedAttribs = {};

        while (rayQuery.Proceed())
        {
            switch (rayQuery.CandidateType())
            {
            case CANDIDATE_NON_OPAQUE_TRIANGLE:
            {
                // Alpha test for non-opaque geometry.
                HitInfo hit = getCandidateTriangleHit(rayQuery);
                VertexData v = gScene.getVertexData(hit);
                uint materialID = gScene.getMaterialID(hit.instanceID);

                if (alphaTest(v, gScene.materials[materialID], gScene.materialResources[materialID], 0.f)) continue;

                rayQuery.CommitNonOpaqueTriangleHit();
                break;
            }
            case CANDIDATE_PROCEDURAL_PRIMITIVE:
            {
                float t;
                HitInfo hit;
                bool valid = getCandidateCurveHit(rayQuery, ray, t, hit);

                if (valid)
                {
                    rayQuery.CommitProceduralPrimitiveHit(t);
                    curveCommittedAttribs = hit.barycentrics;
                }
                break;
            }
            }
        }

        // Process hit/miss.
        float3 color = float3(0);
        float hitT = 1e30;
        switch (rayQuery.CommittedStatus())
        {
        case COMMITTED_TRIANGLE_HIT:
        {
            HitInfo hit = getCommittedTriangleHit(rayQuery);
            hitT = rayQuery.CommittedRayT();
            color = handleHit(pixel, ray.Direction, hit);
            break;
        }
        case COMMITTED_PROCEDURAL_PRIMITIVE_HIT:
        {
            HitInfo hit = getCommittedCurveHit(rayQuery, curveCommittedAttribs);
            hitT = rayQuery.CommittedRayT();
            const float3 curveHitPosW = ray.Origin + ray.Direction * hitT;
            color = handleHit(pixel, ray.Direction, hit, curveHitPosW);
            break;
        }
        default: // COMMITTED_NOTHING
        {
            color = handleMiss(pixel, ray.Direction);
        }
        }

        // Process volumes.
        if (params.showVolumes)
        {
            color = handleVolumes(color, ray.Origin, ray.Direction, hitT);
        }

        // Clamp pixel values if necessary.
        if (params.clamp) color = saturate(color);

        // Write output.
        output[pixel] = float4(color, 1.f);
    }

    float3 remapVector(float3 v)
    {
        if (params.flipSign) v = -v;
        if (params.remapRange) v = 0.5f * v + 0.5f;
        return v;
    }

    float2 remapVector(float2 v)
    {
        return remapVector(float3(v, 0)).xy;
    }

    float3 pseudocolor(uint value)
    {
        uint h = jenkinsHash(value);
        return (uint3(h, h >> 8, h >> 16) & 0xff) / 255.f;
    }

    float3 handleHit(const uint2 pixel, const float3 dir, const HitInfo hit, const float3 curveHitPosW = {})
    {
        uint materialID = PixelData::kInvalidID;

        uint meshID = PixelData::kInvalidID;
        uint meshInstanceID = PixelData::kInvalidID;
        uint blasID = PixelData::kInvalidID;

        uint curveID = PixelData::kInvalidID;
        uint curveInstanceID = PixelData::kInvalidID;

        VertexData v;
        ShadingData sd;

        switch (hit.getType())
        {
        case InstanceType::TriangleMesh:
        {
            meshInstanceID = hit.instanceID;
            meshID = gScene.getMeshInstance(meshInstanceID).meshID;
            blasID = meshToBlasID[meshID];
            materialID = gScene.getMaterialID(meshInstanceID);

            // Load shading attributes.
            float3 barycentrics = float3(1.f - hit.barycentrics.x - hit.barycentrics.y, hit.barycentrics.x, hit.barycentrics.y);
            v = gScene.getVertexData(hit.instanceID, hit.primitiveIndex, barycentrics);
            sd = prepareShadingData(v, materialID, gScene.materials[materialID], gScene.materialResources[materialID], -dir, 0.f);
            break;
        }
        case InstanceType::Curve:
        {
            curveInstanceID = hit.instanceID;
            curveID = gScene.getCurveInstance(curveInstanceID).curveID;
            materialID = gScene.getCurveMaterialID(curveInstanceID);

            // Load shading attributes.
            float radius;
            v = gScene.getVertexDataFromCurve(hit.instanceID, hit.primitiveIndex, hit.barycentrics.x, curveHitPosW, radius);
            sd = prepareShadingData(v, materialID, gScene.materials[materialID], gScene.materialResources[materialID], -dir, 0.f);
            break;
        }
        default:
            // Should not happen. Return an error color.
            return float3(1, 0, 0);
        }

        // Write pixel data for the selected pixel.
        if (all(pixel == params.selectedPixel))
        {
            pixelData[0].meshInstanceID = meshInstanceID;
            pixelData[0].meshID = meshID;
            pixelData[0].materialID = materialID;
            pixelData[0].blasID = blasID;
            pixelData[0].curveInstanceID = curveInstanceID;
            pixelData[0].curveID = curveID;
        }

        // Compute zebra stripes.
        const float z = (pixel.x + pixel.y - params.frameCount) & 0x8 ? 1.f : 0.f;

        switch ((SceneDebuggerMode)params.mode)
        {
        case SceneDebuggerMode::FaceNormal:
            return remapVector(sd.faceN);
        case SceneDebuggerMode::ShadingNormal:
            return remapVector(sd.N);
        case SceneDebuggerMode::ShadingTangent:
            return remapVector(sd.T);
        case SceneDebuggerMode::ShadingBitangent:
            return remapVector(sd.B);
        case SceneDebuggerMode::FrontFacingFlag:
        {
            float v = 0.75f * luminance(abs(sd.faceN)) + 0.25f;
            return sd.frontFacing ? float3(0, v, 0) : float3(v, 0, 0);
        }
        case SceneDebuggerMode::BackfacingShadingNormal:
        {
            float v = 0.75f * luminance(abs(sd.faceN)) + 0.25f;
            bool backFacing = dot(sd.N, sd.V) <= 0.f;
            return backFacing ? float3(z, z, 0) : float3(v, v, v);
        }
        case SceneDebuggerMode::TexCoords:
            return float3(frac(remapVector(sd.uv)), 0.f);
        case SceneDebuggerMode::MeshID:
            return pseudocolor(meshID);
        case SceneDebuggerMode::MeshInstanceID:
            return pseudocolor(meshInstanceID);
        case SceneDebuggerMode::MaterialID:
            return pseudocolor(materialID);
        case SceneDebuggerMode::BlasID:
            return pseudocolor(blasID);
        case SceneDebuggerMode::CurveID:
            return pseudocolor(curveID);
        case SceneDebuggerMode::CurveInstanceID:
            return pseudocolor(curveInstanceID);
        case SceneDebuggerMode::InstancedGeometry:
        {
            float v = 0.75f * luminance(abs(sd.faceN)) + 0.25f;
            if (meshInstanceID != PixelData::kInvalidID)
            {
                bool isInstanced = (meshInstanceInfo[meshInstanceID].flags & (uint)InstanceInfoFlags::IsInstanced) != 0;
                return isInstanced ? float3(0, v, 0) : float3(v, 0, 0);
            }
            else
            {
                // For non-triangle geometry, return grayscale color to indicate instancing status is not available.
                return float3(v, v, v);
            }
        }
        default:
            // Should not happen.
            return float3(1, 0, 0);
        }
    }

    float3 handleMiss(const uint2 pixel, const float3 dir)
    {
        // Draw a checkerboard pattern.
        return ((pixel.x ^ pixel.y) & 0x8) != 0 ? float3(1.f) : float3(0.5f);
    }

    float3 handleVolumes(const float3 color, const float3 pos, const float3 dir, const float hitT)
    {
        float Tr = 1.f;
        for (uint i = 0; i < gScene.getVolumeCount(); ++i)
        {
            Volume volume = gScene.getVolume(i);
            Tr *= evalVolumeTransmittance(volume, pos, dir, 0.f, hitT);
        }

        return Tr * color;
    }

    float evalVolumeTransmittance(Volume volume, const float3 pos, const float3 dir, const float minT, const float maxT)
    {
        if (!volume.hasDensityGrid()) return 1.f;

        // Intersect with volume bounds and get intersection interval along the view ray.
        AABB bounds = volume.getBounds();
        float2 nearFar;
        bool hit = intersectRayAABB(pos, dir, bounds.minPoint, bounds.maxPoint, nearFar);
        nearFar.x = max(nearFar.x, minT);
        nearFar.y = min(nearFar.y, maxT);
        if (nearFar.x >= nearFar.y) return 1.f;

        // Setup access to density grid.
        Grid densityGrid;
        gScene.getGrid(volume.getDensityGrid(), densityGrid);
        Grid::Accessor accessor = densityGrid.createAccessor();

        // Evaluate transmittance using ray-marching.
        const uint kSteps = 500;
        float opticalDepth = 0.f;
        for (uint step = 0; step < kSteps; ++step)
        {
            float t = lerp(nearFar.x, nearFar.y, (step + 0.5f) / kSteps);
            float3 p = pos + t * dir;
            p = mul(float4(p, 1.f), volume.data.invTransform).xyz;
            float density = densityGrid.lookupWorld(p, accessor);
            opticalDepth += density;
        }
        opticalDepth *= (nearFar.y - nearFar.x) / kSteps * volume.data.densityScale * params.densityScale;
        return exp(-opticalDepth);
    }
};

cbuffer CB
{
    SceneDebugger gSceneDebugger;
}

/** Compute shader entry point for scene debugger.
*/
[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    gSceneDebugger.execute(dispatchThreadId.xy);
}
