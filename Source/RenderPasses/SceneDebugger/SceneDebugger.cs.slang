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
import Scene.RaytracingInline;
import Scene.Scene;
import Scene.Shading;
import Utils.Math.HashUtils;
import Utils.Geometry.IntersectionHelpers;
import Utils.Color.ColorMap;
import Utils.Color.ColorHelpers;
import SharedTypes;

struct SceneDebugger
{
    static const int kUseAlphaTest = 1;

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
            pixelData[0].hitType = PixelData::kInvalidID;
            pixelData[0].instanceID = PixelData::kInvalidID;
            pixelData[0].materialID = PixelData::kInvalidID;
            pixelData[0].meshID = PixelData::kInvalidID;
            pixelData[0].blasID = PixelData::kInvalidID;
            pixelData[0].curveID = PixelData::kInvalidID;
        }

        // Trace primary ray.
        RayDesc ray = gScene.camera.computeRayPinhole(pixel, params.frameDim).toRayDesc();

        float3 color = float3(0);

        SceneRayQuery<kUseAlphaTest> sceneRayQuery;
        HitInfo hit;
        float hitT;
        if (sceneRayQuery.traceRay(ray, hit, hitT, RAY_FLAG_NONE, 0xff))
        {
            color = handleHit(pixel, ray.Direction, hit);
        }
        else
        {
            hitT = 1e30f;
            color = handleMiss(pixel, ray.Direction);
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

    float3 handleHit(const uint2 pixel, const float3 dir, const HitInfo hit)
    {
        uint instanceID = PixelData::kInvalidID;
        uint materialID = PixelData::kInvalidID;

        uint meshID = PixelData::kInvalidID;
        uint blasID = PixelData::kInvalidID;

        uint curveID = PixelData::kInvalidID;

        VertexData v;
        ShadingData sd;

        switch (hit.getType())
        {
        case HitType::Triangle:
        {
            const TriangleHit triangleHit = hit.getTriangleHit();
            instanceID = triangleHit.instanceID.index;
            meshID = gScene.getMeshInstance(triangleHit.instanceID).meshID;
            blasID = meshToBlasID[meshID];
            materialID = gScene.getMaterialID(triangleHit.instanceID);

            // Load shading attributes.
            v = gScene.getVertexData(triangleHit);
            sd = prepareShadingData(v, materialID, gScene.materials[materialID], gScene.materialResources[materialID], -dir, 0.f);
            break;
        }
        case HitType::DisplacedTriangle:
        {
            const DisplacedTriangleHit displacedTriangleHit = hit.getDisplacedTriangleHit();
            instanceID = displacedTriangleHit.instanceID.index;
            meshID = gScene.getMeshInstance(displacedTriangleHit.instanceID).meshID;
            blasID = meshToBlasID[meshID];
            materialID = gScene.getMaterialID(displacedTriangleHit.instanceID);

            // Load shading attributes.
            v = gScene.getVertexData(displacedTriangleHit, -dir);
            sd = prepareShadingData(v, materialID, gScene.materials[materialID], gScene.materialResources[materialID], -dir, 0.f);
            break;
        }
        case HitType::Curve:
        {
            const CurveHit curveHit = hit.getCurveHit();
            instanceID = curveHit.instanceID.index;
            curveID = gScene.getCurveInstance(curveHit.instanceID).curveID;
            materialID = gScene.getCurveMaterialID(curveHit.instanceID);

            // Load shading attributes.
            float radius;
            v = gScene.getVertexDataFromCurve(curveHit, radius);
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
            pixelData[0].hitType = uint(hit.getType());
            pixelData[0].instanceID = instanceID;
            pixelData[0].meshID = meshID;
            pixelData[0].materialID = materialID;
            pixelData[0].blasID = blasID;
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
        case SceneDebuggerMode::HitType:
            return pseudocolor(uint(hit.getType()));
        case SceneDebuggerMode::InstanceID:
            return pseudocolor(instanceID);
        case SceneDebuggerMode::MaterialID:
            return pseudocolor(materialID);
        case SceneDebuggerMode::MeshID:
            return pseudocolor(meshID);
        case SceneDebuggerMode::BlasID:
            return pseudocolor(blasID);
        case SceneDebuggerMode::CurveID:
            return pseudocolor(curveID);
        case SceneDebuggerMode::InstancedGeometry:
        {
            float v = 0.75f * luminance(abs(sd.faceN)) + 0.25f;
            if (hit.getType() == HitType::Triangle && instanceID != PixelData::kInvalidID)
            {
                bool isInstanced = (meshInstanceInfo[instanceID].flags & (uint)InstanceInfoFlags::IsInstanced) != 0;
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

        // Transform to index-space.
        const float3 ipos = mul(float4(pos, 1.f), volume.data.invTransform).xyz;
        const float3 idir = mul(float4(dir, 0.f), volume.data.invTransform).xyz;

        // Evaluate transmittance using ray-marching.
        const uint kSteps = 500;
        float opticalDepth = 0.f;
        for (uint step = 0; step < kSteps; ++step)
        {
            float t = lerp(nearFar.x, nearFar.y, (step + 0.5f) / kSteps);
            float density = densityGrid.lookupIndex(ipos + t * idir, accessor);
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
