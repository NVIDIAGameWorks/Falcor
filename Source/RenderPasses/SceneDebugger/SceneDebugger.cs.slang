/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
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
#include "Utils/HostDeviceShared.slangh"
#include "Utils/Math/MathConstants.slangh"

import Utils.Math.HashUtils;
import Utils.Math.ShadingFrame;
import Utils.Geometry.IntersectionHelpers;
import Utils.Debug.PixelDebug;
import Utils.Color.ColorMap;
import Utils.Color.ColorHelpers;
import Utils.Sampling.SampleGenerator;
import SharedTypes;

import Scene.RaytracingInline;
import Scene.Scene;

import Utils.Math.Ray;

struct SceneDebugger
{
    static const int kUseAlphaTest = 1;

    SceneDebuggerParams params;
    StructuredBuffer<uint> meshToBlasID;
    StructuredBuffer<InstanceInfo> instanceInfo;
    Texture2D<PackedHitInfo> vbuffer;

    RWTexture2D<float4> output;
    RWStructuredBuffer<PixelData> pixelData;

    /**
     * Run scene debugger for the given pixel.
     */
    void execute(const uint2 pixel)
    {
        if (any(pixel >= params.frameDim))
            return;

        printSetPixel(pixel);

        // Initialize pixel data for the selected pixel.
        if (all(pixel == params.selectedPixel))
        {
            PixelData d = {};
            d.hitType = (uint)HitType::None;
            d.instanceID = PixelData::kInvalidID;
            d.materialID = PixelData::kInvalidID;
            d.geometryID = PixelData::kInvalidID;
            d.blasID = PixelData::kInvalidID;
            pixelData[0] = d;
        }

        // Trace primary ray.
        const Ray ray = gScene.camera.computeRayPinhole(pixel, params.frameDim);

        float3 color = float3(0);

        HitInfo hit;
        float hitT = 1e30f;

        if ((bool)params.useVBuffer)
        {
            hit = unpackHitInfo(vbuffer[pixel]);
            if (hit.isValid())
            {
                if ((bool)params.showVolumes)
                {
                    float3 posW = getPosW(ray.origin, ray.dir, hit);
                    hitT = length(posW - ray.origin);
                }
            }
        }
        else
        {
            SceneRayQuery<kUseAlphaTest> sceneRayQuery;
            hit = sceneRayQuery.traceRay(ray, hitT);
        }

        if (hit.isValid())
        {
            color = handleHit(pixel, ray.origin, ray.dir, hit);
        }
        else
        {
            hitT = 1e30f;
            color = handleMiss(pixel, ray.dir);
        }

        // Process volumes.
        if ((bool)params.showVolumes)
        {
            color = handleVolumes(color, ray.origin, ray.dir, hitT);
        }

        // Clamp pixel values if necessary.
        if ((bool)params.clamp)
            color = saturate(color);

        // Write output.
        output[pixel] = float4(color, 1.f);
    }

    float3 remapVector(float3 v)
    {
        if ((bool)params.flipSign)
            v = -v;
        if ((bool)params.remapRange)
            v = 0.5f * v + 0.5f;
        return v;
    }

    float2 remapVector(float2 v) { return remapVector(float3(v, 0)).xy; }

    float3 pseudocolor(uint value)
    {
        uint h = jenkinsHash(value);
        return (uint3(h, h >> 8, h >> 16) & 0xff) / 255.f;
    }

    float3 handleHit(const uint2 pixel, const float3 orig, const float3 dir, const HitInfo hit)
    {
        uint instanceID = PixelData::kInvalidID;
        uint materialID = PixelData::kInvalidID;
        uint primitiveID = PixelData::kInvalidID;
        uint geometryID = PixelData::kInvalidID;
        uint blasID = PixelData::kInvalidID;

        VertexData v = {};
        switch (hit.getType())
        {
        case HitType::Triangle:
        {
            const TriangleHit triangleHit = hit.getTriangleHit();
            instanceID = triangleHit.instanceID.index;
            geometryID = gScene.getGeometryInstance(triangleHit.instanceID).geometryID;
            primitiveID = triangleHit.primitiveIndex;
            blasID = meshToBlasID[geometryID];
            materialID = gScene.getMaterialID(triangleHit.instanceID);

            // Load vertex attributes.
            v = gScene.getVertexData(triangleHit);
            break;
        }
        case HitType::DisplacedTriangle:
        {
            const DisplacedTriangleHit displacedTriangleHit = hit.getDisplacedTriangleHit();
            instanceID = displacedTriangleHit.instanceID.index;
            geometryID = gScene.getGeometryInstance(displacedTriangleHit.instanceID).geometryID;
            blasID = meshToBlasID[geometryID];
            materialID = gScene.getMaterialID(displacedTriangleHit.instanceID);

            // Load vertex attributes.
            v = gScene.getVertexData(displacedTriangleHit, -dir);
            break;
        }
        case HitType::Curve:
        {
            const CurveHit curveHit = hit.getCurveHit();
            instanceID = curveHit.instanceID.index;
            geometryID = gScene.getGeometryInstance(curveHit.instanceID).geometryID;
            materialID = gScene.getMaterialID(curveHit.instanceID);

            // Load vertex attributes.
            v = gScene.getVertexDataFromCurve(curveHit);
            break;
        }
        case HitType::SDFGrid:
        {
            const SDFGridHit sdfGridHit = hit.getSDFGridHit();
            instanceID = sdfGridHit.instanceID.index;
            geometryID = gScene.getGeometryInstance(sdfGridHit.instanceID).geometryID;
            materialID = gScene.getMaterialID(sdfGridHit.instanceID);

            // Load vertex attributes.
            v = gScene.getVertexDataFromSDFGrid(sdfGridHit, orig, dir);
            break;
        }
        default:
            // Should not happen. Return an error color.
            return float3(1, 0, 0);
        }

        // Load shading data.
        const ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -dir);

        // Write pixel data for the selected pixel.
        if (all(pixel == params.selectedPixel))
        {
            PixelData d = {};

            // Geometry
            d.hitType = uint(hit.getType());
            d.instanceID = instanceID;
            d.materialID = materialID;
            d.geometryID = geometryID;
            d.blasID = blasID;

            // ShadingData
            d.posW = sd.posW;
            d.V = sd.V;
            d.N = sd.frame.N;
            d.T = sd.frame.T;
            d.B = sd.frame.B;
            d.uv = sd.uv;
            d.faceN = sd.faceN;
            d.tangentW = sd.tangentW;
            d.frontFacing = sd.frontFacing ? 1 : 0;
            d.curveRadius = sd.curveRadius;

            pixelData[0] = d;
        }

        // Profiling of secondary rays.
        if (params.profileSecondaryRays)
        {
            return traceSecondary(sd, pixel);
        }

        // Compute zebra stripes.
        const float z = ((pixel.x + pixel.y - params.frameCount) & 0x8) ? 1.f : 0.f;

        // Compute flat shading.
        // FIXME: Parameterize this
        const float3 frontN = faceforward(sd.faceN, dir, sd.faceN);
        const float flatShaded = 0.8f * saturate(dot(-frontN, dir)) + 0.2;

        switch ((SceneDebuggerMode)params.mode)
        {
        case SceneDebuggerMode::FlatShaded:
            return float3(flatShaded);
        case SceneDebuggerMode::TriangleDensity:
            return colormapViridis(computeTriangleDensity(hit));
        // Geometry
        case SceneDebuggerMode::HitType:
            return pseudocolor(uint(hit.getType()));
        case SceneDebuggerMode::InstanceID:
            return pseudocolor(instanceID);
        case SceneDebuggerMode::MaterialID:
            return pseudocolor(materialID) * flatShaded;
        case SceneDebuggerMode::PrimitiveID:
            return pseudocolor(primitiveID);
        case SceneDebuggerMode::GeometryID:
            return pseudocolor(geometryID);
        case SceneDebuggerMode::BlasID:
            return pseudocolor(blasID);
        case SceneDebuggerMode::InstancedGeometry:
        {
            float v = 0.75f * luminance(abs(sd.faceN)) + 0.25f;
            if (hit.getType() == HitType::Triangle && instanceID != PixelData::kInvalidID)
            {
                bool isInstanced = (instanceInfo[instanceID].flags & (uint)InstanceInfoFlags::IsInstanced) != 0;
                return isInstanced ? float3(0, v, 0) : float3(v, 0, 0);
            }
            else
            {
                // For non-triangle geometry, return grayscale color to indicate instancing status is not available.
                return float3(v, v, v);
            }
        }
        case SceneDebuggerMode::MaterialType:
            return pseudocolor((uint)sd.mtl.getMaterialType()) * flatShaded;
        // Shading data
        case SceneDebuggerMode::FaceNormal:
            return remapVector(sd.faceN);
        case SceneDebuggerMode::ShadingNormal:
            return remapVector(sd.frame.N);
        case SceneDebuggerMode::ShadingTangent:
            return remapVector(sd.frame.T);
        case SceneDebuggerMode::ShadingBitangent:
            return remapVector(sd.frame.B);
        case SceneDebuggerMode::FrontFacingFlag:
        {
            float v = 0.75f * luminance(abs(sd.faceN)) + 0.25f;
            return sd.frontFacing ? float3(0, v, 0) : float3(v, 0, 0);
        }
        case SceneDebuggerMode::BackfacingShadingNormal:
        {
            float v = 0.75f * luminance(abs(sd.faceN)) + 0.25f;
            bool backFacing = dot(sd.frame.N, sd.V) <= 0.f;
            return backFacing ? float3(z, z, 0) : float3(v, v, v);
        }
        case SceneDebuggerMode::TexCoords:
            return float3(frac(remapVector(sd.uv)), 0.f);
        // Material properties
        case SceneDebuggerMode::BSDFProperties:
        {
            // Create material instance and query its properties.
            let lod = ExplicitLodTextureSampler(0.f);
            let mi = gScene.materials.getMaterialInstance(sd, lod);
            let bsdfProperties = mi.getProperties(sd);
            switch ((SceneDebuggerBSDFProperty)params.bsdfProperty)
            {
            case SceneDebuggerBSDFProperty::Emission:
                return bsdfProperties.emission;
            case SceneDebuggerBSDFProperty::Roughness:
                return float3(bsdfProperties.roughness);
            case SceneDebuggerBSDFProperty::GuideNormal:
                return remapVector(bsdfProperties.guideNormal);
            case SceneDebuggerBSDFProperty::DiffuseReflectionAlbedo:
                return bsdfProperties.diffuseReflectionAlbedo;
            case SceneDebuggerBSDFProperty::DiffuseTransmissionAlbedo:
                return bsdfProperties.diffuseTransmissionAlbedo;
            case SceneDebuggerBSDFProperty::SpecularReflectionAlbedo:
                return bsdfProperties.specularReflectionAlbedo;
            case SceneDebuggerBSDFProperty::SpecularTransmissionAlbedo:
                return bsdfProperties.specularTransmissionAlbedo;
            case SceneDebuggerBSDFProperty::SpecularReflectance:
                return bsdfProperties.specularReflectance;
            case SceneDebuggerBSDFProperty::IsTransmissive:
                return bsdfProperties.isTransmissive ? float3(0, 1, 0) : float3(1, 0, 0);
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
        for (uint i = 0; i < gScene.getGridVolumeCount(); ++i)
        {
            GridVolume gridVolume = gScene.getGridVolume(i);
            Tr *= evalGridVolumeTransmittance(gridVolume, pos, dir, 0.f, hitT);
        }

        return Tr * color;
    }

    float evalGridVolumeTransmittance(GridVolume gridVolume, const float3 pos, const float3 dir, const float minT, const float maxT)
    {
        if (!gridVolume.hasDensityGrid())
            return 1.f;

        // Intersect with volume bounds and get intersection interval along the view ray.
        AABB bounds = gridVolume.getBounds();
        float2 nearFar;
        bool hit = intersectRayAABB(pos, dir, bounds.minPoint, bounds.maxPoint, nearFar);
        nearFar.x = max(nearFar.x, minT);
        nearFar.y = min(nearFar.y, maxT);
        if (nearFar.x >= nearFar.y)
            return 1.f;

        // Setup access to density grid.
        Grid densityGrid;
        gScene.getGrid(gridVolume.getDensityGrid(), densityGrid);
        Grid::Accessor accessor = densityGrid.createAccessor();

        // Transform to index-space.
        const float3 ipos = mul(gridVolume.data.invTransform, float4(pos, 1.f)).xyz;
        const float3 idir = mul(gridVolume.data.invTransform, float4(dir, 0.f)).xyz;

        // Evaluate transmittance using ray-marching.
        const uint kSteps = 500;
        float opticalDepth = 0.f;
        for (uint step = 0; step < kSteps; ++step)
        {
            float t = lerp(nearFar.x, nearFar.y, (step + 0.5f) / kSteps);
            float density = densityGrid.lookupIndex(int3(ipos + t * idir), accessor);
            opticalDepth += density;
        }
        opticalDepth *= (nearFar.y - nearFar.x) / kSteps * gridVolume.data.densityScale * params.volumeDensityScale;
        return exp(-opticalDepth);
    }

    static float3 getPosW(const float3 orig, const float3 dir, const HitInfo hit)
    {
        switch (hit.getType())
        {
        case HitType::Triangle:
            return gScene.getVertexData(hit.getTriangleHit()).posW;
        case HitType::DisplacedTriangle:
            return gScene.getVertexData(hit.getDisplacedTriangleHit(), -dir).posW;
        case HitType::Curve:
            return gScene.getVertexDataFromCurve(hit.getCurveHit()).posW;
        case HitType::SDFGrid:
        {
            return gScene.getVertexDataFromSDFGrid(hit.getSDFGridHit(), orig, dir).posW;
        }
        default:
            // Should not happen. Return the origin.
            return orig;
        }
    }

    ShadingData loadHit(const Ray ray, const HitInfo hit)
    {
        uint instanceID = PixelData::kInvalidID;
        uint materialID = PixelData::kInvalidID;
        uint primitiveID = PixelData::kInvalidID;
        uint geometryID = PixelData::kInvalidID;
        uint blasID = PixelData::kInvalidID;

        VertexData v = {};
        switch (hit.getType())
        {
        case HitType::Triangle:
        {
            const TriangleHit triangleHit = hit.getTriangleHit();
            instanceID = triangleHit.instanceID.index;
            geometryID = gScene.getGeometryInstance(triangleHit.instanceID).geometryID;
            primitiveID = triangleHit.primitiveIndex;
            blasID = meshToBlasID[geometryID];
            materialID = gScene.getMaterialID(triangleHit.instanceID);

            // Load vertex attributes.
            v = gScene.getVertexData(triangleHit);
            break;
        }
        case HitType::DisplacedTriangle:
        {
            const DisplacedTriangleHit displacedTriangleHit = hit.getDisplacedTriangleHit();
            instanceID = displacedTriangleHit.instanceID.index;
            geometryID = gScene.getGeometryInstance(displacedTriangleHit.instanceID).geometryID;
            blasID = meshToBlasID[geometryID];
            materialID = gScene.getMaterialID(displacedTriangleHit.instanceID);

            // Load vertex attributes.
            v = gScene.getVertexData(displacedTriangleHit, -ray.dir);
            break;
        }
        case HitType::Curve:
        {
            const CurveHit curveHit = hit.getCurveHit();
            instanceID = curveHit.instanceID.index;
            geometryID = gScene.getGeometryInstance(curveHit.instanceID).geometryID;
            materialID = gScene.getMaterialID(curveHit.instanceID);

            // Load vertex attributes.
            v = gScene.getVertexDataFromCurve(curveHit);
            break;
        }
        case HitType::SDFGrid:
        {
            const SDFGridHit sdfGridHit = hit.getSDFGridHit();
            instanceID = sdfGridHit.instanceID.index;
            geometryID = gScene.getGeometryInstance(sdfGridHit.instanceID).geometryID;
            materialID = gScene.getMaterialID(sdfGridHit.instanceID);

            // Load vertex attributes.
            v = gScene.getVertexDataFromSDFGrid(sdfGridHit, ray.origin, ray.dir);
            break;
        }
        default:
            // Should not happen.
            return {};
        }

        // Load shading data.
        return gScene.materials.prepareShadingData(v, materialID, -ray.dir);
    }

    float3 traceSecondary(const ShadingData sd, const uint2 pixel)
    {
        SampleGenerator sg = SampleGenerator(pixel, params.frameCount);

        // Create frame centered around the face normal oriented towards the viewer.
        const float3 N = sd.getOrientedFaceNormal();
        bool valid;
        let sf = ShadingFrame::createSafe(N, sd.tangentW, valid);

        // Generate random direction in cone centered around +z axis.
        float c = cos(params.profileSecondaryConeAngle * (M_2PI / 360.f));
        float s = sqrt(1.f - c * c);
        float z = c + (1.f - c) * sampleNext1D(sg); // z uniform on [cos(a),1]
        float phi = sampleNext1D(sg) * M_2PI;
        float3 dir = float3(cos(phi) * s, sin(phi) * s, z);
        dir = sf.fromLocal(dir);

        // Trace secondary ray.
        const Ray ray = Ray(sd.computeRayOrigin(), dir);

        float hitT = 1e30f;
        SceneRayQuery<kUseAlphaTest> sceneRayQuery;
        HitInfo hit = sceneRayQuery.traceRay(ray, hitT);

        float3 color = {};

        if (params.profileSecondaryLoadHit)
        {
            if (hit.isValid())
            {
                let hitSD = loadHit(ray, hit);
                color = hitSD.frame.N;
            }
        }
        else
        {
            color = hit.isValid() ? float3(0, 1, 0) : float3(1, 0, 0);
        }

        return color;
    }

    float computeTriangleDensity(const HitInfo hit)
    {
        if (hit.getType() != HitType::Triangle)
            return 0.f;

        const TriangleHit triangleHit = hit.getTriangleHit();
        const float A = gScene.getFaceAreaW(triangleHit.instanceID, triangleHit.primitiveIndex);

        float density = 1.f / A; // Can be inf
        float logDensity = log2(density);

        return (logDensity - params.triangleDensityLogRange.x) / (params.triangleDensityLogRange.y - params.triangleDensityLogRange.x);
    }
};

cbuffer CB
{
    SceneDebugger gSceneDebugger;
}

/**
 * Compute shader entry point for scene debugger.
 */
[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gSceneDebugger.execute(dispatchThreadId.xy);
}
