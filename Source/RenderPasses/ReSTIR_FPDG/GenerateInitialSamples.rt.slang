#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Scene.Raytracing;
import Utils.Sampling.SampleGenerator;
import Utils.Color.ColorHelpers;
import Rendering.RTXDI.RTXDI;
import StructsAndHelpers;
import PhotonDifferentials;

Texture2D<PackedHitInfo> gVBuffer;

StructuredBuffer<PhotonDifferential> gPhotonData[2];
StructuredBuffer<AABB> gPhotonAABB[2];
RaytracingAccelerationStructure gPhotonAS;

RWStructuredBuffer<ReservoirFG> gFinalGatherReservoir;
RWStructuredBuffer<ReservoirCaustic> gCausticReservoir;
RWTexture2D<float4> gEmission;

static const float kPhotonCollectTMax = 1e-7;

cbuffer CB
{
    uint gFrameCount; // Frame count since scene was loaded.
    uint gFGRayMaxPathLength; //Max path length for a Final Gather sample
}

static const bool kAlphaTest = true;
static const float kRoughnessThreshold = ROUGHNESS_THRESHOLD;
static const uint kMIHints = (uint)MaterialInstanceHints::AdjustShadingNormal;

struct RayPayload
{
    PackedHitInfo packedHitInfo; //< Hit Info

    //Init with empty hit
    __init()
    {
        this.packedHitInfo = {};
    }
}

/*  Checks if the point p is inside the sphere.
    \param[in] Center of the sphere in world coordinates
    \param[in] Radius of the sphere
    \param[in] Point P in world coordinates
    \return True if point p is inside of the sphere
*/
bool hitSphere(const float3 center, const float radius, const float3 p)
{
    float3 radiusTest = p - center;
    return dot(radiusTest, radiusTest) < radius * radius;
}

/*  Collects all photons in a radius and estimates radiance for Final Gathering.
    \param[in] Shading data of the gather query hit point
    \param[in] Material instance of the gather query hit point
    \param[in] Shading Normal of the hit point
    \param[in] Ray Tracing instance mask. Can be used to mask out Global or Caustic Photons
    \param[in/out] Random Number Generator
    \return Radiance estimate of the gather point
*/
float3 collectPhotons(ShadingData sd, IMaterialInstance mi, float3 N, uint instanceMask, inout SampleGenerator sg)
{
    float3 collectedRadiance = float3(0);
    //Prepare ray
    RayDesc ray;
    ray.Origin = sd.posW;
    ray.Direction = sd.faceN;
    ray.TMin = 0;
    ray.TMax = kPhotonCollectTMax;

    RayQuery<RAY_FLAG_NONE> rayQuery;

    rayQuery.TraceRayInline(gPhotonAS, RAY_FLAG_NONE, instanceMask, ray);
    while (rayQuery.Proceed()) {
        
        if (rayQuery.CandidateType() != CANDIDATE_PROCEDURAL_PRIMITIVE) continue;

        const uint instanceIdx = rayQuery.CandidateInstanceIndex();
        const uint primIdx = rayQuery.CandidatePrimitiveIndex();

        // Check if ray origin is inside photon radius
        AABB photonAABB = gPhotonAABB[instanceIdx][primIdx];
        float radius = (photonAABB.maxPoint.x - photonAABB.minPoint.x) / 2.0;

        if (!hitSphere(photonAABB.center(), radius, sd.posW)) continue;

        // It is a photon. It uses only dir, flux, pathLenght and normalW from PhotonDifferential.
        // TODO : sepate buffers for Photon (FG) and Photon differentials (FPDG)
        PhotonDifferential photonData = gPhotonData[instanceIdx][primIdx];
        float NdotL = dot(photonData.dir, N);

        // Check for diffent surfaces (e.g. corners)
        if ((dot(photonData.normalW, sd.faceN) < 0.6) || (NdotL <= 0)) continue;

        // Radiance for current photon
        float3 bsdf = mi.eval(sd, photonData.dir, sg);
        collectedRadiance += max(0, (photonData.flux * bsdf) / (M_PI * radius * radius * NdotL)); //NdotL needs to be taken out of the bsdf for Photon Mapping
    }
    return collectedRadiance;
}

/*  Generates a ray at a hit point
    \param[in] Shading data of the hit point
    \param[in] Material instance of the hit point
    \param[in/out] Random Number Generator
    \param[in/out] Throughput of the path
    \param[in/out] PDF for the new sampled direction
    \param[in/out] Description of the ray
    \return True if the sample is valid
*/
bool generateFGSampleRay(ShadingData sd, IMaterialInstance mi, inout SampleGenerator sg, inout float3 thp, inout float pdf, inout RayDesc ray) {
    // Photon Bounce for next iteration
    BSDFSample bsdfSample;
    bool validSample = mi.sample(sd, sg, bsdfSample, true /*ImportanceSampling*/);
    bool transmission = bsdfSample.isLobe(LobeType::Transmission);
    ray.Origin = sd.computeRayOrigin(!transmission); // false is for transmissions
    ray.Direction = bsdfSample.wo;
    thp *= bsdfSample.weight;
    pdf = bsdfSample.pdf;

    return validSample;
}

/*  Reservoir RIS with photons
    \param[in] Shading data of the gather query hit point
    \param[in] Material instance of the gather query hit point
    \param[in] True if global photons with path length == 0 should be collected. Is used when the first camera hit is not diffuse
    \param[in/out] Reservoir for photons
    \param[in/out] Random Number Generator
*/
void photonResampling(ShadingData sd, BSDFProperties bsdfProperties, bool collectGlobalDirect, inout ReservoirCaustic causticReservoir , inout SampleGenerator sg) {
    // Prepare photon collection ray
    RayDesc ray;
    ray.Origin = sd.posW;
    ray.Direction = sd.faceN;
    ray.TMin = 0;
    ray.TMax = kPhotonCollectTMax;

    //Use a ray query to collect the photons
    RayQuery<RAY_FLAG_NONE> rayQuery;
    uint instanceMask = collectGlobalDirect ? 0xFF : 2; //2 is caustic only

    rayQuery.TraceRayInline(gPhotonAS, RAY_FLAG_NONE, instanceMask , ray);
    while (rayQuery.Proceed()) {

        if (rayQuery.CandidateType() != CANDIDATE_PROCEDURAL_PRIMITIVE) continue;

        const uint instanceIdx = rayQuery.CandidateInstanceIndex();
        const uint primIdx = rayQuery.CandidatePrimitiveIndex();
        PhotonType type = (PhotonType)instanceIdx;

        PhotonDifferential photonData = gPhotonData[instanceIdx][primIdx];
        float3 center = gPhotonAABB[instanceIdx][primIdx].center();
        // Check if ray is inside photon footprint
        float footprintArea;
        float kernelWeight = evaluateDifferentialWeight(sd.posW, center, photonData.normalW, photonData.dPdx, photonData.dPdy, footprintArea);
        if (kernelWeight <= 0.0f) continue;

        // If direct global photons should be collected and the photon is global, check the path length
        if (collectGlobalDirect && (type == PhotonType::Global)) {
            if (photonData.pathLength > 0) continue;
        }

        // Check for diffent surfaces (e.g. corners)
        float NdotL = dot(photonData.dir, bsdfProperties.guideNormal);
        if ((dot(photonData.normalW, sd.getOrientedFaceNormal()) < 0.6) || (NdotL <= 0)) continue;

        //Eval Radiance estimate
        float targetFunction = 0;
        if (NdotL > 0)
            targetFunction = evalApproximateBRDFPhoton(sd, bsdfProperties, photonData.dir) * luminance(photonData.flux) * kernelWeight / footprintArea;
        float w = targetFunction;

        //Update reservoir and resplace sample data if the new sample is selected
        causticReservoir.wSum += w;
        bool selectNew = (sampleNext1D(sg) * causticReservoir.wSum) < w;
        if (selectNew) {
            causticReservoir.photonPosW = center;
            causticReservoir.photonDir = photonData.dir;
            causticReservoir.photonRadiance = photonData.flux;
            causticReservoir.targetFunction = targetFunction;
            // For resampling pass
            causticReservoir.dPdx = photonData.dPdx;
            causticReservoir.dPdy = photonData.dPdy;
            causticReservoir.photonNormalW = photonData.normalW;
        }
    }

    // Finalize Reservoir
    if (causticReservoir.targetFunction > 0) {
        causticReservoir.wSum /= causticReservoir.targetFunction;
    }
    else
        causticReservoir.wSum = 0;
    causticReservoir.c = 1;
}

/*  Traces a ray to generate a Final Gather reservoir sample. If (highly) specular materials are hit, the ray is continued.
    Additionally, if the first camera hit was not diffuse, the caustic reservoir is initialized at the first diffuse hit
    \param[in] Ray that is traced
    \param[in/out] Random Number Generator
    \param[in] True if the hit at the ray origin was diffuse
    \param[in/out] Throughput of the path
    \param[in/out] PDF of the last sample
    \param[in/out] Path is valid
    \param[in/out] Reservoir for the final gather sample
    \param[in/out] Reservoir for the (caustic) photon samples
    \param[in/out] Emission for path with a length > 0
*/
void traceFGRay(RayDesc ray, inout SampleGenerator sg, bool prevHitIsRough, inout float3 thp, inout float pdf, inout bool valid, inout ReservoirFG reservoirFG, inout ReservoirCaustic reservoirCaustic, inout float3 emission) {
    RayPayload rayPayload = RayPayload();
    uint pathLength = 0;
    bool collectCaustics = !prevHitIsRough;
    float3 prevThp = float3(1); //Needed for FG reservoir, as throuput from the surface to the final gather point should not be included
    for (pathLength; pathLength < gFGRayMaxPathLength && valid; pathLength++) {
        TraceRay(gScene.rtAccel, 0 /*Flags*/, 0xff /* instanceInclusionMask */, 0 /* hitIdx */, rayTypeCount, 0 /* missIdx */, ray, rayPayload);
        // Get surface hit info
        HitInfo hit = HitInfo(rayPayload.packedHitInfo);
        if (!hit.isValid()) {
            valid = false;
            //TODO EnvMap hits could be added to the reservoir
            return;
        }

        // Get shading infos
        let lod = ExplicitLodTextureSampler(0);

        // Get Material properties
        ShadingData sd = loadShadingData(hit, ray.Direction, lod);
        let mi = gScene.materials.getMaterialInstance(sd, lod, kMIHints);
        let bsdfProperties = mi.getProperties(sd);
        bool hitIsRough = bsdfProperties.roughness > kRoughnessThreshold;
                
        // Emissive hits are undefined for final gather rays
        if (any(bsdfProperties.emission > 0) || (prevHitIsRough && !hitIsRough))
        {
            if (collectCaustics) // Path is completely specular 
                emission += bsdfProperties.emission * thp;
            break;
        }

        // Collect caustics if this was the first diffuse hit on the path
        // Additionally, global photons with path length 0 are collected,
        // as they are missing from RTXDI when the path encountered a (highly) specular surface
        if (collectCaustics && hitIsRough) {
            reservoirCaustic.sampleSurface = hit.pack();
            reservoirCaustic.sampleSurfaceView = ray.Direction;
            reservoirCaustic.pathLength = pathLength + 1;
            reservoirCaustic.thp = thp;
            photonResampling(sd, bsdfProperties, true, reservoirCaustic, sg);
            collectCaustics = false;
        }

        // Check if this is a valid final gather surface
        if (hitIsRough && prevHitIsRough) {
            //Collect photons, estimate outgoing radiance and finalize the reservoir
            reservoirFG.sampleData.radianceOut = collectPhotons(sd, mi, bsdfProperties.guideNormal, 0xFF, sg);
            reservoirFG.thp = prevThp;
            reservoirFG.sampleData.posW = sd.posW;
            reservoirFG.sampleData.normal = sd.faceN; // Face normal?
            reservoirFG.wSum = 1.0 / pdf;
            reservoirFG.c = 1;
            float distanceToSample = distance(ray.Origin, sd.posW);
            reservoirFG.jacobianDeterminant = dot(reservoirFG.sampleData.normal, sd.V) / (distanceToSample * distanceToSample);

            return;
        }

        //Update reservoir
        reservoirFG.sampleSurface = hit.pack(); // Update surface
        reservoirFG.pathLength++;                  
        reservoirFG.sampleData.sampleView = -ray.Direction;
        prevThp = thp;

        //Next ray
        valid = generateFGSampleRay(sd, mi, sg, thp, pdf, ray);
        prevHitIsRough = hitIsRough;
    }
    valid = false;
}

//Insert invalid hit for miss
[shader("miss")]
void miss(inout RayPayload rayData: SV_RayPayload)
{
    HitInfo hitInfo = {};
    rayData.packedHitInfo = hitInfo.pack(); // Invalid hit
}

//For alpha test
[shader("anyhit")]
void anyHit(inout RayPayload rayData: SV_RayPayload, BuiltInTriangleIntersectionAttributes attribs: SV_IntersectionAttributes)
{
    if (kAlphaTest)
    {
        // Alpha test for non-opaque geometry.
        GeometryInstanceID instanceID = getGeometryInstanceID();
        VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
        uint materialID = gScene.getMaterialID(instanceID);
        if (gScene.materials.alphaTest(v, materialID, 0.f))
            IgnoreHit();
    }
}

//Store triangle hit in ray payload
[shader("closesthit")]
void closestHit(inout RayPayload rayData: SV_RayPayload, BuiltInTriangleIntersectionAttributes attribs: SV_IntersectionAttributes) {
    TriangleHit triangleHit;
    triangleHit.instanceID = getGeometryInstanceID();
    triangleHit.primitiveIndex = PrimitiveIndex();
    triangleHit.barycentrics = attribs.barycentrics;

    rayData.packedHitInfo = HitInfo(triangleHit).pack();
}

[shader("raygeneration")]
void rayGen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 dispatchDims = DispatchRaysDimensions().xy;

    HitInfo hitInfo = HitInfo(gVBuffer[pixel]);
    float3 viewDir = -gScene.camera.computeRayPinhole(pixel, dispatchDims).dir;
    float3 outColor = float3(0);
    ReservoirFG finalGatherReservoir = ReservoirFG();
    ReservoirCaustic causticReservoir = ReservoirCaustic();
    float3 emission = float3(0);    //Emission on fully specular paths

    if (hitInfo.isValid()) {
        //Get info for the hit surface
        let lod = ExplicitLodTextureSampler(0);
        SampleGenerator sg = SampleGenerator(pixel, gFrameCount * 5 + 1);

        ShadingData sd = loadShadingData(hitInfo, -viewDir, lod);
        let mi = gScene.materials.getMaterialInstance(sd, lod, kMIHints);
        let bsdfProperties = mi.getProperties(sd);
        float distanceToSurface = distance(sd.posW, gScene.camera.getPosition()); // sd.posW is world position of ray hit from pixel
        
        // Set RTXDI data
        gRTXDI.setSurfaceData(pixel, sd.computeRayOrigin(), bsdfProperties.guideNormal, bsdfProperties.diffuseReflectionAlbedo, bsdfProperties.specularReflectance, bsdfProperties.roughness);

        // Initialize Caustic reservoir if photons can be collected at the current surface
        bool hitIsRough = bsdfProperties.roughness > kRoughnessThreshold;
        if (hitIsRough) { // if surface is diffuse
            causticReservoir.sampleSurface = hitInfo.pack();
            causticReservoir.sampleSurfaceView = -viewDir;
            photonResampling(sd, bsdfProperties, false, causticReservoir, sg);
        }

        // Store surface
        finalGatherReservoir.sampleSurface = hitInfo.pack();
        finalGatherReservoir.sampleData.sampleView = viewDir;

        // Distribute a Final Gather or specular trace ray
        RayDesc ray = {};
        ray.TMin = 0;
        ray.TMax = FLT_MAX;
        float3 fgThp = float3(1.0);
        float fgPdf = 0;
        bool validFGRay = generateFGSampleRay(sd, mi, sg, fgThp, fgPdf, ray);

        traceFGRay(ray, sg, hitIsRough, fgThp, fgPdf, validFGRay, finalGatherReservoir, causticReservoir, emission);
    } else {
        // Set invalid RTXDI data (ray missed scene)
        gRTXDI.setInvalidSurfaceData(pixel);
    }

    gFinalGatherReservoir[index2Dto1D(pixel, dispatchDims.x)] = finalGatherReservoir;
    gCausticReservoir[index2Dto1D(pixel, dispatchDims.x)] = causticReservoir;
    gEmission[pixel] = float4(emission, 1.0);
}
