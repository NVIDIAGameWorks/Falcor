/***************************************************************************
 # Copyright (c) 2015-26, NVIDIA CORPORATION. All rights reserved.
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
import Rendering.Lights.EnvMapSampler;
import Rendering.Lights.EmissiveLightSampler;
import Rendering.Lights.EmissiveLightSamplerHelpers;
import Rendering.Volumes.HomogeneousVolumeSampler;
import Rendering.Utils.PixelStats;
import Utils.Geometry.GeometryHelpers;
import Utils.Debug.PixelDebug;
import RenderPasses.Shared.Denoising.NRDHelpers;

import RenderPasses.PathTracer.PathTracer;
import LoadShadingDataDynamic;

__exported import RenderPasses.PathTracer.PathTracerUtils;
__exported import RenderPasses.PathTracer.PathState;
__exported import Params;
#include "Rendering/Lights/EmissiveLightSamplerType.slangh"

ParameterBlock<PathTracer> gPathTracer;
static const bool kFetchNRDDataPass = FETCH_NRD_DATA_PASS;

/** Interface for querying the closest hit in the scene.
    This is used in `nextHit` to allow using different raytracing APIs.
*/
interface IClosestHitQueryDynamic
{
    /** Trace a ray against the scene and return the closest hit point.
        Note: The content of the path state must not be changed in the implementation of this interface.
        However, for the ReorderTraceRay scheduler it is necessary to pass the state through to
        TraceRayHitObject as ray payload and then re-materialize it after ray traversal.
        \param[in,out] path Path state.
        \param[in] ray Ray.
        \param[out] hitT Hit distance (only valid if function returns true).
        \return Returns hit info, `isValid() == true` indicates valid hit.
    */
    [mutating] HitInfo traceRay(inout PathState path, const Ray ray, out float hitT, bool isPrevFrame);
};

/** Interface for querying visibility in the scene.
    This is used in `handleHit` (and other variants) to allow using different raytracing APIs.
*/
interface IVisibilityQueryDynamic
{
    /** Trace a visibility ray against the scene.
        \param[in] ray Ray.
        \return Returns true if the ray endpoints are mutually visible (i.e. the ray does NOT intersect the scene).
    */
    [mutating] bool traceVisibilityRay(const Ray ray, bool isPrevFrame);
};

struct GeneratePathClosestHitQuery : IClosestHitQueryDynamic
{
    HitInfo primaryHit = {};
    float primaryHitT = {};

    HitInfo traceRay(inout PathState path, const Ray ray, out float hitT, bool isPrevFrame)
    {
        hitT = {};

        if (primaryHit.isValid() && primaryHitT < ray.tMax)
        {
            hitT = primaryHitT;
            return primaryHit;
        }

        return makeInvalidHit();
    }
};

/** Finds the next surface or volume hit.
    If there are volumes in the scene, this function first samples a free flight distance.
    Then a (shortened) ray is traced against the scene to find any surface hits.
    This is done using the provided IClosestHitQuery adaptor to allow using different tracing APIs (TraceRay, TracerRayInline).
    Finally, using both the free flight distance and the closest surface hit, the final hit is determined and stored in the path state.
    \param[in,out] path The path state.
    \param[in,out] chq Closest hit query.
*/
void nextHit<ClosestHitQuery : IClosestHitQueryDynamic>(inout PathState path, inout ClosestHitQuery chq, bool isPrevFrame)
{
    Ray ray = path.getScatterRay();

    path.incrementVertexIndex();

    logTraceRay(PixelStatsRayType::ClosestHit);
    float hitT;

    HitInfo hit = chq.traceRay(path, ray, hitT, isPrevFrame);

    if (hit.isValid())
    {
        path.setHit(hit);

        if (!path.interiorList.isEmpty())
        {
            const uint materialID = path.interiorList.getTopMaterialID();
            let material = gScene.materials.getMaterial(materialID);
            let vp = material.getHomogeneousVolumeProperties(gScene.materials, materialID);
            gPathTracer.updatePathThroughput(path, HomogeneousVolumeSampler::evalTransmittance(vp, hitT));
        }
    }
    else
    {
        path.clearHit();
    }

    // Accumulate path length.
    if (gPathTracer.kOutputGuideData)
    {
        if (path.isHit())
        {
            path.sceneLength += float16_t(hitT);
        }
    }
}

/** Generates a light sample on the environment map.
    \param[in] vertex Path vertex.
    \param[in,out] sg Sample generator.
    \param[out] ls Struct describing valid samples.
    \return True if the sample is valid and has nonzero contribution, false otherwise.
*/
bool generateEnvMapSample(const PathVertex vertex, inout SampleGenerator sg, out LightSample ls, bool isPrevFrame)
{
    ls = {}; // Default initialization to avoid divergence at returns.

    if (!gPathTracer.kUseEnvLight) return false;

    // Sample environment map.
    EnvMapSample lightSample;
    if (!gPathTracer.envMapSampler.sample(sampleNext2D(sg), lightSample, 0.f, isPrevFrame)) return false;

    // Setup returned sample.
    ls.Li = lightSample.pdf > 0.f ? lightSample.Le / lightSample.pdf : float3(0);
    ls.pdf = lightSample.pdf;
    ls.origin = vertex.getRayOrigin(lightSample.dir);
    ls.distance = kRayTMax;
    ls.dir = lightSample.dir;

    return any(ls.Li > 0.f);
}

/** Generates a light sample on the emissive geometry.
    \param[in] vertex Path vertex.
    \param[in] upperHemisphere True if only upper hemisphere (w.r.t. shading normal) should be considered.
    \param[in,out] sg Sample generator.
    \param[out] ls Struct describing valid samples.
    \return True if the sample is valid and has nonzero contribution, false otherwise.
*/
bool generateEmissiveSample( const PathVertex vertex, const bool upperHemisphere, inout SampleGenerator sg, out LightSample ls, bool isPrevFrame )
{
    ls = {}; // Default initialization to avoid divergence at returns.
    if (!gPathTracer.kUseEmissiveLights) return false;

    TriangleLightSample tls;
    if (!gPathTracer.emissiveSampler.sampleLight(vertex.pos, vertex.getOrientedFaceNormal(), upperHemisphere, sg, tls, isPrevFrame)) return false;

    // Setup returned sample.
    ls.Li = tls.pdf > 0.f ? tls.Le / tls.pdf : float3(0);
    ls.pdf = tls.pdf;
    // Offset shading and light position to avoid self-intersection.
    float3 lightPos = computeRayOrigin(tls.posW, tls.normalW);
    ls.origin = vertex.getRayOrigin(lightPos - vertex.pos);
    float3 toLight = lightPos - ls.origin;
    ls.distance = length(toLight);
    ls.dir = normalize(toLight);

    ls.barycentrics = tls.uv;
    ls.triangleId = tls.triangleIndex;
    ls.geometryTerm = tls.geometryTerm;

    return any(ls.Li > 0.f);
}

/** Samples a light source in the scene.
    This function first stochastically selects a type of light source to sample,
    and then calls that the sampling function for the chosen light type.
    The upper/lower hemisphere is defined as the union of the hemispheres w.r.t. to the shading and face normals.
    \param[in] vertex Path vertex.
    \param[in] sampleUpperHemisphere True if the upper hemisphere should be sampled.
    \param[in] sampleLowerHemisphere True if the lower hemisphere should be sampled.
    \param[in,out] sg Sample generator.
    \param[out] ls Struct describing valid samples.
    \return True if the sample is valid and has nonzero contribution, false otherwise.
*/
bool generateLightSample( const PathVertex vertex, const bool sampleUpperHemisphere, const bool sampleLowerHemisphere, inout SampleGenerator sg, out LightSample ls, bool isPrevFrame )
{
    ls = {};

    uint lightType;
    float selectionPdf;
    if (!gPathTracer.selectLightType(lightType, selectionPdf, sg)) return false;

    bool valid = false;
    if (gPathTracer.kUseEnvLight && lightType == (uint)LightType::EnvMap) valid = generateEnvMapSample(vertex, sg, ls, isPrevFrame);
    if (gPathTracer.kUseEmissiveLights && lightType == (uint)LightType::Emissive)
    {
        // Emissive light samplers have an option to exclusively sample the upper hemisphere.
        bool upperHemisphere = sampleUpperHemisphere && !sampleLowerHemisphere;
        valid = generateEmissiveSample(vertex, upperHemisphere, sg, ls, isPrevFrame);
    }
    if (gPathTracer.kUseAnalyticLights && lightType == (uint)LightType::Analytic)
    {
        valid = gPathTracer.generateAnalyticLightSample(vertex, sg, ls);
    }
    if (!valid) return false;

    // Reject samples in non-requested hemispheres.
    float NdotL = dot(vertex.getOrientedFaceNormal(), ls.dir);
    if ((!sampleUpperHemisphere && NdotL >= -kMinCosTheta) || (!sampleLowerHemisphere && NdotL <= kMinCosTheta))
        return false;

    // Account for light type selection.
    ls.lightType = lightType;
    ls.pdf *= selectionPdf;
    ls.Li /= selectionPdf;

    return true;
}

[ForceInline]
void initReplayThresholds(inout SampleGenerator sg, inout PathState path)
{
    if (gPathTracer.restirpt.settings.nearFieldDistanceSigma == 0.f)
        path.restirPTData.distanceThreshold = gPathTracer.restirpt.settings.nearFieldDistanceThreshold;
    else
        path.restirPTData.distanceThreshold = PathRISState::gaussrand(gPathTracer.restirpt.settings.nearFieldDistanceThreshold, gPathTracer.restirpt.settings.nearFieldDistanceSigma, sg);

    if (gPathTracer.restirpt.settings.specularRoughnessSigma == 0.f)
        path.restirPTData.roughnessThreshold = gPathTracer.restirpt.settings.specularRoughnessThreshold;
    else
        path.restirPTData.roughnessThreshold = PathRISState::gaussrand(gPathTracer.restirpt.settings.specularRoughnessThreshold, gPathTracer.restirpt.settings.specularRoughnessSigma, sg);
}

void generatePathForRandomReplay( const HitInfo hit, const ShadingData sd, out PathState path, uint initRandomSeed, ReSTIRPathFlags pathFlags )
{
    path = {};
    path.setActive();
    path.thp = float3(1.f);


    // Create primary ray.
    if (sd.curveRadius > 0.f)
        path.origin = sd.posW - sd.frame.N * sd.curveRadius;
    else
        path.origin = sd.computeRayOrigin();

    path.dir = float3(0.f, 0.f, -1.f);

    path.sg = SampleGenerator(initRandomSeed);

    initReplayThresholds(path.sg, path);

    path.restirPTData.pathRcInfo.pathFlags = pathFlags;

    // Load the primary hit info from the V-buffer.

    // Grid volumes are currently not part of the scene geometry.
    // In order to check for volume scattering events *before* the primary hit,
    // we perform a vertex query and use the primary hit as the result of a "simulated" ray query.

    {
        // If invalid, the path is still active and treated as a miss.
        if (hit.isValid())
        {
            path.setHit(hit);
            path.setVertexIndex(1);
        }
        // !hit.isValid() case shouldn't occur in random replay
    }
}

void generatePathForTemporalUpdate( const HitInfo reconnectionHit, const ShadingData rcSd, out PathState path,
    float3 reconnectionWi, uint rcNextVertexBounceSeed, uint initRandomSeed, ReSTIRPathFlags pathFlags )
{
    path = {};
    path.setActive();
    path.thp = float3(1.f);

    bool isTransmission = pathFlags.decodeIsTransmissionEvent(false);
    path.origin = rcSd.computeRayOrigin(!isTransmission);
    path.dir = reconnectionWi;

    path.sg = SampleGenerator(rcNextVertexBounceSeed);

    // Use a separate generator for thresholds so `path.sg` (consumed by the rc-vertex scatter ray)
    // stays bit-aligned with the original trace.
    SampleGenerator thresholdSg = SampleGenerator(initRandomSeed);
    initReplayThresholds(thresholdSg, path);

    path.restirPTData.pathRcInfo.pathFlags = pathFlags;
    path.restirPTData.pathRcInfo.rcThroughput = float3(1.f);

    // Grid volumes are currently not part of the scene geometry.
    // In order to check for volume scattering events *before* the primary hit,
    // we perform a vertex query and use the primary hit as the result of a "simulated" ray query.
    {
        // If invalid, the path is still active and treated as a miss.
        if (reconnectionHit.isValid())
        {
            path.setHit(reconnectionHit);
            path.setVertexIndex(pathFlags.rcVertexLength() + 1);
        }
        // !reconnectionHit.isValid() case shouldn't occur in random replay
    }
}

// return if has non-zero contribution to path
void updateDirectLightingForRcVertex<VisibilityQuery : IVisibilityQueryDynamic>( inout PathState path, ShadingData sd, const IMaterialInstance mi, inout VisibilityQuery vq,
    bool isRcVertex, bool isPrevFrame )
{
    const bool isTriangleHit = path.hit.getType() == HitType::Triangle;
    const bool isCurveHit = gPathTracer.kUseCurves && (path.hit.getType() == HitType::Curve);
    const bool isHairMaterial = gPathTracer.kUseHairMaterial && (sd.mtl.getMaterialType() == MaterialType::Hair);
    const bool isCurvePolyTubeHit = isTriangleHit && isHairMaterial;

    LightSample ls = {};

    const uint lobes = mi.getLobeTypes(sd);
    // Setup path vertex.
    PathVertex vertex = PathVertex( path.getVertexIndex(), (isCurveHit ? sd.posW - sd.frame.N * sd.curveRadius : sd.posW), sd.faceN, sd.frontFacing, isCurveHit );

    // Determine if upper/lower hemispheres need to be sampled.
    bool sampleUpperHemisphere = isCurveHit || isCurvePolyTubeHit || ((lobes & (uint)LobeType::NonDeltaReflection) != 0);
    sampleUpperHemisphere = (!kUseLightsInDielectricVolumes && path.isInsideDielectricVolume()) ? false : sampleUpperHemisphere;

    bool sampleLowerHemisphere = isCurveHit || isCurvePolyTubeHit || ((lobes & (uint)LobeType::NonDeltaTransmission) != 0);

    // Sample a light.
    bool validSample = generateLightSample(vertex, sampleUpperHemisphere, sampleLowerHemisphere, path.sg, ls, isPrevFrame);

    path.setLightSampled(sampleUpperHemisphere, sampleLowerHemisphere);

    path.dir = ls.dir;

    if (validSample)
    {
        // for ReSTIR PT
        if (ls.lightType == (uint)LightType::Analytic && ls.pdf == 0.f)
            ls.pdf = gPathTracer.getAnalyicSelectionProbability(); // analytic light doesn't have a solid angle PDF.
        float3 Le = ls.Li * ls.pdf;

        // Apply MIS weight. (we don't care here)
        if (kUseMIS && ls.lightType != (uint)LightType::Analytic && !isRcVertex)
        {
            float scatterPdf = mi.evalPdf(sd, ls.dir, kUseBSDFSampling);
            ls.Li *= evalMIS(1, ls.pdf, 1, scatterPdf);
        }

        float3 weight = mi.eval(sd, ls.dir, path.sg);
        float3 Lr = weight * ls.Li;
        if (any(Lr > 0.f))
        {
            Ray ray = ls.getVisibilityRay();
            bool visible = true;
            if (isCurvePolyTubeHit)
            {
                // For curves tessellated into poly-tubes, we make sure that the origin is on the same side as light
                // so there is no self-shadowing (transmission lobe of hair BSDF takes care of that).
                if (dot(sd.frame.N, ray.dir) < 0.f)
                {
                    ray.origin = ray.origin - sd.frame.N * sd.curveRadius * 2.1f;
                }
            }

            logTraceRay(PixelStatsRayType::Visibility);
            visible = vq.traceVisibilityRay(ray, isPrevFrame);

            if (visible)
            {
                gPathTracer.addToPathContribution(path, Lr);
                float3 reconnectionIncidentRadiance = isRcVertex ? Le : path.restirPTData.pathRcInfo.rcThroughput * Lr;
                gPathTracer.restirpt.WriteTemporalRetraceData(path, ls.lightType, ls.pdf, reconnectionIncidentRadiance);
            }
        }
    }
}

bool generateScatterRayGivenDirection(ShadingData sd, const IMaterialInstance mi, inout PathState path)
{
    uint componentType = path.restirPTData.pathRcInfo.pathFlags.decodeBSDFComponent(false);
    bool isDeltaEvent = path.restirPTData.pathRcInfo.pathFlags.decodeIsDeltaEvent(false);
    bool isTransmission = path.restirPTData.pathRcInfo.pathFlags.decodeIsTransmissionEvent(false);
    if (isDeltaEvent) return false;

    BSDFSample result = {};

    // hack the material system for sampled lobe
    sd.mtl.setBSDFIndex(componentType); // if this is non-zero, result.pdf will only return the pdf of the sampled lobe
    float3 f = mi.eval(sd, path.dir, path.sg);
    float pdf = mi.evalPdf(sd, path.dir);

    result.weight = pdf <= 0.f ? 0.f : f / pdf;

    if (all(f == 0.f) || pdf == 0.f) return false;

    result.wo = path.dir;


    // we should only compute this pdf if next bounce we are going to terminate with bsdf hit
    if ((path.getVertexIndex() - 1) == path.restirPTData.pathRcInfo.pathFlags.pathLength() &&
        !path.restirPTData.pathRcInfo.pathFlags.lastVertexNEE())
    {
        sd.mtl.setBSDFIndex(-1); // if this is non-zero, result.pdf will only return the pdf of the sampled lobe
        result.pdf = mi.evalPdf(sd, path.dir);
        // we need to query pdf of unsampled components
    }
    path.setSampledBSDFComponent(componentType);

    result.lobeType = componentType == 0 ? uint(LobeType::DiffuseReflection) :
    (componentType == 1 ? uint(LobeType::DiffuseTransmission) : ((componentType == 2 || !isTransmission) ?
        uint(LobeType::SpecularReflection) : uint(LobeType::SpecularTransmission)));

    // ReSTIR PT
    path.setRough(mi.isSampleRough(sd, componentType, result.pdf, path.restirPTData.roughnessThreshold));
    path.setSupportBSDFIndexing(mi.supportBSDFIndexing());
    path.setSupportSimpleRoughnessThreshold(mi.supportSimpleRoughnessThreshold());

    path.setLightSampled(true, true);

    return gPathTracer.generateScatterRay(result, sd, mi, path);
}

bool generateScatterRayRandomReplay(const ShadingData sd, const IMaterialInstance mi, inout PathState path)
{
    bool needToQueryAllLobes = (path.getVertexIndex() - 1) == path.restirPTData.pathRcInfo.pathFlags.pathLength() &&
                               !path.restirPTData.pathRcInfo.pathFlags.lastVertexNEE();
    BSDFSample result = {};
    // hack the material system for sampled lobe
    bool valid = mi.sample(sd, path.sg, result, kUseBSDFSampling);

    // set result.pdf to a large value for delta
    if (result.isLobe(LobeType::Delta)) result.pdf = 1e20f;

    valid &= !isinf(result.pdf) && all(!isinf(result.weight));
    if (!valid) return false;


    // get total pdf in a separate call
    uint componentType = result.BSDFIndex;

    path.setSampledBSDFComponent(componentType);

    // ReSTIR PT
    path.setRough(mi.isSampleRough(sd, componentType, result.pdf, path.restirPTData.roughnessThreshold));
    path.setSupportBSDFIndexing(mi.supportBSDFIndexing());
    path.setSupportSimpleRoughnessThreshold(mi.supportSimpleRoughnessThreshold());


    // Ignore valid on purpose for now.
    if (kFetchNRDDataPass)
    {
        const uint lobeTypes = mi.getLobeTypes(sd);
        const bool hasDeltaTransmissionLobe = (lobeTypes & (uint)LobeType::DeltaTransmission) != 0;
        const bool hasNonDeltaLobes = (lobeTypes & (uint)LobeType::NonDelta) != 0;

        if (path.getVertexIndex() == 1)
        {
            // Mark path as delta-only if it followed delta lobe on the primary hit, even though there might have been non-delta lobes.
            path.setDeltaOnlyPath(result.isLobe(LobeType::DeltaReflection) || result.isLobe(LobeType::DeltaTransmission));

            path.setDeltaReflectionPrimaryHit(result.isLobe(LobeType::DeltaReflection));
            path.setDeltaTransmissionPath(result.isLobe(LobeType::DeltaTransmission));
        }

        if (path.getVertexIndex() > 1)
        {
            if (hasNonDeltaLobes) path.setDeltaOnlyPath(false);

            if (path.isDeltaTransmissionPath() && path.isDeltaOnlyPath() && hasDeltaTransmissionLobe)
            {
                if (result.isLobe(LobeType::DeltaReflection) && !isDeltaReflectionAllowedAlongDeltaTransmissionPath(sd))
                {
                    path.setDeltaTransmissionPath(false);
                }
            }
        }
    }

    return gPathTracer.generateScatterRay(result, sd, mi, path);
}

[ForceInline]
float3 computeReplayPathOriginForHit(ShadingData sd, bool isCurveHit, bool isCurvePolyTubeHit
)
{
    if (isCurveHit)
    {
        // For curves, set the new origin at the sphere center.
        return sd.posW - sd.frame.N * sd.curveRadius;
    }
    else if (isCurvePolyTubeHit)
    {
        // For curves tessellated into poly-tubes, offset slightly away from the curve center.
        return sd.posW + sd.frame.N * sd.curveRadius * 0.1f;
    }
    return sd.computeRayOrigin();
}


void handleSurfaceHitReplayPath<VisibilityQuery : IVisibilityQueryDynamic>(inout PathState path, inout VisibilityQuery vq)
{
    const HitType hitType = path.hit.getType();
    const bool isTriangleHit = hitType == HitType::Triangle;
    const bool isCurveHit = gPathTracer.kUseCurves && (hitType == HitType::Curve);
    let lod = gPathTracer.createTextureSampler(path, false);

    // Load shading data. This is a long latency operation.
    ShadingData sd = loadShadingData(path.hit, path.origin, path.dir, path.isPrevFrame());

    const bool isHairMaterial = gPathTracer.kUseHairMaterial && (sd.mtl.getMaterialType() == MaterialType::Hair);
    const bool isCurvePolyTubeHit = isTriangleHit && isHairMaterial;


    // Reject false hits in nested dielectrics.
    if (!gPathTracer.handleNestedDielectrics(sd, path)) return;

    int currentPathLength = path.getVertexIndex() - 1;
    int rcVertexLength = path.restirPTData.pathRcInfo.pathFlags.rcVertexLength();

    // Create BSDF instance and query its properties.
    const IMaterialInstance mi = gScene.materials.getMaterialInstance(sd, lod);
    BSDFProperties bsdfProperties = mi.getProperties(sd);

    gPathTracer.evaluateLightProfile(bsdfProperties, sd, path);

    // Disable specular lobes if caustics are disabled and path already contains a diffuse vertex.
    bool isSpecular = bsdfProperties.roughness <= gPathTracer.params.specularRoughnessThreshold;
    if (kDisableCaustics && path.getBounces(BounceType::Diffuse) > 0 && isSpecular)
    {
        sd.mtl.setActiveLobes((uint)LobeType::Diffuse);
    }

    // Optionally disable emission inside volumes.
    if (!kUseLightsInDielectricVolumes && path.isInsideDielectricVolume())
    {
        bsdfProperties.emission = float3(0.f);
    }

    // Check if the scatter event is samplable by the light sampling technique.
    const bool isLightSamplable = path.isLightSamplable();

    // Add emitted radiance.
    // The primary hit is always included, secondary hits only if emissive lights are enabled and the full light contribution hasn't been sampled elsewhere.
    bool computeEmissive = gPathTracer.kUseEmissiveLights && (!kUseNEE || kUseMIS || !path.isLightSampled() || !isLightSamplable);

    // kUseReSTIRPT stuff
    float distanceToPrevVertex = length(sd.posW - path.origin);
    float geometryFactor = abs(dot(sd.faceN, sd.V)) / (distanceToPrevVertex * distanceToPrevVertex);
    LastVertexState lastVertexState;
    float inverseGeometryFactor = distanceToPrevVertex * distanceToPrevVertex / abs(dot(path.normal, sd.V));
    float rayFootprint = 1.f / (geometryFactor * path.pdf);

    lastVertexState.Init( path.isDelta(), path.isTransmission(), path.getSampledBSDFComponent(), path.supportBSDFIndexing(), path.isRough(),
        rayFootprint >= path.restirPTData.primaryRayFootprint * path.restirPTData.distanceThreshold );

    bool terminateByHittingLight = currentPathLength - 1 == path.restirPTData.pathRcInfo.pathFlags.pathLength() &&
                                        !path.restirPTData.pathRcInfo.pathFlags.lastVertexNEE();
    // terminateByNEE is currently impossible because of the forced NEE reconnection
    bool terminateByNEE = currentPathLength == path.restirPTData.pathRcInfo.pathFlags.pathLength() &&
                                         path.restirPTData.pathRcInfo.pathFlags.lastVertexNEE();
    // skip random number to align between PSS sequences when reuse
    sampleNext1D(path.sg);

    if (computeEmissive && terminateByHittingLight && any(bsdfProperties.emission > 0.f))
    {
        float misWeight = 1.f;
        float lightPdf = 0.f;

        if (gPathTracer.kUseEmissiveLights && kUseNEE && kUseMIS && isTriangleHit && path.isLightSampled() && isLightSamplable)
        {
            // If NEE and MIS are enabled, and we've already sampled emissive lights,
            // then we need to evaluate the MIS weight here to account for the remaining contribution.
            // Note that MIS is only applied for hits on emissive triangles (other emissive geometry is not supported).

            // Prepare hit point struct with data needed for emissive light PDF evaluation.
            TriangleHit triangleHit = path.hit.getTriangleHit();
            TriangleLightHit hit;
            hit.triangleIndex = gScene.lightCollection.getTriangleIndex(triangleHit.instanceID, triangleHit.primitiveIndex);
            hit.posW = sd.posW;
            hit.normalW = sd.frontFacing ? sd.faceN : -sd.faceN;

            // Evaluate PDF at the hit, had it been generated with light sampling.
            // Emissive light samplers have an option to exclusively sample the upper hemisphere.
            bool upperHemisphere = path.isLightSampledUpper() && !path.isLightSampledLower();
            lightPdf = gPathTracer.getEmissiveSelectionProbability() * gPathTracer.emissiveSampler.evalPdf(path.origin, path.normal, upperHemisphere, hit);

            // Compute MIS weight by combining this with BSDF sampling.
            // Note we can assume path.pdf > 0.f since we shouldn't have got here otherwise.

            int initialLightSampleCount = max(1, gPathTracer.restirpt.computeInitialLightSampleCount( gPathTracer.restirpt.initialLightSampleCount, currentPathLength));
            misWeight = evalMIS(1, path.pdf, (kUseLightTileRIS ? initialLightSampleCount : 1), lightPdf);
        }

        // Accumulate emitted radiance weighted by path throughput and MIS weight.
        gPathTracer.addToPathContribution(path, misWeight * bsdfProperties.emission);

        if (gPathTracer.restirpt.kTemporalUpdateForDynamicScene)
        {
            float3 reconnectionIncidentRadiance = currentPathLength - 1 == rcVertexLength ?
            bsdfProperties.emission : path.restirPTData.pathRcInfo.rcThroughput * misWeight * bsdfProperties.emission;
            gPathTracer.restirpt.WriteTemporalRetraceData(path, (uint)LightType::Emissive, lightPdf, reconnectionIncidentRadiance);
        }
        if (!kFetchNRDDataPass)
            path.L = currentPathLength < rcVertexLength && (
                min(rayFootprint, inverseGeometryFactor / gPathTracer.restirpt.getDiffusePseudoPdf()) >= path.restirPTData.primaryRayFootprint * path.restirPTData.distanceThreshold
                && (!path.supportSimpleRoughnessThreshold() || lastVertexState.isLastVertexRough())) ? 0.f : path.L;
    }

    // Terminate after scatter ray on last vertex has been processed.
    bool lastVertex = false;
    if (!(kMaxSurfaceBounces == kMaxDiffuseBounces && kMaxSurfaceBounces == kMaxSpecularBounces && kMaxSurfaceBounces == kMaxTransmissionBounces))
        lastVertex = gPathTracer.hasFinishedSurfaceBounces(path);

    if (lastVertex || terminateByHittingLight || gPathTracer.restirpt.kTemporalUpdateForDynamicScene && currentPathLength >= rcVertexLength && terminateByNEE)
    {
        if (kFetchNRDDataPass) path.thp = 0.f;
        path.terminate();
        return;
    }

    if (kFetchNRDDataPass)
    {        
        // if we are about to terminate due to length limit or reconnection, also write NRD data
        if (currentPathLength == rcVertexLength - 1) // reached Vertex Before reconnection
        {
            if (path.isDeltaReflectionPrimaryHit() || path.isDeltaTransmissionPath())
            {
                const MaterialType materialType = sd.mtl.getMaterialType();
                float3 deltaReflectance = getMaterialReflectanceForDeltaPaths(materialType, true, sd, bsdfProperties);
                path.thp = deltaReflectance;
            }
            else path.thp = 0.f;

            path.terminate();
            return;
        }
    }

    path.origin = computeReplayPathOriginForHit(sd, isCurveHit, isCurvePolyTubeHit
    );
    // Determine if BSDF has non-delta lobes.
    const uint lobes = mi.getLobeTypes(sd);
    const bool hasNonDeltaLobes = (lobes & (uint)LobeType::NonDelta) != 0;

    // this is always true if we are not doing temporal update
    if (!kFetchNRDDataPass && (!gPathTracer.restirpt.kTemporalUpdateForDynamicScene || currentPathLength < rcVertexLength))
    {
        bool shouldTerminate = currentPathLength == rcVertexLength - 1;
        bool nonInvertible = shouldTerminate && !hasNonDeltaLobes;

        if (shouldTerminate && !nonInvertible)
        {
            gPathTracer.restirpt.WriteReconnectionData( path, sd.V, lastVertexState, rayFootprint, inverseGeometryFactor );
        }

        path.setVertexIndex(nonInvertible ? 0 : path.getVertexIndex());
        path.setFlag(PathFlags::active, !shouldTerminate);

        if (!path.isActive()) return;
    }

    if (terminateByNEE)
    {
        path.terminate();
        return;
    }

    // we can use stored light vertex if we use a position-independent light sampler

    {
#if _EMISSIVE_LIGHT_SAMPLER_TYPE == EMISSIVE_LIGHT_SAMPLER_LIGHT_BVH
        // Check if we should apply NEE.
        const bool sampleLights = kUseNEE && hasNonDeltaLobes;
        if (sampleLights)
        {
            bool sampleUpperHemisphere = isCurveHit || isCurvePolyTubeHit || ((lobes & (uint)LobeType::NonDeltaReflection) != 0);
            bool sampleLowerHemisphere = isCurveHit || isCurvePolyTubeHit || ((lobes & (uint)LobeType::NonDeltaTransmission) != 0);
            if (!kUseLightsInDielectricVolumes && path.isInsideDielectricVolume()) sampleUpperHemisphere = false;
            path.setLightSampled(sampleUpperHemisphere, sampleLowerHemisphere);
        }
#endif

        // make sure that random numbers match
        sampleNext1D(path.sg);
        gPathTracer.skipLightSampleRandomNumbers(path.sg);
    }

    if (kUseRussianRoulette) sampleNext1D(path.sg);

    // Russian roulette to terminate paths early.

    bool wasDeltaOnlyPathBeforeScattering = path.isDeltaOnlyPath();
    bool wasDeltaTransmissionPathBeforeScattering = path.isDeltaTransmissionPath();

    // Generate the next path segment or terminate.
    bool valid = generateScatterRayRandomReplay(sd, mi, path);

    if (kFetchNRDDataPass)
    {        
        // write NRD data when non-delta scattering is sampled        
        if (!path.isDelta())
        {
            if (path.isDeltaReflectionPrimaryHit() || wasDeltaTransmissionPathBeforeScattering)
            {
                const MaterialType materialType = sd.mtl.getMaterialType();
                const bool hasDeltaLobes = (lobes & (uint)LobeType::Delta) != 0;
                float3 deltaReflectance = getMaterialReflectanceForDeltaPaths(materialType, hasDeltaLobes, sd, bsdfProperties);
                path.thp = deltaReflectance;
            }
            else path.thp = 0.f;
            path.terminate();
            return;
        }
    }

    // non-invertible (current vertex being rough before the replay ends)
    if (!kFetchNRDDataPass && ((!gPathTracer.restirpt.kTemporalUpdateForDynamicScene || currentPathLength < rcVertexLength) &&
        (min(rayFootprint, inverseGeometryFactor / path.pdf) >= path.restirPTData.primaryRayFootprint * path.restirPTData.distanceThreshold && lastVertexState.isLastVertexRough()
        )))
    {
        path.setVertexIndex(0);
        valid = false;
    }

    if (!(kMaxSurfaceBounces == kMaxDiffuseBounces && kMaxSurfaceBounces == kMaxSpecularBounces && kMaxSurfaceBounces == kMaxTransmissionBounces))
    {
        // Check if this is the last path vertex. Don't terminate at delta vertices as it messes with NRC predictions).
        bool isLastVertex = gPathTracer.hasFinishedSurfaceBounces(path);
        // Terminate if this is the last path vertex and light sampling already completely sampled incident radiance.
        if (kUseNEE && !kUseMIS && isLastVertex && path.isLightSamplable()) valid = false;
    }

    // Terminate caustics paths.
    if (kDisableCaustics && path.getBounces(BounceType::Diffuse) > 0 && path.isSpecular()) valid = false;

    // Terminate the ray if we've generated an invalid segment
    path.setFlag(PathFlags::active, valid);
}

void handleHitReplayPath<VisibilityQuery : IVisibilityQueryDynamic>(inout PathState path, inout VisibilityQuery vq)
{
    // doesn't support volume hit
    if (path.hit.getType() == HitType::Volume)
    {
    }
    else
    {
        handleSurfaceHitReplayPath(path, vq);
    }
}
/** Handle the case when a scatter ray hits a volume.
    After handling the hit, a new scatter ray is generated or the path is terminated.
    \param[in,out] path The path state.
    \param[in,out] vq Visibility query.
*/

/** Handle hit on specular materials.
    After handling the hit, a new scatter ray is generated or the path is terminated.
    \param[in,out] path The path state.
*/

void handleMissReplayPath(inout PathState path)
{
    // Upon miss:
    // - Compute MIS weight if previous path vertex sampled a light
    // - Evaluate environment map
    // - Write guiding data
    // - Terminate the path

    // Check if the scatter event is samplable by the light sampling technique.
    const bool isLightSamplable = path.isLightSamplable();

    // Add env radiance.
    bool computeEnv = gPathTracer.kUseEnvLight && (!kUseNEE || kUseMIS || !path.isLightSampled() || !isLightSamplable);

    // Disable all constributions to direct illumination if requested.
    if (/*kDisableDirectIllumination &&*/ path.getVertexIndex() == 2 && !gPathTracer.restirpt.sampleDI) computeEnv = false;

    float3 emitterRadiance = 0.f;

    int currentPathLength = path.getVertexIndex() - 1;
    int rcVertexLength = path.restirPTData.pathRcInfo.pathFlags.rcVertexLength();

    bool terminateByHittingLight = currentPathLength - 1 == path.restirPTData.pathRcInfo.pathFlags.pathLength() &&
                                        !path.restirPTData.pathRcInfo.pathFlags.lastVertexNEE(); // && !isRcVertexEmissive;

    if (computeEnv && terminateByHittingLight)
    {

        float misWeight = 1.f;
        float lightPdf = 0.f;
        if (kUseNEE && kUseMIS && path.isLightSampled() && isLightSamplable)
        {
            // If NEE and MIS are enabled, and we've already sampled the env map,
            // then we need to evaluate the MIS weight here to account for the remaining contribution.

            // Evaluate PDF, had it been generated with light sampling.
            lightPdf = gPathTracer.getEnvMapSelectionProbability() * gPathTracer.envMapSampler.evalPdf(path.dir);

            // Compute MIS weight by combining this with BSDF sampling.
            // Note we can assume path.pdf > 0.f since we shouldn't have got here otherwise.
            int initialLightSampleCount = gPathTracer.restirpt.computeInitialLightSampleCount( gPathTracer.restirpt.initialLightSampleCount, currentPathLength);
            misWeight = evalMIS(1, path.pdf, (kUseLightTileRIS ? initialLightSampleCount : 1), lightPdf);
        }

        float3 Le = gPathTracer.envMapSampler.eval(path.dir, 0.f, path.isPrevFrame());
        emitterRadiance = misWeight * Le;
        gPathTracer.addToPathContribution(path, emitterRadiance);
        // for temporal update
        float3 reconnectionIncidentRadiance = currentPathLength - 1 == rcVertexLength ? Le : path.restirPTData.pathRcInfo.rcThroughput * emitterRadiance;
        gPathTracer.restirpt.WriteTemporalRetraceData(path, (uint)LightType::EnvMap, lightPdf, reconnectionIncidentRadiance);

        // differentiate between hybrid shift replay and temporal update
        // we are suppose to terminate at this vertex, so this shouldn't be connectible. otherwise it is non-invertible
        if (!kFetchNRDDataPass)
            path.L = currentPathLength < rcVertexLength && path.isRough() ? 0.f : path.L;
    }

    path.terminate();
}

