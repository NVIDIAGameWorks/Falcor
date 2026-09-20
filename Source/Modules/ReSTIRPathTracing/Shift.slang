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
__exported import PathReservoir;
__exported import ReSTIRPT;
import Scene.ShadingData;
import LoadShadingDataDynamic; // currently using a copy from InlinePathtracer
import PathTracer;
import Utils.Sampling.TinyUniformSampleGenerator;
import Scene.RaytracingInline;
import Utils.Debug.PixelDebug;
__exported import HybridShiftTraceRays;
import Rendering.Lights.EmissiveLightSampler;
#include "Rendering/Lights/EmissiveLightSamplerType.slangh"

static const float kShadowRayTMaxRatio      = 0.999f;
static const float kShadowOriginAdditionalOffsetRatio = 0.005f;

bool isJacobianInvalid(float Jacobian)
{
    return Jacobian <= 0.f || isnan(Jacobian) || isinf(Jacobian);
}

bool isIntegrandInvalid(float3 Integrand)
{
    return all(Integrand == 0.f) || any(isnan(Integrand) || isinf(Integrand) || Integrand < 0.f);
}

struct PairwiseFactor
{
    float c_sum;
    float c_c;
}

bool shiftAndMergeReservoirTemporal( const ReSTIRPT restirpt, const ReSTIRPathTracingParams params,
    inout PathReservoir dstReservoir, PathReservoir srcReservoir, ReconnectionData rcData, float primaryRayFootprint, inout SampleGenerator sg,
    float c_c, float splatOrLensShiftJacobian = 1.f, bool temporalUpdateForDynamicScene = false )
{
    float Jacobian = 1.f;
    float3 dstIntegrand = shiftPath(restirpt, params, Jacobian,
                                    srcReservoir, rcData, primaryRayFootprint, false, temporalUpdateForDynamicScene);

    Jacobian *= splatOrLensShiftJacobian;
    float srcTerm = PathRISState::toScalar(srcReservoir.integrand) * srcReservoir.M;
    float misWeight = srcTerm / (srcTerm + Jacobian * PathRISState::toScalar(dstIntegrand) * c_c);

    bool selected = dstReservoir.merge(dstIntegrand, Jacobian, srcReservoir, sg, misWeight);

    return selected;
}

// shouldn't use this for canonical samples
bool shiftAndMergeReservoirSpatial( const ReSTIRPT restirpt, const ReSTIRPathTracingParams params,
    inout PathReservoir dstReservoir, PathReservoir srcReservoir, ReconnectionData rcData, float primaryRayFootprint, inout SampleGenerator sg,
    PairwiseFactor C, float splatOrLensShiftJacobian
#if USE_DECOUPLED_SHADING
    , inout float3 outColor
#endif

)
{
    float Jacobian = 1.f;
    float3 dstIntegrand = shiftPath(restirpt, params, Jacobian,
                                    srcReservoir, rcData, primaryRayFootprint, false);

    Jacobian *= splatOrLensShiftJacobian;
    float srcTerm = PathRISState::toScalar(srcReservoir.integrand) * (C.c_sum - C.c_c);
    float misWeight = srcReservoir.M / C.c_sum * srcTerm / (srcTerm + Jacobian * PathRISState::toScalar(dstIntegrand) * C.c_c);

    bool selected = dstReservoir.merge(dstIntegrand, Jacobian, srcReservoir, sg, misWeight);

#if USE_DECOUPLED_SHADING
    float3 wvec = dstIntegrand * Jacobian * srcReservoir.weight * (isnan(misWeight) ? 0.f : misWeight);
    outColor += wvec;
#endif    

    return selected;
}

bool shiftAndMergeReservoirSpatialWithPrecomputedInfo(
    inout PathReservoir dstReservoir, PathReservoir srcReservoir, PrecomputedShiftData spData, inout SampleGenerator sg,
    PairwiseFactor C, float splatOrLensShiftJacobian
#if USE_DECOUPLED_SHADING
    , inout float3 outColor
#endif
)
{
    float Jacobian = 1.f;
    float3 dstIntegrand = shiftPathWithPrecomputedInfo(Jacobian,
                                    srcReservoir, spData);
    Jacobian *= splatOrLensShiftJacobian;
    float srcTerm = PathRISState::toScalar(srcReservoir.integrand) * (C.c_sum - C.c_c);
    float misWeight = srcReservoir.M / C.c_sum * srcTerm / (srcTerm + Jacobian * PathRISState::toScalar(dstIntegrand) * C.c_c);

    bool selected = dstReservoir.merge(dstIntegrand, Jacobian, srcReservoir, sg, misWeight);

#if USE_DECOUPLED_SHADING
    float3 wvec = dstIntegrand * Jacobian * srcReservoir.weight * (isnan(misWeight) ? 0.f : misWeight);
    outColor += wvec;
#endif

    return selected;
}

// this function doesn't overwrite external srcReservoir
float3 computeShiftedIntegrand(
    const ReSTIRPT restirpt, const ReSTIRPathTracingParams params, inout float Jacobian,
    PathReservoir srcReservoir, ReconnectionData rcData, float primaryRayFootprint, bool isPrevFrame = false, bool temporalUpdateForDynamicScene = false)
{
    return shiftPath(restirpt, params, Jacobian,
                     srcReservoir, rcData, primaryRayFootprint, isPrevFrame, temporalUpdateForDynamicScene);
}

float3 computeShiftedIntegrandWithPrecomputedInfo(
    inout float Jacobian,
    PathReservoir srcReservoir, PrecomputedShiftData spData )
{
    return shiftPathWithPrecomputedInfo(Jacobian,
                     srcReservoir, spData);
}

float3 shiftPathWithPrecomputedInfo(  inout float Jacobian,
    inout PathReservoir srcReservoir, PrecomputedShiftData spData)
{
    //  4+4+12 = 20 bytes
    srcReservoir.reconnectionJacobian = spData.reconnectionJacobian;
    Jacobian = spData.Jacobian;
    return spData.pathThroughput;
}

float3 shiftPath(const ReSTIRPT restirpt, const ReSTIRPathTracingParams params, inout float Jacobian,
                 inout PathReservoir srcReservoir, ReconnectionData rcData, float primaryRayFootprint,
                 bool isPrevFrame = false, bool temporalUpdateForDynamicScene = false)
{
    float3 pathThroughput = (srcReservoir.weight != 0.f) ? rcData.pathThroughput : float3(0.f);
    bool isRcVertexLengthOne = srcReservoir.pathFlags.rcVertexLength() == 1;

    // the reconnection vertex exists
    if (restirpt.kUseAreaReservoirs && !rcData.rcPrevHit.isValid() && srcReservoir.pathFlags.rcVertexLength() < restirpt.kMaximumRcLength) // return emissive value
    {
        if (!srcReservoir.pathFlags.isVeryDirectLight()) pathThroughput = 0.f;
        else pathThroughput = gPathTracer.envMapSampler.eval(-rcData.rcPrevWo, 0.f, isPrevFrame);
        Jacobian = 1.f;
    }

    if (!restirpt.kUsePrevFrameSceneData) isPrevFrame = false;

    if (any(pathThroughput > 0.f) && rcData.rcPrevHit.isValid())
    {
        ShadingData dstRcPrevSd = loadShadingData(rcData.rcPrevHit, float3(0.f), -rcData.rcPrevWo, isPrevFrame);

        if (temporalUpdateForDynamicScene && restirpt.kTemporalUpdateForDynamicScene && !srcReservoir.pathFlags.isVeryDirectLight())
        {
            TraceTemporalUpdate(dstRcPrevSd, srcReservoir, isPrevFrame);
            if (all(srcReservoir.reconnectionIncidentRadiance == 0.f)) pathThroughput = 0.f;
        }

        float reconnectionJacobian = 1.f;
        pathThroughput *= shiftPathReconnection( restirpt, params, primaryRayFootprint, rcData.rcFootprintFactor, reconnectionJacobian, dstRcPrevSd, srcReservoir, isPrevFrame);
        Jacobian *= reconnectionJacobian;
    }

    return pathThroughput;
}

[ForceInline]
void sampleHybridShiftThresholds(const ReSTIRPT restirpt, uint initRandomSeed, out float distanceThreshold, out float roughnessThreshold)
{
    SampleGenerator thresholdSg = SampleGenerator(initRandomSeed);
    distanceThreshold  = (restirpt.settings.nearFieldDistanceSigma == 0.f)
        ? restirpt.settings.nearFieldDistanceThreshold
        : PathRISState::gaussrand(restirpt.settings.nearFieldDistanceThreshold, restirpt.settings.nearFieldDistanceSigma, thresholdSg);
    roughnessThreshold = (restirpt.settings.specularRoughnessSigma == 0.f)
        ? restirpt.settings.specularRoughnessThreshold
        : PathRISState::gaussrand(restirpt.settings.specularRoughnessThreshold, restirpt.settings.specularRoughnessSigma, thresholdSg);
}

[ForceInline]
void recomputeLightPdfForEmissiveRcVertex(inout float lightPdf, HitInfo rcVertexHit, ShadingData rcVertexSd, ShadingData dstPrimarySd, const IMaterialInstance dstMI, bool isCurveHit)
{
#if _EMISSIVE_LIGHT_SAMPLER_TYPE == EMISSIVE_LIGHT_SAMPLER_LIGHT_BVH
    TriangleLightHit hit;
    TriangleHit triangleHit = rcVertexHit.getTriangleHit();
    hit.triangleIndex = gScene.lightCollection.getTriangleIndex(triangleHit.instanceID, triangleHit.primitiveIndex);
    hit.posW = rcVertexSd.posW;
    hit.normalW = rcVertexSd.getOrientedFaceNormal();

    uint lobeTypes = dstMI.getLobeTypes(dstPrimarySd);
    bool sampleUpperHemisphere = isCurveHit || (lobeTypes & (uint)LobeType::NonDeltaReflection) != 0;
    bool sampleLowerHemisphere = isCurveHit || (lobeTypes & (uint)LobeType::NonDeltaTransmission) != 0;
    lightPdf = gPathTracer.getEmissiveSelectionProbability() * gPathTracer.emissiveSampler.evalPdf( dstPrimarySd.posW, dstPrimarySd.frame.N, sampleUpperHemisphere && !sampleLowerHemisphere, hit );
#endif
}

/**
    Algorithmic references:
      - Mixed PSS / solid-angle Jacobian (See the comments in Line 333 PathReservoir.slang for more details):
       The reservoir stores `reconnectionJacobian = G(x_{k-1} -> x_k)` (geometry factor at the source pixel) 
       and we accumulate `Jacobian *= dstGeometryTerm / srcReservoir.reconnectionJacobian`
       at the end. No scatter-PDF ratios appear because the integrand around the reconnection vertex 
       is parameterized in solid-angle.
      - Dual ray-footprint reconnection criteria: Lin et al. 2026 (ReSTIR PT Enhanced, Eq. 5).
            min( 1/(p_scatter_{k-1} * G(x_{k-1}->x_k)), 1/(p_scatter_k * G(x_k->x_{k-1})) )
              >= (c/100) * primaryRayFootprint
        The ratio `(c/100)` is rolled into `restirpt.settings.nearFieldDistanceThreshold`
        on the host (see `ReSTIRPathTracing.cpp`, `setShaderData`).
      - Single-vertex roughness threshold at x_{k-1}:  See ReSTIR PT Enhanced Section 4.2

    Returns the shifted integrand at the destination, scales `Jacobian` by the shift Jacobian, and
    overwrites `srcReservoir.reconnectionJacobian` with the destination geometry factor (used by
    later GRIS rounds to chain shifts).
*/
float3 shiftPathReconnection(const ReSTIRPT restirpt, const ReSTIRPathTracingParams params, float primaryRayFootprint, float rcFootprintFactor,
                             inout float Jacobian, ShadingData dstPrimarySd, inout PathReservoir srcReservoir, bool isPrevFrame)
{
    // dummy sg (stochastic BSDF is not supported yet)
    var sg = TinyUniformSampleGenerator(1);

    // ---- Unpack reservoir + flag decoding ----
    const int rcVertexLength = srcReservoir.pathFlags.rcVertexLength();
    const HitInfo rcVertexHit = unpackReservoirHitInfo(srcReservoir.reconnectionHit);
    const float3 rcVertexIrradiance = srcReservoir.reconnectionIncidentRadiance;
    const float3 rcVertexWi = srcReservoir.reconnectionWi;
    bool rcVertexHitExists = rcVertexHit.isValid();
    const float3 reconnectionJacobian = srcReservoir.reconnectionJacobian;

    const bool isTransmission1 = srcReservoir.pathFlags.decodeIsTransmissionEvent(true);
    const uint componentType1 = srcReservoir.pathFlags.decodeBSDFComponent(true);

    const bool isDelta1 = srcReservoir.pathFlags.decodeIsDeltaEvent(true);
    const bool isDelta2 = srcReservoir.pathFlags.decodeIsDeltaEvent(false);
    // delta bounce before/after rcVertex (if isRcVertexNEE, deltaAfterRc won't be set)
    if (isDelta1 || isDelta2) return 0.f;

    const bool isRcVertexConnectedtoLight = srcReservoir.pathFlags.pathLength() == rcVertexLength;
    const bool isRcVertexEmissive = srcReservoir.pathFlags.pathLength() + 1 == rcVertexLength;
    const bool isLightPathNEE = srcReservoir.pathFlags.lastVertexNEE();
    const bool isRcVertexNEE = isRcVertexConnectedtoLight && isLightPathNEE;

    bool isCurveHit = dstPrimarySd.curveRadius > 0.f;
    if (isCurveHit) dstPrimarySd.posW = dstPrimarySd.posW - dstPrimarySd.frame.N * dstPrimarySd.curveRadius;
    else dstPrimarySd.posW = dstPrimarySd.computeRayOrigin(!isTransmission1); // a simple way to set ray offset. 
    //Note that this will disallow SpecularReflectionTransmission to switch side (reflection<->refraction)
    //during reuse as otherwise self-shadowing will happen

    const IMaterialInstance dstMI = gScene.materials.getMaterialInstance( dstPrimarySd, ExplicitLodTextureSampler(rcVertexLength == 1 ? 0.f : params.lodBias),
            getMaterialInstanceHints(dstPrimarySd.curveRadius == 0.f, rcVertexLength == 1));

    if (restirpt.kUseAreaReservoirs && srcReservoir.pathFlags.isVeryDirectLight())
    {
        Jacobian = 1.f;
        return dstMI.getProperties(dstPrimarySd).emission;
    }
    // cannot support reuse between these too cases
    if (dstMI.supportBSDFIndexing() != srcReservoir.pathFlags.decodeSupportBSDFIndexingBeforeRcVertex()) return 0.f;

    // ---- Resolve the rc-vertex shading data + connection vector ----
    ShadingData rcVertexSd = {};
    float3 shiftedDisp = 0.f;
    float shifted_dist2 = 0.f;

    // direction point from dst primary hit point to reconnection vertex
    float3 dstConnectionV;
    rcVertexSd = loadShadingData( rcVertexHit, dstPrimarySd.posW, float3(0.f), isPrevFrame );
    bool isPointLight = false;
    if (!rcVertexHitExists && isRcVertexEmissive && isLightPathNEE && (rcVertexHit.data.x == 1 || rcVertexHit.data.x == 2))
    {
        rcVertexSd.posW = rcVertexWi;
#if !HIT_INFO_USE_COMPRESSION
        rcVertexSd.faceN = asfloat(rcVertexHit.data.yzw);
#endif
        rcVertexHitExists = true;
        isPointLight = rcVertexHit.data.x == 1;
    }
    shiftedDisp = rcVertexSd.posW - dstPrimarySd.posW;
    shifted_dist2 = dot(shiftedDisp, shiftedDisp);

    // ---- Determine hybrid-shift thresholds ----
    float distanceThreshold, roughnessThreshold;
    sampleHybridShiftThresholds(restirpt, srcReservoir.initRandomSeed, distanceThreshold, roughnessThreshold);

    dstConnectionV = rcVertexHitExists ? shiftedDisp / sqrt(shifted_dist2) : rcVertexWi;
    float shifted_cosine = isPointLight ? 1.f : abs(dot(rcVertexSd.faceN, -dstConnectionV));

    // Destination geometry factor G(y_{k-1} -> x_k). Used both as the numerator of the shift Jacobian
    // (committed at the bottom of this function) and inside the footprint tests below.
    float dstGeometryTerm = shifted_cosine / shifted_dist2;
    float dstJacobian = rcVertexHitExists ? dstGeometryTerm : 1.f;

    float3 dstIntegrand = rcVertexIrradiance;

    // assuming bsdf sampling
    // setup lobe mask to query only sampled components
    // does not matter for non-BSDF indexing material
    dstPrimarySd.mtl.setBSDFIndex(isRcVertexEmissive && isLightPathNEE ? -1 : componentType1);
    float3 dstF1 = dstMI.eval(dstPrimarySd, dstConnectionV, sg);
    dstPrimarySd.mtl.setBSDFIndex(-1);
    float dstPDF1BSDF = dstMI.evalPdf(dstPrimarySd, dstConnectionV);

    // lightPdf stored in area measure if rcVertexHitExists
    float lightPDF1 = srcReservoir.lightPdf / dstJacobian;

    // If we use a position-dependent NEE sampler, the NEE PDF must be re-evaluated at the dst point.
    if (rcVertexHit.isValid() && isRcVertexEmissive)
        recomputeLightPdfForEmissiveRcVertex(lightPDF1, rcVertexHit, rcVertexSd, dstPrimarySd, dstMI, isCurveHit);
    dstIntegrand *= dstF1; // f_{k-1}

    float inverseGeometryTerm = shifted_dist2 / abs(dot(dstPrimarySd.frame.N, -dstConnectionV));
    float rayFootprint = 1.f / (dstPDF1BSDF * dstGeometryTerm);

    // Invertibility test: `rcFootprintFactor` which stores G(x_{k-1}->x_{k-2})) / p_scatter_{k-1} cannot satisfy inverse footprint threshold 
    // otherwise, x_{k-1} could have been a rcVertex.
    Jacobian *= rcVertexLength > 1 && rcFootprintFactor > 0.f && rcFootprintFactor / dstPDF1BSDF >= primaryRayFootprint * distanceThreshold ? 0.f : 1.f;

    // Single-vertex roughness threshold at x_{k-1}
    // Skipped for emissive-BSDF-sampled rc vertices on materials without a simple roughness proxy.
    bool violateRoughnessCondition = !(isRcVertexEmissive && !isLightPathNEE && !dstMI.supportSimpleRoughnessThreshold()) &&
                                    !dstMI.isSampleRough(dstPrimarySd, componentType1, dstPDF1BSDF, roughnessThreshold);
    // Ray footprint threshold for x_k
    bool violateDistanceCondition = rcVertexHitExists && rayFootprint < primaryRayFootprint * distanceThreshold;

    dstIntegrand *= uint(!rcVertexHitExists || !isRcVertexEmissive || rcVertexSd.frontFacing);

    int initialLightSampleCount = restirpt.computeInitialLightSampleCount(restirpt.initialLightSampleCount, rcVertexLength);
    int NEELightSampleCount = kUseLightTileRIS ? initialLightSampleCount : 1;
    float MISForEmissive = (!kUseMIS || srcReservoir.pathFlags.lightType() == (uint)LightType::Analytic) ? 1.f : evalMIS( (isLightPathNEE ? NEELightSampleCount : 1), (isLightPathNEE ? lightPDF1 : dstPDF1BSDF),
                                     (isLightPathNEE ? 1 : NEELightSampleCount), (isLightPathNEE ? dstPDF1BSDF : lightPDF1) );
    dstIntegrand *= isRcVertexEmissive ? MISForEmissive : 1.f; // MIS weight to sample emissive x_k

    const bool needSecondPart = rcVertexHitExists && !isRcVertexEmissive;
    const IMaterialInstance rcMI = gScene.materials.getMaterialInstance( rcVertexSd, ExplicitLodTextureSampler(params.lodBias), getMaterialInstanceHints(rcVertexHit, false));
    const uint componentType2 = srcReservoir.pathFlags.decodeBSDFComponent(false);

    // does not matter for non-BSDF indexing material
    rcVertexSd.mtl.setBSDFIndex(isRcVertexNEE ? -1 : componentType2);
    float3 dstF2 = rcMI.eval(rcVertexSd, rcVertexWi, sg);
    rcVertexSd.mtl.setBSDFIndex(-1);
    float dstPDF2BSDF = rcMI.evalPdf(rcVertexSd, rcVertexWi);

    initialLightSampleCount = restirpt.computeInitialLightSampleCount(restirpt.initialLightSampleCount, rcVertexLength+1);
    NEELightSampleCount = kUseLightTileRIS ? initialLightSampleCount : 1;

    float MISForLightVertex = evalMIS( (isRcVertexNEE ? NEELightSampleCount : 1), (isRcVertexNEE ? srcReservoir.lightPdf : dstPDF2BSDF),
        (isRcVertexNEE ? 1 : NEELightSampleCount), (isRcVertexNEE ? dstPDF2BSDF : srcReservoir.lightPdf) );

    // Inverse Ray footprint threshold for x_k
    // For emissive rc vertices reached by BSDF sampling (if it's NEE sampled it's a force reconnection) we use a diffuse-pseudo-PDF as a proxy. 
    // When the test fails the Jacobian is zeroed -- the shift would not be invertible from the destination's side.
    Jacobian *= ReSTIRPT::ShiftMapping(restirpt.kShiftMapping) == ReSTIRPT::ShiftMapping::Hybrid &&
                !(isRcVertexEmissive && isLightPathNEE) && ((violateRoughnessCondition || violateDistanceCondition) || !(isRcVertexEmissive && !rcVertexHitExists) &&
                     (inverseGeometryTerm / (isRcVertexEmissive && !isLightPathNEE ? restirpt.getDiffusePseudoPdf() : dstPDF2BSDF) < primaryRayFootprint * distanceThreshold)) ? 0.f : 1.f;

    // MIS weight to sample emissive x_{k+1}
    dstIntegrand *= (kUseMIS && isRcVertexConnectedtoLight && srcReservoir.pathFlags.lightType() != (uint)LightType::Analytic) ? MISForLightVertex : 1.f;
    dstIntegrand *= needSecondPart ? dstF2 : 1.f; // f_k

    // Mixed-PSS Jacobian: G(y_{k-1} -> x_k) / G(x_{k-1} -> x_k) for the shift.
    Jacobian *= dstJacobian / srcReservoir.reconnectionJacobian;

    float3 shadowDestination = rcVertexHitExists ? rcVertexSd.posW : rcVertexWi;
    bool isVisible = evalSegmentVisibility(isPrevFrame, dstPrimarySd.posW, shadowDestination, !rcVertexHitExists, restirpt.settings.additionalShadowRayOffset);
    bool isInvalidJacobianOrIntegrand = isJacobianInvalid(Jacobian) || isIntegrandInvalid(dstIntegrand);
    bool invalidShift = !isVisible || isInvalidJacobianOrIntegrand;

    Jacobian = invalidShift ? 0.f : Jacobian;
    dstIntegrand = invalidShift ? 0.f : dstIntegrand;

    srcReservoir.reconnectionJacobian = dstJacobian;

    return dstIntegrand;
}

/// Evaluates the visibility term to the reconnection vertex from the previous vertex.
bool evalSegmentVisibility(bool isPrevFrame,const float3 sourceVertex, const float3 targetVertex, bool isDirection, bool additionalOffset)
{
    float3 dir = isDirection ? targetVertex : targetVertex - sourceVertex;
    float shadowRayLength = length(dir);
    float tMax = isDirection ? kRayTMax : kShadowRayTMaxRatio * shadowRayLength; // alternatively, compute tMax based on offseted rcVertex position
    Ray ray = Ray(sourceVertex, dir / shadowRayLength, (additionalOffset ? kShadowOriginAdditionalOffsetRatio : 0.f) * shadowRayLength, tMax);

    SceneRayQuery<kUseAlphaTest> sceneRayQuery;
    return sceneRayQuery.traceVisibilityRay(ray, RAY_FLAG_NONE, 0xff, isPrevFrame);
}
