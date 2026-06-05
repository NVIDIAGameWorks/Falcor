#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Scene.RaytracingInline;
import Utils.Sampling.SampleGenerator;
import Utils.Color.ColorHelpers;
import Utils.Math.MathHelpers;
import Rendering.RTXDI.RTXDI;
import StructsAndHelpers;

cbuffer CB
{
    uint gFrameCount;                       // Current Iteration for the Sample Generator
    uint2 gFrameDim;                        // Frame Dimensions
    uint gConfidenceLimit;                  // Confidence Limit

    uint gSpatialSamples;                   // Number of spatial samples
    uint gDisocclusionBoostSpatialSamples;  // How many extra spatial samples if temporal resampling fails
    float gSpatialRadius;                   // (Pixel)radius for spatial resampling
    float gNormalThreshold;                 // Normal Threshold

    float2 gPhotonRadius;                   // Collection radius Global / Caustic
    uint2 _pad;
}

Texture2D<float2> gMVec;
StructuredBuffer<ReservoirCaustic> gCausticReservoirPrev;

RWStructuredBuffer<ReservoirCaustic> gCausticReservoir;

static const uint kMIHints = (uint)MaterialInstanceHints::AdjustShadingNormal;

inline float getPhotonRadius(ReservoirCaustic r) {
    return r.pathLength == 0 ? gPhotonRadius.y : gPhotonRadius.x;
}

/*  Generalized Balance Heuristic with two samples
    \param[in] Reservoir which sample is used for MIS
    \param[in] Shading data of the resampling surface from the other sample
    \param[in] BSDF Properties of the resampling surface from the other sample
    \param[in] Cofidence weight of the other sample
    \return MIS weight
*/
float generalizedBalanceHeuristic(ReservoirCaustic r, ShadingData otherSD, BSDFProperties otherProps, uint otherC) {
    float p0 = r.targetFunction * r.c;
    float p1 = 0;
    if (length(otherSD.posW - r.photonPosW) < getPhotonRadius(r))   //Check if sample could have been collected from the surface of the other reservoir
        p1 = evalApproximateBRDFPhoton(otherSD, otherProps, r.photonDir) * luminance(r.photonRadiance) * otherC;

    float sum = (p0 + p1);
    if (sum > 0)
        return p0 / sum;
    else
        return 0;
}

/*  Resampling with 2 reservoirs. Finalizes the reservoir afterwards.
    \param[in/out] Current reservoir for the pixel. Is the finalized reservoir at the end of the function
    \param[in] Shading data for current reservoir
    \param[in] BSDF Properties for current reservoir
    \param[in] Other (spatial or temporal) reservoir
    \param[in] Shading data for other reservoir
    \param[in] BSDF Properties for other reservoir
    \param[in/out] Random Number Generator
    \return True if resampling was successful
*/
bool combineReservoirsCaustic(inout ReservoirCaustic current, ShadingData currentSD, BSDFProperties currentProps,
                                ReservoirCaustic other, ShadingData otherSD, BSDFProperties otherProps, inout SampleGenerator sg)
{
    //Check if surfaces are similar and resampling can be performed
    bool normalThreshold = dot(otherSD.faceN, currentSD.faceN) < gNormalThreshold;
    bool pathThreshold = (other.pathLength != current.pathLength);
    bool surfacesPosThreshold = length(currentSD.posW - otherSD.posW) >= (2 * getPhotonRadius(current));
    if (normalThreshold || pathThreshold || surfacesPosThreshold)
        return false;

    //Needed reservoir variables
    float w_0, w_1 = 0;
    float wSum = 0;
   
    //Get ReSTIR weight of current reservoir
    if (current.isValid()) {
        float mis = generalizedBalanceHeuristic(current, otherSD, otherProps, other.c);
        w_0 = mis * current.targetFunction * current.wSum;
        wSum += w_0;
    }

    //Calculate ReSTIR weight for other reservoir sample
    float targetFunctionOther = 0.0;
    bool otherReservoirValid = other.isValid();
    if (otherReservoirValid) {
        // Check if sample is part of the domain
        if (length(currentSD.posW - other.photonPosW) < getPhotonRadius(other)) {
            float mis = generalizedBalanceHeuristic(other, currentSD, currentProps, current.c);
            targetFunctionOther = evalApproximateBRDFPhoton(currentSD, currentProps, other.photonDir) * luminance(other.photonRadiance);
            w_1 = mis * targetFunctionOther * other.wSum;
        }
        wSum += w_1;
    }

    //Check if other sample is selected and replace in reservoir if it was selected
    bool selectOtherSample = (sampleNext1D(sg) * wSum) < w_1 && otherReservoirValid;
    if (selectOtherSample) {
        current.photonPosW = other.photonPosW;
        current.photonDir = other.photonDir;
        current.photonRadiance = other.photonRadiance;
        current.targetFunction = targetFunctionOther;
    }

    //Increase confidence weights
    current.c += other.c;
    current.c = min(current.c, gConfidenceLimit);

    //Finalize reservoir (wSum -> W)
    if (current.targetFunction > 0)
        current.wSum = wSum / current.targetFunction;
    else
        current.wSum = 0;

    //If something went wrong, set reservoir to 0
    if (isnan(current.wSum) || isinf(current.wSum))
        current.wSum = 0;

    return true;
}

/*  Loads the necessary data for the other reservoir and resamples
    \param[in/out] Current reservoir for the pixel.
    \param[in] Shading data for current reservoir
    \param[in] BSDF Properties for current reservoir
    \param[in/out] Random Number Generator
    \param[in] Pixel location where the other reservoir is located
    \return True if resampling was successful
*/
bool resampleReservoir(inout ReservoirCaustic current, ShadingData currentSD, BSDFProperties currentBsdfProperties, inout SampleGenerator sg , int2 resamplePixel) {

    //Check if reservoir is inside the screen. 
    if (any(resamplePixel < 0) || any(resamplePixel >= gFrameDim))
        return false;

    //Fetch the (spatio)temporal reservoir
    ReservoirCaustic temporal = gCausticReservoirPrev[index2Dto1D(resamplePixel, gFrameDim.x)];
    HitInfo temporalHit = HitInfo(temporal.sampleSurface);
    if (!temporalHit.isValid()) {
        return false;
    }

    //Get shading datas
    let lod = ExplicitLodTextureSampler(0.f);
    ShadingData temporalSD = loadShadingData(temporalHit, temporal.sampleSurfaceView, lod);
    let temporalMi = gScene.materials.getMaterialInstance(temporalSD, lod, kMIHints);
    let temporalBsdfProperties = temporalMi.getProperties(temporalSD);

    //Combine both reservoirs
    bool valid = combineReservoirsCaustic(current, currentSD, currentBsdfProperties, temporal, temporalSD, temporalBsdfProperties, sg);

    return valid;
}

[numthreads(16, 16, 1)]
void main(uint2 pixel: SV_DispatchThreadID)
{
    let lod = ExplicitLodTextureSampler(0.f);
    SampleGenerator sg = SampleGenerator(pixel, gFrameCount * 5 + 3);

    ReservoirCaustic current = gCausticReservoir[index2Dto1D(pixel, gFrameDim.x)];
    HitInfo currentHit = HitInfo(current.sampleSurface);

    if (!currentHit.isValid())
        return;

    //Get Shading Data
    ShadingData currentSD = loadShadingData(currentHit, current.sampleSurfaceView, lod);
    let currentMi = gScene.materials.getMaterialInstance(currentSD, lod, kMIHints);
    let currentBsdfProperties = currentMi.getProperties(currentSD);

    // Precalculate the targetFunction
    current.targetFunction = evalApproximateBRDFPhoton(currentSD, currentBsdfProperties, current.photonDir) * luminance(current.photonRadiance);

    //
    // Temporal Resampling
    //

    // Reproject to the previous pixel using the motion vectors
    float2 motionOffset = gMVec[pixel] * float2(gFrameDim);
    float2 prevPixelF = motionOffset + float2(pixel) + float2(0.5);
    float2 rndOffset = sampleNext2D(sg) - 0.5;
    prevPixelF += rndOffset;
    int2 prevPixel = int2(prevPixelF);

    bool resamplingSuccesfull = resampleReservoir(current, currentSD, currentBsdfProperties, sg, prevPixel);

    //
    // Spatiotemporal Resampling
    //
    uint spatialSamples = resamplingSuccesfull ? gSpatialSamples : gSpatialSamples + gDisocclusionBoostSpatialSamples;
    for (uint i = 0; i < spatialSamples; i++) {
        // Sample in a disc radius around the reprojected sample
        float2 spatialOffset = gSpatialRadius * sample_disk_concentric(sampleNext2D(sg));
        float2 spatialPixelF = prevPixelF + spatialOffset;
        int2 spatialPixel = int2(spatialPixelF);

        resampleReservoir(current, currentSD, currentBsdfProperties, sg, spatialPixel);
    }

    //Store final reservoir
    gCausticReservoir[index2Dto1D(pixel, gFrameDim.x)] = current;
}
