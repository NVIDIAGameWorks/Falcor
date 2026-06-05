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

    float gJacobianDistanceThreshold;       // Jacobian distance threshold
    bool gUsePathThreshold;                 // Enables and disables path threshold
    uint2 _pad;
}

Texture2D<float2> gMVec;
StructuredBuffer<ReservoirFG> gFinalGatherReservoirPrev;

RWStructuredBuffer<ReservoirFG> gFinalGatherReservoir;

static const uint kMIHints = (uint)MaterialInstanceHints::AdjustShadingNormal;

/*  Generalized Balance Heuristic with two samples
    \param[in] Reservoir which sample is used for MIS
    \param[in] Shading data of the resampling surface from the other sample
    \param[in] BSDF Properties of the resampling surface from the other sample
    \param[in] Cofidence weight of the other sample
    \param[in] If true, skips the visibility test for the other sample
    \return MIS weight
*/
float generalizedBalanceHeuristic(ReservoirFG r, ShadingData otherSD, BSDFProperties otherProps, uint otherC, bool assumeVisible = false) {
    float p0 = r.targetFunction * r.c; // No visibility or Jacobian Determinant needed for this sample
    float p1 = evaluateFinalGatherSampleApproximateUnbiased(r, otherSD, otherProps, assumeVisible) * otherC;  // targetFunction * vis * Jacobian Determinant

    float sum = (p0 + p1);
    if(sum > 0)
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
bool combineReservoirsFG(inout ReservoirFG current, ShadingData currentSD, BSDFProperties currentProps,
                                ReservoirFG other, ShadingData otherSD, BSDFProperties otherProps, inout SampleGenerator sg)
{
    //Check if surfaces are similar and resampling can be performed
    float3 toSurfaceOther = currentSD.posW - other.sampleData.posW;
    float distToSurfaceOther = length(toSurfaceOther);
    toSurfaceOther /= distToSurfaceOther; //Normalize
    bool jacobianThreshold = distToSurfaceOther <= gJacobianDistanceThreshold;
    bool normalThreshold = dot(otherSD.faceN, currentSD.faceN) < gNormalThreshold;
    bool pathThreshold = gUsePathThreshold ? (other.pathLength != current.pathLength) : false;
    if (normalThreshold || pathThreshold || jacobianThreshold)
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
    bool otherReservoirValid = other.isValid();
    float targetFunctionOther = 0.0;
    if (otherReservoirValid) {
        //Check if sample can be shifted
        if (dispatchShadowRay(currentSD, -toSurfaceOther, distToSurfaceOther)) {
            // We can assume p1 is visible, as this was checked by the visibility ray in the line above
            float mis = generalizedBalanceHeuristic(other, currentSD, currentProps, current.c, true);
            targetFunctionOther = evalApproximateBRDF(currentSD, currentProps, -toSurfaceOther) * luminance(other.sampleData.radianceOut);
            float jacobianDeterminant = dot(other.sampleData.normal, toSurfaceOther) / (distToSurfaceOther * distToSurfaceOther);
            jacobianDeterminant /= other.jacobianDeterminant;
            w_1 = mis * targetFunctionOther * jacobianDeterminant * other.wSum;
        }
        wSum += w_1;
    }

    //Check if other sample is selected and replace in reservoir if it was selected
    bool selectOtherSample = (sampleNext1D(sg) * wSum) <= w_1 && otherReservoirValid;
    if (selectOtherSample) {
        current.sampleData.posW = other.sampleData.posW;
        current.sampleData.normal = other.sampleData.normal;
        current.sampleData.radianceOut = other.sampleData.radianceOut;
        current.targetFunction = targetFunctionOther;
        current.jacobianDeterminant = dot(current.sampleData.normal, toSurfaceOther) / (distToSurfaceOther * distToSurfaceOther);
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
bool resampleReservoir(inout ReservoirFG current, ShadingData currentSD, BSDFProperties currentBsdfProperties, inout SampleGenerator sg , int2 resamplePixel) {

    //Check if reservoir is inside the screen. 
    if (any(resamplePixel < 0) || any(resamplePixel >= gFrameDim))
        return false;

    //Fetch the (spatio)temporal reservoir
    ReservoirFG temporal = gFinalGatherReservoirPrev[index2Dto1D(resamplePixel, gFrameDim.x)];
    HitInfo temporalHit = HitInfo(temporal.sampleSurface);
    if (!temporalHit.isValid()) {
        return false;
    }

    //Get shading datas
    let lod = ExplicitLodTextureSampler(0.f);
    ShadingData temporalSD = loadShadingData(temporalHit, -temporal.sampleData.sampleView, lod);
    let temporalMi = gScene.materials.getMaterialInstance(temporalSD, lod, kMIHints);
    let temporalBsdfProperties = temporalMi.getProperties(temporalSD);

    //Combine both reservoirs
    bool valid = combineReservoirsFG(current, currentSD, currentBsdfProperties, temporal, temporalSD, temporalBsdfProperties, sg);

    return valid;
}

[numthreads(16, 16, 1)]
void main(uint2 pixel: SV_DispatchThreadID)
{
    let lod = ExplicitLodTextureSampler(0.f);
    SampleGenerator sg = SampleGenerator(pixel, gFrameCount * 5 + 2);

    ReservoirFG current = gFinalGatherReservoir[index2Dto1D(pixel, gFrameDim.x)];
    HitInfo currentHit = HitInfo(current.sampleSurface);

    //Current camera hit misses the scene
    if (!currentHit.isValid())
        return;

    //Get Shading Data
    ShadingData currentSD = loadShadingData(currentHit, -current.sampleData.sampleView, lod);
    let currentMi = gScene.materials.getMaterialInstance(currentSD, lod, kMIHints);
    let currentBsdfProperties = currentMi.getProperties(currentSD);

    // Precalculate the targetFunction
    float3 toLight = normalize(current.sampleData.posW - currentSD.posW);
    current.targetFunction = evalApproximateBRDF(currentSD, currentBsdfProperties, toLight) * luminance(current.sampleData.radianceOut);

    //
    // Temporal Resampling
    //

    // Reproject to the previous pixel using the motion vectors
    float2 motionOffset = gMVec[pixel] * float2(gFrameDim);
    float2 prevPixelF = motionOffset + float2(pixel) + float2(0.5);
    float2 rndOffset = sampleNext2D(sg) - 0.5;
    prevPixelF += rndOffset;
    int2 prevPixel = int2(prevPixelF);

    bool resamplingSuccessfull = resampleReservoir(current, currentSD, currentBsdfProperties, sg, prevPixel);

    //
    // Spatiotemporal Resampling
    //
    uint spatialSamples = resamplingSuccessfull ? gSpatialSamples : gSpatialSamples + gDisocclusionBoostSpatialSamples;
    for (uint i = 0; i < spatialSamples; i++) {
        // Sample in a disc radius around the reprojected sample
        float2 spatialOffset = gSpatialRadius * sample_disk_concentric(sampleNext2D(sg));
        float2 spatialPixelF = prevPixelF + spatialOffset;
        int2 spatialPixel = int2(spatialPixelF);

        resampleReservoir(current, currentSD, currentBsdfProperties, sg, spatialPixel);
    }

    //Store final reservoir
    gFinalGatherReservoir[index2Dto1D(pixel, gFrameDim.x)] = current;
}
