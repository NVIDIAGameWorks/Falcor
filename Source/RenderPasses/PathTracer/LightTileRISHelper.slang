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
import PathState;
import Scene.Shading;
import LightTileResampling.Lights;
import LightTileResampling.StreamingRIS;
import PathTracerUtils;
import Params;
import Utils.Geometry.GeometryHelpers;
import Utils.Color.ColorHelpers;

struct LightTileRISData
{
    float3 selectedBrdf = {};
    float selectedPdf = {};
    float pdfWarpFactor = 1.f;
}

struct LightTileRIS
{
    static const uint kLightTileCount = LIGHT_TILE_COUNT;
    static const uint kLightTileSize = LIGHT_TILE_SIZE;
    static const uint kScreenTileSize = SCREEN_TILE_SIZE;
    static const uint kInitialLightSampleCount = INITIAL_LIGHT_SAMPLE_COUNT;

    StructuredBuffer<PackedLoadedLightSample> lightTileData;
    Lights lights;

    int computeInitialLightSampleCount(int vertexIndex)
    {
        return max(1, kInitialLightSampleCount / (vertexIndex * vertexIndex));
    }

    bool GenerateSample( inout PathState path, ShadingData sd, const IMaterialInstance mi, inout LightSample ls, bool isCurveHit, bool applyScreenSpaceReSTIR, uint neeSeed, PathTracerParams params, inout LightTileRISData risData)
    {
        int initialLightSampleCount = computeInitialLightSampleCount(path.getVertexIndex());

        uint2 screenTile = path.getPixel() / kScreenTileSize;

        SampleGenerator tileSg = SampleGenerator(screenTile, params.seed);
        uint tileIndex = min(uint(sampleNext1D(tileSg) * kLightTileCount), kLightTileCount - 1);
        uint lightTileOffset = tileIndex * kLightTileSize;

        SampleGenerator initialLightSampleSg = SampleGenerator(neeSeed);
        RisState risState = RisState::createEmpty();

        uint stride = (kLightTileSize + initialLightSampleCount - 1) / initialLightSampleCount;
        uint offset = min(uint(sampleNext1D(initialLightSampleSg) * stride), stride - 1);

        for (uint i = 0; i < initialLightSampleCount; ++i)
        {
            uint index = lightTileOffset + (offset + i * stride) % kLightTileSize;
            const LoadedLightSample loadedLightSample = LoadedLightSample::createFromPacked(lightTileData[index]);
            const EvaluatedLightSample evaluatedLightSample = loadedLightSample.eval(sd.posW);

            float3 brdf = mi.eval(sd, evaluatedLightSample.dir, initialLightSampleSg);

            float sourcePdf = evaluatedLightSample.geomFactor == 0.f ? 0.f : evaluatedLightSample.pdf / evaluatedLightSample.geomFactor;

            float misWeight = 1.f;
            float brdfPdf = 0.f;
            if (kUseMIS && !applyScreenSpaceReSTIR)
            {
                brdfPdf = mi.evalPdf(sd, evaluatedLightSample.dir, kUseBSDFSampling);
                if (loadedLightSample.lightSample.getKind() == LightKind::Analytic) brdfPdf = 0.f;
                misWeight = sourcePdf + brdfPdf == 0.f ? 0.f : evalMIS(initialLightSampleCount, sourcePdf, 1, brdfPdf);
            }

            float targetPdf = sourcePdf == 0.f ? 0.f : luminance(brdf * evaluatedLightSample.emission) * misWeight;
            sourcePdf = sourcePdf == 0.f ? 1.f : sourcePdf;

            bool selected = streamingResampleStep(risState, loadedLightSample.lightSample, targetPdf, sourcePdf, initialLightSampleSg);
            if (selected)
            {
                risData.selectedBrdf = brdf;
                risData.selectedPdf = brdfPdf;
            }
        }

        risState.weight = risState.weight > 0.f ? (risState.weightSum / risState.M) / risState.weight : 0.f;

        ls.origin = isCurveHit ? path.origin : sd.posW;
        ls.origin = isCurveHit ? ls.origin : sd.computeRayOrigin();

        EvaluatedLightSample evaluatedLightSample = lights.evalLightSample(risState.lightSample, ls.origin);

        ls.dir = evaluatedLightSample.dir;
        ls.distance = evaluatedLightSample.distance - Lights::kRayEpsilon;

        ls.lightType = ((uint)risState.lightSample.getKind()) >> 30;
        ls.pdf = evaluatedLightSample.geomFactor == 0.f ? 0.f : evaluatedLightSample.pdf / evaluatedLightSample.geomFactor;
#if USE_RESTIRPT        
        ls.geometryTerm = evaluatedLightSample.geomFactor;
#endif
        risData.pdfWarpFactor = 1.f / (risState.weight * ls.pdf);

        ls.Li = lights.evalEmission(risState.lightSample, -ls.dir) / ls.pdf;

#if USE_RESTIRPT        
        if (risState.lightSample.getKind() == LightKind::Emissive)
        {
            ls.barycentrics = risState.lightSample.getEmissiveBarycentrics().yz;
            ls.triangleId = risState.lightSample.getEmissiveIndex();
        }
#endif        

        bool validSample = risState.weight > 0.f && ls.pdf > 0.f && risState.lightSample.isValid();
        return validSample;
    }
}
