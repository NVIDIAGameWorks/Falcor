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
import Shift;
import Scene.HitInfo;
import PathReservoir;
import PathTracer;
import Scene.Scene;
import Utils.Debug.PixelDebug;
import Rendering.Utils.PixelStats;
import Utils.Sampling.TinyUniformSampleGenerator;
import Utils.Math.Ray;
import Utils.Math.MathHelpers;
import LoadShadingDataDynamic;
import ResamplingCommon;

struct Pass
{
    ReSTIRPathTracingParams params;
    ReSTIRPT restirpt; 

    Texture2D<PackedHitInfo> vbuffer;
    Texture2D<PackedHitInfo> temporalVBuffer;
    Texture2D<float2> originalMotionVectors;
    Texture2D<float2> deltaReflectionMotionVectors;
    Texture2D<float2> deltaTransmissionMotionVectors;
    RWTexture2D<float2> motionVectors;
    StructuredBuffer<PathReservoir> prevReservoirs;
    RWStructuredBuffer<PathReservoir> mergedPrevReservoirs;
    void DoPass(const int2 pixel)
    {
        var sg = TinyUniformSampleGenerator(pixel, restirpt.sgCount() * params.seed + restirpt.specMVecSgOffset());

        float2 motionVector = originalMotionVectors[pixel];

        HitInfo centralPrimaryHit;
        ShadingData centralPrimarySd = getPixelShadingData(params, vbuffer, pixel, centralPrimaryHit);

        if (!centralPrimaryHit.isValid())
        {
            motionVectors[pixel] = motionVector;
            return;
        }

        // Temporally Reliable Motion Vectors for Real-time Ray Tracing [Zeng et al. 2021]
        if (restirpt.enableDisocclusionMotionVectors)
        {
            int2 prevPixel = pixel + motionVector * params.frameDim + 0.5;

            HitInfo prevPrimaryHit;
            ShadingData prevPrimarySd = getPixelTemporalShadingData(params, temporalVBuffer, prevPixel, prevPrimaryHit);

            if (!isValidScreenRegion(params, prevPixel)) // zero motion vector
            {
                motionVector = -motionVector;
            }
            else if (prevPrimaryHit.isValid() && isReprojectedToOccluder(centralPrimarySd, prevPrimarySd))
            {
                VertexData curFrameVertexData = getVertexData(prevPrimaryHit, 0.f, 0.f);
                float3 occluderCurrPosW = curFrameVertexData.posW;
                float4 currPosH = mul(gScene.camera.data.viewProjMatNoJitter, float4(occluderCurrPosW, 1.f));
                float2 currCrd = currPosH.xy / currPosH.w;
            #ifdef FALCOR_FLIP_Y
                currCrd *= float2(0.5, 0.5);
            #else
                currCrd *= float2(0.5, -0.5);
            #endif
                currCrd += 0.5f;
                float2 occluderPixelPos = currCrd * params.frameDim;
                motionVector = float2(-(occluderPixelPos - (prevPixel+0.5))) / params.frameDim;
            }
        }

        const IMaterialInstance centralMi = gScene.materials.getMaterialInstance( centralPrimarySd, ExplicitLodTextureSampler(0.f), getMaterialInstanceHints(centralPrimaryHit, true));

        // if rcVertexLength == 1, we know from reservoir
        // otherwise, we replay the initRandomSeed to see what we sample
        float2 motionVectorSelectionProbabilities = centralMi.getReflectionMotionVectorProbabilities(centralPrimarySd);
        float u = sampleNext1D(sg);

        // has both diffuse and refraction -> disable diffuse
        if (motionVectorSelectionProbabilities.x > 0.f && motionVectorSelectionProbabilities.x + motionVectorSelectionProbabilities.y < 0.99f)
        {
            motionVectorSelectionProbabilities.y = motionVectorSelectionProbabilities.y / (1.f - motionVectorSelectionProbabilities.x);
            motionVectorSelectionProbabilities.x = 0.f;
        }


        bool isSmooth = restirpt.enableSpecularMotionVectors && centralMi.getProperties(centralPrimarySd).roughness <= restirpt.specularMotionVectorRoughnessThreshold; 

        if (!isSmooth)
            motionVectorSelectionProbabilities = float2(1.f, 0.f);

        if (u < motionVectorSelectionProbabilities.x)
        {} // do nothing
        else if (u < motionVectorSelectionProbabilities.x + motionVectorSelectionProbabilities.y)
        {
            motionVector = deltaReflectionMotionVectors[pixel];
        }
        else
        {
            motionVector = deltaTransmissionMotionVectors[pixel];
        }
        motionVectors[pixel] = motionVector;

    }

    void execute(const uint2 _pixel)
    {
        const uint2 pixel = _pixel;

        if (any(pixel >= params.frameDim)) return;

        printSetPixel(pixel);
        logSetPixel(pixel);

        DoPass(pixel);
    }
};

cbuffer CB
{
    Pass gPass;
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    gPass.execute(dispatchThreadId.xy);
}
