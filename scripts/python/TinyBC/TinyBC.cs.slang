/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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

import BCTypes;

static bool kUseAdam = USE_ADAM;
static int kNumOptimizationSteps = NUM_OPTIMIZATION_STEPS;

uniform float lr;
uniform float adamBeta1;
uniform float adamBeta2;
uniform int2  textureDim;

// Uncompressed input texture
Texture2D<float4> gInputTex;

// Decoded BC texture
RWTexture2D<float4> gDecodedTex;

[BackwardDifferentiable]
void l2Loss(BCTile tile, no_diff float4 target[16], out float loss[16])
{
    [ForceUnroll]
    for (int i = 0; i < 16; i++)
    {
        let delta = target[i] - tile.decode(i);
        loss[i] = dot(delta, delta);
    }
}

[numthreads(4, 8, 1)]
void encoder(int3 dispatchThreadId : SV_DispatchThreadID)
{
    let texel = 4 * dispatchThreadId.xy;
    if (any(texel >= textureDim)) return;

    // Adam moments
    BCTile m1, m2;

    // Load target values and initialize with mid-point interpolation
    // and minmax endpoints computed over the tile
    float4 target[16];
    BCTile tile = BCTile(0.5, float4(1), float4(0));
    for (int i = 0; i < 16; i++)
    {
        target[i] = gInputTex[texel + int2(i % 4, i / 4)];
        tile.minEndPoints = min(tile.minEndPoints, target[i]);
        tile.maxEndPoints = max(tile.maxEndPoints, target[i]);
    }

    // To compress the texture we perform a number of gradient descent steps
    // to joint-optimize interpolation weights and endpoints for each tile.
    // Eventually the interpolation weights are quantized and frozen,
    // while the endpoints are fine tuned for a few more steps.
    bool frozenWeight = false;
    let endPointsFineTuninigThres = 0.85f;
    for (int s = 0; s < kNumOptimizationSteps; s++)
    {
        // Forward pass - compute error
        float loss[16];
        l2Loss(tile, target, loss);

        // Backward pass - compute gradients
        var dp_tile = diffPair(tile);
        __bwd_diff(l2Loss)(dp_tile, target, {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1});

        // Use Adam optimizer to update gradients
        BCTile grad = BCTile(dp_tile.d);
        if (kUseAdam)
        {
            m1 = adamBeta1 * m1 + (1 - adamBeta1) * grad;
            m2 = adamBeta2 * m2 + (1 - adamBeta2) * (grad * grad);
            let unbiased_m1 = rcp(1.f - pow(adamBeta1, 1 + s)) * m1;
            let unbiased_m2 = rcp(1.f - pow(adamBeta2, 1 + s)) * m2;
            grad = unbiased_m1 * (1E-10f + unbiased_m2).rsqrt();
        }

        // If weights are frozen zero their gradient
        if (frozenWeight)
            grad.weights = BCWeights(0);

        // Update BC tile via gradient descent & clamp its values to a valid range
        tile = tile - lr * grad;
        tile.clamp();

        // Time to freeze the BC weights?
        if (frozenWeight == false && float(s) / kNumOptimizationSteps >= endPointsFineTuninigThres)
        {
            // Quantize & freeze weights
            // We will continue to fine tune the endpoints for a few more steps
            frozenWeight = true;
            tile.weights.quantize();
        }
    }

    // Quantize & write out decoded tile
    tile.quantize();
    for (int i = 0; i < 16; i++)
        gDecodedTex[texel + int2(i % 4, i / 4)] = tile.decodeExact(i);;
}
