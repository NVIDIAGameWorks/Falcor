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
import DiffRendering.SceneGradients;
import DiffRendering.DiffDebug;
import Scene.Material.MaterialParamLayout;
import Utils.Math.HashUtils;

// Avoid compilation warnings if not running in a differentiable mode.
#ifndef DIFF_MODE
#define DIFF_MODE 0
#endif

// Simple hash function.
uint hashFunction(uint index, uint hashSize)
{
    return jenkinsHash(index) % hashSize;
}

/**
 * The wrapper function has custom derivatives for setting up the data flow of scene gradients during backpropagation.
 * It indicates where to accumulate the gradients by providing `offset` in the gradient buffer.
 */
struct GradientIOWrapper
{
    GradientType gradType;
    uint baseOffset;
    uint hashIndex;

    __init(GradientType _gradType, uint _baseOffset, uint _hashIndex)
    {
        this.gradType = _gradType;
        this.baseOffset = _baseOffset;
        this.hashIndex = _hashIndex;
    }

    [ForwardDerivative(__fwd_d_getFloat)]
    [BackwardDerivative(__bwd_d_getFloat)]
    float getFloat(float val, uint offset) { return val; }

    [TreatAsDifferentiable]
    DifferentialPair<float> __fwd_d_getFloat(DifferentialPair<float> dpVal, uint offset) { return diffPair(dpVal.p, 0.f); }

    [TreatAsDifferentiable]
    void __bwd_d_getFloat(inout DifferentialPair<float> dpVal, uint offset, float dOut)
    {
        offset += baseOffset;
        gSceneGradients.atomicAddGrad(gradType, offset, hashIndex, dOut);
    }

    [ForwardDerivative(__fwd_d_getFloats)]
    [BackwardDerivative(__bwd_d_getFloats)]
    __generic<let N : int> vector<float, N> getFloat(vector<float, N> val, uint offset) { return val; }

    [TreatAsDifferentiable]
    __generic<let N : int> DifferentialPair<vector<float, N>> __fwd_d_getFloats(DifferentialPair<vector<float, N>> dpVal, uint offset)
    {
        vector<float, N> d;
        [ForceUnroll]
        for (int i = 0; i < N; i++)
            d[i] = 0.f;

#if DIFF_MODE == 2 // ForwardDiffDebug
        if (gDiffDebug.varType == DiffVariableType::Material && baseOffset == gDiffDebug.id.x * 20 && offset == gDiffDebug.offset)
        {
            [ForceUnroll]
            for (int i = 0; i < N; i++)
                d[i] = gDiffDebug.grad[i];
        }
#endif

        return diffPair(dpVal.p, d);
    }

    [TreatAsDifferentiable]
    __generic<let N : int> void __bwd_d_getFloats(inout DifferentialPair<vector<float, N>> dpVal, uint offset, vector<float, N> dOut)
    {
        offset += baseOffset;
        [ForceUnroll]
        for (uint i = 0; i < N; i++)
            gSceneGradients.atomicAddGrad(gradType, offset + i, hashIndex, dOut[i]);
    }

    [Differentiable]
    float getFloat(float val, const MaterialParamLayoutEntry entry) { return getFloat(val, entry.offset); }

    [Differentiable]
    __generic<let N : int> vector<float, N> getFloat(vector<float, N> val, const MaterialParamLayoutEntry entry)
    {
        return getFloat(val, entry.offset);
    }
};
