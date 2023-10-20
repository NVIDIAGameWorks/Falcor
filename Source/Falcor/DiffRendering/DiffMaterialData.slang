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
import Scene.Material.ShadingUtils;

/**
 * Stores differentiable parameters used for BSDF evaluation.
 */
struct DiffMaterialData : IDifferentiable
{
    static const uint kMaterialParamCount = 20;

    ShadingFrame sf;

    // Material parameters.
    float data[kMaterialParamCount];

    [Differentiable]
    __init()
    {
        sf = ShadingFrame.createIdentity();
        [ForceUnroll]
        for (uint i = 0; i < kMaterialParamCount; i++)
            data[i] = 0.f;
    }

    [Differentiable]
    float read(inout uint offset) { return data[offset++]; }

    [Differentiable]
    void read<let N : int>(out vector<float, N> value, inout uint offset)
    {
        [ForceUnroll]
        for (uint i = 0; i < N; i++)
            value[i] = read(offset);
    }

    [Differentiable]
    vector<float, N> read<let N : int>(inout uint offset)
    {
        vector<float, N> value;
        this.read(value, offset);
        return value;
    }

    [mutating]
    [Differentiable]
    void write(float value, inout uint offset) { data[offset++] = value; }

    [mutating]
    [Differentiable]
    void write<let N : int>(vector<float, N> value, inout uint offset)
    {
        [ForceUnroll]
        for (uint i = 0; i < N; i++)
            this.write(value[i], offset);
    }
};
