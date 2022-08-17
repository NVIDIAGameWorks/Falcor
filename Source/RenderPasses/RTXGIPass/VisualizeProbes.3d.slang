/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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
import Scene.Raster;
import Scene.Camera.Camera;
import Rendering.RTXGI.RTXGIVolume;
import VisualizerDefines;

cbuffer PerFrameCB
{
    Camera gCamera;
    uint gVisualizerProbeMode; // TODO: Replace by enum VisualizerProbeMode when Slang supports enums in structs
    bool gShowProbeStates;
    bool gHighlightProbe;
    int gProbeIndex;
    float gProbeRadius;
    uint gFrameCount;
};

static const uint kFlashFreq = 0x8;

struct VertexOut
{
    float3 posW                     : POSW;
    float3 normalW                  : NORMAL;
    nointerpolation uint probeIndex : PROBE_ID;

    float4 posH                     : SV_Position;
};

VertexOut vsMain(VSIn vIn, uint probeIndex : SV_InstanceID)
{
    VertexOut vOut;
    const GeometryInstanceID instanceID = { vIn.instanceID };

    // Get the probe's world position.
    float3 probePos = gRTXGIVolume.getProbeWorldPosition(probeIndex);

    float4x4 worldMat = gScene.getWorldMatrix(instanceID); // Should be identity transform
    float3 posW = mul(worldMat, float4(vIn.pos, 1.f)).xyz;
    posW *= gProbeRadius;
    posW += probePos;

    vOut.posW = posW;
    vOut.posH = mul(gCamera.data.viewProjMat, float4(posW, 1.f));
    vOut.normalW = mul(gScene.getInverseTransposeWorldMatrix(instanceID), vIn.unpack().normal);
    vOut.probeIndex = probeIndex;

    return vOut;
}

float4 psMain(VertexOut vOut) : SV_Target0
{
    const int probeIndex = vOut.probeIndex;
    const float3 dir = normalize(vOut.normalW);
    float3 color = float3(0);

    if (gVisualizerProbeMode == (uint)VisualizerProbeMode::Direction)
    {
        color = saturate(dir);
    }
    else if (gVisualizerProbeMode == (uint)VisualizerProbeMode::Irradiance)
    {
        // The irradiance is expected to be evaluated based on a surface normal direction.
        // For each direction on the probe, we visualize the stored irradiance value.
        color = gRTXGIVolume.evalProbeIrradiance(probeIndex, dir);
    }

    if (gHighlightProbe)
    {
        // Color highlighted probe flashing red/green.
        if (probeIndex == gProbeIndex)
            color = (gFrameCount & kFlashFreq) ? float3(1, 0, 0) : float3(0, 1, 0);
    }

    if (gShowProbeStates)
    {
        const float3 kColorInactive = float3(1, 0, 0);  // Red
        const float3 kColorActive = float3(0, 1, 0);    // Green

        uint state = gRTXGIVolume.getProbeState(probeIndex);
        float3 probePos = gRTXGIVolume.getProbeWorldPosition(probeIndex);

        // Border visualization for probe states.
        if (abs(dot(normalize(probePos - gCamera.data.posW), dir)) < 0.5f)
        {
            if (state == RTXGIVolume::kProbeStateActive) color = kColorActive;
            else if (state == RTXGIVolume::kProbeStateInactive) color = kColorInactive;
            else color = (gFrameCount & kFlashFreq) ? kColorActive : kColorInactive; // Error: flashing colors
        }
    }

    return float4(color, 1);
}
