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
import Scene.Lights.EnvMapData;
import Utils.Math.MathHelpers;

/** Struct for accessing an environment map.
*/
struct EnvMap
{
    Texture2D       envMap;         ///< Environment map texture.
    SamplerState    envSampler;     ///< Environment map sampler.

    EnvMapData      data;           ///< Environment map data.

    /** Returns the dimensions of the env map texture.
    */
    uint2 getDimensions()
    {
        uint2 dim;
        envMap.GetDimensions(dim.x, dim.y);
        return dim;
    }

    /** Evaluates the radiance at a given texel.
    */
    float3 evalTexel(uint2 coord)
    {
        return envMap[coord].rgb * getIntensity();
    }

    /** Evaluates the radiance at a given uv coordinate.
    */
    float3 eval(float2 uv, float lod = 0.f)
    {
        return envMap.SampleLevel(envSampler, uv, lod).rgb * getIntensity();
    }

    /** Evaluates the radiance coming from world space direction 'dir'.
    */
    float3 eval(float3 dir, float lod = 0.f)
    {
        return eval(worldToUv(dir), lod);
    }

    /** Transform direction in local space to uv coordinates.
    */
    float2 localToUv(float3 dir)
    {
        return world_to_latlong_map(dir);
    }

    /** Transform uv coordinates to direction in local space.
    */
    float3 uvToLocal(float2 uv)
    {
        return latlong_map_to_world(uv);
    }

    /** Transform direction in world space to uv coordinates.
    */
    float2 worldToUv(float3 dir)
    {
        return localToUv(toLocal(dir));
    }

    /** Transform uv coordinates to direction in world space.
    */
    float3 uvToWorld(float2 uv)
    {
        return toWorld(uvToLocal(uv));
    }

    /** Transform direction from local to world space.
    */
    float3 toWorld(float3 dir)
    {
        // TODO: For identity transform we might want to skip this statically.
        return mul((float3x3)data.transform, dir);
    }

    /** Transform direction from world to local space.
    */
    float3 toLocal(float3 dir)
    {
        // TODO: For identity transform we might want to skip this statically.
        return mul((float3x3)data.invTransform, dir);
    }

    /** Get the intensity scaling factor (including tint).
    */
    float3 getIntensity()
    {
        return data.intensity * data.tint;
    }
};
