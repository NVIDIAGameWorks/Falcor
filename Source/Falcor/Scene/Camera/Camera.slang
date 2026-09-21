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
import Scene.Camera.CameraData;
__exported import Utils.Math.Ray;
import Utils.Math.MathHelpers;

struct Camera
{
    CameraData data;
    CameraData prevData;

    float3 getPosition() { return data.posW; }
    float4x4 getViewProj() { return data.viewProjMat; }

    /** Computes a camera ray for a given pixel assuming a pinhole camera model.
        The camera jitter is taken into account to compute the sample position on the image plane.
        \param[in] pixel Pixel coordinates with origin in top-left.
        \param[in] frameDim Image plane dimensions in pixels.
        \param[in] applyJitter true if jitter should be applied else false.
        \return Returns the camera ray.
    */
    Ray computeRayPinhole(uint2 pixel, uint2 frameDim, bool applyJitter = true)
    {
        return computeRayPinhole(data, pixel, frameDim, applyJitter);
    }

    Ray computeRayPinholePrevFrame(uint2 pixel, uint2 frameDim, bool applyJitter = true)
    {
        return computeRayPinhole(prevData, pixel, frameDim, applyJitter);
    }

    Ray computeRayPinhole(const CameraData camData, uint2 pixel, uint2 frameDim, bool applyJitter = true)
    {
        Ray ray;

        // Compute the normalized ray direction assuming a pinhole camera.
        ray.origin = camData.posW;
        ray.dir = normalize(computeNonNormalizedRayDirPinhole(pixel, frameDim, applyJitter));

        float invCos = 1.f / dot(normalize(camData.cameraW), ray.dir);
        ray.tMin = camData.nearZ * invCos;
        ray.tMax = camData.farZ * invCos;

        return ray;
    }

    /** Computes the primary ray's direction, non-normalized assuming pinhole camera model.
        The camera jitter is taken into account to compute the sample position on the image plane.
        \param[in] pixel Pixel coordinates with origin in top-left.
        \param[in] frameDim Image plane dimensions in pixels.
        \param[in] applyJitter True if jitter should be applied else false.
        \return Returns the non-normalized ray direction
    */
    float3 computeNonNormalizedRayDirPinhole(uint2 pixel, uint2 frameDim, bool applyJitter = true)
    {
        // Compute sample position in screen space in [0,1] with origin at the top-left corner.
        // The camera jitter offsets the sample by +-0.5 pixels from the pixel center.
        float2 p = (pixel + float2(0.5f, 0.5f)) / frameDim;
        if (applyJitter) p += float2(-data.jitterX, data.jitterY);
        float2 ndc = float2(2, -2) * p + float2(-1, 1);

        // Compute the non-normalized ray direction assuming a pinhole camera.
        return ndc.x * data.cameraU + ndc.y * data.cameraV + data.cameraW;
    }


    /** Computes a camera ray for a given pixel assuming a thin-lens camera model.
        The camera jitter is taken into account to compute the sample position on the image plane.
        \param[in] pixel Pixel coordinates with origin in top-left.
        \param[in] frameDim Image plane dimensions in pixels.
        \param[in] u Uniform 2D sample.
        \return Returns the camera ray.
    */
    Ray computeRayThinlens(uint2 pixel, uint2 frameDim, float2 u, bool applyJitter = true)
    {
        float2 u0 = float2(0.5f, 0.5f);
        if (applyJitter) u0 += frameDim * float2(-data.jitterX, data.jitterY);
        return computeRayGeneral(data, pixel, frameDim, u0, u);
    }

    Ray computeRayThinlensPrevFrame(uint2 pixel, uint2 frameDim, float2 u, bool applyJitter = true)
    {
        float2 u0 = float2(0.5f, 0.5f);
        if (applyJitter) u0 += frameDim * float2(-prevData.jitterX, prevData.jitterY);
        return computeRayGeneral(prevData, pixel, frameDim, u0, u);
    }

    // u0 is subpixel uv and u1 is lens uv
    Ray computeRayThinlensSubPixelRandom(float2 pixel, uint2 frameDim, float2 u0, float2 u1)
    {
        return computeRayGeneral(data, pixel, frameDim, u0, u1);
    }

    Ray computeRayThinlensSubPixelRandomPrevFrame(float2 pixel, uint2 frameDim, float2 u0, float2 u1)
    {
        return computeRayGeneral(prevData, pixel, frameDim, u0, u1);
    }

    Ray computeRayGeneral(const CameraData camData, float2 pixel, uint2 frameDim, float2 u0, float2 u1)
    {
        Ray ray;

        // Sample position in screen space in [0,1] with origin at the top-left corner.
        // The camera jitter offsets the sample by +-0.5 pixels from the pixel center.
        float2 p = float2(pixel + u0) / frameDim;
        float2 ndc = float2(2, -2) * p + float2(-1, 1);

        // Compute the normalized ray direction assuming a thin-lens camera.
        ray.origin = camData.posW;
        ray.dir = ndc.x * camData.cameraU + ndc.y * camData.cameraV + camData.cameraW;
        float2 apertureSample = sample_disk(u1); // Sample lies in the unit disk [-1,1]^2
        float3 rayTarget = ray.origin + ray.dir;
        ray.origin += camData.apertureRadius * (apertureSample.x * normalize(camData.cameraU) + apertureSample.y * normalize(camData.cameraV));
        ray.dir = normalize(rayTarget - ray.origin);

        float invCos = 1.f / dot(normalize(camData.cameraW), ray.dir);
        ray.tMin = camData.nearZ * invCos;
        ray.tMax = camData.farZ * invCos;

        return ray;
    }
};
