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
#pragma once
#include "Vector.h"
#include "Matrix.h"
#include "Core/Assert.h"
#include <algorithm>
#include <cmath>

namespace Falcor
{
    /*!
    *  \addtogroup Falcor
    *  @{
    */

    /** Creates a quaternion representing rotation between 2 vectors
        \param[in] from The source vector
        \param[in] to The destination vector
    */
    inline glm::quat createQuaternionFromVectors(const float3& from, const float3& to)
    {
        glm::quat quat;
        float3 nFrom = glm::normalize(from);
        float3 nTo = glm::normalize(to);

        float dot = glm::dot(nFrom, nTo);
        dot = std::clamp(dot, -1.0f, 1.0f);
        if(dot != 1)
        {
            float angle = std::acos(dot);

            float3 cross = glm::cross(nFrom, nTo);
            float3 axis = glm::normalize(cross);

            quat = glm::angleAxis(angle, axis);
        }

        return quat;
    }

    /** Calculates a world-space ray direction from a screen-space mouse pos.
        \param[in] mousePos Normalized coordinates in the range [0, 1] with (0, 0) being the top-left of the screen. Same coordinate space as MouseEvent.
        \param[in] viewMat View matrix from the camera.
        \param[in] projMat Projection matrix from the camera.
        \return World space ray direction coming from the camera position in the direction of the mouse position
    */
    inline float3 mousePosToWorldRay(const float2& mousePos, const rmcv::mat4& viewMat, const rmcv::mat4& projMat)
    {
        // Convert from [0, 1] to [-1, 1] range
        const float x = mousePos.x * 2.0f - 1.0f;

#ifdef FALCOR_FLIP_Y
        // NDC Y is top-to-bottom
        const float y = mousePos.y * 2.0f - 1.0f;
#else
        // NDC Y is bottom-to-top
        const float y = (1.0f - mousePos.y) * 2.0f - 1.0f;
#endif

        // NDC/Clip
        float4 ray(x, y, -1.0f, 1.0f);

        // View
        ray = rmcv::inverse(projMat) * ray;
        ray.z = -1.0f;
        ray.w = 0.0f;

        // World
        return glm::normalize(rmcv::inverse(viewMat) * ray);
    }

    /** Creates a rotation matrix from individual basis vectors.
        \param[in] forward Forward vector.
        \param[in] up Up vector.
        \return 3x3 rotation matrix.
    */
    inline rmcv::mat3 createMatrixFromBasis(const float3& forward, const float3& up)
    {
        float3 f = glm::normalize(forward);
        float3 s = glm::normalize(glm::cross(up, forward));
        float3 u = glm::cross(f, s);

        return rmcv::make_mat3_fromCols(s, u, f);
    }

    /** Creates a rotation matrix from look-at coordinates.
        \param[in] position Object's position.
        \param[in] target Object's look-at target.
        \param[in] up Object's up vector.
        \return 3x3 rotation matrix.
    */
    inline rmcv::mat3 createMatrixFromLookAt(const float3& position, const float3& target, const float3& up)
    {
        return createMatrixFromBasis(target - position, up);
    }

    /** Projects a 2D coordinate onto a unit sphere
        \param xy The 2D coordinate. if x and y are in the [0,1) range, then a z value can be calculate. Otherwise, xy is normalized and z is zero.
    */
    inline float3 project2DCrdToUnitSphere(float2 xy)
    {
        float xyLengthSquared = glm::dot(xy, xy);

        float z = 0;
        if(xyLengthSquared < 1)
        {
            z = std::sqrt(1 - xyLengthSquared);
        }
        else
        {
            xy = glm::normalize(xy);
        }
        return float3(xy.x, xy.y, z);
    }

    /** Calculates vertical FOV in radians from camera parameters.
        \param[in] focalLength Focal length in mm.
        \param[in] frameHeight Height of film/sensor in mm.
    */
    inline float focalLengthToFovY(float focalLength, float frameHeight)
    {
        return 2.0f * std::atan(0.5f * frameHeight / focalLength);
    }

    /** Calculates camera focal length from vertical FOV.
        \param[in] fovY Vertical FOV in radians.
        \param[in] frameHeight Height of film/sensor in mm.
    */
    inline float fovYToFocalLength(float fovY, float frameHeight)
    {
        return frameHeight / (2.0f * std::tan(0.5f * fovY));
    }

    /** Calculates camera aperture radius in scene units.
        \param[in] fNumber Aperture f-number.
        \param[in] focalLength Focal length in mm.
        \param[in] sceneUnit Scene unit in meters.
    */
    inline float apertureFNumberToRadius(float fNumber, float focalLength, float sceneUnit)
    {
        FALCOR_ASSERT(fNumber > 0.0f && focalLength > 0.f && sceneUnit > 0.f);
        float radius = 0.5f * focalLength / fNumber; // in mm
        return radius * 0.001f / sceneUnit;
    }

    /** Calculates camera aperture f-number from camera parameters.
        \param[in] apertureRadius Aperture radius in scene units.
        \param[in] focalLength Focal length in mm.
        \param[in] sceneUnit Scene unit in meters.
    */
    inline float apertureRadiusToFNumber(float apertureRadius, float focalLength, float sceneUnit)
    {
        FALCOR_ASSERT(focalLength > 0.f && sceneUnit > 0.f);
        float radius = apertureRadius * sceneUnit * 1000.f; // in mm
        return 0.5f * focalLength / radius;
    }

    // Base 2 Van der Corput radical inverse
    inline float radicalInverse(uint32_t i)
    {
        i = (i & 0x55555555) << 1 | (i & 0xAAAAAAAA) >> 1;
        i = (i & 0x33333333) << 2 | (i & 0xCCCCCCCC) >> 2;
        i = (i & 0x0F0F0F0F) << 4 | (i & 0xF0F0F0F0) >> 4;
        i = (i & 0x00FF00FF) << 8 | (i & 0xFF00FF00) >> 8;
        i = (i << 16) | (i >> 16);
        return float(i) * 2.3283064365386963e-10f;
    }

    inline float3 hammersleyUniform(uint32_t i, uint32_t n)
    {
        float2 uv((float)i / (float)n, radicalInverse(i));

        // Map to radius 1 hemisphere
        float phi = uv.y * 2.0f * (float)M_PI;
        float t = 1.0f - uv.x;
        float s = std::sqrt(1.0f - t * t);
        return float3(s * std::cos(phi), s * std::sin(phi), t);
    }

    inline float3 hammersleyCosine(uint32_t i, uint32_t n)
    {
        float2 uv((float)i / (float)n, radicalInverse(i));

        // Map to radius 1 hemisphere
        float phi = uv.y * 2.0f * (float)M_PI;
        float t = std::sqrt(1.0f - uv.x);
        float s = std::sqrt(1.0f - t * t);
        return float3(s * std::cos(phi), s * std::sin(phi), t);
    }

    inline float smoothstep(const float t)
    {
        const float s = std::clamp(t, 0.0f, 1.0f);
        return s * s * (3.0f - 2.0f * s);
    }

    inline float smoothstep(const float start, const float end, const float t)
    {
        float s = (t - start) / (end - start);
        return smoothstep(s);
    }

/*! @} */
}
