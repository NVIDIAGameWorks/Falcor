/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
import Utils.Helpers;

#ifndef ENABLE_CURVE_SPHERE_JOINTS
#define ENABLE_CURVE_SPHERE_JOINTS 0
#endif

/** Linear swept sphere intersection function wrapper.
    \param[in] rayOrigin Ray origin position.
    \param[in] rayDir Unit ray direction vector.
    \param[in] sphereA Sphere (3D position + radius) at one end point.
    \param[in] sphereB Sphere at the other end point.
    \param[out] result The closest intersection distance t, and a parameter u for linear interpolation (between 0 and 1).
    \return True if the ray intersects the linear swept sphere segment.
*/
bool intersectLinearSweptSphere(float3 rayOrigin, float3 rayDir, float4 sphereA, float4 sphereB, out float2 result)
{
    bool useSphereJoints;
#if ENABLE_CURVE_SPHERE_JOINTS
    useSphereJoints = true;
#else
    useSphereJoints = false;
#endif

    return intersectLinearSweptSphereHan19(rayOrigin, rayDir, sphereA, sphereB, useSphereJoints, result);
}
