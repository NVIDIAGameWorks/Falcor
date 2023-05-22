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
#pragma once
#include "Vector.h"

namespace Falcor
{
/**
 * Ray type.
 * This should match the layout of DXR RayDesc.
 */
struct Ray
{
    float3 origin;
    float tMin;
    float3 dir;
    float tMax;

    Ray() = default;
    explicit Ray(float3 origin, float3 dir, float tMin = 0.f, float tMax = std::numeric_limits<float>::max())
        : origin(origin), tMin(tMin), dir(dir), tMax(tMax)
    {}
};

// These are to ensure that the struct Ray match DXR RayDesc.
static_assert(offsetof(Ray, origin) == 0);
static_assert(offsetof(Ray, tMin) == sizeof(float3));
static_assert(offsetof(Ray, dir) == offsetof(Ray, tMin) + sizeof(float));
static_assert(offsetof(Ray, tMax) == offsetof(Ray, dir) + sizeof(float3));
static_assert(sizeof(Ray) == 32);
} // namespace Falcor
