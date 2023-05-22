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

#include "ScalarTypes.h"
#include "VectorTypes.h"

#include <type_traits>

namespace Falcor
{
namespace math
{

/**
 * Quaternion type.
 *
 * A quaternion is an expression of the form:
 *
 * q = w + xi + yj + zk
 *
 * where w, x, y, and z are real numbers and i, j, and k are the imaginary units.
 *
 * The quaternion is normalized if:
 * w^2 + x^2 + y^2 + z^2 = 1
 *
 * Quaternions are stored as (x, y, z, w) to make them better for interop with the GPU.
 */
template<typename T>
struct quat
{
    using value_type = T;
    static_assert(std::disjunction_v<std::is_same<T, float>, std::is_same<T, double>>, "Invalid quaternion type");

    T x, y, z, w;

    quat() : x{T(0)}, y{T(0)}, z{T(0)}, w{T(1)} {}

    explicit quat(const vector<T, 3>& xyz, const T& w) : x{xyz.x}, y{xyz.y}, z{xyz.z}, w{w} {}
    explicit quat(const T& x, const T& y, const T& z, const T& w) : x{x}, y{y}, z{z}, w{w} {}

    /// Identity quaternion.
    [[nodiscard]] static quat identity() { return quat(T(0), T(0), T(0), T(1)); }

    // Accesses
    value_type& operator[](size_t i) { return (&x)[i]; }
    const value_type& operator[](size_t i) const { return (&x)[i]; }
};

using quatf = quat<float>;

} // namespace math

using quatf = math::quatf;

} // namespace Falcor
