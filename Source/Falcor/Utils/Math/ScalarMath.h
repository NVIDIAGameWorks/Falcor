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
#include <fstd/bit.h> // TODO C++20: Replace with <bit>
#include <cmath>

namespace Falcor
{
namespace math
{

// ----------------------------------------------------------------------------
// Boolean reductions
// ----------------------------------------------------------------------------

template<typename T>
[[nodiscard]] constexpr bool any(T x)
{
    return x != T(0);
}

template<typename T>
[[nodiscard]] constexpr bool all(T x)
{
    return x != T(0);
}

// ----------------------------------------------------------------------------
// Basic functions
// ----------------------------------------------------------------------------

template<typename T, std::enable_if_t<is_arithmetic_v<T>, bool> = false>
[[nodiscard]] constexpr T min(T x, T y) noexcept
{
    return x < y ? x : y;
}

template<typename T, std::enable_if_t<is_arithmetic_v<T>, bool> = false>
[[nodiscard]] constexpr T max(T x, T y) noexcept
{
    return x > y ? x : y;
}

template<typename T, std::enable_if_t<is_arithmetic_v<T>, bool> = false>
[[nodiscard]] constexpr T clamp(T x, T min_, T max_) noexcept
{
    return max(min_, min(max_, x));
}

template<typename T, std::enable_if_t<is_signed_v<T>, bool> = false>
[[nodiscard]] constexpr T abs(T x) noexcept
{
    return std::abs(x);
}

template<typename T, std::enable_if_t<is_signed_v<T>, bool> = false>
[[nodiscard]] constexpr T sign(T x) noexcept
{
    return x < T(0) ? T(-1) : (x > T(0) ? T(1) : T(0));
}

// clang-format off

// ----------------------------------------------------------------------------
// Floating point checks
// ----------------------------------------------------------------------------

template<typename T> [[nodiscard]] bool isfinite(T x) noexcept;
template<> [[nodiscard]] inline bool isfinite(float x) noexcept { return std::isfinite(x); }
template<> [[nodiscard]] inline bool isfinite(double x) noexcept { return std::isfinite(x); }
template<> [[nodiscard]] inline bool isfinite(float16_t x) noexcept { return x.isFinite(); }

template<typename T> [[nodiscard]] bool isinf(T x) noexcept;
template<> [[nodiscard]] inline bool isinf(float x) noexcept { return std::isinf(x); }
template<> [[nodiscard]] inline bool isinf(double x) noexcept { return std::isinf(x); }
template<> [[nodiscard]] inline bool isinf(float16_t x) noexcept { return x.isInf(); }

template<typename T> [[nodiscard]] bool isnan(T x) noexcept;
template<> [[nodiscard]] inline bool isnan(float x) noexcept { return std::isnan(x); }
template<> [[nodiscard]] inline bool isnan(double x) noexcept { return std::isnan(x); }
template<> [[nodiscard]] inline bool isnan(float16_t x) noexcept { return x.isNan(); }

// ----------------------------------------------------------------------------
// Rounding
// ----------------------------------------------------------------------------

template<typename T> [[nodiscard]] T floor(T x) noexcept;
template<> [[nodiscard]] inline float floor(float x) noexcept { return std::floor(x); }
template<> [[nodiscard]] inline double floor(double x) noexcept { return std::floor(x); }

template<typename T> [[nodiscard]] T ceil(T x) noexcept;
template<> [[nodiscard]] inline float ceil(float x) noexcept { return std::ceil(x); }
template<> [[nodiscard]] inline double ceil(double x) noexcept { return std::ceil(x); }

template<typename T> [[nodiscard]] T trunc(T x) noexcept;
template<> [[nodiscard]] inline float trunc(float x) noexcept { return std::trunc(x); }
template<> [[nodiscard]] inline double trunc(double x) noexcept { return std::trunc(x); }

template<typename T> [[nodiscard]] T round(T x) noexcept;
template<> [[nodiscard]] inline float round(float x) noexcept { return std::round(x); }
template<> [[nodiscard]] inline double round(double x) noexcept { return std::round(x); }

// ----------------------------------------------------------------------------
// Exponential
// ----------------------------------------------------------------------------

template<typename T> [[nodiscard]] T pow(T x, T y) noexcept;
template<> [[nodiscard]] inline float pow(float x, float y) noexcept { return std::pow(x, y); }
template<> [[nodiscard]] inline double pow(double x, double y) noexcept { return std::pow(x, y); }

template<typename T> [[nodiscard]] T sqrt(T x) noexcept;
template<> [[nodiscard]] inline float sqrt(float x) noexcept { return std::sqrt(x); }
template<> [[nodiscard]] inline double sqrt(double x) noexcept { return std::sqrt(x); }

template<typename T> [[nodiscard]] T rsqrt(T x) noexcept;
template<> [[nodiscard]] inline float rsqrt(float x) noexcept { return 1.f / std::sqrt(x); }
template<> [[nodiscard]] inline double rsqrt(double x) noexcept { return 1.0 / std::sqrt(x); }

template<typename T> [[nodiscard]] T exp(T x) noexcept;
template<> [[nodiscard]] inline float exp(float x) noexcept { return std::exp(x); }
template<> [[nodiscard]] inline double exp(double x) noexcept { return std::exp(x); }
template<> [[nodiscard]] inline float16_t exp(float16_t x) noexcept { return float16_t(std::exp(float(x))); }

template<typename T> [[nodiscard]] T exp2(T x) noexcept;
template<> [[nodiscard]] inline float exp2(float x) noexcept { return std::exp2(x); }
template<> [[nodiscard]] inline double exp2(double x) noexcept { return std::exp2(x); }
template<> [[nodiscard]] inline float16_t exp2(float16_t x) noexcept { return float16_t(std::exp2(float(x))); }

template<typename T> [[nodiscard]] T log(T x) noexcept;
template<> [[nodiscard]] inline float log(float x) noexcept { return std::log(x); }
template<> [[nodiscard]] inline double log(double x) noexcept { return std::log(x); }
template<> [[nodiscard]] inline float16_t log(float16_t x) noexcept { return float16_t(std::log(float(x))); }

template<typename T> [[nodiscard]] T log2(T x) noexcept;
template<> [[nodiscard]] inline float log2(float x) noexcept { return std::log2(x); }
template<> [[nodiscard]] inline double log2(double x) noexcept { return std::log2(x); }

template<typename T> [[nodiscard]] T log10(T x) noexcept;
template<> [[nodiscard]] inline float log10(float x) noexcept { return std::log10(x); }
template<> [[nodiscard]] inline double log10(double x) noexcept { return std::log10(x); }

// ----------------------------------------------------------------------------
// Trigonometry
// ----------------------------------------------------------------------------

template<typename T> [[nodiscard]] T radians(T x) noexcept;
template<> [[nodiscard]] inline float radians(float x) noexcept { return x * 0.01745329251994329576923690768489f; }
template<> [[nodiscard]] inline double radians(double x) noexcept { return x * 0.01745329251994329576923690768489; }

template<typename T> [[nodiscard]] T degrees(T x) noexcept;
template<> [[nodiscard]] inline float degrees(float x) noexcept { return x * 57.295779513082320876798154814105f; }
template<> [[nodiscard]] inline double degrees(double x) noexcept { return x * 57.295779513082320876798154814105; }

template<typename T> [[nodiscard]] T sin(T x) noexcept;
template<> [[nodiscard]] inline float sin(float x) noexcept { return std::sin(x); }
template<> [[nodiscard]] inline double sin(double x) noexcept { return std::sin(x); }

template<typename T> [[nodiscard]] T cos(T x) noexcept;
template<> [[nodiscard]] inline float cos(float x) noexcept { return std::cos(x); }
template<> [[nodiscard]] inline double cos(double x) noexcept { return std::cos(x); }

template<typename T> [[nodiscard]] T tan(T x) noexcept;
template<> [[nodiscard]] inline float tan(float x) noexcept { return std::tan(x); }
template<> [[nodiscard]] inline double tan(double x) noexcept { return std::tan(x); }

template<typename T> [[nodiscard]] T asin(T x) noexcept;
template<> [[nodiscard]] inline float asin(float x) noexcept { return std::asin(x); }
template<> [[nodiscard]] inline double asin(double x) noexcept { return std::asin(x); }

template<typename T> [[nodiscard]] T acos(T x) noexcept;
template<> [[nodiscard]] inline float acos(float x) noexcept { return std::acos(x); }
template<> [[nodiscard]] inline double acos(double x) noexcept { return std::acos(x); }

template<typename T> [[nodiscard]] T atan(T x) noexcept;
template<> [[nodiscard]] inline float atan(float x) noexcept { return std::atan(x); }
template<> [[nodiscard]] inline double atan(double x) noexcept { return std::atan(x); }

template<typename T> [[nodiscard]] T atan2(T y, T x) noexcept;
template<> [[nodiscard]] inline float atan2(float y, float x) noexcept { return std::atan2(y, x); }
template<> [[nodiscard]] inline double atan2(double y, double x) noexcept { return std::atan2(y, x); }

template<typename T> [[nodiscard]] T sinh(T x) noexcept;
template<> [[nodiscard]] inline float sinh(float x) noexcept { return std::sinh(x); }
template<> [[nodiscard]] inline double sinh(double x) noexcept { return std::sinh(x); }

template<typename T> [[nodiscard]] T cosh(T x) noexcept;
template<> [[nodiscard]] inline float cosh(float x) noexcept { return std::cosh(x); }
template<> [[nodiscard]] inline double cosh(double x) noexcept { return std::cosh(x); }

template<typename T> [[nodiscard]] T tanh(T x) noexcept;
template<> [[nodiscard]] inline float tanh(float x) noexcept { return std::tanh(x); }
template<> [[nodiscard]] inline double tanh(double x) noexcept { return std::tanh(x); }

// ----------------------------------------------------------------------------
// Misc
// ----------------------------------------------------------------------------

template<typename T> [[nodiscard]] T fmod(T x, T y) noexcept;
template<> [[nodiscard]] inline float fmod(float x, float y) noexcept { return std::fmod(x, y); }
template<> [[nodiscard]] inline double fmod(double x, double y) noexcept { return std::fmod(x, y); }

template<typename T> [[nodiscard]] T frac(T x) noexcept;
template<> [[nodiscard]] inline float frac(float x) noexcept { return x - floor(x); }
template<> [[nodiscard]] inline double frac(double x) noexcept { return x - floor(x); }

template<typename T> [[nodiscard]] T lerp(T x, T y, T s) noexcept;
template<> [[nodiscard]] inline float lerp(float x, float y, float s) noexcept { return (1.f - s) * x + s * y; }
template<> [[nodiscard]] inline double lerp(double x, double y, double s) noexcept { return (1.0 - s) * x + s * y; }

template<typename T> [[nodiscard]] T rcp(T x) noexcept;
template<> [[nodiscard]] inline float rcp(float x) noexcept { return 1.f / x; }
template<> [[nodiscard]] inline double rcp(double x) noexcept { return 1.0 / x; }

template<typename T> [[nodiscard]] T saturate(T x) noexcept;
template<> [[nodiscard]] inline float saturate(float x) noexcept { return max(0.f, min(1.f, x)); }
template<> [[nodiscard]] inline double saturate(double x) noexcept { return max(0.0, min(1.0, x)); }

template<typename T> [[nodiscard]] T step(T x, T y) noexcept;
template<> [[nodiscard]] inline float step(float x, float y) noexcept { return x >= y ? 1.f : 0.f; }
template<> [[nodiscard]] inline double step(double x, double y) noexcept { return x >= y ? 1.0 : 0.0; }

// clang-format on

template<typename T, std::enable_if_t<is_floating_point_v<T>, bool> = false>
[[nodiscard]] T smoothstep(T min_, T max_, T x) noexcept
{
    x = saturate((x - min_) / (max_ - min_));
    return x * x * (T(3) - T(2) * x);
}

// ----------------------------------------------------------------------------
// Conversion
// ----------------------------------------------------------------------------

// clang-format off
[[nodiscard]] inline float f16tof32(uint v) noexcept { return float16ToFloat32(v & 0xffff); }
[[nodiscard]] inline uint f32tof16(float v) noexcept { return float32ToFloat16(v); }

[[nodiscard]] inline float asfloat(uint32_t i) noexcept { return fstd::bit_cast<float>(i); }
[[nodiscard]] inline float asfloat(int32_t i) noexcept { return fstd::bit_cast<float>(i); }
[[nodiscard]] inline float16_t asfloat16(uint16_t i) noexcept { return fstd::bit_cast<float16_t>(i); }

[[nodiscard]] inline uint32_t asuint(float f) noexcept { return fstd::bit_cast<uint32_t>(f); }
[[nodiscard]] inline int32_t asint(float f) noexcept { return fstd::bit_cast<int32_t>(f); }
[[nodiscard]] inline uint16_t asuint16(float16_t f) noexcept { return fstd::bit_cast<uint16_t>(f); }
// clang-format on

} // namespace math
} // namespace Falcor
