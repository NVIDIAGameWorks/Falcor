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

#include "VectorTypes.h"
#include "ScalarMath.h"

#include <fmt/format.h>

#include <string>

namespace Falcor
{
namespace math
{

// ----------------------------------------------------------------------------
// Boolean reductions
// ----------------------------------------------------------------------------

// clang-format off
[[nodiscard]] constexpr bool any(const bool1 v) noexcept { return v.x; }
[[nodiscard]] constexpr bool any(const bool2 v) noexcept { return v.x || v.y; }
[[nodiscard]] constexpr bool any(const bool3 v) noexcept { return v.x || v.y || v.z; }
[[nodiscard]] constexpr bool any(const bool4 v) noexcept { return v.x || v.y || v.z || v.w; }

[[nodiscard]] constexpr bool all(const bool1 v) noexcept { return v.x; }
[[nodiscard]] constexpr bool all(const bool2 v) noexcept { return v.x && v.y; }
[[nodiscard]] constexpr bool all(const bool3 v) noexcept { return v.x && v.y && v.z; }
[[nodiscard]] constexpr bool all(const bool4 v) noexcept { return v.x && v.y && v.z && v.w; }

[[nodiscard]] constexpr bool none(const bool1 v) noexcept { return !any(v); }
[[nodiscard]] constexpr bool none(const bool2 v) noexcept { return !any(v); }
[[nodiscard]] constexpr bool none(const bool3 v) noexcept { return !any(v); }
[[nodiscard]] constexpr bool none(const bool4 v) noexcept { return !any(v); }
// clang-format on

// ----------------------------------------------------------------------------
// Unary operators
// ----------------------------------------------------------------------------

/// Unary plus operator
template<typename T, int N, std::enable_if_t<is_arithmetic_v<T>, bool> = false>
[[nodiscard]] constexpr auto operator+(const vector<T, N> v) noexcept
{
    return v;
}

#if FALCOR_MSVC
#pragma warning(push)
#pragma warning(disable : 4146) // unary minus operator applied to unsigned type, result still unsigned
#endif
/// Unary minus operator
template<typename T, int N, std::enable_if_t<is_arithmetic_v<T>, bool> = false>
[[nodiscard]] constexpr auto operator-(const vector<T, N> v) noexcept
{
    if constexpr (N == 1)
        return vector<T, N>{-v.x};
    else if constexpr (N == 2)
        return vector<T, N>{-v.x, -v.y};
    else if constexpr (N == 3)
        return vector<T, N>{-v.x, -v.y, -v.z};
    else
        return vector<T, N>{-v.x, -v.y, -v.z, -v.w};
}
#if FALCOR_MSVC
#pragma warning(pop)
#endif

/// Unary not operator
template<typename T, int N>
[[nodiscard]] constexpr auto operator!(const vector<T, N> v) noexcept
{
    if constexpr (N == 1)
        return bool1{!v.x};
    else if constexpr (N == 2)
        return bool2{!v.x, !v.y};
    else if constexpr (N == 3)
        return bool3{!v.x, !v.y, !v.z};
    else
        return bool4{!v.x, !v.y, !v.z, !v.w};
}

/// Unary not operator
template<typename T, int N, std::enable_if_t<is_integral_v<T>, bool> = false>
[[nodiscard]] constexpr auto operator~(const vector<T, N> v) noexcept
{
    if constexpr (N == 1)
        return vector<T, N>{~v.x};
    else if constexpr (N == 2)
        return vector<T, N>{~v.x, ~v.y};
    else if constexpr (N == 3)
        return vector<T, N>{~v.x, ~v.y, ~v.z};
    else
        return vector<T, N>{~v.x, ~v.y, ~v.z, ~v.w};
}

// ----------------------------------------------------------------------------
// Binary operators
// ----------------------------------------------------------------------------

// clang-format off
/* <<<PYMACRO
def print_binary_operator(op, enable_if):
    print(f"""/// Binary {op} operator
template<typename T, int N, std::enable_if_t<{enable_if}, bool> = false>
[[nodiscard]] constexpr vector<T, N> operator{op}(const vector<T, N>& lhs, const vector<T, N>& rhs)
{{
    if constexpr (N == 1)
        return vector<T, N>{{lhs.x {op} rhs.x}};
    else if constexpr (N == 2)
        return vector<T, N>{{lhs.x {op} rhs.x, lhs.y {op} rhs.y}};
    else if constexpr (N == 3)
        return vector<T, N>{{lhs.x {op} rhs.x, lhs.y {op} rhs.y, lhs.z {op} rhs.z}};
    else if constexpr (N == 4)
        return vector<T, N>{{lhs.x {op} rhs.x, lhs.y {op} rhs.y, lhs.z {op} rhs.z, lhs.w {op} rhs.w}};
}}

/// Binary {op} operator
template<typename T, int N, std::enable_if_t<{enable_if}, bool> = false>
[[nodiscard]] constexpr vector<T, N> operator{op}(const vector<T, N>& lhs, T rhs)
{{
    return lhs {op} vector<T, N>(rhs);
}}

/// Binary {op} operator
template<typename T, int N, std::enable_if_t<{enable_if}, bool> = false>
[[nodiscard]] constexpr vector<T, N> operator{op}(T lhs, const vector<T, N>& rhs)
{{
    return vector<T, N>(lhs) {op} rhs;
}}
""")

print_binary_operator("+", "is_arithmetic_v<T>")
print_binary_operator("-", "is_arithmetic_v<T>")
print_binary_operator("*", "is_arithmetic_v<T>")
print_binary_operator("/", "is_arithmetic_v<T>")
print_binary_operator("%", "is_integral_v<T>")
print_binary_operator("<<", "is_integral_v<T>")
print_binary_operator(">>", "is_integral_v<T>")
print_binary_operator("|", "is_integral_v<T>")
print_binary_operator("&", "is_integral_v<T>")
print_binary_operator("^", "is_integral_v<T>")
>>> */
/// Binary + operator
template<typename T, int N, std::enable_if_t<is_arithmetic_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> operator+(const vector<T, N>& lhs, const vector<T, N>& rhs)
{
    if constexpr (N == 1)
        return vector<T, N>{lhs.x + rhs.x};
    else if constexpr (N == 2)
        return vector<T, N>{lhs.x + rhs.x, lhs.y + rhs.y};
    else if constexpr (N == 3)
        return vector<T, N>{lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
    else if constexpr (N == 4)
        return vector<T, N>{lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w};
}

/// Binary + operator
template<typename T, int N, std::enable_if_t<is_arithmetic_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> operator+(const vector<T, N>& lhs, T rhs)
{
    return lhs + vector<T, N>(rhs);
}

/// Binary + operator
template<typename T, int N, std::enable_if_t<is_arithmetic_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> operator+(T lhs, const vector<T, N>& rhs)
{
    return vector<T, N>(lhs) + rhs;
}

/// Binary - operator
template<typename T, int N, std::enable_if_t<is_arithmetic_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> operator-(const vector<T, N>& lhs, const vector<T, N>& rhs)
{
    if constexpr (N == 1)
        return vector<T, N>{lhs.x - rhs.x};
    else if constexpr (N == 2)
        return vector<T, N>{lhs.x - rhs.x, lhs.y - rhs.y};
    else if constexpr (N == 3)
        return vector<T, N>{lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
    else if constexpr (N == 4)
        return vector<T, N>{lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w};
}

/// Binary - operator
template<typename T, int N, std::enable_if_t<is_arithmetic_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> operator-(const vector<T, N>& lhs, T rhs)
{
    return lhs - vector<T, N>(rhs);
}

/// Binary - operator
template<typename T, int N, std::enable_if_t<is_arithmetic_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> operator-(T lhs, const vector<T, N>& rhs)
{
    return vector<T, N>(lhs) - rhs;
}

/// Binary * operator
template<typename T, int N, std::enable_if_t<is_arithmetic_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> operator*(const vector<T, N>& lhs, const vector<T, N>& rhs)
{
    if constexpr (N == 1)
        return vector<T, N>{lhs.x * rhs.x};
    else if constexpr (N == 2)
        return vector<T, N>{lhs.x * rhs.x, lhs.y * rhs.y};
    else if constexpr (N == 3)
        return vector<T, N>{lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z};
    else if constexpr (N == 4)
        return vector<T, N>{lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w};
}

/// Binary * operator
template<typename T, int N, std::enable_if_t<is_arithmetic_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> operator*(const vector<T, N>& lhs, T rhs)
{
    return lhs * vector<T, N>(rhs);
}

/// Binary * operator
template<typename T, int N, std::enable_if_t<is_arithmetic_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> operator*(T lhs, const vector<T, N>& rhs)
{
    return vector<T, N>(lhs) * rhs;
}

/// Binary / operator
template<typename T, int N, std::enable_if_t<is_arithmetic_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> operator/(const vector<T, N>& lhs, const vector<T, N>& rhs)
{
    if constexpr (N == 1)
        return vector<T, N>{lhs.x / rhs.x};
    else if constexpr (N == 2)
        return vector<T, N>{lhs.x / rhs.x, lhs.y / rhs.y};
    else if constexpr (N == 3)
        return vector<T, N>{lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z};
    else if constexpr (N == 4)
        return vector<T, N>{lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z, lhs.w / rhs.w};
}

/// Binary / operator
template<typename T, int N, std::enable_if_t<is_arithmetic_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> operator/(const vector<T, N>& lhs, T rhs)
{
    return lhs / vector<T, N>(rhs);
}

/// Binary / operator
template<typename T, int N, std::enable_if_t<is_arithmetic_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> operator/(T lhs, const vector<T, N>& rhs)
{
    return vector<T, N>(lhs) / rhs;
}

/// Binary % operator
template<typename T, int N, std::enable_if_t<is_integral_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> operator%(const vector<T, N>& lhs, const vector<T, N>& rhs)
{
    if constexpr (N == 1)
        return vector<T, N>{lhs.x % rhs.x};
    else if constexpr (N == 2)
        return vector<T, N>{lhs.x % rhs.x, lhs.y % rhs.y};
    else if constexpr (N == 3)
        return vector<T, N>{lhs.x % rhs.x, lhs.y % rhs.y, lhs.z % rhs.z};
    else if constexpr (N == 4)
        return vector<T, N>{lhs.x % rhs.x, lhs.y % rhs.y, lhs.z % rhs.z, lhs.w % rhs.w};
}

/// Binary % operator
template<typename T, int N, std::enable_if_t<is_integral_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> operator%(const vector<T, N>& lhs, T rhs)
{
    return lhs % vector<T, N>(rhs);
}

/// Binary % operator
template<typename T, int N, std::enable_if_t<is_integral_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> operator%(T lhs, const vector<T, N>& rhs)
{
    return vector<T, N>(lhs) % rhs;
}

/// Binary << operator
template<typename T, int N, std::enable_if_t<is_integral_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> operator<<(const vector<T, N>& lhs, const vector<T, N>& rhs)
{
    if constexpr (N == 1)
        return vector<T, N>{lhs.x << rhs.x};
    else if constexpr (N == 2)
        return vector<T, N>{lhs.x << rhs.x, lhs.y << rhs.y};
    else if constexpr (N == 3)
        return vector<T, N>{lhs.x << rhs.x, lhs.y << rhs.y, lhs.z << rhs.z};
    else if constexpr (N == 4)
        return vector<T, N>{lhs.x << rhs.x, lhs.y << rhs.y, lhs.z << rhs.z, lhs.w << rhs.w};
}

/// Binary << operator
template<typename T, int N, std::enable_if_t<is_integral_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> operator<<(const vector<T, N>& lhs, T rhs)
{
    return lhs << vector<T, N>(rhs);
}

/// Binary << operator
template<typename T, int N, std::enable_if_t<is_integral_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> operator<<(T lhs, const vector<T, N>& rhs)
{
    return vector<T, N>(lhs) << rhs;
}

/// Binary >> operator
template<typename T, int N, std::enable_if_t<is_integral_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> operator>>(const vector<T, N>& lhs, const vector<T, N>& rhs)
{
    if constexpr (N == 1)
        return vector<T, N>{lhs.x >> rhs.x};
    else if constexpr (N == 2)
        return vector<T, N>{lhs.x >> rhs.x, lhs.y >> rhs.y};
    else if constexpr (N == 3)
        return vector<T, N>{lhs.x >> rhs.x, lhs.y >> rhs.y, lhs.z >> rhs.z};
    else if constexpr (N == 4)
        return vector<T, N>{lhs.x >> rhs.x, lhs.y >> rhs.y, lhs.z >> rhs.z, lhs.w >> rhs.w};
}

/// Binary >> operator
template<typename T, int N, std::enable_if_t<is_integral_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> operator>>(const vector<T, N>& lhs, T rhs)
{
    return lhs >> vector<T, N>(rhs);
}

/// Binary >> operator
template<typename T, int N, std::enable_if_t<is_integral_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> operator>>(T lhs, const vector<T, N>& rhs)
{
    return vector<T, N>(lhs) >> rhs;
}

/// Binary | operator
template<typename T, int N, std::enable_if_t<is_integral_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> operator|(const vector<T, N>& lhs, const vector<T, N>& rhs)
{
    if constexpr (N == 1)
        return vector<T, N>{lhs.x | rhs.x};
    else if constexpr (N == 2)
        return vector<T, N>{lhs.x | rhs.x, lhs.y | rhs.y};
    else if constexpr (N == 3)
        return vector<T, N>{lhs.x | rhs.x, lhs.y | rhs.y, lhs.z | rhs.z};
    else if constexpr (N == 4)
        return vector<T, N>{lhs.x | rhs.x, lhs.y | rhs.y, lhs.z | rhs.z, lhs.w | rhs.w};
}

/// Binary | operator
template<typename T, int N, std::enable_if_t<is_integral_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> operator|(const vector<T, N>& lhs, T rhs)
{
    return lhs | vector<T, N>(rhs);
}

/// Binary | operator
template<typename T, int N, std::enable_if_t<is_integral_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> operator|(T lhs, const vector<T, N>& rhs)
{
    return vector<T, N>(lhs) | rhs;
}

/// Binary & operator
template<typename T, int N, std::enable_if_t<is_integral_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> operator&(const vector<T, N>& lhs, const vector<T, N>& rhs)
{
    if constexpr (N == 1)
        return vector<T, N>{lhs.x & rhs.x};
    else if constexpr (N == 2)
        return vector<T, N>{lhs.x & rhs.x, lhs.y & rhs.y};
    else if constexpr (N == 3)
        return vector<T, N>{lhs.x & rhs.x, lhs.y & rhs.y, lhs.z & rhs.z};
    else if constexpr (N == 4)
        return vector<T, N>{lhs.x & rhs.x, lhs.y & rhs.y, lhs.z & rhs.z, lhs.w & rhs.w};
}

/// Binary & operator
template<typename T, int N, std::enable_if_t<is_integral_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> operator&(const vector<T, N>& lhs, T rhs)
{
    return lhs & vector<T, N>(rhs);
}

/// Binary & operator
template<typename T, int N, std::enable_if_t<is_integral_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> operator&(T lhs, const vector<T, N>& rhs)
{
    return vector<T, N>(lhs) & rhs;
}

/// Binary ^ operator
template<typename T, int N, std::enable_if_t<is_integral_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> operator^(const vector<T, N>& lhs, const vector<T, N>& rhs)
{
    if constexpr (N == 1)
        return vector<T, N>{lhs.x ^ rhs.x};
    else if constexpr (N == 2)
        return vector<T, N>{lhs.x ^ rhs.x, lhs.y ^ rhs.y};
    else if constexpr (N == 3)
        return vector<T, N>{lhs.x ^ rhs.x, lhs.y ^ rhs.y, lhs.z ^ rhs.z};
    else if constexpr (N == 4)
        return vector<T, N>{lhs.x ^ rhs.x, lhs.y ^ rhs.y, lhs.z ^ rhs.z, lhs.w ^ rhs.w};
}

/// Binary ^ operator
template<typename T, int N, std::enable_if_t<is_integral_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> operator^(const vector<T, N>& lhs, T rhs)
{
    return lhs ^ vector<T, N>(rhs);
}

/// Binary ^ operator
template<typename T, int N, std::enable_if_t<is_integral_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> operator^(T lhs, const vector<T, N>& rhs)
{
    return vector<T, N>(lhs) ^ rhs;
}

/* <<<PYMACROEND>>> */
// clang-format on

// ----------------------------------------------------------------------------
// Binary logic operators
// ----------------------------------------------------------------------------

// clang-format off
/* <<<PYMACRO
def print_binary_operator(op, enable_if):
    print(f"""/// Binary {op} operator
template<typename T, int N, std::enable_if_t<{enable_if}, bool> = false>
[[nodiscard]] constexpr auto operator{op}(const vector<T, N>& lhs, const vector<T, N>& rhs)
{{
    if constexpr (N == 1)
        return bool1{{lhs.x {op} rhs.x}};
    else if constexpr (N == 2)
        return bool2{{lhs.x {op} rhs.x, lhs.y {op} rhs.y}};
    else if constexpr (N == 3)
        return bool3{{lhs.x {op} rhs.x, lhs.y {op} rhs.y, lhs.z {op} rhs.z}};
    else if constexpr (N == 4)
        return bool4{{lhs.x {op} rhs.x, lhs.y {op} rhs.y, lhs.z {op} rhs.z, lhs.w {op} rhs.w}};
}}

/// Binary {op} operator
template<typename T, int N, std::enable_if_t<{enable_if}, bool> = false>
[[nodiscard]] constexpr auto operator{op}(const vector<T, N>& lhs, T rhs)
{{
    return lhs {op} vector<T, N>(rhs);
}}

/// Binary {op} operator
template<typename T, int N, std::enable_if_t<{enable_if}, bool> = false>
[[nodiscard]] constexpr auto operator{op}(T lhs, const vector<T, N>& rhs)
{{
    return vector<T, N>(lhs) {op} rhs;
}}
""")

print_binary_operator("||", "is_bool_v<T>")
print_binary_operator("&&", "is_bool_v<T>")
print_binary_operator("==", "true")
print_binary_operator("!=", "true")
print_binary_operator("<", "is_arithmetic_v<T>")
print_binary_operator(">", "is_arithmetic_v<T>")
print_binary_operator("<=", "is_arithmetic_v<T>")
print_binary_operator(">=", "is_arithmetic_v<T>")
>>> */
/// Binary || operator
template<typename T, int N, std::enable_if_t<is_bool_v<T>, bool> = false>
[[nodiscard]] constexpr auto operator||(const vector<T, N>& lhs, const vector<T, N>& rhs)
{
    if constexpr (N == 1)
        return bool1{lhs.x || rhs.x};
    else if constexpr (N == 2)
        return bool2{lhs.x || rhs.x, lhs.y || rhs.y};
    else if constexpr (N == 3)
        return bool3{lhs.x || rhs.x, lhs.y || rhs.y, lhs.z || rhs.z};
    else if constexpr (N == 4)
        return bool4{lhs.x || rhs.x, lhs.y || rhs.y, lhs.z || rhs.z, lhs.w || rhs.w};
}

/// Binary || operator
template<typename T, int N, std::enable_if_t<is_bool_v<T>, bool> = false>
[[nodiscard]] constexpr auto operator||(const vector<T, N>& lhs, T rhs)
{
    return lhs || vector<T, N>(rhs);
}

/// Binary || operator
template<typename T, int N, std::enable_if_t<is_bool_v<T>, bool> = false>
[[nodiscard]] constexpr auto operator||(T lhs, const vector<T, N>& rhs)
{
    return vector<T, N>(lhs) || rhs;
}

/// Binary && operator
template<typename T, int N, std::enable_if_t<is_bool_v<T>, bool> = false>
[[nodiscard]] constexpr auto operator&&(const vector<T, N>& lhs, const vector<T, N>& rhs)
{
    if constexpr (N == 1)
        return bool1{lhs.x && rhs.x};
    else if constexpr (N == 2)
        return bool2{lhs.x && rhs.x, lhs.y && rhs.y};
    else if constexpr (N == 3)
        return bool3{lhs.x && rhs.x, lhs.y && rhs.y, lhs.z && rhs.z};
    else if constexpr (N == 4)
        return bool4{lhs.x && rhs.x, lhs.y && rhs.y, lhs.z && rhs.z, lhs.w && rhs.w};
}

/// Binary && operator
template<typename T, int N, std::enable_if_t<is_bool_v<T>, bool> = false>
[[nodiscard]] constexpr auto operator&&(const vector<T, N>& lhs, T rhs)
{
    return lhs && vector<T, N>(rhs);
}

/// Binary && operator
template<typename T, int N, std::enable_if_t<is_bool_v<T>, bool> = false>
[[nodiscard]] constexpr auto operator&&(T lhs, const vector<T, N>& rhs)
{
    return vector<T, N>(lhs) && rhs;
}

/// Binary == operator
template<typename T, int N, std::enable_if_t<true, bool> = false>
[[nodiscard]] constexpr auto operator==(const vector<T, N>& lhs, const vector<T, N>& rhs)
{
    if constexpr (N == 1)
        return bool1{lhs.x == rhs.x};
    else if constexpr (N == 2)
        return bool2{lhs.x == rhs.x, lhs.y == rhs.y};
    else if constexpr (N == 3)
        return bool3{lhs.x == rhs.x, lhs.y == rhs.y, lhs.z == rhs.z};
    else if constexpr (N == 4)
        return bool4{lhs.x == rhs.x, lhs.y == rhs.y, lhs.z == rhs.z, lhs.w == rhs.w};
}

/// Binary == operator
template<typename T, int N, std::enable_if_t<true, bool> = false>
[[nodiscard]] constexpr auto operator==(const vector<T, N>& lhs, T rhs)
{
    return lhs == vector<T, N>(rhs);
}

/// Binary == operator
template<typename T, int N, std::enable_if_t<true, bool> = false>
[[nodiscard]] constexpr auto operator==(T lhs, const vector<T, N>& rhs)
{
    return vector<T, N>(lhs) == rhs;
}

/// Binary != operator
template<typename T, int N, std::enable_if_t<true, bool> = false>
[[nodiscard]] constexpr auto operator!=(const vector<T, N>& lhs, const vector<T, N>& rhs)
{
    if constexpr (N == 1)
        return bool1{lhs.x != rhs.x};
    else if constexpr (N == 2)
        return bool2{lhs.x != rhs.x, lhs.y != rhs.y};
    else if constexpr (N == 3)
        return bool3{lhs.x != rhs.x, lhs.y != rhs.y, lhs.z != rhs.z};
    else if constexpr (N == 4)
        return bool4{lhs.x != rhs.x, lhs.y != rhs.y, lhs.z != rhs.z, lhs.w != rhs.w};
}

/// Binary != operator
template<typename T, int N, std::enable_if_t<true, bool> = false>
[[nodiscard]] constexpr auto operator!=(const vector<T, N>& lhs, T rhs)
{
    return lhs != vector<T, N>(rhs);
}

/// Binary != operator
template<typename T, int N, std::enable_if_t<true, bool> = false>
[[nodiscard]] constexpr auto operator!=(T lhs, const vector<T, N>& rhs)
{
    return vector<T, N>(lhs) != rhs;
}

/// Binary < operator
template<typename T, int N, std::enable_if_t<is_arithmetic_v<T>, bool> = false>
[[nodiscard]] constexpr auto operator<(const vector<T, N>& lhs, const vector<T, N>& rhs)
{
    if constexpr (N == 1)
        return bool1{lhs.x < rhs.x};
    else if constexpr (N == 2)
        return bool2{lhs.x < rhs.x, lhs.y < rhs.y};
    else if constexpr (N == 3)
        return bool3{lhs.x < rhs.x, lhs.y < rhs.y, lhs.z < rhs.z};
    else if constexpr (N == 4)
        return bool4{lhs.x < rhs.x, lhs.y < rhs.y, lhs.z < rhs.z, lhs.w < rhs.w};
}

/// Binary < operator
template<typename T, int N, std::enable_if_t<is_arithmetic_v<T>, bool> = false>
[[nodiscard]] constexpr auto operator<(const vector<T, N>& lhs, T rhs)
{
    return lhs < vector<T, N>(rhs);
}

/// Binary < operator
template<typename T, int N, std::enable_if_t<is_arithmetic_v<T>, bool> = false>
[[nodiscard]] constexpr auto operator<(T lhs, const vector<T, N>& rhs)
{
    return vector<T, N>(lhs) < rhs;
}

/// Binary > operator
template<typename T, int N, std::enable_if_t<is_arithmetic_v<T>, bool> = false>
[[nodiscard]] constexpr auto operator>(const vector<T, N>& lhs, const vector<T, N>& rhs)
{
    if constexpr (N == 1)
        return bool1{lhs.x > rhs.x};
    else if constexpr (N == 2)
        return bool2{lhs.x > rhs.x, lhs.y > rhs.y};
    else if constexpr (N == 3)
        return bool3{lhs.x > rhs.x, lhs.y > rhs.y, lhs.z > rhs.z};
    else if constexpr (N == 4)
        return bool4{lhs.x > rhs.x, lhs.y > rhs.y, lhs.z > rhs.z, lhs.w > rhs.w};
}

/// Binary > operator
template<typename T, int N, std::enable_if_t<is_arithmetic_v<T>, bool> = false>
[[nodiscard]] constexpr auto operator>(const vector<T, N>& lhs, T rhs)
{
    return lhs > vector<T, N>(rhs);
}

/// Binary > operator
template<typename T, int N, std::enable_if_t<is_arithmetic_v<T>, bool> = false>
[[nodiscard]] constexpr auto operator>(T lhs, const vector<T, N>& rhs)
{
    return vector<T, N>(lhs) > rhs;
}

/// Binary <= operator
template<typename T, int N, std::enable_if_t<is_arithmetic_v<T>, bool> = false>
[[nodiscard]] constexpr auto operator<=(const vector<T, N>& lhs, const vector<T, N>& rhs)
{
    if constexpr (N == 1)
        return bool1{lhs.x <= rhs.x};
    else if constexpr (N == 2)
        return bool2{lhs.x <= rhs.x, lhs.y <= rhs.y};
    else if constexpr (N == 3)
        return bool3{lhs.x <= rhs.x, lhs.y <= rhs.y, lhs.z <= rhs.z};
    else if constexpr (N == 4)
        return bool4{lhs.x <= rhs.x, lhs.y <= rhs.y, lhs.z <= rhs.z, lhs.w <= rhs.w};
}

/// Binary <= operator
template<typename T, int N, std::enable_if_t<is_arithmetic_v<T>, bool> = false>
[[nodiscard]] constexpr auto operator<=(const vector<T, N>& lhs, T rhs)
{
    return lhs <= vector<T, N>(rhs);
}

/// Binary <= operator
template<typename T, int N, std::enable_if_t<is_arithmetic_v<T>, bool> = false>
[[nodiscard]] constexpr auto operator<=(T lhs, const vector<T, N>& rhs)
{
    return vector<T, N>(lhs) <= rhs;
}

/// Binary >= operator
template<typename T, int N, std::enable_if_t<is_arithmetic_v<T>, bool> = false>
[[nodiscard]] constexpr auto operator>=(const vector<T, N>& lhs, const vector<T, N>& rhs)
{
    if constexpr (N == 1)
        return bool1{lhs.x >= rhs.x};
    else if constexpr (N == 2)
        return bool2{lhs.x >= rhs.x, lhs.y >= rhs.y};
    else if constexpr (N == 3)
        return bool3{lhs.x >= rhs.x, lhs.y >= rhs.y, lhs.z >= rhs.z};
    else if constexpr (N == 4)
        return bool4{lhs.x >= rhs.x, lhs.y >= rhs.y, lhs.z >= rhs.z, lhs.w >= rhs.w};
}

/// Binary >= operator
template<typename T, int N, std::enable_if_t<is_arithmetic_v<T>, bool> = false>
[[nodiscard]] constexpr auto operator>=(const vector<T, N>& lhs, T rhs)
{
    return lhs >= vector<T, N>(rhs);
}

/// Binary >= operator
template<typename T, int N, std::enable_if_t<is_arithmetic_v<T>, bool> = false>
[[nodiscard]] constexpr auto operator>=(T lhs, const vector<T, N>& rhs)
{
    return vector<T, N>(lhs) >= rhs;
}

/* <<<PYMACROEND>>> */
// clang-format on

// ----------------------------------------------------------------------------
// Assignment operators
// ----------------------------------------------------------------------------

// clang-format off
/* <<<PYMACRO
def print_assignment_operator(op, enable_if):
    print(f"""/// {op} assignment operator
template<typename T, int N, std::enable_if_t<{enable_if}, bool> = false>
constexpr vector<T, N> operator{op}(vector<T, N>& lhs, const vector<T, N>& rhs)
{{
    lhs.x {op} rhs.x;
    if constexpr (N >= 2)
        lhs.y {op} rhs.y;
    if constexpr (N >= 3)
        lhs.z {op} rhs.z;
    if constexpr (N >= 4)
        lhs.w {op} rhs.w;
    return lhs;
}}

/// {op} assignment operator
template<typename T, int N, std::enable_if_t<{enable_if}, bool> = false>
constexpr vector<T, N> operator{op}(vector<T, N>& lhs, T rhs)
{{
    return (lhs {op} vector<T, N>(rhs));
}}
""")

print_assignment_operator("+=", "is_arithmetic_v<T>")
print_assignment_operator("-=", "is_arithmetic_v<T>")
print_assignment_operator("*=", "is_arithmetic_v<T>")
print_assignment_operator("/=", "is_arithmetic_v<T>")
print_assignment_operator("%=", "is_integral_v<T>")
print_assignment_operator("<<=", "is_integral_v<T>")
print_assignment_operator(">>=", "is_integral_v<T>")
print_assignment_operator("|=", "is_integral_v<T>")
print_assignment_operator("&=", "is_integral_v<T>")
print_assignment_operator("^=", "is_integral_v<T>")
>>> */
/// += assignment operator
template<typename T, int N, std::enable_if_t<is_arithmetic_v<T>, bool> = false>
constexpr vector<T, N> operator+=(vector<T, N>& lhs, const vector<T, N>& rhs)
{
    lhs.x += rhs.x;
    if constexpr (N >= 2)
        lhs.y += rhs.y;
    if constexpr (N >= 3)
        lhs.z += rhs.z;
    if constexpr (N >= 4)
        lhs.w += rhs.w;
    return lhs;
}

/// += assignment operator
template<typename T, int N, std::enable_if_t<is_arithmetic_v<T>, bool> = false>
constexpr vector<T, N> operator+=(vector<T, N>& lhs, T rhs)
{
    return (lhs += vector<T, N>(rhs));
}

/// -= assignment operator
template<typename T, int N, std::enable_if_t<is_arithmetic_v<T>, bool> = false>
constexpr vector<T, N> operator-=(vector<T, N>& lhs, const vector<T, N>& rhs)
{
    lhs.x -= rhs.x;
    if constexpr (N >= 2)
        lhs.y -= rhs.y;
    if constexpr (N >= 3)
        lhs.z -= rhs.z;
    if constexpr (N >= 4)
        lhs.w -= rhs.w;
    return lhs;
}

/// -= assignment operator
template<typename T, int N, std::enable_if_t<is_arithmetic_v<T>, bool> = false>
constexpr vector<T, N> operator-=(vector<T, N>& lhs, T rhs)
{
    return (lhs -= vector<T, N>(rhs));
}

/// *= assignment operator
template<typename T, int N, std::enable_if_t<is_arithmetic_v<T>, bool> = false>
constexpr vector<T, N> operator*=(vector<T, N>& lhs, const vector<T, N>& rhs)
{
    lhs.x *= rhs.x;
    if constexpr (N >= 2)
        lhs.y *= rhs.y;
    if constexpr (N >= 3)
        lhs.z *= rhs.z;
    if constexpr (N >= 4)
        lhs.w *= rhs.w;
    return lhs;
}

/// *= assignment operator
template<typename T, int N, std::enable_if_t<is_arithmetic_v<T>, bool> = false>
constexpr vector<T, N> operator*=(vector<T, N>& lhs, T rhs)
{
    return (lhs *= vector<T, N>(rhs));
}

/// /= assignment operator
template<typename T, int N, std::enable_if_t<is_arithmetic_v<T>, bool> = false>
constexpr vector<T, N> operator/=(vector<T, N>& lhs, const vector<T, N>& rhs)
{
    lhs.x /= rhs.x;
    if constexpr (N >= 2)
        lhs.y /= rhs.y;
    if constexpr (N >= 3)
        lhs.z /= rhs.z;
    if constexpr (N >= 4)
        lhs.w /= rhs.w;
    return lhs;
}

/// /= assignment operator
template<typename T, int N, std::enable_if_t<is_arithmetic_v<T>, bool> = false>
constexpr vector<T, N> operator/=(vector<T, N>& lhs, T rhs)
{
    return (lhs /= vector<T, N>(rhs));
}

/// %= assignment operator
template<typename T, int N, std::enable_if_t<is_integral_v<T>, bool> = false>
constexpr vector<T, N> operator%=(vector<T, N>& lhs, const vector<T, N>& rhs)
{
    lhs.x %= rhs.x;
    if constexpr (N >= 2)
        lhs.y %= rhs.y;
    if constexpr (N >= 3)
        lhs.z %= rhs.z;
    if constexpr (N >= 4)
        lhs.w %= rhs.w;
    return lhs;
}

/// %= assignment operator
template<typename T, int N, std::enable_if_t<is_integral_v<T>, bool> = false>
constexpr vector<T, N> operator%=(vector<T, N>& lhs, T rhs)
{
    return (lhs %= vector<T, N>(rhs));
}

/// <<= assignment operator
template<typename T, int N, std::enable_if_t<is_integral_v<T>, bool> = false>
constexpr vector<T, N> operator<<=(vector<T, N>& lhs, const vector<T, N>& rhs)
{
    lhs.x <<= rhs.x;
    if constexpr (N >= 2)
        lhs.y <<= rhs.y;
    if constexpr (N >= 3)
        lhs.z <<= rhs.z;
    if constexpr (N >= 4)
        lhs.w <<= rhs.w;
    return lhs;
}

/// <<= assignment operator
template<typename T, int N, std::enable_if_t<is_integral_v<T>, bool> = false>
constexpr vector<T, N> operator<<=(vector<T, N>& lhs, T rhs)
{
    return (lhs <<= vector<T, N>(rhs));
}

/// >>= assignment operator
template<typename T, int N, std::enable_if_t<is_integral_v<T>, bool> = false>
constexpr vector<T, N> operator>>=(vector<T, N>& lhs, const vector<T, N>& rhs)
{
    lhs.x >>= rhs.x;
    if constexpr (N >= 2)
        lhs.y >>= rhs.y;
    if constexpr (N >= 3)
        lhs.z >>= rhs.z;
    if constexpr (N >= 4)
        lhs.w >>= rhs.w;
    return lhs;
}

/// >>= assignment operator
template<typename T, int N, std::enable_if_t<is_integral_v<T>, bool> = false>
constexpr vector<T, N> operator>>=(vector<T, N>& lhs, T rhs)
{
    return (lhs >>= vector<T, N>(rhs));
}

/// |= assignment operator
template<typename T, int N, std::enable_if_t<is_integral_v<T>, bool> = false>
constexpr vector<T, N> operator|=(vector<T, N>& lhs, const vector<T, N>& rhs)
{
    lhs.x |= rhs.x;
    if constexpr (N >= 2)
        lhs.y |= rhs.y;
    if constexpr (N >= 3)
        lhs.z |= rhs.z;
    if constexpr (N >= 4)
        lhs.w |= rhs.w;
    return lhs;
}

/// |= assignment operator
template<typename T, int N, std::enable_if_t<is_integral_v<T>, bool> = false>
constexpr vector<T, N> operator|=(vector<T, N>& lhs, T rhs)
{
    return (lhs |= vector<T, N>(rhs));
}

/// &= assignment operator
template<typename T, int N, std::enable_if_t<is_integral_v<T>, bool> = false>
constexpr vector<T, N> operator&=(vector<T, N>& lhs, const vector<T, N>& rhs)
{
    lhs.x &= rhs.x;
    if constexpr (N >= 2)
        lhs.y &= rhs.y;
    if constexpr (N >= 3)
        lhs.z &= rhs.z;
    if constexpr (N >= 4)
        lhs.w &= rhs.w;
    return lhs;
}

/// &= assignment operator
template<typename T, int N, std::enable_if_t<is_integral_v<T>, bool> = false>
constexpr vector<T, N> operator&=(vector<T, N>& lhs, T rhs)
{
    return (lhs &= vector<T, N>(rhs));
}

/// ^= assignment operator
template<typename T, int N, std::enable_if_t<is_integral_v<T>, bool> = false>
constexpr vector<T, N> operator^=(vector<T, N>& lhs, const vector<T, N>& rhs)
{
    lhs.x ^= rhs.x;
    if constexpr (N >= 2)
        lhs.y ^= rhs.y;
    if constexpr (N >= 3)
        lhs.z ^= rhs.z;
    if constexpr (N >= 4)
        lhs.w ^= rhs.w;
    return lhs;
}

/// ^= assignment operator
template<typename T, int N, std::enable_if_t<is_integral_v<T>, bool> = false>
constexpr vector<T, N> operator^=(vector<T, N>& lhs, T rhs)
{
    return (lhs ^= vector<T, N>(rhs));
}

/* <<<PYMACROEND>>> */
// clang-format on

// ----------------------------------------------------------------------------
// Intrinsics
// ----------------------------------------------------------------------------

// clang-format off
/* <<<PYMACRO
def print_section(name):
    print(f"""// ----------------------------------------------------------------------------
// {name}
// ----------------------------------------------------------------------------
""")
def print_unary_func(func, enable_if, arg="x", return_type="T"):
    print(f"""/// {func}
template<typename T, int N, std::enable_if_t<{enable_if}, bool> = false>
[[nodiscard]] constexpr vector<{return_type}, N> {func}(const vector<T, N>& {arg})
{{
    if constexpr (N == 1)
        return vector<{return_type}, N>{{{func}({arg}.x)}};
    else if constexpr (N == 2)
        return vector<{return_type}, N>{{{func}({arg}.x), {func}({arg}.y)}};
    else if constexpr (N == 3)
        return vector<{return_type}, N>{{{func}({arg}.x), {func}({arg}.y), {func}({arg}.z)}};
    else if constexpr (N == 4)
        return vector<{return_type}, N>{{{func}({arg}.x), {func}({arg}.y), {func}({arg}.z), {func}({arg}.w)}};
}}
""")
def print_binary_func(func, enable_if, arg0="x", arg1="y", return_type="T"):
    print(f"""/// {func}
template<typename T, int N, std::enable_if_t<{enable_if}, bool> = false>
[[nodiscard]] constexpr vector<{return_type}, N> {func}(const vector<T, N>& {arg0}, const vector<T, N>& {arg1})
{{
    if constexpr (N == 1)
        return vector<{return_type}, N>{{{func}({arg0}.x, {arg1}.x)}};
    else if constexpr (N == 2)
        return vector<{return_type}, N>{{{func}({arg0}.x, {arg1}.x), {func}({arg0}.y, {arg1}.y)}};
    else if constexpr (N == 3)
        return vector<{return_type}, N>{{{func}({arg0}.x, {arg1}.x), {func}({arg0}.y, {arg1}.y), {func}({arg0}.z, {arg1}.z)}};
    else if constexpr (N == 4)
        return vector<{return_type}, N>{{{func}({arg0}.x, {arg1}.x), {func}({arg0}.y, {arg1}.y), {func}({arg0}.z, {arg1}.z), {func}({arg0}.w, {arg1}.w)}};
}}
""")
def print_ternary_func(func, enable_if, arg0="x", arg1="y", arg2="z", return_type="T"):
    print(f"""/// {func}
template<typename T, int N, std::enable_if_t<{enable_if}, bool> = false>
[[nodiscard]] constexpr vector<{return_type}, N> {func}(const vector<T, N>& {arg0}, const vector<T, N>& {arg1}, const vector<T, N>& {arg2})
{{
    if constexpr (N == 1)
        return vector<{return_type}, N>{{{func}({arg0}.x, {arg1}.x, {arg2}.x)}};
    else if constexpr (N == 2)
        return vector<{return_type}, N>{{{func}({arg0}.x, {arg1}.x, {arg2}.x), {func}({arg0}.y, {arg1}.y, {arg2}.y)}};
    else if constexpr (N == 3)
        return vector<{return_type}, N>{{{func}({arg0}.x, {arg1}.x, {arg2}.x), {func}({arg0}.y, {arg1}.y, {arg2}.y), {func}({arg0}.z, {arg1}.z, {arg2}.z)}};
    else if constexpr (N == 4)
        return vector<{return_type}, N>{{{func}({arg0}.x, {arg1}.x, {arg2}.x), {func}({arg0}.y, {arg1}.y, {arg2}.y), {func}({arg0}.z, {arg1}.z, {arg2}.z), {func}({arg0}.w, {arg1}.w, {arg2}.w)}};
}}
""")
print_section("Basic functions")
print_binary_func("min", "is_arithmetic_v<T>")
print_binary_func("max", "is_arithmetic_v<T>")
print_ternary_func("clamp", "is_arithmetic_v<T>", "x", "min_", "max_")
print_unary_func("abs", "is_signed_v<T>")
print_unary_func("sign", "is_signed_v<T>")
print_section("Floating point checks")
print_unary_func("isfinite", "is_floating_point_v<T>", "x", "bool")
print_unary_func("isinf", "is_floating_point_v<T>", "x", "bool")
print_unary_func("isnan", "is_floating_point_v<T>", "x", "bool")
print_section("Rounding")
print_unary_func("floor", "is_floating_point_v<T>")
print_unary_func("ceil", "is_floating_point_v<T>")
print_unary_func("trunc", "is_floating_point_v<T>")
print_unary_func("round", "is_floating_point_v<T>")
print_section("Exponential")
print_binary_func("pow", "is_floating_point_v<T>", "x", "y")
print_unary_func("sqrt", "is_floating_point_v<T>")
print_unary_func("rsqrt", "is_floating_point_v<T>")
print_unary_func("exp", "is_floating_point_v<T>")
print_unary_func("exp2", "is_floating_point_v<T>")
print_unary_func("log", "is_floating_point_v<T>")
print_unary_func("log2", "is_floating_point_v<T>")
print_unary_func("log10", "is_floating_point_v<T>")
print_section("Trigonometry")
print_unary_func("radians", "is_floating_point_v<T>")
print_unary_func("degrees", "is_floating_point_v<T>")
print_unary_func("sin", "is_floating_point_v<T>")
print_unary_func("cos", "is_floating_point_v<T>")
print_unary_func("tan", "is_floating_point_v<T>")
print_unary_func("asin", "is_floating_point_v<T>")
print_unary_func("acos", "is_floating_point_v<T>")
print_unary_func("atan", "is_floating_point_v<T>")
print_binary_func("atan2", "is_floating_point_v<T>", "y", "x")
print_unary_func("sinh", "is_floating_point_v<T>")
print_unary_func("cosh", "is_floating_point_v<T>")
print_unary_func("tanh", "is_floating_point_v<T>")
print_section("Misc")
print_binary_func("fmod", "is_floating_point_v<T>", "x", "y")
print_unary_func("frac", "is_floating_point_v<T>")
print_ternary_func("lerp", "is_floating_point_v<T>", "x", "y", "s")
print_unary_func("rcp", "is_floating_point_v<T>")
print_unary_func("saturate", "is_floating_point_v<T>")
print_ternary_func("smoothstep", "is_floating_point_v<T>", "min_", "max_", "x")
print_binary_func("step", "is_floating_point_v<T>", "x", "y")
>>> */
// ----------------------------------------------------------------------------
// Basic functions
// ----------------------------------------------------------------------------

/// min
template<typename T, int N, std::enable_if_t<is_arithmetic_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> min(const vector<T, N>& x, const vector<T, N>& y)
{
    if constexpr (N == 1)
        return vector<T, N>{min(x.x, y.x)};
    else if constexpr (N == 2)
        return vector<T, N>{min(x.x, y.x), min(x.y, y.y)};
    else if constexpr (N == 3)
        return vector<T, N>{min(x.x, y.x), min(x.y, y.y), min(x.z, y.z)};
    else if constexpr (N == 4)
        return vector<T, N>{min(x.x, y.x), min(x.y, y.y), min(x.z, y.z), min(x.w, y.w)};
}

/// max
template<typename T, int N, std::enable_if_t<is_arithmetic_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> max(const vector<T, N>& x, const vector<T, N>& y)
{
    if constexpr (N == 1)
        return vector<T, N>{max(x.x, y.x)};
    else if constexpr (N == 2)
        return vector<T, N>{max(x.x, y.x), max(x.y, y.y)};
    else if constexpr (N == 3)
        return vector<T, N>{max(x.x, y.x), max(x.y, y.y), max(x.z, y.z)};
    else if constexpr (N == 4)
        return vector<T, N>{max(x.x, y.x), max(x.y, y.y), max(x.z, y.z), max(x.w, y.w)};
}

/// clamp
template<typename T, int N, std::enable_if_t<is_arithmetic_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> clamp(const vector<T, N>& x, const vector<T, N>& min_, const vector<T, N>& max_)
{
    if constexpr (N == 1)
        return vector<T, N>{clamp(x.x, min_.x, max_.x)};
    else if constexpr (N == 2)
        return vector<T, N>{clamp(x.x, min_.x, max_.x), clamp(x.y, min_.y, max_.y)};
    else if constexpr (N == 3)
        return vector<T, N>{clamp(x.x, min_.x, max_.x), clamp(x.y, min_.y, max_.y), clamp(x.z, min_.z, max_.z)};
    else if constexpr (N == 4)
        return vector<T, N>{clamp(x.x, min_.x, max_.x), clamp(x.y, min_.y, max_.y), clamp(x.z, min_.z, max_.z), clamp(x.w, min_.w, max_.w)};
}

/// abs
template<typename T, int N, std::enable_if_t<is_signed_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> abs(const vector<T, N>& x)
{
    if constexpr (N == 1)
        return vector<T, N>{abs(x.x)};
    else if constexpr (N == 2)
        return vector<T, N>{abs(x.x), abs(x.y)};
    else if constexpr (N == 3)
        return vector<T, N>{abs(x.x), abs(x.y), abs(x.z)};
    else if constexpr (N == 4)
        return vector<T, N>{abs(x.x), abs(x.y), abs(x.z), abs(x.w)};
}

/// sign
template<typename T, int N, std::enable_if_t<is_signed_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> sign(const vector<T, N>& x)
{
    if constexpr (N == 1)
        return vector<T, N>{sign(x.x)};
    else if constexpr (N == 2)
        return vector<T, N>{sign(x.x), sign(x.y)};
    else if constexpr (N == 3)
        return vector<T, N>{sign(x.x), sign(x.y), sign(x.z)};
    else if constexpr (N == 4)
        return vector<T, N>{sign(x.x), sign(x.y), sign(x.z), sign(x.w)};
}

// ----------------------------------------------------------------------------
// Floating point checks
// ----------------------------------------------------------------------------

/// isfinite
template<typename T, int N, std::enable_if_t<is_floating_point_v<T>, bool> = false>
[[nodiscard]] constexpr vector<bool, N> isfinite(const vector<T, N>& x)
{
    if constexpr (N == 1)
        return vector<bool, N>{isfinite(x.x)};
    else if constexpr (N == 2)
        return vector<bool, N>{isfinite(x.x), isfinite(x.y)};
    else if constexpr (N == 3)
        return vector<bool, N>{isfinite(x.x), isfinite(x.y), isfinite(x.z)};
    else if constexpr (N == 4)
        return vector<bool, N>{isfinite(x.x), isfinite(x.y), isfinite(x.z), isfinite(x.w)};
}

/// isinf
template<typename T, int N, std::enable_if_t<is_floating_point_v<T>, bool> = false>
[[nodiscard]] constexpr vector<bool, N> isinf(const vector<T, N>& x)
{
    if constexpr (N == 1)
        return vector<bool, N>{isinf(x.x)};
    else if constexpr (N == 2)
        return vector<bool, N>{isinf(x.x), isinf(x.y)};
    else if constexpr (N == 3)
        return vector<bool, N>{isinf(x.x), isinf(x.y), isinf(x.z)};
    else if constexpr (N == 4)
        return vector<bool, N>{isinf(x.x), isinf(x.y), isinf(x.z), isinf(x.w)};
}

/// isnan
template<typename T, int N, std::enable_if_t<is_floating_point_v<T>, bool> = false>
[[nodiscard]] constexpr vector<bool, N> isnan(const vector<T, N>& x)
{
    if constexpr (N == 1)
        return vector<bool, N>{isnan(x.x)};
    else if constexpr (N == 2)
        return vector<bool, N>{isnan(x.x), isnan(x.y)};
    else if constexpr (N == 3)
        return vector<bool, N>{isnan(x.x), isnan(x.y), isnan(x.z)};
    else if constexpr (N == 4)
        return vector<bool, N>{isnan(x.x), isnan(x.y), isnan(x.z), isnan(x.w)};
}

// ----------------------------------------------------------------------------
// Rounding
// ----------------------------------------------------------------------------

/// floor
template<typename T, int N, std::enable_if_t<is_floating_point_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> floor(const vector<T, N>& x)
{
    if constexpr (N == 1)
        return vector<T, N>{floor(x.x)};
    else if constexpr (N == 2)
        return vector<T, N>{floor(x.x), floor(x.y)};
    else if constexpr (N == 3)
        return vector<T, N>{floor(x.x), floor(x.y), floor(x.z)};
    else if constexpr (N == 4)
        return vector<T, N>{floor(x.x), floor(x.y), floor(x.z), floor(x.w)};
}

/// ceil
template<typename T, int N, std::enable_if_t<is_floating_point_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> ceil(const vector<T, N>& x)
{
    if constexpr (N == 1)
        return vector<T, N>{ceil(x.x)};
    else if constexpr (N == 2)
        return vector<T, N>{ceil(x.x), ceil(x.y)};
    else if constexpr (N == 3)
        return vector<T, N>{ceil(x.x), ceil(x.y), ceil(x.z)};
    else if constexpr (N == 4)
        return vector<T, N>{ceil(x.x), ceil(x.y), ceil(x.z), ceil(x.w)};
}

/// trunc
template<typename T, int N, std::enable_if_t<is_floating_point_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> trunc(const vector<T, N>& x)
{
    if constexpr (N == 1)
        return vector<T, N>{trunc(x.x)};
    else if constexpr (N == 2)
        return vector<T, N>{trunc(x.x), trunc(x.y)};
    else if constexpr (N == 3)
        return vector<T, N>{trunc(x.x), trunc(x.y), trunc(x.z)};
    else if constexpr (N == 4)
        return vector<T, N>{trunc(x.x), trunc(x.y), trunc(x.z), trunc(x.w)};
}

/// round
template<typename T, int N, std::enable_if_t<is_floating_point_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> round(const vector<T, N>& x)
{
    if constexpr (N == 1)
        return vector<T, N>{round(x.x)};
    else if constexpr (N == 2)
        return vector<T, N>{round(x.x), round(x.y)};
    else if constexpr (N == 3)
        return vector<T, N>{round(x.x), round(x.y), round(x.z)};
    else if constexpr (N == 4)
        return vector<T, N>{round(x.x), round(x.y), round(x.z), round(x.w)};
}

// ----------------------------------------------------------------------------
// Exponential
// ----------------------------------------------------------------------------

/// pow
template<typename T, int N, std::enable_if_t<is_floating_point_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> pow(const vector<T, N>& x, const vector<T, N>& y)
{
    if constexpr (N == 1)
        return vector<T, N>{pow(x.x, y.x)};
    else if constexpr (N == 2)
        return vector<T, N>{pow(x.x, y.x), pow(x.y, y.y)};
    else if constexpr (N == 3)
        return vector<T, N>{pow(x.x, y.x), pow(x.y, y.y), pow(x.z, y.z)};
    else if constexpr (N == 4)
        return vector<T, N>{pow(x.x, y.x), pow(x.y, y.y), pow(x.z, y.z), pow(x.w, y.w)};
}

/// sqrt
template<typename T, int N, std::enable_if_t<is_floating_point_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> sqrt(const vector<T, N>& x)
{
    if constexpr (N == 1)
        return vector<T, N>{sqrt(x.x)};
    else if constexpr (N == 2)
        return vector<T, N>{sqrt(x.x), sqrt(x.y)};
    else if constexpr (N == 3)
        return vector<T, N>{sqrt(x.x), sqrt(x.y), sqrt(x.z)};
    else if constexpr (N == 4)
        return vector<T, N>{sqrt(x.x), sqrt(x.y), sqrt(x.z), sqrt(x.w)};
}

/// rsqrt
template<typename T, int N, std::enable_if_t<is_floating_point_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> rsqrt(const vector<T, N>& x)
{
    if constexpr (N == 1)
        return vector<T, N>{rsqrt(x.x)};
    else if constexpr (N == 2)
        return vector<T, N>{rsqrt(x.x), rsqrt(x.y)};
    else if constexpr (N == 3)
        return vector<T, N>{rsqrt(x.x), rsqrt(x.y), rsqrt(x.z)};
    else if constexpr (N == 4)
        return vector<T, N>{rsqrt(x.x), rsqrt(x.y), rsqrt(x.z), rsqrt(x.w)};
}

/// exp
template<typename T, int N, std::enable_if_t<is_floating_point_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> exp(const vector<T, N>& x)
{
    if constexpr (N == 1)
        return vector<T, N>{exp(x.x)};
    else if constexpr (N == 2)
        return vector<T, N>{exp(x.x), exp(x.y)};
    else if constexpr (N == 3)
        return vector<T, N>{exp(x.x), exp(x.y), exp(x.z)};
    else if constexpr (N == 4)
        return vector<T, N>{exp(x.x), exp(x.y), exp(x.z), exp(x.w)};
}

/// exp2
template<typename T, int N, std::enable_if_t<is_floating_point_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> exp2(const vector<T, N>& x)
{
    if constexpr (N == 1)
        return vector<T, N>{exp2(x.x)};
    else if constexpr (N == 2)
        return vector<T, N>{exp2(x.x), exp2(x.y)};
    else if constexpr (N == 3)
        return vector<T, N>{exp2(x.x), exp2(x.y), exp2(x.z)};
    else if constexpr (N == 4)
        return vector<T, N>{exp2(x.x), exp2(x.y), exp2(x.z), exp2(x.w)};
}

/// log
template<typename T, int N, std::enable_if_t<is_floating_point_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> log(const vector<T, N>& x)
{
    if constexpr (N == 1)
        return vector<T, N>{log(x.x)};
    else if constexpr (N == 2)
        return vector<T, N>{log(x.x), log(x.y)};
    else if constexpr (N == 3)
        return vector<T, N>{log(x.x), log(x.y), log(x.z)};
    else if constexpr (N == 4)
        return vector<T, N>{log(x.x), log(x.y), log(x.z), log(x.w)};
}

/// log2
template<typename T, int N, std::enable_if_t<is_floating_point_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> log2(const vector<T, N>& x)
{
    if constexpr (N == 1)
        return vector<T, N>{log2(x.x)};
    else if constexpr (N == 2)
        return vector<T, N>{log2(x.x), log2(x.y)};
    else if constexpr (N == 3)
        return vector<T, N>{log2(x.x), log2(x.y), log2(x.z)};
    else if constexpr (N == 4)
        return vector<T, N>{log2(x.x), log2(x.y), log2(x.z), log2(x.w)};
}

/// log10
template<typename T, int N, std::enable_if_t<is_floating_point_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> log10(const vector<T, N>& x)
{
    if constexpr (N == 1)
        return vector<T, N>{log10(x.x)};
    else if constexpr (N == 2)
        return vector<T, N>{log10(x.x), log10(x.y)};
    else if constexpr (N == 3)
        return vector<T, N>{log10(x.x), log10(x.y), log10(x.z)};
    else if constexpr (N == 4)
        return vector<T, N>{log10(x.x), log10(x.y), log10(x.z), log10(x.w)};
}

// ----------------------------------------------------------------------------
// Trigonometry
// ----------------------------------------------------------------------------

/// radians
template<typename T, int N, std::enable_if_t<is_floating_point_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> radians(const vector<T, N>& x)
{
    if constexpr (N == 1)
        return vector<T, N>{radians(x.x)};
    else if constexpr (N == 2)
        return vector<T, N>{radians(x.x), radians(x.y)};
    else if constexpr (N == 3)
        return vector<T, N>{radians(x.x), radians(x.y), radians(x.z)};
    else if constexpr (N == 4)
        return vector<T, N>{radians(x.x), radians(x.y), radians(x.z), radians(x.w)};
}

/// degrees
template<typename T, int N, std::enable_if_t<is_floating_point_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> degrees(const vector<T, N>& x)
{
    if constexpr (N == 1)
        return vector<T, N>{degrees(x.x)};
    else if constexpr (N == 2)
        return vector<T, N>{degrees(x.x), degrees(x.y)};
    else if constexpr (N == 3)
        return vector<T, N>{degrees(x.x), degrees(x.y), degrees(x.z)};
    else if constexpr (N == 4)
        return vector<T, N>{degrees(x.x), degrees(x.y), degrees(x.z), degrees(x.w)};
}

/// sin
template<typename T, int N, std::enable_if_t<is_floating_point_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> sin(const vector<T, N>& x)
{
    if constexpr (N == 1)
        return vector<T, N>{sin(x.x)};
    else if constexpr (N == 2)
        return vector<T, N>{sin(x.x), sin(x.y)};
    else if constexpr (N == 3)
        return vector<T, N>{sin(x.x), sin(x.y), sin(x.z)};
    else if constexpr (N == 4)
        return vector<T, N>{sin(x.x), sin(x.y), sin(x.z), sin(x.w)};
}

/// cos
template<typename T, int N, std::enable_if_t<is_floating_point_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> cos(const vector<T, N>& x)
{
    if constexpr (N == 1)
        return vector<T, N>{cos(x.x)};
    else if constexpr (N == 2)
        return vector<T, N>{cos(x.x), cos(x.y)};
    else if constexpr (N == 3)
        return vector<T, N>{cos(x.x), cos(x.y), cos(x.z)};
    else if constexpr (N == 4)
        return vector<T, N>{cos(x.x), cos(x.y), cos(x.z), cos(x.w)};
}

/// tan
template<typename T, int N, std::enable_if_t<is_floating_point_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> tan(const vector<T, N>& x)
{
    if constexpr (N == 1)
        return vector<T, N>{tan(x.x)};
    else if constexpr (N == 2)
        return vector<T, N>{tan(x.x), tan(x.y)};
    else if constexpr (N == 3)
        return vector<T, N>{tan(x.x), tan(x.y), tan(x.z)};
    else if constexpr (N == 4)
        return vector<T, N>{tan(x.x), tan(x.y), tan(x.z), tan(x.w)};
}

/// asin
template<typename T, int N, std::enable_if_t<is_floating_point_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> asin(const vector<T, N>& x)
{
    if constexpr (N == 1)
        return vector<T, N>{asin(x.x)};
    else if constexpr (N == 2)
        return vector<T, N>{asin(x.x), asin(x.y)};
    else if constexpr (N == 3)
        return vector<T, N>{asin(x.x), asin(x.y), asin(x.z)};
    else if constexpr (N == 4)
        return vector<T, N>{asin(x.x), asin(x.y), asin(x.z), asin(x.w)};
}

/// acos
template<typename T, int N, std::enable_if_t<is_floating_point_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> acos(const vector<T, N>& x)
{
    if constexpr (N == 1)
        return vector<T, N>{acos(x.x)};
    else if constexpr (N == 2)
        return vector<T, N>{acos(x.x), acos(x.y)};
    else if constexpr (N == 3)
        return vector<T, N>{acos(x.x), acos(x.y), acos(x.z)};
    else if constexpr (N == 4)
        return vector<T, N>{acos(x.x), acos(x.y), acos(x.z), acos(x.w)};
}

/// atan
template<typename T, int N, std::enable_if_t<is_floating_point_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> atan(const vector<T, N>& x)
{
    if constexpr (N == 1)
        return vector<T, N>{atan(x.x)};
    else if constexpr (N == 2)
        return vector<T, N>{atan(x.x), atan(x.y)};
    else if constexpr (N == 3)
        return vector<T, N>{atan(x.x), atan(x.y), atan(x.z)};
    else if constexpr (N == 4)
        return vector<T, N>{atan(x.x), atan(x.y), atan(x.z), atan(x.w)};
}

/// atan2
template<typename T, int N, std::enable_if_t<is_floating_point_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> atan2(const vector<T, N>& y, const vector<T, N>& x)
{
    if constexpr (N == 1)
        return vector<T, N>{atan2(y.x, x.x)};
    else if constexpr (N == 2)
        return vector<T, N>{atan2(y.x, x.x), atan2(y.y, x.y)};
    else if constexpr (N == 3)
        return vector<T, N>{atan2(y.x, x.x), atan2(y.y, x.y), atan2(y.z, x.z)};
    else if constexpr (N == 4)
        return vector<T, N>{atan2(y.x, x.x), atan2(y.y, x.y), atan2(y.z, x.z), atan2(y.w, x.w)};
}

/// sinh
template<typename T, int N, std::enable_if_t<is_floating_point_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> sinh(const vector<T, N>& x)
{
    if constexpr (N == 1)
        return vector<T, N>{sinh(x.x)};
    else if constexpr (N == 2)
        return vector<T, N>{sinh(x.x), sinh(x.y)};
    else if constexpr (N == 3)
        return vector<T, N>{sinh(x.x), sinh(x.y), sinh(x.z)};
    else if constexpr (N == 4)
        return vector<T, N>{sinh(x.x), sinh(x.y), sinh(x.z), sinh(x.w)};
}

/// cosh
template<typename T, int N, std::enable_if_t<is_floating_point_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> cosh(const vector<T, N>& x)
{
    if constexpr (N == 1)
        return vector<T, N>{cosh(x.x)};
    else if constexpr (N == 2)
        return vector<T, N>{cosh(x.x), cosh(x.y)};
    else if constexpr (N == 3)
        return vector<T, N>{cosh(x.x), cosh(x.y), cosh(x.z)};
    else if constexpr (N == 4)
        return vector<T, N>{cosh(x.x), cosh(x.y), cosh(x.z), cosh(x.w)};
}

/// tanh
template<typename T, int N, std::enable_if_t<is_floating_point_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> tanh(const vector<T, N>& x)
{
    if constexpr (N == 1)
        return vector<T, N>{tanh(x.x)};
    else if constexpr (N == 2)
        return vector<T, N>{tanh(x.x), tanh(x.y)};
    else if constexpr (N == 3)
        return vector<T, N>{tanh(x.x), tanh(x.y), tanh(x.z)};
    else if constexpr (N == 4)
        return vector<T, N>{tanh(x.x), tanh(x.y), tanh(x.z), tanh(x.w)};
}

// ----------------------------------------------------------------------------
// Misc
// ----------------------------------------------------------------------------

/// fmod
template<typename T, int N, std::enable_if_t<is_floating_point_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> fmod(const vector<T, N>& x, const vector<T, N>& y)
{
    if constexpr (N == 1)
        return vector<T, N>{fmod(x.x, y.x)};
    else if constexpr (N == 2)
        return vector<T, N>{fmod(x.x, y.x), fmod(x.y, y.y)};
    else if constexpr (N == 3)
        return vector<T, N>{fmod(x.x, y.x), fmod(x.y, y.y), fmod(x.z, y.z)};
    else if constexpr (N == 4)
        return vector<T, N>{fmod(x.x, y.x), fmod(x.y, y.y), fmod(x.z, y.z), fmod(x.w, y.w)};
}

/// frac
template<typename T, int N, std::enable_if_t<is_floating_point_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> frac(const vector<T, N>& x)
{
    if constexpr (N == 1)
        return vector<T, N>{frac(x.x)};
    else if constexpr (N == 2)
        return vector<T, N>{frac(x.x), frac(x.y)};
    else if constexpr (N == 3)
        return vector<T, N>{frac(x.x), frac(x.y), frac(x.z)};
    else if constexpr (N == 4)
        return vector<T, N>{frac(x.x), frac(x.y), frac(x.z), frac(x.w)};
}

/// lerp
template<typename T, int N, std::enable_if_t<is_floating_point_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> lerp(const vector<T, N>& x, const vector<T, N>& y, const vector<T, N>& s)
{
    if constexpr (N == 1)
        return vector<T, N>{lerp(x.x, y.x, s.x)};
    else if constexpr (N == 2)
        return vector<T, N>{lerp(x.x, y.x, s.x), lerp(x.y, y.y, s.y)};
    else if constexpr (N == 3)
        return vector<T, N>{lerp(x.x, y.x, s.x), lerp(x.y, y.y, s.y), lerp(x.z, y.z, s.z)};
    else if constexpr (N == 4)
        return vector<T, N>{lerp(x.x, y.x, s.x), lerp(x.y, y.y, s.y), lerp(x.z, y.z, s.z), lerp(x.w, y.w, s.w)};
}

/// rcp
template<typename T, int N, std::enable_if_t<is_floating_point_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> rcp(const vector<T, N>& x)
{
    if constexpr (N == 1)
        return vector<T, N>{rcp(x.x)};
    else if constexpr (N == 2)
        return vector<T, N>{rcp(x.x), rcp(x.y)};
    else if constexpr (N == 3)
        return vector<T, N>{rcp(x.x), rcp(x.y), rcp(x.z)};
    else if constexpr (N == 4)
        return vector<T, N>{rcp(x.x), rcp(x.y), rcp(x.z), rcp(x.w)};
}

/// saturate
template<typename T, int N, std::enable_if_t<is_floating_point_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> saturate(const vector<T, N>& x)
{
    if constexpr (N == 1)
        return vector<T, N>{saturate(x.x)};
    else if constexpr (N == 2)
        return vector<T, N>{saturate(x.x), saturate(x.y)};
    else if constexpr (N == 3)
        return vector<T, N>{saturate(x.x), saturate(x.y), saturate(x.z)};
    else if constexpr (N == 4)
        return vector<T, N>{saturate(x.x), saturate(x.y), saturate(x.z), saturate(x.w)};
}

/// smoothstep
template<typename T, int N, std::enable_if_t<is_floating_point_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> smoothstep(const vector<T, N>& min_, const vector<T, N>& max_, const vector<T, N>& x)
{
    if constexpr (N == 1)
        return vector<T, N>{smoothstep(min_.x, max_.x, x.x)};
    else if constexpr (N == 2)
        return vector<T, N>{smoothstep(min_.x, max_.x, x.x), smoothstep(min_.y, max_.y, x.y)};
    else if constexpr (N == 3)
        return vector<T, N>{smoothstep(min_.x, max_.x, x.x), smoothstep(min_.y, max_.y, x.y), smoothstep(min_.z, max_.z, x.z)};
    else if constexpr (N == 4)
        return vector<T, N>{smoothstep(min_.x, max_.x, x.x), smoothstep(min_.y, max_.y, x.y), smoothstep(min_.z, max_.z, x.z), smoothstep(min_.w, max_.w, x.w)};
}

/// step
template<typename T, int N, std::enable_if_t<is_floating_point_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> step(const vector<T, N>& x, const vector<T, N>& y)
{
    if constexpr (N == 1)
        return vector<T, N>{step(x.x, y.x)};
    else if constexpr (N == 2)
        return vector<T, N>{step(x.x, y.x), step(x.y, y.y)};
    else if constexpr (N == 3)
        return vector<T, N>{step(x.x, y.x), step(x.y, y.y), step(x.z, y.z)};
    else if constexpr (N == 4)
        return vector<T, N>{step(x.x, y.x), step(x.y, y.y), step(x.z, y.z), step(x.w, y.w)};
}

/* <<<PYMACROEND>>> */
// clang-format on

// ----------------------------------------------------------------------------
// Conversion
// ----------------------------------------------------------------------------

[[nodiscard]] inline float2 f16tof32(const uint2& value) noexcept
{
    return float2{f16tof32(value.x), f16tof32(value.y)};
}

[[nodiscard]] inline float3 f16tof32(const uint3& value) noexcept
{
    return float3{f16tof32(value.x), f16tof32(value.y), f16tof32(value.z)};
}

[[nodiscard]] inline float4 f16tof32(const uint4& value) noexcept
{
    return float4{f16tof32(value.x), f16tof32(value.y), f16tof32(value.z), f16tof32(value.w)};
}

[[nodiscard]] inline uint2 f32tof16(const float2& value) noexcept
{
    return uint2{f32tof16(value.x), f32tof16(value.y)};
}

[[nodiscard]] inline uint3 f32tof16(const float3& value) noexcept
{
    return uint3{f32tof16(value.x), f32tof16(value.y), f32tof16(value.z)};
}

[[nodiscard]] inline uint4 f32tof16(const float4& value) noexcept
{
    return uint4{f32tof16(value.x), f32tof16(value.y), f32tof16(value.z), f32tof16(value.w)};
}

// TODO(@skallweit) should we have implicit scalar -> vector conversion?
template<typename T, int N, std::enable_if_t<is_floating_point_v<T>, bool> = false>
[[nodiscard]] constexpr vector<T, N> lerp(const vector<T, N>& a, const vector<T, N>& b, const T& s)
{
    if constexpr (N == 1)
        return vector<T, N>{lerp(a.x, b.x, s)};
    else if constexpr (N == 2)
        return vector<T, N>{lerp(a.x, b.x, s), lerp(a.y, b.y, s)};
    else if constexpr (N == 3)
        return vector<T, N>{lerp(a.x, b.x, s), lerp(a.y, b.y, s), lerp(a.z, b.z, s)};
    else if constexpr (N == 4)
        return vector<T, N>{lerp(a.x, b.x, s), lerp(a.y, b.y, s), lerp(a.z, b.z, s), lerp(a.w, b.w, s)};
}

/// dot
template<typename T, int N>
[[nodiscard]] constexpr T dot(const vector<T, N>& lhs, const vector<T, N>& rhs)
{
    T result = lhs.x * rhs.x;
    if constexpr (N >= 2)
        result += lhs.y * rhs.y;
    if constexpr (N >= 3)
        result += lhs.z * rhs.z;
    if constexpr (N >= 4)
        result += lhs.w * rhs.w;
    return result;
}

/// cross
template<typename T>
[[nodiscard]] constexpr vector<T, 3> cross(const vector<T, 3>& lhs, const vector<T, 3>& rhs)
{
    return vector<T, 3>(lhs.y * rhs.z - lhs.z * rhs.y, lhs.z * rhs.x - lhs.x * rhs.z, lhs.x * rhs.y - lhs.y * rhs.x);
}

/// length
template<typename T, int N>
[[nodiscard]] constexpr T length(const vector<T, N>& v)
{
    return sqrt(dot(v, v));
}

/// normalize
template<typename T, int N>
[[nodiscard]] constexpr vector<T, N> normalize(const vector<T, N>& v)
{
    return v * rsqrt(dot(v, v));
}

/// reflect
template<typename T, int N>
[[nodiscard]] constexpr vector<T, N> reflect(const vector<T, N>& v, const vector<T, N>& n)
{
    return v - T(2) * dot(v, n) * n;
}

/// Convert vector to string.
template<typename T, int N>
[[nodiscard]] std::string to_string(const vector<T, N>& v)
{
    return ::fmt::format("{}", v);
}

} // namespace math
} // namespace Falcor

// Specialize std::less to allow using vectors as key in std::map for example.
template<typename T, int N>
struct std::less<::Falcor::math::vector<T, N>>
{
    constexpr bool operator()(const ::Falcor::math::vector<T, N>& lhs, const ::Falcor::math::vector<T, N>& rhs) const
    {
        for (int i = 0; i < N; ++i)
            if (lhs[i] != rhs[i])
                return lhs[i] < rhs[i];
        return false;
    }
};

template<typename T, int N>
struct std::equal_to<::Falcor::math::vector<T, N>>
{
    constexpr bool operator()(const ::Falcor::math::vector<T, N>& lhs, const ::Falcor::math::vector<T, N>& rhs) const
    {
        for (int i = 0; i < N; ++i)
            if (lhs[i] != rhs[i])
                return false;
        return true;
    }
};
template<typename T, int N>
struct std::not_equal_to<::Falcor::math::vector<T, N>>
{
    constexpr bool operator()(const ::Falcor::math::vector<T, N>& lhs, const ::Falcor::math::vector<T, N>& rhs) const
    {
        for (int i = 0; i < N; ++i)
            if (lhs[i] == rhs[i])
                return false;
        return true;
    }
};

template<typename T, int N>
struct std::hash<::Falcor::math::vector<T, N>>
{
    constexpr int operator()(const ::Falcor::math::vector<T, N>& v) const
    {
        size_t result = 0;
        for (int i = 0; i < N; ++i)
            result ^= std::hash<T>()(v[i]) + 0x9e3779b9 + (result << 6) + (result >> 2);
        return result;
    }
};

/// Vector string formatter.
template<typename T, int N>
struct fmt::formatter<Falcor::math::vector<T, N>> : formatter<T>
{
    template<typename FormatContext>
    auto format(const Falcor::math::vector<T, N>& v, FormatContext& ctx) const
    {
        auto out = ctx.out();
        for (int i = 0; i < N; ++i)
        {
            out = ::fmt::format_to(out, "{}", (i == 0) ? "{" : ", ");
            out = formatter<T>::format(v[i], ctx);
        }
        out = ::fmt::format_to(out, "}}");
        return out;
    }
};
