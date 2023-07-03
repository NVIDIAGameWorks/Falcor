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

#include "Float16.h"

#include <fmt/core.h>

#include <string>
#include <cstdint>

namespace Falcor
{
namespace math
{

enum class Handedness
{
    RightHanded,
    LeftHanded,
};

using uint = uint32_t;

// clang-format off
template<typename T> [[nodiscard]] std::string to_string(T v);
template<> [[nodiscard]] inline std::string to_string(bool v) { return v ? "1" : "0"; }
template<> [[nodiscard]] inline std::string to_string(int v) { return std::to_string(v); }
template<> [[nodiscard]] inline std::string to_string(uint v) { return std::to_string(v); }
template<> [[nodiscard]] inline std::string to_string(float v) { return std::to_string(v); }
template<> [[nodiscard]] inline std::string to_string(double v) { return std::to_string(v); }
template<> [[nodiscard]] inline std::string to_string(float16_t v) { return std::to_string(float(v)); }
// clang-format on

// clang-format off
template<typename T> struct is_bool : ::std::is_same<T, bool> {};
template<typename T> struct is_int : ::std::is_same<T, int32_t> {};
template<typename T> struct is_uint : ::std::is_same<T, uint32_t> {};
template<typename T> struct is_float : ::std::is_same<T, float> {};
template<typename T> struct is_double : ::std::is_same<T, double> {};
template<typename T> struct is_float16_t : ::std::is_same<T, float16_t> {};

template<typename T> constexpr bool is_bool_v = is_bool<T>::value;
template<typename T> constexpr bool is_int_v = is_int<T>::value;
template<typename T> constexpr bool is_uint_v = is_uint<T>::value;
template<typename T> constexpr bool is_float_v = is_float<T>::value;
template<typename T> constexpr bool is_double_v = is_double<T>::value;
template<typename T> constexpr bool is_float16_t_v = is_float16_t<T>::value;

template<typename T> constexpr bool is_arithmetic_v = std::is_arithmetic_v<T> || is_float16_t_v<T>;
template<typename T> constexpr bool is_floating_point_v = is_float_v<T> || is_double_v<T> || is_float16_t_v<T>;
using std::is_integral_v;
using std::is_signed_v;
using std::is_unsigned_v;
// clang-format on

template<typename T>
struct ScalarTraits
{};

template<>
struct ScalarTraits<bool>
{
    static constexpr const char* name{"bool"};
};

template<>
struct ScalarTraits<int>
{
    static constexpr const char* name{"int"};
};

template<>
struct ScalarTraits<uint>
{
    static constexpr const char* name{"uint"};
};

template<>
struct ScalarTraits<float>
{
    static constexpr const char* name{"float"};
};

template<>
struct ScalarTraits<double>
{
    static constexpr const char* name{"double"};
};

template<>
struct ScalarTraits<float16_t>
{
    static constexpr const char* name{"float16_t"};
};

} // namespace math

using uint = math::uint;
using float16_t = math::float16_t;

#if FALCOR_MSVC
#pragma warning(push)
#pragma warning(disable : 4455) // disable warning about literal suffixes not starting with an underscore
#endif

using math::operator""h;

#if FALCOR_MSVC
#pragma warning(pop)
#endif

} // namespace Falcor

// Formatter for the float16_t.
template<>
struct fmt::formatter<Falcor::math::float16_t> : formatter<float>
{
    template<typename FormatContext>
    auto format(Falcor::math::float16_t value, FormatContext& ctx) const
    {
        return formatter<float>::format(float(value), ctx);
    }
};
