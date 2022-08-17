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
#include "Core/Assert.h"
#include "Vector.h"
#include <glm/detail/type_half.hpp>
#include <string>

namespace Falcor
{
    /** Represents a IEEE 754-2008 compatible binary16 type (half precision).
        Numbers outside the representable range +-65504 are stored as +-inf.
    */
    class float16_t
    {
    public:
        float16_t() = default;

        // Float conversion
        explicit float16_t(float v) : bits(glm::detail::toFloat16(v)) {}
        explicit operator float() const { return glm::detail::toFloat32(bits); }

        bool operator==(const float16_t& other) const { return bits == other.bits; }
        bool operator!=(const float16_t& other) const { return bits != other.bits; }

    private:
        glm::detail::hdata bits;
    };

    inline std::string to_string(const float16_t& v) { return std::to_string((float)v); }


    // Vector types

    template<size_t N>
    struct tfloat16_vec
    {
    };

    template<>
    struct tfloat16_vec<2>
    {
        using value_type = float16_t;

        float16_t x, y;

        // Constructors
        tfloat16_vec() = default;
        tfloat16_vec(const float16_t& v) : x(v), y(v) {}
        tfloat16_vec(const float16_t& v1, const float16_t& v2) : x(v1), y(v2) {}

        // Float conversion
        explicit tfloat16_vec(float v) : x(v), y(v) {}
        explicit tfloat16_vec(const float2& v) : x(v.x), y(v.y) {}
        explicit tfloat16_vec(float v1, float v2) : x(v1), y(v2) {}
        explicit operator float2() const { return float2(float(x), float(y)); }

        // Access
        float16_t& operator[](size_t i) { FALCOR_ASSERT(i < length()); return (&x)[i]; }
        const float16_t& operator[](size_t i) const { FALCOR_ASSERT(i < length()); return (&x)[i]; }

        bool operator==(const tfloat16_vec& other) const { return x == other.x && y == other.y; }
        bool operator!=(const tfloat16_vec& other) const { return x != other.x || y != other.y; }

        static constexpr size_t length() { return 2; }
    };

    template<>
    struct tfloat16_vec<3>
    {
        using value_type = float16_t;

        float16_t x, y, z;

        // Constructors
        tfloat16_vec() = default;
        tfloat16_vec(const float16_t& v) : x(v), y(v), z(v) {}
        tfloat16_vec(const float16_t& v1, const float16_t& v2, const float16_t& v3) : x(v1), y(v2), z(v3) {}

        // Float conversion
        explicit tfloat16_vec(float v) : x(v), y(v), z(v) {}
        explicit tfloat16_vec(const float3& v) : x(v.x), y(v.y), z(v.z) {}
        explicit tfloat16_vec(float v1, float v2, float v3) : x(v1), y(v2), z(v3) {}
        explicit operator float3() const { return float3(float(x), float(y), float(z)); }

        // Access
        float16_t& operator[](size_t i) { FALCOR_ASSERT(i < length()); return (&x)[i]; }
        const float16_t& operator[](size_t i) const { FALCOR_ASSERT(i < length()); return (&x)[i]; }

        bool operator==(const tfloat16_vec& other) const { return x == other.x && y == other.y && z == other.z; }
        bool operator!=(const tfloat16_vec& other) const { return x != other.x || y != other.y || z != other.z; }

        static constexpr size_t length() { return 3; }
    };

    template<>
    struct tfloat16_vec<4>
    {
        using value_type = float16_t;

        float16_t x, y, z, w;

        // Constructors
        tfloat16_vec() = default;
        tfloat16_vec(const float16_t& v) : x(v), y(v), z(v), w(v) {}
        tfloat16_vec(const float16_t& v1, const float16_t& v2, const float16_t& v3, const float16_t& v4) : x(v1), y(v2), z(v3), w(v4) {}

        // Float conversion
        explicit tfloat16_vec(float v) : x(v), y(v), z(v), w(v) {}
        explicit tfloat16_vec(const float4& v) : x(v.x), y(v.y), z(v.z), w(v.w) {}
        explicit tfloat16_vec(float v1, float v2, float v3, float v4) : x(v1), y(v2), z(v3), w(v4) {}
        explicit operator float4() const { return float4(float(x), float(y), float(z), float(w)); }

        // Access
        float16_t& operator[](size_t i) { FALCOR_ASSERT(i < length()); return (&x)[i]; }
        const float16_t& operator[](size_t i) const { FALCOR_ASSERT(i < length()); return (&x)[i]; }

        bool operator==(const tfloat16_vec& other) const { return x == other.x && y == other.y && z == other.z && w == other.w; }
        bool operator!=(const tfloat16_vec& other) const { return x != other.x || y != other.y || z != other.z || w != other.w; }

        static constexpr size_t length() { return 4; }
    };

    using float16_t2 = tfloat16_vec<2>;
    using float16_t3 = tfloat16_vec<3>;
    using float16_t4 = tfloat16_vec<4>;

    inline std::string to_string(const float16_t2& v) { return "float16_t2(" + to_string(v.x) + "," + to_string(v.y) + ")"; }
    inline std::string to_string(const float16_t3& v) { return "float16_t3(" + to_string(v.x) + "," + to_string(v.y) + "," + to_string(v.z) + ")"; }
    inline std::string to_string(const float16_t4& v) { return "float16_t4(" + to_string(v.x) + "," + to_string(v.y) + "," + to_string(v.z) + "," + to_string(v.w) + ")"; }
}
