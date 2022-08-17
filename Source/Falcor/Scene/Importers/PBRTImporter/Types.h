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

// This code is based on pbrt:
// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#pragma once
#include "Utils/Math/Vector.h"
#include "Utils/Color/Spectrum.h"
#include <fmt/format.h>
#include <filesystem>
#include <functional>
#include <string_view>
#include <string>
#include <variant>

namespace Falcor
{
    namespace pbrt
    {
        using Float = float;

        /** File location.
        */
        struct FileLoc
        {
            FileLoc() = default;
            FileLoc(std::string_view filename) : filename(filename) {}
            std::string toString() const
            {
                return fmt::format("{}:{}:{}", filename, line, column);
            }

            std::string_view filename;
            uint32_t line = 1;
            uint32_t column = 0;
        };

        /** Placeholder for representing RGB color space.
            RGB in Rec.709. is currently always used.
        */
        struct RGBColorSpace;

        /** Resolve a relative file path.
        */
        using Resolver = std::function<std::filesystem::path(const std::filesystem::path& path)>;

        /** Spectrum types.
        */
        enum class SpectrumType { Illuminant, Albedo, Unbounded };

        /** Spectrum holder.
        */
        using Spectrum = std::variant<
            float3, // RGB value
            PiecewiseLinearSpectrum,
            BlackbodySpectrum
        >;
    }
}
