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

#include "Parameters.h"
#include "Helpers.h"
#include "Core/Assert.h"

namespace Falcor
{
    namespace pbrt
    {
        // --------------------------------------------------------------------
        // ParsedParameter
        // --------------------------------------------------------------------

        void ParsedParameter::addFloat(Float f)
        {
            FALCOR_ASSERT(ints.empty() && strings.empty() && bools.empty());
            floats.push_back(f);
        }

        void ParsedParameter::addInt(int i)
        {
            FALCOR_ASSERT(floats.empty() && strings.empty() && bools.empty());
            ints.push_back(i);
        }

        void ParsedParameter::addString(std::string_view s)
        {
            FALCOR_ASSERT(floats.empty() && ints.empty() && bools.empty());
            strings.push_back({s.begin(), s.end()});
        }

        void ParsedParameter::addBool(bool b)
        {
            FALCOR_ASSERT(floats.empty() && ints.empty() && strings.empty());
            bools.push_back(b);
        }

        std::string ParsedParameter::toString() const
        {
            std::string str;
            str += fmt::format("{} {} [ ", type, name);
            if (!floats.empty()) for (Float f : floats) str += fmt::format("{} ", f);
            else if (!ints.empty()) for (int i : ints) str += fmt::format("{} ", i);
            else if (!strings.empty()) for (const auto& s : strings) str += '\"' + s + "\" ";
            else if (!bools.empty()) for (bool b : bools) str += b ? "true " : "false ";
            str += "]";
            return str;
        }

        // --------------------------------------------------------------------
        // ParameterTypeTraits
        // --------------------------------------------------------------------

        template<>
        struct ParameterTypeTraits<ParameterType::Float>
        {
            static constexpr char typeName[] = "float";
            static constexpr size_t perItemCount = 1;
            using ReturnType = Float;
            static Float convert(const Float* v) { return *v; }
            static const auto& getValues(const ParsedParameter& param) { return param.floats; }
        };

        template<>
        struct ParameterTypeTraits<ParameterType::Int>
        {
            static constexpr char typeName[] = "integer";
            static constexpr size_t perItemCount = 1;
            using ReturnType = int;
            static int convert(const int* v) { return *v; }
            static const auto& getValues(const ParsedParameter& param) { return param.ints; }
        };

        template<>
        struct ParameterTypeTraits<ParameterType::Bool>
        {
            static constexpr char typeName[] = "bool";
            static constexpr size_t perItemCount = 1;
            using ReturnType = uint8_t;
            static bool convert(const uint8_t* v) { return *v; }
            static const auto& getValues(const ParsedParameter& param) { return param.bools; }
        };

        template<>
        struct ParameterTypeTraits<ParameterType::String>
        {
            static constexpr char typeName[] = "string";
            static constexpr size_t perItemCount = 1;
            using ReturnType = std::string;
            static std::string convert(const std::string* v) { return *v; }
            static const auto& getValues(const ParsedParameter& param) { return param.strings; }
        };

        template<>
        struct ParameterTypeTraits<ParameterType::Point2>
        {
            static constexpr char typeName[] = "point2";
            static constexpr size_t perItemCount = 2;
            using ReturnType = float2;
            static float2 convert(const float* v) { return float2(v[0], v[1]); }
            static const auto& getValues(const ParsedParameter& param) { return param.floats; }
        };

        template<>
        struct ParameterTypeTraits<ParameterType::Vector2>
        {
            static constexpr char typeName[] = "vector2";
            static constexpr size_t perItemCount = 2;
            using ReturnType = float2;
            static float2 convert(const float* v) { return float2(v[0], v[1]); }
            static const auto& getValues(const ParsedParameter& param) { return param.floats; }
        };

        template<>
        struct ParameterTypeTraits<ParameterType::Point3>
        {
            static constexpr char typeName[] = "point3";
            static constexpr size_t perItemCount = 3;
            using ReturnType = float3;
            static float3 convert(const float* v) { return float3(v[0], v[1], v[2]); }
            static const auto& getValues(const ParsedParameter& param) { return param.floats; }
        };

        template<>
        struct ParameterTypeTraits<ParameterType::Vector3>
        {
            static constexpr char typeName[] = "vector3";
            static constexpr size_t perItemCount = 3;
            using ReturnType = float3;
            static float3 convert(const float* v) { return float3(v[0], v[1], v[2]); }
            static const auto& getValues(const ParsedParameter& param) { return param.floats; }
        };

        template<>
        struct ParameterTypeTraits<ParameterType::Normal>
        {
            static constexpr char typeName[] = "normal";
            static constexpr size_t perItemCount = 3;
            using ReturnType = float3;
            static float3 convert(const float* v) { return float3(v[0], v[1], v[2]); }
            static const auto& getValues(const ParsedParameter& param) { return param.floats; }
        };

        // --------------------------------------------------------------------
        // ParameterDictionary
        // --------------------------------------------------------------------

        ParameterDictionary::ParameterDictionary(ParsedParameterVector params, const RGBColorSpace* pColorSpace)
            : mParams(params)
            , mpColorSpace(pColorSpace)
        {
        }

        ParameterDictionary::ParameterDictionary(ParsedParameterVector params1, ParsedParameterVector params2, const RGBColorSpace* pColorSpace)
            : mParams(params1)
            , mpColorSpace(pColorSpace)
        {
            mParams.insert(mParams.end(), params2.begin(), params2.end());
        }

        FileLoc ParameterDictionary::getParameterLoc(const std::string& name) const
        {
            auto p = findParameter(name);
            if (!p) throw RuntimeError("Parameter not found!");
            return p->loc;
        }

        bool ParameterDictionary::hasParameter(const std::string& name) const
        {
            auto p = findParameter(name);
            return p != nullptr;
        }

        template<ParameterType PT>
        bool ParameterDictionary::hasParameterWithType(const std::string& name) const
        {
            using traits = ParameterTypeTraits<PT>;
            return std::any_of(mParams.begin(), mParams.end(), [=](const auto& param) { return param.name == name && param.type == traits::typeName; });
        }

        bool ParameterDictionary::hasFloat(const std::string& name) const
        {
            return hasParameterWithType<ParameterType::Float>(name);
        }

        bool ParameterDictionary::hasInt(const std::string& name) const
        {
            return hasParameterWithType<ParameterType::Int>(name);
        }

        bool ParameterDictionary::hasString(const std::string& name) const
        {
            return hasParameterWithType<ParameterType::String>(name);
        }

        bool ParameterDictionary::hasBool(const std::string& name) const
        {
            return hasParameterWithType<ParameterType::Bool>(name);
        }

        bool ParameterDictionary::hasPoint2(const std::string& name) const
        {
            return hasParameterWithType<ParameterType::Point2>(name);
        }

        bool ParameterDictionary::hasVector2(const std::string& name) const
        {
            return hasParameterWithType<ParameterType::Vector2>(name);
        }

        bool ParameterDictionary::hasPoint3(const std::string& name) const
        {
            return hasParameterWithType<ParameterType::Point3>(name);
        }

        bool ParameterDictionary::hasVector3(const std::string& name) const
        {
            return hasParameterWithType<ParameterType::Vector3>(name);
        }

        bool ParameterDictionary::hasNormal(const std::string& name) const
        {
            return hasParameterWithType<ParameterType::Normal>(name);
        }

        bool ParameterDictionary::hasSpectrum(const std::string& name) const
        {
            auto p = findParameter(name);
            return p && (p->type == "rgb" || p->type == "blackbody" || p->type == "spectrum");
        }

        bool ParameterDictionary::hasTexture(const std::string& name) const
        {
            auto p = findParameter(name);
            return p && p->type == "texture";
        }

        Float ParameterDictionary::getFloat(const std::string& name, Float def) const
        {
            return lookupSingle<ParameterType::Float>(name, def);
        }

        int ParameterDictionary::getInt(const std::string& name, int def) const
        {
            return lookupSingle<ParameterType::Int>(name, def);
        }

        std::string ParameterDictionary::getString(const std::string& name, const std::string& def) const
        {
            return lookupSingle<ParameterType::String>(name, def);
        }

        bool ParameterDictionary::getBool(const std::string& name, bool def) const
        {
            return lookupSingle<ParameterType::Bool>(name, def);
        }

        float2 ParameterDictionary::getPoint2(const std::string& name, float2 def) const
        {
            return lookupSingle<ParameterType::Point2>(name, def);
        }

        float2 ParameterDictionary::getVector2(const std::string& name, float2 def) const
        {
            return lookupSingle<ParameterType::Vector2>(name, def);
        }

        float3 ParameterDictionary::getPoint3(const std::string& name, float3 def) const
        {
            return lookupSingle<ParameterType::Point3>(name, def);
        }

        float3 ParameterDictionary::getVector3(const std::string& name, float3 def) const
        {
            return lookupSingle<ParameterType::Vector3>(name, def);
        }

        float3 ParameterDictionary::getNormal(const std::string& name, float3 def) const
        {
            return lookupSingle<ParameterType::Normal>(name, def);
        }

        Spectrum ParameterDictionary::getSpectrum(const std::string& name, const Spectrum& def, Resolver resolver) const
        {
            if (const ParsedParameter* p = findParameter(name))
            {
                auto s = extractSpectrumArray(*p, resolver);
                if (!s.empty())
                {
                    if (s.size() > 1)
                    {
                        throwError(p->loc, "More than one value provided for parameter '{}'.", p->name);
                    }
                    return s[0];
                }
            }

            return def;
        }

        std::string ParameterDictionary::getTexture(const std::string& name) const
        {
            if (const ParsedParameter* p = findParameter(name); p && p->type == "texture")
            {
                if (p->strings.empty())
                {
                    throwError(p->loc, "No string values provided for parameter '{}'.", name);
                }
                if (p->strings.size() > 1)
                {
                    throwError(p->loc, "More than one value provided for parameter '{}'.", name);
                }
                return p->strings[0];
            }

            return "";
        }

        std::vector<Float> ParameterDictionary::getFloatArray(const std::string& name) const
        {
            return lookupArray<ParameterType::Float>(name);
        }

        std::vector<int> ParameterDictionary::getIntArray(const std::string& name) const
        {
            return lookupArray<ParameterType::Int>(name);
        }

        std::vector<std::string> ParameterDictionary::getStringArray(const std::string& name) const
        {
            return lookupArray<ParameterType::String>(name);
        }

        std::vector<uint8_t> ParameterDictionary::getBoolArray(const std::string& name) const
        {
            return lookupArray<ParameterType::Bool>(name);
        }

        std::vector<float2> ParameterDictionary::getPoint2Array(const std::string& name) const
        {
            return lookupArray<ParameterType::Point2>(name);
        }

        std::vector<float2> ParameterDictionary::getVector2Array(const std::string& name) const
        {
            return lookupArray<ParameterType::Vector2>(name);
        }

        std::vector<float3> ParameterDictionary::getPoint3Array(const std::string& name) const
        {
            return lookupArray<ParameterType::Point3>(name);
        }

        std::vector<float3> ParameterDictionary::getVector3Array(const std::string& name) const
        {
            return lookupArray<ParameterType::Vector3>(name);
        }

        std::vector<float3> ParameterDictionary::getNormalArray(const std::string& name) const
        {
            return lookupArray<ParameterType::Normal>(name);
        }

        std::vector<Spectrum> ParameterDictionary::getSpectrumArray(const std::string& name, Resolver resolver) const
        {
            if (const ParsedParameter* p = findParameter(name))
            {
                auto s = extractSpectrumArray(*p, resolver);
                if (!s.empty())
                {
                    return s;
                }
            }

            return {};
        }

        std::string ParameterDictionary::toString() const
        {
            std::string str;
            str += "[\n";
            for (const auto& p : mParams)
            {
                str += "  " + p.toString() + "\n";
            }
            str += "]";
            return str;
        }

        const ParsedParameter* ParameterDictionary::findParameter(const std::string& name) const
        {
            for (const auto& p : mParams)
            {
                if (p.name == name) return &p;
            }

            return nullptr;
        }

        template<typename ReturnType, typename ValuesType, typename C>
        static std::vector<ReturnType> returnArray(const ParsedParameter& param, const ValuesType& values, size_t perItemCount, C convert)
        {
            if (values.empty()) throwError(param.loc, "No values provided for parameter '{}'.", param.name);
            if (values.size() % perItemCount) throwError(param.loc, "Number of values provided for '{}' not a multiple of {}.", param.name, perItemCount);

            size_t count = values.size() / perItemCount;
            std::vector<ReturnType> v(count);
            for (size_t i = 0; i < count; ++i)
            {
                v[i] = convert(&values[perItemCount * i]);
            }
            return v;
        }

        template<ParameterType PT>
        typename ParameterTypeTraits<PT>::ReturnType ParameterDictionary::lookupSingle(const std::string& name, typename ParameterTypeTraits<PT>::ReturnType defaultValue) const
        {
            using traits = ParameterTypeTraits<PT>;
            for (const auto& param : mParams)
            {
                if (param.name != name || param.type != traits::typeName) continue;

                // Extract parameter values.
                const auto& values = traits::getValues(param);

                // Issue error if incorrect number of parameter values were provided.
                if (values.empty()) throwError(param.loc, "No values provided for parameter '{}'.", name);
                if (values.size() > traits::perItemCount) throwError(param.loc, "Expected {} values for parameter '{}'.", traits::perItemCount, name);

                // Return values.
                return traits::convert(values.data());
            }

            return defaultValue;
        }

        template<ParameterType PT>
        std::vector<typename ParameterTypeTraits<PT>::ReturnType> ParameterDictionary::lookupArray(const std::string& name) const
        {
            using traits = ParameterTypeTraits<PT>;
            for (const auto& param : mParams)
            {
                if (param.name != name || param.type != traits::typeName) continue;

                return returnArray<typename traits::ReturnType>(param, traits::getValues(param), traits::perItemCount, traits::convert);
            }

            return {};
        }

        std::vector<Spectrum> ParameterDictionary::extractSpectrumArray(const ParsedParameter& param, Resolver resolver) const
        {
            if (param.type == "rgb")
            {
                return returnArray<Spectrum>(
                    param, param.floats, 3,
                    [&param](const Float* v) -> Spectrum {
                        Falcor::float3 rgb(v[0], v[1], v[2]);
                        if (rgb.r < 0.f || rgb.g < 0.f || rgb.b < 0.f)
                        {
                            throwError(param.loc, "RGB parameter '{}' has negative component.", param.name);
                        }
                        return Spectrum(rgb);
                    });
            }
            else if (param.type == "blackbody")
            {
                return returnArray<Spectrum>(
                    param, param.floats, 1,
                    [](const Float* v) -> Spectrum {
                        return Spectrum(BlackbodySpectrum(v[0]));
                    });
            }
            else if (param.type == "spectrum" && !param.floats.empty())
            {
                if (param.floats.size() % 2 != 0)
                {
                    throwError(param.loc, "Found odd number of values for '{}'.", param.name);
                }
                size_t sampleCount = param.floats.size() / 2;
                return returnArray<Spectrum>(
                    param, param.floats, param.floats.size(),
                    [&param, sampleCount](const Float* v) -> Spectrum {
                        std::vector<Float> wavelengths(sampleCount), values(sampleCount);
                        for (size_t i = 0; i < sampleCount; ++i)
                        {
                            if (i > 0 && v[2 * i] <= wavelengths[i - 1])
                            {
                                throwError(param.loc, "Spectrum description invalid: at %d'th entry, wavelengths aren't increasing: %f >= %f.",
                                           i - 1, wavelengths[i - 1], v[2 * i]);
                            }
                            wavelengths[i] = v[2 * i];
                            values[i] = v[2 * i + 1];
                        }
                        return Spectrum(PiecewiseLinearSpectrum(wavelengths, values));
                    });
            }
            else if (param.type == "spectrum" && !param.strings.empty())
            {
                return returnArray<Spectrum>(
                    param, param.strings, 1,
                    [&param, &resolver](const std::string* s) -> Spectrum {
                        auto namedSpectrum = Spectra::getNamedSpectrum(*s);
                        if (namedSpectrum) return Spectrum(*namedSpectrum);

                        auto spectrum = PiecewiseLinearSpectrum::fromFile(resolver(*s));
                        if (!spectrum)
                        {
                            throwError(param.loc, "Unable to read spectrum file '{}'.", *s);
                        }
                        return Spectrum(*spectrum);
                    });
            }

            return {};
        }
    }
}
