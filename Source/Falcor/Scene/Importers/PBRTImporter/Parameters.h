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

#include "Types.h"
#include "Utils/Math/Vector.h"
#include "Utils/Color/Spectrum.h"
#include <string>
#include <string_view>
#include <vector>

namespace Falcor
{
    namespace pbrt
    {
        class ParsedParameter
        {
        public:
            ParsedParameter(FileLoc loc) : loc(loc) {}

            void addFloat(Float f);
            void addInt(int i);
            void addString(std::string_view s);
            void addBool(bool b);

            std::string toString() const;

            // ParsedParameter Public Members
            std::string type;
            std::string name;
            FileLoc loc;
            std::vector<Float> floats;
            std::vector<int> ints;
            std::vector<std::string> strings;
            std::vector<uint8_t> bools;
            mutable bool lookedUp = false;
            mutable const RGBColorSpace* colorSpace = nullptr;
            bool mayBeUnused = false;
        };

        using ParsedParameterVector = std::vector<ParsedParameter>;

        enum class ParameterType
        {
            Float,
            Int,
            String,
            Bool,
            Point2,
            Vector2,
            Point3,
            Vector3,
            Normal,
            Spectrum,
            Texture
        };

        template<ParameterType PT>
        struct ParameterTypeTraits {};

        class ParameterDictionary
        {
        public:
            // ParameterDictionary Public Methods
            ParameterDictionary() = default;
            ParameterDictionary(ParsedParameterVector params, const RGBColorSpace* pColorSpace);
            ParameterDictionary(ParsedParameterVector params1, ParsedParameterVector params2, const RGBColorSpace* pColorSpace);

            const ParsedParameterVector& getParameters() const { return mParams; }
            FileLoc getParameterLoc(const std::string& name) const;

            bool hasParameter(const std::string& name) const;

            template<ParameterType PT>
            bool hasParameterWithType(const std::string& name) const;

            bool hasFloat(const std::string& name) const;
            bool hasInt(const std::string& name) const;
            bool hasString(const std::string& name) const;
            bool hasBool(const std::string& name) const;
            bool hasPoint2(const std::string& name) const;
            bool hasVector2(const std::string& name) const;
            bool hasPoint3(const std::string& name) const;
            bool hasVector3(const std::string& name) const;
            bool hasNormal(const std::string& name) const;
            bool hasSpectrum(const std::string& name) const;
            bool hasTexture(const std::string& name) const;

            Float getFloat(const std::string& name, Float def) const;
            int getInt(const std::string& name, int def) const;
            std::string getString(const std::string& name, const std::string& def) const;
            bool getBool(const std::string& name, bool def) const;
            float2 getPoint2(const std::string& name, float2 def) const;
            float2 getVector2(const std::string& name, float2 def) const;
            float3 getPoint3(const std::string& name, float3 def) const;
            float3 getVector3(const std::string& name, float3 def) const;
            float3 getNormal(const std::string& name, float3 def) const;
            Spectrum getSpectrum(const std::string& name, const Spectrum& def, Resolver resolver) const;
            std::string getTexture(const std::string& name) const;

            std::vector<Float> getFloatArray(const std::string& name) const;
            std::vector<int> getIntArray(const std::string& name) const;
            std::vector<std::string> getStringArray(const std::string& name) const;
            std::vector<uint8_t> getBoolArray(const std::string& name) const;
            std::vector<float2> getPoint2Array(const std::string& name) const;
            std::vector<float2> getVector2Array(const std::string& name) const;
            std::vector<float3> getPoint3Array(const std::string& name) const;
            std::vector<float3> getVector3Array(const std::string& name) const;
            std::vector<float3> getNormalArray(const std::string& name) const;
            std::vector<Spectrum> getSpectrumArray(const std::string& name, Resolver resolver) const;

            std::string toString() const;

        private:
            const ParsedParameter* findParameter(const std::string& name) const;

            template<ParameterType PT>
            typename ParameterTypeTraits<PT>::ReturnType lookupSingle(const std::string& name, typename ParameterTypeTraits<PT>::ReturnType defaultValue) const;

            template<ParameterType PT>
            std::vector<typename ParameterTypeTraits<PT>::ReturnType> lookupArray(const std::string& name) const;

            std::vector<Spectrum> extractSpectrumArray(const ParsedParameter& param, Resolver resolver) const;

            ParsedParameterVector mParams;
            const RGBColorSpace* mpColorSpace;
        };
    }
}
