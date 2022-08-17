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
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace Falcor
{
    /** Class representing a measured material file from the RGL BRDF database.
    */
    class RGLFile
    {
    public:
        /** There are many more field types, but none that we need. Ignore all other types.
        */
        enum FieldType
        {
            UInt8 = 1,
            UInt32 = 5,
            Float32 = 10,
        };

        struct Field
        {
            std::string name;
            FieldType type;
            uint32_t dim;
            int64_t numElems;
            std::unique_ptr<uint64_t[]> shape;
            std::unique_ptr<uint8_t[]> data;
        };

        /** Collected set of fields necessary to render the BRDF.
        */
        struct MeasurementData
        {
            const Field* thetaI;
            const Field* phiI;
            const Field* sigma;
            const Field* ndf;
            const Field* vndf;
            const Field* rgb;
            const Field* luminance;
            bool isotropic;
            std::string description;
        };

        RGLFile() = default;

        /** Loads RGL measured BRDF file and validates contents. Throws RuntimeError on failure.
        */
        RGLFile(std::ifstream& in);

        void saveFile(std::ofstream& out) const;

        const MeasurementData& data() const
        {
            return mMeasurement;
        }

        void addField(const std::string& name, FieldType type, const std::vector<uint32_t>& shape, const void* data);

    private:
        std::unordered_map<std::string, int> mFieldMap;
        std::vector<Field> mFields;
        MeasurementData mMeasurement;

        /** Make sure all required fields are present and have correct shape and dimensions,
            then populates mMeasurement field if all fields are correct.
            Throws RuntimeError on validation error.
        */
        void validate();

        const Field* getField(const std::string& name) const;

        static size_t fieldSize(FieldType type);
    };
}
