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
#include "Core/API/fwd.h"
#include "Core/API/Formats.h"
#include "Utils/Math/Vector.h"
#include "Scene/Material/DiffuseSpecularData.slang"
#include <filesystem>
#include <memory>

namespace Falcor
{
    class Device;

    /** Class for loading a measured material from the MERL BRDF database.
        Additional metadata is loaded along with the BRDF if available.
    */
    class FALCOR_API MERLFile
    {
    public:
        struct Desc
        {
            std::string name;                   ///< Name of the BRDF.
            std::filesystem::path path;         ///< Full path to the loaded BRDF.
            DiffuseSpecularData extraData = {}; ///< Parameters for a best fit BRDF approximation.
        };

        static constexpr ResourceFormat kAlbedoLUTFormat = ResourceFormat::RGBA32Float;

        MERLFile() = default;

        /** Constructs a new object and loads a MERL BRDF. Throws on error.
            \param[in] path Path to the binary MERL file.
        */
        MERLFile(const std::filesystem::path& path);

        /** Loads a MERL BRDF.
            \param[in] path Path to the binary MERL file.
            \return True if the BRDF was successfully loaded.
        */
        bool loadBRDF(const std::filesystem::path& path);

        /** Prepare an albedo lookup table.
            The table is loaded from disk or recomputed if needed.
            \param[in] pDevice The device.
            \return Albedo lookup table that can be used with `kAlbedoLUTFormat`.
        */
        const std::vector<float4>& prepareAlbedoLUT(ref<Device> pDevice);

        const Desc& getDesc() const { return mDesc; }
        const std::vector<float3>& getData() const { return mData; }

    private:
        void prepareData(const int dims[3], const std::vector<double>& data);
        void computeAlbedoLUT(ref<Device> pDevice, const size_t binCount);

        Desc mDesc;                     ///< BRDF description and sampling parameters.
        std::vector<float3> mData;      ///< BRDF data in RGB float format.
        std::vector<float4> mAlbedoLUT; ///< Precomputed albedo lookup table.
    };
}
