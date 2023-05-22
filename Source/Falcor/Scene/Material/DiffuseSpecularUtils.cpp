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
#include "DiffuseSpecularUtils.h"
#include "DiffuseSpecularData.slang"
#include "Utils/Logger.h"
#include "Utils/Color/ColorHelpers.slang"
#include <nlohmann/json.hpp>
#include <fstream>

using json = nlohmann::json;

namespace Falcor
{
    namespace
    {
        bool get_to(const json& j, float3& v)
        {
            if (!j.is_array())
                return false;
            if (j.size() != v.length())
                return false;
            j[0].get_to(v.x);
            j[1].get_to(v.y);
            j[2].get_to(v.z);
            return true;
        }
    }

    bool DiffuseSpecularUtils::loadJSONData(const std::filesystem::path& path, DiffuseSpecularData& data)
    {
        // Set default parameters for json data is not available.
        // The default is a mix between diffuse and specular with medium roughness.
        data = {};
        data.baseColor = float3(0.5f);
        data.roughness = 0.5f;
        data.metallic = 0.5f;

        // Try loading data from json file.
        std::ifstream ifs(path);
        if (!ifs.good())
        {
            logWarning("DiffuseSpecularUtils: Failed to open file '{}' for reading.", path);
            return false;
        }

        try
        {
            json doc = json::parse(ifs);

            DiffuseSpecularData d = {};
            float3 baseColorSRGB = {};
            get_to(doc["base_color_srgb"], baseColorSRGB);
            d.baseColor = sRGBToLinear(baseColorSRGB);
            d.roughness = doc["roughness"];
            d.specular = doc["specular"];
            d.metallic = doc["metallic"];
            d.lossValue = doc["loss_value"];

            data = d;
        }
        catch (const json::exception& e)
        {
            logWarning("DiffuseSpecularUtils: Error ({}) when parsing file '{}'.", std::string(e.what()), path);
            return false;
        }

        return true;
    }

    bool DiffuseSpecularUtils::renderUI(Gui::Widgets& widget, DiffuseSpecularData& data)
    {
        DiffuseSpecularData prevData = data;
        widget.rgbColor("baseColor", data.baseColor);
        widget.var("roughness", data.roughness, 0.f, 1.f);
        widget.var("metallic", data.metallic, 0.f, 1.f);
        widget.var("specular", data.specular, 0.f, 1.f);

        return data != prevData;
    }
}
