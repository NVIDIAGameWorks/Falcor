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
#include "LightProfile.h"
#include "Utils/Algorithm/ComputeParallelReduction.h"
#include "Core/Platform/OS.h"

#include <filesystem>
#include <fstream>

namespace Falcor
{
    namespace
    {
        const uint32_t kBakeResolution = 256;

        const char kBakeIesProfileFile[] = "Scene/Lights/BakeIesProfile.cs.slang";
        ComputePass::SharedPtr pBakePass;

        const char* kSupportedProfiles[] = {
            "IESNA:LM-63-1986",
            "IESNA:LM-63-1991",
            "IESNA91",
            "IESNA:LM-63-1995",
            "IESNA:LM-63-2002",
            "ERCO Leuchten GmbH  BY: ERCO/LUM650/8701",
            "ERCO Leuchten GmbH"
        };

        // See https://docs.agi32.com/PhotometricToolbox/Content/Open_Tool/iesna_lm-63_format.htm for the format reference. Pasted below.

        /*
        Each line marked with an asterisk must begin a new line.
        Descriptions enclosed by the brackets"<" and ">" refer to the actual data stored on that line.
        Lines marked with an "at sign" @ appear only if TILT=INCLUDE.

        All data is in standard ASCII format.

        *IESNA:LM-63-2002
        *<keyword [TEST]>
        *<keyword [TESTLAB]>
        *<keyword [ISSUEDATE]>
        *<keyword [MANUFAC]>
        *<keyword 5>
        "
        *<keyword n>
        *TILT=<filespec> or INCLUDE or NONE
        @ *<lamp to luminaire geometry>
        @ *<# of pairs of angles and multiplying factors>
        @ *<angles>
        @ *<multiplying factors>
        * <# lamps> <lumens/lamp> <multiplier> <# vertical angles> <# horizontal angles>
        <photometric type> <units type> <width> <length> <height>
        * <ballast factor> <ballast lamp factor> <input watts>
        * <vertical angles>
        * <horizontal angles>
        * <candela values for all vertical angles at first horizontal angle>
        * <candela values for all vertical angles at second horizontal angle>
        * "
        * "
        <candela values for all vertical angles at last horizontal angle>
        */

        enum class IesStatus
        {
            Success,
            UnsupportedProfile,
            UnsupportedTilt,
            WrongDataSize,
            InvalidData
        };

        IesStatus parseIesFile(char* fileData, std::vector<float>& numericData, float& maxCandelas)
        {
            // count whitespace to get a rough estimate of the number of floats stored
            int numWhitespace = 0;
            for (char* p = fileData; *p; p++)
            {
                if (*p == ' ')
                    ++numWhitespace;
            }

            // parse the header line by line
            const char* lineDelimiters = "\r\n";
            const char* dataDelimiters = "\r\n\t ";
            char* line = strtok(fileData, lineDelimiters);
            int lineNumber = 1;

            while (line)
            {
                if (lineNumber == 1)
                {
                    bool profileFound = false;
                    for (const char* profile : kSupportedProfiles)
                    {
                        if (strstr(line, profile))
                        {
                            profileFound = true;
                            break;
                        }
                    }

                    if (!profileFound)
                    {
                        return IesStatus::UnsupportedProfile;
                    }
                }
                else
                {
                    if (strstr(line, "TILT=NONE") == line)
                    {
                        break;
                    }
                    else if (strstr(line, "TILT=") == line)
                    {
                        return IesStatus::UnsupportedTilt;
                    }
                }

                line = strtok(NULL, lineDelimiters);
                ++lineNumber;
            }

            numericData.reserve(numWhitespace);
            while ((line = strtok(NULL, dataDelimiters)))
            {
                float value = 0.f;
                if (sscanf(line, "%f", &value) == 1)
                    numericData.push_back(value);
            }

            if (numericData.size() < 16)
            {
                return IesStatus::WrongDataSize;
            }

            int numLamps = int(numericData[0]);
            int numVerticalAngles = int(numericData[3]);
            int numHorizontalAngles = int(numericData[4]);
            int headerSize = 13;

            int expectedDataSize = headerSize + numHorizontalAngles + numVerticalAngles + numHorizontalAngles * numVerticalAngles;
            if (numericData.size() != expectedDataSize)
            {
                return IesStatus::WrongDataSize;
            }

            maxCandelas = 0.f;
            for (int index = headerSize + numHorizontalAngles + numVerticalAngles; index < expectedDataSize; index++)
                maxCandelas = std::max(maxCandelas, numericData[index]);

            return IesStatus::Success;
        }
    }


    LightProfile::LightProfile(const std::string& name, const std::vector<float>& rawData)
        : mName(name)
        , mRawData(rawData)
    {}

    LightProfile::SharedPtr LightProfile::createFromIesProfile(const std::filesystem::path& filename, bool normalize)
    {
        std::filesystem::path fullpath;
        if (!findFileInDataDirectories(filename, fullpath))
        {
            logWarning("Error when loading light profile. Can't find file '{}'", filename);
            return nullptr;
        }

        std::ifstream ifs(fullpath);
        std::string str;
        ifs.seekg(0, std::ios::end);
        str.reserve(ifs.tellg());
        ifs.seekg(0, std::ios::beg);
        str.assign((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());

        std::vector<float> numericData;
        float maxCandelas;
        IesStatus status = parseIesFile(str.data(), numericData, maxCandelas);
        switch (status)
        {
        case IesStatus::UnsupportedProfile:
        case IesStatus::UnsupportedTilt:
        case IesStatus::WrongDataSize:
        case IesStatus::InvalidData:
            logWarning("Error while loading IES profile from '{}'.", fullpath);
            return nullptr;
        }

        // Stash the normalization factor in data[0], we don't use that anyway
        numericData[0] = normalize ? (1.f / maxCandelas) : 1.f;

        std::string name = fullpath.filename().string();

        return SharedPtr(new LightProfile(name, numericData));
    }

    void LightProfile::bake(RenderContext* pRenderContext)
    {
        if (!pBakePass)
        {
            pBakePass = ComputePass::create(kBakeIesProfileFile, "main");
        }

        auto pBuffer = Buffer::createTyped<float>((uint32_t)mRawData.size(), ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, mRawData.data());
        mpTexture = Texture::create2D(kBakeResolution, kBakeResolution, ResourceFormat::R16Float, 1, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
        auto pFluxTexture = Texture::create2D(kBakeResolution, kBakeResolution, ResourceFormat::R32Float, 1, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);

        auto var = pBakePass->getRootVar();
        var["gIesData"] = pBuffer;
        var["gTexture"] = mpTexture;
        var["gFluxTexture"] = pFluxTexture;
        var["CB"]["gBakeResolution"] = kBakeResolution;
        pBakePass->execute(pRenderContext, kBakeResolution, kBakeResolution);

        float4 fluxFactor;
        auto pReduction = ComputeParallelReduction::create();
        pReduction->execute<float4>(pRenderContext, pFluxTexture, ComputeParallelReduction::Type::Sum, &fluxFactor);
        mFluxFactor = fluxFactor.x;

        Sampler::Desc desc;
        desc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear);
        mpSampler = Sampler::create(desc);
    }

    void LightProfile::setShaderData(const ShaderVar& var) const
    {
        var["fluxFactor"] = mFluxFactor;
        var["texture"] = mpTexture;
        var["sampler"] = mpSampler;
    }

    void LightProfile::renderUI(Gui::Widgets& widget)
    {
        widget.text("Light Profile: " + mName);
        if (mpTexture)
        {
            widget.text("Texture info: " + std::to_string(mpTexture->getWidth()) + "x" + std::to_string(mpTexture->getHeight()) + " (" + to_string(mpTexture->getFormat()) + ")");
            widget.image("Texture", mpTexture, float2(100.f));
        }
        widget.text("Flux factor: " + std::to_string(mFluxFactor));
    }
}
