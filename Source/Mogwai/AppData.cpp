/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include "stdafx.h"
#include "AppData.h"
#define RAPIDJSON_HAS_STDSTRING 1
#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"
#include "rapidjson/ostreamwrapper.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/error/en.h"
#include <fstream>

using namespace Falcor;

namespace Mogwai
{
    namespace
    {
        const char kRecentScripts[] = "recentScripts";
        const char kRecentScenes[] = "recentScenes";

        size_t kMaxRecentFiles = 25;
    }

    AppData::AppData(const std::filesystem::path& path)
        : mPath(path.lexically_normal())
    {
        // Make sure directories exist.
        std::filesystem::create_directories(mPath.parent_path());

        loadFromFile(mPath);
    }

    void AppData::addRecentScript(const std::string& filename)
    {
        addRecentFile(mRecentScripts, filename);
    }

    void AppData::addRecentScene(const std::string& filename)
    {
        addRecentFile(mRecentScenes, filename);
    }

    void AppData::addRecentFile(std::vector<std::string>& recentFiles, const std::string& filename)
    {
        std::filesystem::path path = std::filesystem::absolute(filename);
        if (!std::filesystem::exists(path)) return;
        std::string entry = canonicalizeFilename(path.string());
        recentFiles.erase(std::remove(recentFiles.begin(), recentFiles.end(), entry), recentFiles.end());
        recentFiles.insert(recentFiles.begin(), entry);
        if (recentFiles.size() > kMaxRecentFiles) recentFiles.resize(kMaxRecentFiles);
        save();
    }

    void AppData::removeNonExistingFiles(std::vector<std::string>& files)
    {
        files.erase(std::remove_if(files.begin(), files.end(), [](const std::string& filename) {
            return !doesFileExist(filename);
        }), files.end());
    }

    void AppData::save()
    {
        saveToFile(mPath);
    }

    void AppData::loadFromFile(const std::filesystem::path& path)
    {
        rapidjson::Document document;

        std::ifstream ifs(path);
        if (!ifs.good()) return;

        rapidjson::IStreamWrapper isw(ifs);
        document.ParseStream(isw);

        if (document.HasParseError())
        {
            logWarning("Failed to parse Mogwai settings file " + path.string() + ": " + rapidjson::GetParseError_En(document.GetParseError()));
            return;
        }

        auto readStringArray = [](const rapidjson::Value& value)
        {
            std::vector<std::string> strings;
            if (value.IsArray())
            {
                for (const auto& item : value.GetArray())
                {
                    if (item.IsString()) strings.push_back(item.GetString());
                }
            }
            return strings;
        };

        mRecentScripts = readStringArray(document[kRecentScripts]);
        mRecentScenes = readStringArray(document[kRecentScenes]);

        removeNonExistingFiles(mRecentScripts);
        removeNonExistingFiles(mRecentScenes);
    }

    void AppData::saveToFile(const std::filesystem::path& path)
    {
        rapidjson::Document document;
        document.SetObject();
        auto& allocator = document.GetAllocator();

        auto writeStringArray = [&allocator](const std::vector<std::string>& strings)
        {
            rapidjson::Value value(rapidjson::kArrayType);
            for (const auto& item : strings) value.PushBack(rapidjson::StringRef(item), allocator);
            return value;
        };

        document.AddMember(kRecentScripts, writeStringArray(mRecentScripts), allocator);
        document.AddMember(kRecentScenes, writeStringArray(mRecentScenes), allocator);

        std::ofstream ofs(path);
        if (!ofs.good()) return;

        rapidjson::OStreamWrapper osw(ofs);
        rapidjson::PrettyWriter<rapidjson::OStreamWrapper> writer(osw);
        document.Accept(writer);
    }
}
