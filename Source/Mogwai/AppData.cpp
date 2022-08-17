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
#include "Falcor.h"
#include "AppData.h"
#define RAPIDJSON_HAS_STDSTRING 1
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/error/en.h>
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

    void AppData::addRecentScript(const std::filesystem::path& path)
    {
        addRecentPath(mRecentScripts, path);
    }

    void AppData::addRecentScene(const std::filesystem::path& path)
    {
        addRecentPath(mRecentScenes, path);
    }

    void AppData::addRecentPath(std::vector<std::filesystem::path>& paths, const std::filesystem::path& path)
    {
        if (!std::filesystem::exists(path)) return;
        std::filesystem::path fullPath = std::filesystem::canonical(path);
        paths.erase(std::remove(paths.begin(), paths.end(), fullPath), paths.end());
        paths.insert(paths.begin(), fullPath);
        if (paths.size() > kMaxRecentFiles) paths.resize(kMaxRecentFiles);
        save();
    }

    void AppData::removeNonExistingPaths(std::vector<std::filesystem::path>& paths)
    {
        paths.erase(std::remove_if(paths.begin(), paths.end(), [](const auto& path) {
            // Remove path if file does not exist.
            if (!std::filesystem::exists(path)) return true;
            auto canonicalPath = std::filesystem::canonical(path);
            // Remove path if not in canonical form.
            return path != canonicalPath;
        }), paths.end());
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
            logWarning("Failed to parse Mogwai settings file '{}': {}", path, rapidjson::GetParseError_En(document.GetParseError()));
            return;
        }

        auto readPathArray = [](const rapidjson::Value& value)
        {
            std::vector<std::filesystem::path> paths;
            if (value.IsArray())
            {
                for (const auto& item : value.GetArray())
                {
                    if (item.IsString()) paths.push_back(item.GetString());
                }
            }
            return paths;
        };

        mRecentScripts = readPathArray(document[kRecentScripts]);
        mRecentScenes = readPathArray(document[kRecentScenes]);

        removeNonExistingPaths(mRecentScripts);
        removeNonExistingPaths(mRecentScenes);
    }

    void AppData::saveToFile(const std::filesystem::path& path)
    {
        rapidjson::Document document;
        document.SetObject();
        auto& allocator = document.GetAllocator();

        auto writePathArray = [&allocator](const std::vector<std::filesystem::path>& paths)
        {
            rapidjson::Value value(rapidjson::kArrayType);
            for (const auto& path : paths) value.PushBack(rapidjson::Value(path.string(), allocator), allocator);
            return value;
        };

        document.AddMember(kRecentScripts, writePathArray(mRecentScripts), allocator);
        document.AddMember(kRecentScenes, writePathArray(mRecentScenes), allocator);

        std::ofstream ofs(path);
        if (!ofs.good()) return;

        rapidjson::OStreamWrapper osw(ofs);
        rapidjson::PrettyWriter<rapidjson::OStreamWrapper> writer(osw);
        document.Accept(writer);
    }
}
