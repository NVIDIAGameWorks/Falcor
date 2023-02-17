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
#include <nlohmann/json.hpp>
#include <fstream>

using json = nlohmann::json;

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
        std::ifstream ifs(path);
        if (!ifs.good()) return;

        auto readPathArray = [](const json& j)
        {
            std::vector<std::string> strings = j.get<std::vector<std::string>>();
            std::vector<std::filesystem::path> paths;
            std::transform(strings.begin(), strings.end(), std::back_inserter(paths), [](const std::string& str) { return str; });
            return paths;
        };

        try
        {
            const json j = json::parse(ifs);
            mRecentScripts = readPathArray(j[kRecentScripts]);
            mRecentScenes = readPathArray(j[kRecentScenes]);
        }
        catch (const std::exception& e)
        {
            logWarning("Failed to parse Mogwai settings file '{}': {}", path, e.what());
        }

        removeNonExistingPaths(mRecentScripts);
        removeNonExistingPaths(mRecentScenes);
    }

    void AppData::saveToFile(const std::filesystem::path& path)
    {
        auto getPathArray = [](const std::vector<std::filesystem::path>& paths)
        {
            std::vector<std::string> strings;
            std::transform(paths.begin(), paths.end(), std::back_inserter(strings), [](const std::filesystem::path& path) { return path.string(); });
            return json(strings);
        };

        json j = json::object();
        j[kRecentScripts] = getPathArray(mRecentScripts);
        j[kRecentScenes] = getPathArray(mRecentScenes);

        std::ofstream ofs(path);
        if (!ofs.good()) return;

        ofs << j.dump(4);
    }
}
