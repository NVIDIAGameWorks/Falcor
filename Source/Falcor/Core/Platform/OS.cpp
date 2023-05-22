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
#include "OS.h"
#include "SearchDirectories.h"
#include "Core/Errors.h"
#include "Utils/StringUtils.h"
#include "Utils/StringFormatters.h"
#include <backward/backward.hpp> // TODO: Replace with C++20 <stacktrace> when available.
#include <zlib.h>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <regex>

namespace Falcor
{
const std::filesystem::path& getExecutableDirectory()
{
    static std::filesystem::path directory{getExecutablePath().parent_path()};
    return directory;
}

const std::string& getExecutableName()
{
    static std::string name{getExecutablePath().filename().string()};
    return name;
}

inline std::vector<std::filesystem::path> getInitialShaderDirectories()
{
    std::filesystem::path projectDir(_PROJECT_DIR_);

    std::vector<std::filesystem::path> developmentDirectories = {
        // First we search in source folders.
        projectDir,
        projectDir / "..",
        projectDir / ".." / "Tools" / "FalcorTest",
        // Then we search in deployment folder (necessary to pickup NVAPI and other third-party shaders).
        getRuntimeDirectory() / "shaders",
    };

    std::vector<std::filesystem::path> deploymentDirectories = {
        getRuntimeDirectory() / "shaders",
    };

    return isDevelopmentMode() ? developmentDirectories : deploymentDirectories;
}

static std::vector<std::filesystem::path> gShaderDirectories = getInitialShaderDirectories(); // TODO: REMOVEGLOBAL

inline std::vector<std::filesystem::path> getInitialDataDirectories()
{
    std::filesystem::path projectDir(_PROJECT_DIR_);

    std::vector<std::filesystem::path> developmentDirectories = {
        projectDir / ".." / ".." / "data",
        getRuntimeDirectory() / "data",
    };

    std::vector<std::filesystem::path> deploymentDirectories = {getRuntimeDirectory() / "data"};

    std::vector<std::filesystem::path> directories = isDevelopmentMode() ? developmentDirectories : deploymentDirectories;

    // Add development media folder.
    directories.push_back(projectDir / ".." / ".." / "media");

    // Add additional media folders.
    if (auto mediaFolders = getEnvironmentVariable("FALCOR_MEDIA_FOLDERS"))
    {
        auto folders = splitString(*mediaFolders, ";");
        directories.insert(directories.end(), folders.begin(), folders.end());
    }

    return directories;
}

static std::vector<std::filesystem::path> gDataDirectories = getInitialDataDirectories(); // TODO: REMOVEGLOBAL

const std::vector<std::filesystem::path>& getDataDirectoriesList()
{
    return gDataDirectories;
}

void addDataDirectory(const std::filesystem::path& dir, bool addToFront)
{
    FALCOR_CHECK_ARG_MSG(!dir.empty(), "Do not add empty directories to search paths");

    if (std::find_if(
            gDataDirectories.begin(), gDataDirectories.end(), [&dir](const std::filesystem::path& d) { return isSamePath(dir, d); }
        ) == gDataDirectories.end())
    {
        if (addToFront)
        {
            gDataDirectories.insert(gDataDirectories.begin(), dir);
        }
        else
        {
            gDataDirectories.push_back(dir);
        }
    }
}

void removeDataDirectory(const std::filesystem::path& dir)
{
    FALCOR_CHECK_ARG_MSG(!dir.empty(), "Do not remove empty directories to search paths");

    auto it = std::find_if(
        gDataDirectories.begin(), gDataDirectories.end(), [&dir](const std::filesystem::path& d) { return isSamePath(dir, d); }
    );
    if (it != gDataDirectories.end())
    {
        gDataDirectories.erase(it);
    }
}

bool isDevelopmentMode()
{
    static bool devMode = []()
    {
        auto value = getEnvironmentVariable("FALCOR_DEVMODE");
        return value && *value == "1";
    }();

    return devMode;
}

bool isSamePath(const std::filesystem::path& lhs, const std::filesystem::path& rhs)
{
    return std::filesystem::weakly_canonical(lhs) == std::filesystem::weakly_canonical(rhs);
}

bool findFileInDataDirectories(const std::filesystem::path& path, std::filesystem::path& fullPath)
{
    return findFileInDirectories(path, fullPath, getDataDirectoriesList());
}

bool findFileInDirectories(const std::filesystem::path& path, std::filesystem::path& fullPath, const SearchDirectories& directories)
{
    // Check if this is an absolute path.
    if (path.is_absolute())
    {
        if (std::filesystem::exists(path))
        {
            fullPath = std::filesystem::canonical(path);
            return true;
        }
    }

    // Search in other paths.
    for (const auto& dir : directories.get())
    {
        fullPath = dir / path;
        if (std::filesystem::exists(fullPath))
        {
            fullPath = std::filesystem::canonical(fullPath);
            return true;
        }
    }

    return false;
}

std::vector<std::filesystem::path> globFilesInDirectory(
    const std::filesystem::path& path,
    const std::regex& regexPattern,
    bool firstHitOnly
)
{
    std::vector<std::filesystem::path> result;
    if (!std::filesystem::exists(path))
        return result;
    for (const auto& entry : std::filesystem::directory_iterator(path))
    {
        if (!entry.is_regular_file())
            continue;
        std::string filename = entry.path().filename().string();
        if (std::regex_match(filename, regexPattern))
        {
            result.push_back(entry.path());
            if (firstHitOnly)
                return result;
        }
    }

    return result;
}

std::vector<std::filesystem::path> globFilesInDataDirectories(
    const std::filesystem::path& path,
    const std::regex& regexPattern,
    bool firstHitOnly
)
{
    return globFilesInDirectories(path, regexPattern, getDataDirectoriesList(), firstHitOnly);
}
std::vector<std::filesystem::path> globFilesInDirectories(
    const std::filesystem::path& path,
    const std::regex& regexPattern,
    const SearchDirectories& directories,
    bool firstHitOnly
)
{
    std::vector<std::filesystem::path> result;

    // Check if this is an absolute path.
    if (path.is_absolute())
    {
        result = globFilesInDirectory(path, regexPattern, firstHitOnly);
    }
    else
    {
        // Search in other paths.
        for (const auto& dir : directories.get())
        {
            auto fullPath = dir / path;
            std::vector<std::filesystem::path> local = globFilesInDirectory(fullPath, regexPattern, firstHitOnly);
            result.reserve(result.size() + local.size());
            for (auto&& it : local)
                result.push_back(std::move(it));
            if (firstHitOnly && !result.empty())
                break;
        }
    }

    for (auto& it : result)
        it = std::filesystem::canonical(it);

    return result;
}

const std::vector<std::filesystem::path>& getShaderDirectoriesList()
{
    return gShaderDirectories;
}

bool findFileInShaderDirectories(const std::filesystem::path& path, std::filesystem::path& fullPath)
{
    // Check if this is an absolute path.
    if (path.is_absolute())
    {
        if (std::filesystem::exists(path))
        {
            fullPath = std::filesystem::canonical(path);
            return true;
        }
    }

    // Search in other paths.
    for (const auto& dir : gShaderDirectories)
    {
        fullPath = dir / path;
        if (std::filesystem::exists(fullPath))
        {
            fullPath = std::filesystem::canonical(fullPath);
            return true;
        }
    }

    return false;
}

std::filesystem::path findAvailableFilename(const std::string& prefix, const std::filesystem::path& directory, const std::string& extension)
{
    for (uint32_t i = 0; i < (uint32_t)-1; i++)
    {
        std::string newPrefix = prefix + '.' + std::to_string(i);
        std::filesystem::path path = directory / (newPrefix + "." + extension);
        if (!std::filesystem::exists(path))
            return path;
    }
    throw RuntimeError("Failed to find available filename.");
}

std::filesystem::path stripDataDirectories(const std::filesystem::path& path)
{
    if (path.is_relative())
        return path;

    auto canonicalPath = std::filesystem::weakly_canonical(path);

    for (const auto& dir : gDataDirectories)
    {
        auto canonicalDir = std::filesystem::weakly_canonical(dir);

        if (hasPrefix(canonicalPath.string(), canonicalDir.string()))
        {
            return std::filesystem::relative(canonicalPath, canonicalDir);
        }
    }

    return path;
}

bool hasExtension(const std::filesystem::path& path, std::string_view ext)
{
    // Remove leading '.' from ext.
    if (ext.size() > 0 && ext[0] == '.')
        ext.remove_prefix(1);

    std::string pathExt = getExtensionFromPath(path);

    if (ext.size() != pathExt.size())
        return false;

    return std::equal(ext.rbegin(), ext.rend(), pathExt.rbegin(), [](char a, char b) { return std::tolower(a) == std::tolower(b); });
}

std::string getExtensionFromPath(const std::filesystem::path& path)
{
    std::string ext;
    if (path.has_extension())
    {
        ext = path.extension().string();
        // Remove the leading '.' from ext.
        if (ext.size() > 0 && ext[0] == '.')
            ext.erase(0, 1);
        // Convert to lower-case.
        std::transform(ext.begin(), ext.end(), ext.begin(), [](char c) { return std::tolower(c); });
    }
    return ext;
}

std::filesystem::path getTempFilePath()
{
    static std::mutex mutex;
    std::lock_guard<std::mutex> guard(mutex);
    char* name = std::tmpnam(nullptr);
    if (name == nullptr)
    {
        throw RuntimeError("Failed to create temporary file path.");
    }
    return name;
}

std::string readFile(const std::filesystem::path& path)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs)
        throw RuntimeError("Failed to read from file '{}'.", path);
    return std::string((std::istreambuf_iterator<char>(ifs)), (std::istreambuf_iterator<char>()));
}

std::string decompressFile(const std::filesystem::path& path)
{
    std::string compressed = readFile(path);

    z_stream zs = {};
    // MAX_WBITS | 32 to support both zlib or gzip files.
    if (inflateInit2(&zs, MAX_WBITS | 32) != Z_OK)
        throw RuntimeError("inflateInit2 failed while decompressing.");

    zs.next_in = reinterpret_cast<Bytef*>(compressed.data());
    zs.avail_in = (uInt)compressed.size();

    int ret;
    std::vector<char> buffer(128 * 1024);
    std::string decompressed;

    // We can probably assume that the decompressed file is at least as large as the compressed one.
    decompressed.reserve(compressed.size());

    // Get the decompressed bytes blockwise using repeated calls to inflate.
    do
    {
        zs.next_out = reinterpret_cast<Bytef*>(buffer.data());
        zs.avail_out = (uInt)buffer.size();

        ret = inflate(&zs, 0);

        if (decompressed.size() < zs.total_out)
        {
            decompressed.append(buffer.data(), zs.total_out - decompressed.size());
        }
    } while (ret == Z_OK);

    inflateEnd(&zs);

    // Check for errors.
    if (ret != Z_STREAM_END)
    {
        throw RuntimeError("Failure to decompress file '{}' (error: {}).", path, ret);
    }

    return decompressed;
}

std::string getStackTrace(size_t skip, size_t maxDepth)
{
    // Capture stack trace.
    backward::StackTrace st;
    st.load_here(maxDepth == 0 ? 1000 : maxDepth);
    st.skip_n_firsts(skip);

    // We implement our own stack trace formatting here as the default printer in backward is not printing
    // source locations in a way that is parsable by typical IDEs (file:line).
    backward::TraceResolver resolver;
    resolver.load_stacktrace(st);
    std::string result;
    for (size_t i = 0; i < st.size(); ++i)
    {
        auto trace = resolver.resolve(st[i]);

        result += fmt::format(" {}#", i);
        if (!trace.source.filename.empty())
        {
            result += fmt::format(" {} at {}:{}", trace.source.function, trace.source.filename, trace.source.line);
        }
        else
        {
            result += fmt::format(" 0x{:016x} ({})", reinterpret_cast<uintptr_t>(trace.addr), trace.object_function);
            if (!trace.object_filename.empty())
                result += fmt::format(" in {}", trace.object_filename);
        }
        result += "\n";
    }

    return result;
}

} // namespace Falcor
