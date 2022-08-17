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
#include "OS.h"
#include "Core/Errors.h"
#include "Utils/StringUtils.h"
#include "Utils/StringFormatters.h"
#include <zlib.h>
#include <filesystem>
#include <fstream>
#include <mutex>


namespace Falcor
{
    std::string gMsgBoxTitle = "Falcor";

    void msgBoxTitle(const std::string& title)
    {
        gMsgBoxTitle = title;
    }

    const std::filesystem::path& getExecutableDirectory()
    {
        static std::filesystem::path directory;
        if (directory.empty()) directory = getExecutablePath().parent_path();
        return directory;
    }

    const std::string& getExecutableName()
    {
        static std::string name;
        if (name.empty()) name = getExecutablePath().filename().string();
        return name;
    }

    inline std::vector<std::filesystem::path> getInitialShaderDirectories()
    {
        std::filesystem::path projectDir(_PROJECT_DIR_);

        std::vector<std::filesystem::path> developmentDirectories =
        {
            // First we search in source folders.
            projectDir,
            projectDir / "..",
            projectDir / ".." / "Tools" / "FalcorTest",
            // Then we search in deployment folder (necessary to pickup NVAPI and other third-party shaders).
            getExecutableDirectory() / "Shaders",
        };

        std::vector<std::filesystem::path> deploymentDirectories =
        {
            getExecutableDirectory() / "Shaders",
        };

        return isDevelopmentMode() ? developmentDirectories : deploymentDirectories;
    }

    static std::vector<std::filesystem::path> gShaderDirectories = getInitialShaderDirectories();

    inline std::vector<std::filesystem::path> getInitialDataDirectories()
    {
        std::filesystem::path projectDir(_PROJECT_DIR_);

        std::vector<std::filesystem::path> developmentDirectories =
        {
            projectDir / "Data",
            getExecutableDirectory() / "Data",
        };

        std::vector<std::filesystem::path> deploymentDirectories =
        {
            getExecutableDirectory() / "Data"
        };

        std::vector<std::filesystem::path> directories = isDevelopmentMode() ? developmentDirectories : deploymentDirectories;

        // Add development media folder.
        directories.push_back(projectDir / ".." / ".." / "media");

        // Add additional media folders.
        std::string mediaFolders;
        if (getEnvironmentVariable("FALCOR_MEDIA_FOLDERS", mediaFolders))
        {
            auto folders = splitString(mediaFolders, ";");
            directories.insert(directories.end(), folders.begin(), folders.end());
        }

        return directories;
    }

    static std::vector<std::filesystem::path> gDataDirectories = getInitialDataDirectories();

    const std::vector<std::filesystem::path>& getDataDirectoriesList()
    {
        return gDataDirectories;
    }

    void addDataDirectory(const std::filesystem::path& dir, bool addToFront)
    {
        if (std::find(gDataDirectories.begin(), gDataDirectories.end(), dir) == gDataDirectories.end())
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
        auto it = std::find(gDataDirectories.begin(), gDataDirectories.end(), dir);
        if (it != gDataDirectories.end())
        {
            gDataDirectories.erase(it);
        }
    }

    bool isDevelopmentMode()
    {
        static bool initialized = false;
        static bool devMode = false;

        if (!initialized)
        {
            std::string value;
            devMode = getEnvironmentVariable("FALCOR_DEVMODE", value) && value == "1";
            initialized = true;
        }

        return devMode;
    }

    bool findFileInDataDirectories(const std::filesystem::path& path, std::filesystem::path& fullPath)
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
        for (const auto& dir : gDataDirectories)
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
            if (!std::filesystem::exists(path)) return path;
        }
        throw RuntimeError("Failed to find available filename.");
    }

    std::filesystem::path stripDataDirectories(const std::filesystem::path& path)
    {
        if (path.is_relative()) return path;

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
        if (ext.size() > 0 && ext[0] == '.') ext.remove_prefix(1);

        std::string pathExt = getExtensionFromPath(path);

        if (ext.size() != pathExt.size()) return false;

        return std::equal(ext.rbegin(), ext.rend(), pathExt.rbegin(),
            [](char a, char b) { return std::tolower(a) == std::tolower(b); });
    }

    std::string getExtensionFromPath(const std::filesystem::path& path)
    {
        std::string ext;
        if (path.has_extension())
        {
            ext = path.extension().string();
            // Remove the leading '.' from ext.
            if (ext.size() > 0 && ext[0] == '.') ext.erase(0, 1);
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
        if (!ifs) throw RuntimeError("Failed to read from file '{}'.", path);
        return std::string((std::istreambuf_iterator<char>(ifs)), (std::istreambuf_iterator<char>()));
    }

    std::string decompressFile(const std::filesystem::path& path)
    {
        std::string compressed = readFile(path);

        z_stream zs = {};
        // MAX_WBITS | 32 to support both zlib or gzip files.
        if (inflateInit2(&zs, MAX_WBITS | 32) != Z_OK) throw RuntimeError("inflateInit2 failed while decompressing.");

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
}
