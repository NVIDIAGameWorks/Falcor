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
#include "Utils/StringUtils.h"
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

namespace Falcor
{
    std::string gMsgBoxTitle = "Falcor";

    void msgBoxTitle(const std::string& title)
    {
        gMsgBoxTitle = title;
    }

    uint32_t getLowerPowerOf2(uint32_t a)
    {
        assert(a != 0);
        return 1 << bitScanReverse(a);
    }

    inline std::vector<std::string> getInitialShaderDirectories()
    {
        std::vector<std::string> developmentDirectories =
        {
            // First we search in source folders.
            std::string(_PROJECT_DIR_),
            std::string(_PROJECT_DIR_) + "../",
            std::string(_PROJECT_DIR_) + "../Tools/FalcorTest/",
            // Then we search in deployment folder (necessary to pickup NVAPI and other third-party shaders).
            getExecutableDirectory() + "/Shaders"
        };

        std::vector<std::string> deploymentDirectories =
        {
            getExecutableDirectory() + "/Shaders"
        };

        return isDevelopmentMode() ? developmentDirectories : deploymentDirectories;
    }

    static std::vector<std::string> gShaderDirectories = getInitialShaderDirectories();

    inline std::vector<std::string> getInitialDataDirectories()
    {
        std::vector<std::string> developmentDirectories =
        {
            std::string(_PROJECT_DIR_) + "/Data",
            getExecutableDirectory() + "/Data",
        };

        std::vector<std::string> deploymentDirectories =
        {
            getExecutableDirectory() + "/Data"
        };

        std::vector<std::string> directories = isDevelopmentMode() ? developmentDirectories : deploymentDirectories;

        // Add development media folder.
#ifdef _MSC_VER
        directories.push_back(getExecutableDirectory() + "/../../../Media"); // Relative to Visual Studio output folder
#else
        directories.push_back(getExecutableDirectory() + "/../Media"); // Relative to Makefile output folder
#endif

        // Add additional media folders.
        std::string mediaFolders;
        if (getEnvironmentVariable("FALCOR_MEDIA_FOLDERS", mediaFolders))
        {
            auto folders = splitString(mediaFolders, ";");
            directories.insert(directories.end(), folders.begin(), folders.end());
        }

        return directories;
    }

    static std::vector<std::string> gDataDirectories = getInitialDataDirectories();

    const std::vector<std::string>& getDataDirectoriesList()
    {
        return gDataDirectories;
    }

    void addDataDirectory(const std::string& dir, bool addToFront)
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

    void removeDataDirectory(const std::string& dir)
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

    std::string canonicalizeFilename(const std::string& filename)
    {
        fs::path path(replaceSubstring(filename, "\\", "/"));
        return fs::exists(path) ? fs::canonical(path).string() : "";
    }

    bool findFileInDataDirectories(const std::string& filename, std::string& fullPath)
    {
        // Check if this is an absolute path
        if (fs::path(filename).is_absolute())
        {
            fullPath = canonicalizeFilename(filename);
            return !fullPath.empty(); // Empty fullPath means path doesn't exist
        }

        for (const auto& dir : gDataDirectories)
        {
            fullPath = canonicalizeFilename((fs::path(dir) / filename).string());
            if (doesFileExist(fullPath))
            {
                return true;
            }
        }

        return false;
    }

    const std::vector<std::string>& getShaderDirectoriesList()
    {
        return gShaderDirectories;
    }

    bool findFileInShaderDirectories(const std::string& filename, std::string& fullPath)
    {
        for (const auto& dir : gShaderDirectories)
        {
            fullPath = canonicalizeFilename((fs::path(dir) / filename).string());
            if (doesFileExist(fullPath))
            {
                return true;
            }
        }

        return false;
    }

    bool findAvailableFilename(const std::string& prefix, const std::string& directory, const std::string& extension, std::string& filename)
    {
        for (uint32_t i = 0; i < (uint32_t)-1; i++)
        {
            std::string newPrefix = prefix + '.' + std::to_string(i);
            filename = (fs::path(directory) / newPrefix).string() + "." + extension;

            if (doesFileExist(filename) == false)
            {
                return true;
            }
        }
        should_not_get_here();
        filename = "";
        return false;
    }

    std::string stripDataDirectories(const std::string& filename)
    {
        std::string stripped = filename;
        std::string canonFile = canonicalizeFilename(filename);
        for (const auto& dir : gDataDirectories)
        {
            std::string canonDir = canonicalizeFilename(dir);
            if (canonDir.size() && hasPrefix(canonFile, canonDir, false))
            {
                // canonicalizeFilename adds trailing \\ to drive letters and removes them from paths containing folders
                // The entire prefix directory including the slash should be removed
                bool trailingSlash = canonDir.back() == '\\' || canonDir.back() == '/';
                size_t len = trailingSlash ? canonDir.length() : canonDir.length() + 1;
                std::string tmp = canonFile.erase(0, len);
                if (tmp.length() < stripped.length())
                {
                    stripped = tmp;
                }
            }
        }

        return stripped;
    }

    std::string swapFileExtension(const std::string& str, const std::string& currentExtension, const std::string& newExtension)
    {
        if (hasSuffix(str, currentExtension))
        {
            std::string ret = str;
            return (ret.erase(ret.rfind(currentExtension)) + newExtension);
        }
        else
        {
            return str;
        }
    }

    std::string getDirectoryFromFile(const std::string& filename)
    {
        fs::path path = filename;
        return path.has_filename() ? path.parent_path().string() : filename;
    }

    std::string getExtensionFromFile(const std::string& filename)
    {
        fs::path path = filename;
        std::string ext;
        if (path.has_extension())
        {
            // remove the leading '.' that filesystem gives us
            ext = path.extension().string();
            if (hasPrefix(ext, "."))   ext = ext.substr(1, ext.size());
        }
        return ext;
    }

    std::string getFilenameFromPath(const std::string& filename)
    {
        return fs::path(filename).filename().string();
    }

    std::string readFile(const std::string& filename)
    {
        std::ifstream filestream(filename);
        std::string str;
        filestream.seekg(0, std::ios::end);
        str.reserve(filestream.tellg());
        filestream.seekg(0, std::ios::beg);
        str.assign(std::istreambuf_iterator<char>(filestream), std::istreambuf_iterator<char>());
        return str;
    }
}
