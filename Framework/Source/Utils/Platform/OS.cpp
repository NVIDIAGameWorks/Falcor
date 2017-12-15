/***************************************************************************
# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
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
***************************************************************************/
#include "Framework.h"
#include "Utils/Platform/OS.h"
#include "Utils/StringUtils.h"
#include <fstream>
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;

namespace Falcor
{
    template<bool bOpen>
    bool fileDialogCommon(const char* pFilters, std::string& filename);

    bool openFileDialog(const char* pFilters, std::string& filename)
    {
        return fileDialogCommon<true>(pFilters, filename);
    }

    bool saveFileDialog(const char* pFilters, std::string& filename)
    {
        return fileDialogCommon<false>(pFilters, filename);
    }

    uint32_t getLowerPowerOf2(uint32_t a)
    {
        assert(a != 0);
        return 1 << bitScanReverse(a);
    }

    std::vector<std::string> gDataDirectories =
    {
        // Ordering matters here, we want that while developing, resources will be loaded from the development media directory
        std::string(getWorkingDirectory()),
        std::string(getWorkingDirectory() + "/Data"),
        std::string(_PROJECT_DIR_) + "/ShadingUtils",
        std::string(getExecutableDirectory()),
        std::string(getExecutableDirectory() + "/Data"),

        // The local solution media folder
#ifdef _MSC_VER
        std::string(getExecutableDirectory() + "/../../../Media"), // Relative to Visual Studio output folder
#else
        std::string(getExecutableDirectory() + "/../Media"), // Relative to Makefile output folder
#endif
    };

    const std::vector<std::string>& getDataDirectoriesList()
    {
        return gDataDirectories;
    }

    void addDataDirectory(const std::string& dataDir)
    {
        //Insert unique elements
        if (std::find(gDataDirectories.begin(), gDataDirectories.end(), dataDir) == gDataDirectories.end())
        {
            gDataDirectories.push_back(dataDir);
        }
    }

    std::string canonicalizeFilename(const std::string& filename)
    {
        fs::path path(replaceSubstring(filename, "\\", "/"));
        return fs::exists(path) ? fs::canonical(path).string() : "";
    }

    bool findFileInDataDirectories(const std::string& filename, std::string& fullpath)
    {
        static bool bInit = false;
        if (bInit == false)
        {
            std::string dataDirs;
            if (getEnvironmentVariable("FALCOR_MEDIA_FOLDERS", dataDirs))
            {
                auto folders = splitString(dataDirs, ";");
                gDataDirectories.insert(gDataDirectories.end(), folders.begin(), folders.end());
            }
            bInit = true;
        }

        // Check if this is an absolute path
        if (doesFileExist(filename))
        {
            fullpath = canonicalizeFilename(filename);
            return true;
        }

        for (const auto& Dir : gDataDirectories)
        {
            fullpath = canonicalizeFilename(Dir + '/' + filename);
            if (doesFileExist(fullpath))
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
            filename = directory + '/' + newPrefix + "." + extension;

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
            if (hasPrefix(canonFile, canonDir, false))
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

    bool readFileToString(const std::string& fullpath, std::string& str)
    {
        std::ifstream t(fullpath.c_str());
        if ((t.rdstate() & std::ifstream::failbit) == 0)
        {
            str = std::string((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
            return true;
        }
        return false;
    }
    
    std::string getDirectoryFromFile(const std::string& filename)
    {
        fs::path path = filename;
        return path.has_filename() ? path.parent_path().string() : filename;
    }

    std::string getFilenameFromPath(const std::string& filename)
    {
        return fs::path(filename).filename().string();
    }
}