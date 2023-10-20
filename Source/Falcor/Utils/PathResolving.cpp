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
#include "PathResolving.h"

#include "Core/Error.h"
#include "Utils/StringUtils.h"
#include "Core/Platform/OS.h"

namespace Falcor
{

bool resolveEnvVariables(std::string& str, EnvVarResolver envResolver, std::string_view beginToken, std::string_view endToken)
{
    FALCOR_ASSERT(envResolver);

    std::string::size_type begin = str.find_first_of(beginToken);
    if (begin == std::string::npos)
        return true;

    std::string result = str.substr(0, begin);
    while (begin != std::string::npos)
    {
        std::string::size_type end = str.find_first_of(endToken, begin);
        // Didn't find a matching end
        if (end == std::string::npos)
            return false;
        std::string envVar = str.substr(begin + 2, end - begin - 2);
        std::string resolved = envResolver(envVar).value_or("");
        result += resolved;

        begin = str.find_first_of(beginToken, end);
        result += str.substr(end + 1, begin - end - 1);
    }

    str = std::move(result);
    return true;
}

ResolvedPaths resolveSearchPaths(
    const std::vector<std::filesystem::path>& current,
    const std::vector<std::string>& update,
    const std::vector<std::filesystem::path>& standard,
    EnvVarResolver envResolver
)
{
    ResolvedPaths result;

    for (std::string it : update)
    {
        if (!resolveEnvVariables(it, envResolver))
        {
            result.invalid.push_back(std::move(it));
            continue;
        }

        std::vector<std::string> temp = splitString(it, ";");
        for (std::string& it2 : temp)
        {
            if (it2.empty())
                continue;
            if (it2 == "&")
            {
                result.resolved.insert(result.resolved.end(), current.begin(), current.end());
                continue;
            }
            if (it2 == "@")
            {
                result.resolved.insert(result.resolved.end(), standard.begin(), standard.end());
                continue;
            }
            std::filesystem::path path = std::filesystem::weakly_canonical(it2);
            if (path.is_absolute())
                result.resolved.push_back(std::move(path));
            else
                result.invalid.push_back(std::move(it2));
        }
    }

    return result;
}

std::filesystem::path resolvePath(
    const std::vector<std::filesystem::path>& searchPaths,
    const std::filesystem::path& currentWorkingDirectory,
    const std::string& filePath,
    FileChecker fileChecker
)
{
    FALCOR_ASSERT(currentWorkingDirectory.is_absolute());

    if (filePath.empty())
        return std::filesystem::path();

    std::filesystem::path path(filePath);
    if (path.is_absolute())
    {
        if (fileChecker(path))
            return std::filesystem::weakly_canonical(path);
        return std::filesystem::path();
    }

    // This is the relative case
    if (filePath[0] == '.')
    {
        std::filesystem::path result = currentWorkingDirectory / path;
        if (fileChecker(result))
            return std::filesystem::weakly_canonical(result);
        return std::filesystem::path();
    }

    // This is the searchpath case
    for (const auto& searchpath : searchPaths)
    {
        std::filesystem::path result = searchpath / path;
        if (fileChecker(result))
            return std::filesystem::weakly_canonical(result);
    }
    return std::filesystem::path();
}

} // namespace Falcor
