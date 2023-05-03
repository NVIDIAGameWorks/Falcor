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
#pragma once
#include "Core/Macros.h"
#include "Core/Platform/OS.h"

#include <vector>
#include <string>
#include <filesystem>

namespace Falcor
{

using EnvVarResolver = std::optional<std::string>(const std::string&);
using FileChecker = bool(const std::filesystem::path&);

static bool checkFileExists(const std::filesystem::path& path)
{
    return path.has_filename() && std::filesystem::exists(path);
}

/**
 * Replace all occurances of environment variable names between `beginToken` and `endToken` with the environment variable value
 * resolved by calling the `envResolver`.
 *
 * If `envResolver` is not defined, standard OS env resolver is used.
 * If env variable is not found, it is replaced with an empty string.
 * If the beginToken and endToken do not match in the string, the call fails returning a false, leaving the string resolved until the
 * failure point.
 *
 * @param str String in which env variables will be in-place resolved.
 * @param envResolver Resolver used to resolve the env variables. When not defined, OS resolver is used. (Mostly defined for testing).
 * @param beginToken Token marking the start of the env variable.
 * @param endToken Token marking the end of the env variable. Env variable is between the two tokens, e.g., `${VARIABLE}`.
 * @return false if the begin/end tokens do not match, true otherwise. Not resolving an env variable does not result in false.
 */
FALCOR_API
bool resolveEnvVariables(
    std::string& str,
    EnvVarResolver envResolver = getEnvironmentVariable,
    std::string_view beginToken = "${",
    std::string_view endToken = "}"
);

/**
 * Result of the resolveSearchPaths.
 */
struct ResolvedPaths
{
    std::vector<std::filesystem::path> resolved; ///< Resolved paths.
    std::vector<std::string> invalid;            ///< Paths that could not be resolved (only for error reporting).
};

/**
 * Given the current search paths and the update search paths as vectors, modifies the current search paths
 * in place to reflect the new set of search paths.
 *
 * This follows the Renderman semantic.
 * All search paths must be absolute paths.
 * ${ENV_VARIABLE} get expanded using the envResolver. If one isn't provided, a standard implementation is used (this is mostly for
 * testing). Each searchpath end up as its own string. The update or the env variables can contain semi-colon separated list of paths.
 * Special character `@` will be replaced with the `standard` search path.
 * Special character `&` will be replaced with the `current` search path.
 *
 * @param current Current search paths, get used in place of `&` in update. Can be empty.
 * @param update The update to the search paths, can be empty, a single semi-colon separate string, and contain `${ENV_VARIABLES}`.
 * Non-absolute paths will be stripped away
 * @param standard Standard search paths, usually sourced from another settings, gets used in place of `@` in update. Can be empty.
 * @param envResolver Optionally provided custom env variable resolver, used for testing.
 * @return Returns vector of strings that haven't been found as valid search paths (i.e., not absolute)
 */
FALCOR_API
ResolvedPaths resolveSearchPaths(
    const std::vector<std::filesystem::path>& current,
    const std::vector<std::string>& update,
    const std::vector<std::filesystem::path>& standard,
    EnvVarResolver envResolver = getEnvironmentVariable
);

/**
 * Resolve a path into an absolute file path, using search paths and current work directory.
 *
 * This follows the Renderman semantic as well.
 * Paths starting with `/` or `<driveletter>:` are considered absolute paths and as such already resolved.
 * Paths starting with `.` are considered relative paths, and resolved against currentWorkingDirectory only.
 * All other paths are resolved using the searchPaths, in the order in which they appear,
 * returning the first path that exists according to fileChecker.
 *
 * @param searchPaths Search paths used to look for the filePath when it is neither absolute nor relative.
 * @param currentWorkingDirectory Current working directory, used when filePath is relative.
 * @param filePath Requested file path.
 * @param fileChecker Function that checks if the file path is valid.
 * @return Returns the fully resolved absolute path, or empty path when it could not be resolved.
 */
FALCOR_API
std::filesystem::path resolvePath(
    const std::vector<std::filesystem::path>& searchPaths,
    const std::filesystem::path& currentWorkingDirectory,
    const std::string& filePath,
    FileChecker fileChecker = checkFileExists
);

} // namespace Falcor
