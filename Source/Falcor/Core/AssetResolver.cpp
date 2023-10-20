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
#include "AssetResolver.h"
#include "Core/Platform/OS.h"
#include "Utils/Scripting/ScriptBindings.h"

namespace Falcor
{

AssetResolver::AssetResolver()
{
    mSearchContexts.resize(size_t(AssetCategory::Count));
}

std::filesystem::path AssetResolver::resolvePath(const std::filesystem::path& path, AssetCategory category) const
{
    FALCOR_CHECK(category < AssetCategory::Count, "Invalid asset category.");

    // If this is an existing absolute path, or a relative path to the working directory, return it.
    std::filesystem::path absolute = std::filesystem::absolute(path);
    if (std::filesystem::exists(absolute))
        return std::filesystem::canonical(absolute);

    // Otherwise, try to resolve using search paths.
    // First try resolving for the specified asset category.
    std::filesystem::path resolved = mSearchContexts[size_t(category)].resolvePath(path);

    // If not resolved, try resolving for the Any asset category.
    if (category != AssetCategory::Any && resolved.empty())
        resolved = mSearchContexts[size_t(AssetCategory::Any)].resolvePath(path);

    if (resolved.empty())
        logWarning("Failed to resolve path '{}' for asset type '{}'.", path, category);

    return resolved;
}

std::vector<std::filesystem::path> AssetResolver::resolvePathPattern(
    const std::filesystem::path& path,
    const std::string& pattern,
    bool firstMatchOnly,
    AssetCategory category
) const
{
    FALCOR_CHECK(category < AssetCategory::Count, "Invalid asset category.");

    std::regex regex(pattern);

    // If this is an existing absolute path, or a relative path to the working directory, search it.
    std::filesystem::path absolute = std::filesystem::absolute(path);
    std::vector<std::filesystem::path> resolved = globFilesInDirectory(absolute, regex, firstMatchOnly);
    if (!resolved.empty())
        return resolved;

    // Otherwise, try to resolve using search paths.
    // First try resolving for the specified asset category.
    resolved = mSearchContexts[size_t(category)].resolvePathPattern(path, regex, firstMatchOnly);

    // If not resolved, try resolving for the Any asset category.
    if (category != AssetCategory::Any && resolved.empty())
        resolved = mSearchContexts[size_t(AssetCategory::Any)].resolvePathPattern(path, regex, firstMatchOnly);

    if (resolved.empty())
        logWarning("Failed to resolve path pattern '{}/{}' for asset type '{}'.", path, pattern, category);

    return resolved;
}

void AssetResolver::addSearchPath(const std::filesystem::path& path, SearchPathPriority priority, AssetCategory category)
{
    FALCOR_CHECK(path.is_absolute(), "Search path must be absolute.");
    FALCOR_CHECK(category < AssetCategory::Count, "Invalid asset category.");
    mSearchContexts[size_t(category)].addSearchPath(path, priority);
}

AssetResolver& AssetResolver::getDefaultResolver()
{
    static AssetResolver defaultResolver;
    return defaultResolver;
}

std::filesystem::path AssetResolver::SearchContext::resolvePath(const std::filesystem::path& path) const
{
    for (const auto& searchPath : searchPaths)
    {
        std::filesystem::path absolutePath = searchPath / path;
        if (std::filesystem::exists(absolutePath))
            return std::filesystem::canonical(absolutePath);
    }

    return {};
}

std::vector<std::filesystem::path> AssetResolver::SearchContext::resolvePathPattern(
    const std::filesystem::path& path,
    const std::regex& regex,
    bool firstMatchOnly
) const
{
    for (const auto& searchPath : searchPaths)
    {
        std::filesystem::path absolutePath = searchPath / path;
        std::vector<std::filesystem::path> resolved = globFilesInDirectory(absolutePath, regex, firstMatchOnly);
        if (!resolved.empty())
            return resolved;
    }
    return {};
}

void AssetResolver::SearchContext::addSearchPath(const std::filesystem::path& path, SearchPathPriority priority)
{
    FALCOR_ASSERT(path.is_absolute());
    auto it = std::find_if(searchPaths.begin(), searchPaths.end(), [&path](const std::filesystem::path& p) { return isSamePath(path, p); });
    if (it != searchPaths.end())
        searchPaths.erase(it);
    switch (priority)
    {
    case SearchPathPriority::First:
        searchPaths.insert(searchPaths.begin(), path);
        return;
    case SearchPathPriority::Last:
        searchPaths.push_back(path);
        return;
    }
    FALCOR_THROW("Invalid search path priority.");
}

FALCOR_SCRIPT_BINDING(AssetResolver)
{
    using namespace pybind11::literals;

    pybind11::falcor_enum<AssetCategory>(m, "AssetCategory");
    pybind11::falcor_enum<SearchPathPriority>(m, "SearchPathPriority");

    pybind11::class_<AssetResolver> assetResolver(m, "AssetResolver");
    assetResolver.def("resolve_path", &AssetResolver::resolvePath, "path"_a, "category"_a = AssetCategory::Any);
    assetResolver.def(
        "resolve_path_pattern",
        &AssetResolver::resolvePathPattern,
        "path"_a,
        "pattern"_a,
        "first_match_only"_a = false,
        "category"_a = AssetCategory::Any
    );

    assetResolver.def(
        "add_search_path",
        &AssetResolver::addSearchPath,
        "path"_a,
        "priority"_a = SearchPathPriority::Last,
        "category"_a = AssetCategory::Any
    );

    assetResolver.def_property_readonly_static("default_resolver", [](pybind11::object) { return AssetResolver::getDefaultResolver(); });
}

} // namespace Falcor
