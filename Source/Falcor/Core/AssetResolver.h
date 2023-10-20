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
#pragma once

#include "Macros.h"
#include "Enum.h"
#include <filesystem>
#include <regex>
#include <string>
#include <vector>

namespace Falcor
{

/// Asset categories.
enum class AssetCategory
{
    Any,
    Scene,
    Texture,
    Count,
};

FALCOR_ENUM_INFO(
    AssetCategory,
    {
        {AssetCategory::Any, "Any"},
        {AssetCategory::Scene, "Scene"},
        {AssetCategory::Texture, "Texture"},
    }
);
FALCOR_ENUM_REGISTER(AssetCategory);

/// Search path priorities.
enum class SearchPathPriority
{
    First, ///< Add to the beginning of the search path list.
    Last,  ///< Add to the end of the search path list.
};

FALCOR_ENUM_INFO(
    SearchPathPriority,
    {
        {SearchPathPriority::First, "First"},
        {SearchPathPriority::Last, "Last"},
    }
);
FALCOR_ENUM_REGISTER(SearchPathPriority);

/**
 * @brief Class for resolving paths to asset files.
 *
 * The AssetResolver class is used to resolve relative paths of assets to absolute paths.
 * Paths are resolved with the following logic:
 * - If the path is absolute and exists, it is returned in its canonical form.
 * - If the path is relative to the working directory and exists, it is returned in its canonical form.
 * - If the path is relative to a search directory and exists, it is returned in its canonical form.
 * The resolver supports resolving assets of different categories. Each asset category has its own list of
 * search paths. When resolving a path, the resolver will first try to resolve the path
 * for the specified category, and if that fails, it will try to resolve it for the \c AssetCategory::Any category.
 * If no asset category is specified, the \c AssetCategory::Any category is used by default.
 */
class FALCOR_API AssetResolver
{
public:
    /// Default constructor.
    AssetResolver();

    /**
     * Resolve \c path to an existing absolute file path.
     * If \c path is absolute or relative to the working directory and exists, it is returned in its canonical form.
     * If \c path is relative and resolves to some \c <searchpath>/<path> it is returned.
     * If the path cannot be resolved, an empty path is returned.
     * @param path Path to resolve.
     * @param category Asset category.
     * @return The resolved path, or an empty path if the path could not be resolved.
     */
    std::filesystem::path resolvePath(const std::filesystem::path& path, AssetCategory category = AssetCategory::Any) const;

    /**
     * Resolve \c <path>/<pattern> to a list of existing absolute file paths.
     * If \c path is absolute or relative to the working directory and resolves to some \c <path>/<pattern>
     * then the list of matching files is returned.
     * If \c path is relative and resolves to some \c <searchpath>/<path>/<pattern>
     * then the list of matching files is returned.
     * If the path cannot be resolved, an empty list is returned.
     * @param path Path prefix to resolve.
     * @param pattern Filename pattern to match (regular expression in ECMAScript format).
     * @param firstMatchOnly If true, only the first found match is returned.
     * @param category Asset category.
     * @return Returns an unordered list of resolved paths, or an empty list if the path could not be resolved.
     */
    std::vector<std::filesystem::path> resolvePathPattern(
        const std::filesystem::path& path,
        const std::string& pattern,
        bool firstMatchOnly = false,
        AssetCategory category = AssetCategory::Any
    ) const;

    /**
     * Add a search path to the resolver.
     * The path needs to be absolute and exist.
     * An optional priority can be specified, which determines whether the path is added to the beginning or end of the search path list.
     * If the search path already exists, it is moved to the specified priority.
     * @param path Path to add.
     * @param priority Search path priority.
     * @param category Asset category.
     */
    void addSearchPath(
        const std::filesystem::path& path,
        SearchPathPriority priority = SearchPathPriority::Last,
        AssetCategory category = AssetCategory::Any
    );

    /// Return the global default asset resolver.
    static AssetResolver& getDefaultResolver();

private:
    struct SearchContext
    {
        /// List of search paths. Resolving is done by searching these paths in order.
        std::vector<std::filesystem::path> searchPaths;

        std::filesystem::path resolvePath(const std::filesystem::path& path) const;

        std::vector<std::filesystem::path> resolvePathPattern(
            const std::filesystem::path& path,
            const std::regex& regex,
            bool firstMatchOnly
        ) const;

        void addSearchPath(const std::filesystem::path& path, SearchPathPriority priority);
    };

    std::vector<SearchContext> mSearchContexts;
};
} // namespace Falcor
