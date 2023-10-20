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

#include "Core/Macros.h"
#include "Core/Error.h"
#include "Core/Platform/OS.h"

#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

namespace Falcor
{

/**
 * @brief Plugin manager for loading plugin libraries and creating plugin instances.
 *
 * The plugin system is based around the following principles:
 * - Plugins are compiled into shared libraries known as _plugin libraries_.
 * - A _plugin library_ registers one or more _plugin classes_ when being loaded.
 * - A _plugin class_ needs to inherit from a known _plugin base class_, implementing it's interface.
 * - A _plugin base class_ defines a `PluginInfo` struct, describing meta information about each
 *   _plugin class_ as well as a `PluginCreate` typedef used for instantiating plugin classes.
 *
 * A _plugin base class_ can be any class that is exported from the main Falcor library.
 * It needs to define a `PluginInfo` struct as well as a `PluginCreate` typedef.
 * The FALCOR_PLUGIN_BASE_CLASS macro extends the class with the required members for the plugin system.
 * For example:
 *
 * @code
 * class FALCOR_API PluginBase
 * {
 * public:
 *     struct PluginInfo { const char* desc };
 *     using PluginCreate = (PluginBase *)(*)();
 *     FALCOR_PLUGIN_BASE_CLASS(PluginBase);
 *
 *     virtual void doStuff() = 0;
 * };
 * @endcode
 *
 * To implement a _plugin class_ type, we simply inherit from the base class.
 * The FALCOR_PLUGIN_CLASS macro extends the class with the required members for the plugin system.
 * The class also needs to implement a static create function, matching the `PluginCreate` type of the base class.
 * For example:
 *
 * @code
 * class PluginA : public PluginBase
 * {
 * public:
 *     FALCOR_PLUGIN_CLASS(PluginA, "PluginA", PluginInfo({"plugin description string"}));
 *
 *     static PluginBase* create() { return new PluginA(); }
 *
 *     void doStuff() override { ... }
 * };
 * @endcode
 *
 * The _plugin library_ must export a `registerPlugin` function for registering all the plugin class types when called.
 *
 * @code
 * extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
 * {
 *     registry.registerClass<PluginBase, PluginA>();
 * }
 * @endcode
 *
 * At runtime, the plugin manager can load plugin libraries using `loadPluginByName` and `loadPlugin`.
 * New plugin instances can be created using `createClass`, for example:
 *
 * @code
 * PluginBase* p = PluginManager::instance().createClass<PluginBase>("PluginA");
 * @endcode
 *
 * The `getInfos` function returns a list of plugin infos for all loaded plugin types of a given plugin base class.
 */
class FALCOR_API PluginManager
{
public:
    /// Singleton accessor.
    static PluginManager& instance();

    /**
     * @brief Create an instance of a plugin class.
     *
     * @tparam BaseT The plugin base class.
     * @tparam Args The argument type pack used for construction.
     * @param type The plugin type name.
     * @param args Additional arguments passed on construction.
     * @return Returns a new instance of the requested plugin type or nullptr if not registered.
     */
    template<typename BaseT, typename... Args>
    std::invoke_result_t<typename BaseT::PluginCreate, Args...> createClass(std::string_view type, Args... args) const
    {
        std::lock_guard<std::mutex> lock(mClassDescsMutex);
        const ClassDesc<BaseT>* classDesc = findClassDesc<BaseT>(type);
        return classDesc ? classDesc->create(args...) : std::invoke_result_t<typename BaseT::PluginCreate, Args...>{nullptr};
    }

    /**
     * @brief Check if a given type of a plugin is available.
     *
     * @tparam BaseT The plugin base class.
     * @param type The plugin type name.
     * @return True if plugin type is available.
     */
    template<typename BaseT>
    bool hasClass(std::string_view type) const
    {
        std::lock_guard<std::mutex> lock(mClassDescsMutex);
        const ClassDesc<BaseT>* classDesc = findClassDesc<BaseT>(type);
        return classDesc != nullptr;
    }

    /**
     * @brief Get infos for all registered plugin types for a given plugin base class.
     *
     * @tparam BaseT The plugin base class.
     * @return A list of infos.
     */
    template<typename BaseT>
    std::vector<std::pair<std::string, typename BaseT::PluginInfo>> getInfos() const
    {
        std::lock_guard<std::mutex> lock(mClassDescsMutex);
        std::vector<std::pair<std::string, typename BaseT::PluginInfo>> result;

        for (const auto& [name, desc] : mClassDescs)
            if (auto matchDesc = dynamic_cast<const ClassDesc<BaseT>*>(desc.get()))
                result.push_back(std::make_pair(matchDesc->type, matchDesc->info));

        return result;
    }

    /**
     * @brief Load a plugin library by name.
     * This will automatically determine the plugin path and file extension.
     * @param name Name of the plugin library.
     * @return True if successful.
     */
    bool loadPluginByName(std::string_view name);

    /**
     * Load a plugin library.
     * @param path File path of the plugin library.
     * @return True if successful.
     */
    bool loadPlugin(const std::filesystem::path& path);

    /**
     * Release a previously loaded plugin library.
     * @param path File path of the plugin library.
     * @return True if successful.
     */
    bool releasePlugin(const std::filesystem::path& path);

    /**
     * Load all plugin libraries.
     */
    void loadAllPlugins();

    /**
     * Release all loaded plugin libraries.
     */
    void releaseAllPlugins();

private:
    struct ClassDescBase
    {
        ClassDescBase(SharedLibraryHandle library, std::string_view type) : library(library), type(type) {}
        virtual ~ClassDescBase() {}

        SharedLibraryHandle library;
        std::string type;
    };

    template<typename BaseT>
    struct ClassDesc : public ClassDescBase
    {
        typename BaseT::PluginInfo info;
        typename BaseT::PluginCreate create;

        ClassDesc(SharedLibraryHandle library, std::string_view type, typename BaseT::PluginInfo info, typename BaseT::PluginCreate create)
            : ClassDescBase(library, type), info(info), create(create)
        {}
    };

    template<typename BaseT>
    void registerClass(
        SharedLibraryHandle library,
        std::string_view type,
        typename BaseT::PluginInfo info,
        typename BaseT::PluginCreate create
    )
    {
        std::lock_guard<std::mutex> lock(mClassDescsMutex);

        if (findClassDesc<BaseT>(type) != nullptr)
            FALCOR_THROW(
                "A plugin class with type name '{}' (base class type '{}') has already been registered.", type, BaseT::getPluginBaseType()
            );

        auto desc = std::make_shared<ClassDesc<BaseT>>(library, type, info, create);
        mClassDescs.emplace(type, std::move(desc));
    }

    template<typename BaseT>
    const ClassDesc<BaseT>* findClassDesc(std::string_view type) const
    {
        if (auto it = mClassDescs.find(std::string(type)); it != mClassDescs.end())
            if (auto desc = dynamic_cast<const ClassDesc<BaseT>*>(it->second.get()); desc && desc->type == type)
                return desc;

        return nullptr;
    }

    std::map<std::filesystem::path, SharedLibraryHandle> mLibraries;
    std::map<std::string, std::shared_ptr<ClassDescBase>> mClassDescs;

    mutable std::mutex mLibrariesMutex;
    mutable std::mutex mClassDescsMutex;

    friend class PluginRegistry;
};

/**
 * @brief Helper class passed to plugin libraries to register plugin classes.
 */
class PluginRegistry
{
public:
    PluginRegistry(PluginManager& pluginManager, SharedLibraryHandle library) : mPluginManager(pluginManager), mLibrary(library) {}
    PluginRegistry(const PluginRegistry&) = delete;
    PluginRegistry& operator=(const PluginRegistry&) = delete;

    /**
     * @brief Register a plugin class.
     * Throws if a class with the same type name (and same base class) has already been registered.
     *
     * @tparam BaseT The plugin base class.
     * @param type The plugin type name.
     * @param info The plugin info (type of BaseT::PluginInfo).
     * @param create The plugin create function (type of BaseT::PluginCreate).
     */
    template<typename BaseT>
    void registerClass(std::string_view type, typename BaseT::PluginInfo info, typename BaseT::PluginCreate create)
    {
        mPluginManager.registerClass<BaseT>(mLibrary, type, info, create);
    }

    /**
     * @brief Register a plugin class.
     * This helper assumes that the plugin class has a `create` function matching the BaseT::PluginCreate type.
     *
     * @tparam BaseT The plugin base class.
     * @tparam T The plugin class.
     */
    template<typename BaseT, typename T>
    void registerClass()
    {
        registerClass<BaseT>(T::kPluginType, T::kPluginInfo, T::create);
    }

private:
    PluginManager& mPluginManager;
    SharedLibraryHandle mLibrary;
};

/**
 * @brief Macro for extending a class to be used as a plugin base class.
 *
 * This macro must be applied in the class declaration of the plugin base class.
 * It assumes that both a public PluginInfo struct type and a PluginCreate typedef
 * are defined when invoked.
 */
#define FALCOR_PLUGIN_BASE_CLASS(cls)                        \
public:                                                      \
    static_assert(std::is_class_v<cls> == true);             \
    /* TODO: check for existance of cls::PluginCreate */     \
    static_assert(std::is_class_v<cls::PluginInfo> == true); \
    static const std::string& getPluginBaseType()            \
    {                                                        \
        static std::string type{#cls};                       \
        return type;                                         \
    }                                                        \
    virtual const std::string& getPluginType() const = 0;    \
    virtual const PluginInfo& getPluginInfo() const = 0;

/**
 * @brief Macro for extending a class to be used as a plugin class.
 *
 * This macro must be applied in the class declaration of the plugin class.
 */
#define FALCOR_PLUGIN_CLASS(cls, type, info)               \
public:                                                    \
    static_assert(std::is_class_v<cls> == true);           \
    /* TODO: check inheritance of base plugin class */     \
    static inline const std::string kPluginType{type};     \
    static inline const PluginInfo kPluginInfo{info};      \
    virtual const std::string& getPluginType() const final \
    {                                                      \
        return kPluginType;                                \
    }                                                      \
    virtual const PluginInfo& getPluginInfo() const final  \
    {                                                      \
        return kPluginInfo;                                \
    }

} // namespace Falcor
