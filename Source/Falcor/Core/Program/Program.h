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
#include "ProgramVersion.h"
#include "DefineList.h"
#include "Core/Macros.h"
#include "Core/Object.h"
#include "Core/API/fwd.h"
#include "Core/API/Types.h"
#include "Core/API/Raytracing.h"
#include "Core/API/RtStateObject.h"
#include "Core/State/StateGraph.h"
#include <filesystem>
#include <memory>
#include <string_view>
#include <string>
#include <map>
#include <unordered_map>
#include <vector>
#include <tuple>

namespace Falcor
{

class RtStateObject;
class RtProgramVars;

/**
 * Representing a shader implementation of an interface.
 * When linked into a `ProgramVersion`, the specialized shader will contain
 * the implementation of the specified type in a dynamic dispatch function.
 */
struct TypeConformance
{
    std::string typeName;
    std::string interfaceName;
    TypeConformance() = default;
    TypeConformance(const std::string& typeName_, const std::string& interfaceName_) : typeName(typeName_), interfaceName(interfaceName_) {}
    bool operator<(const TypeConformance& other) const
    {
        return typeName < other.typeName || (typeName == other.typeName && interfaceName < other.interfaceName);
    }
    bool operator==(const TypeConformance& other) const { return typeName == other.typeName && interfaceName == other.interfaceName; }
    struct HashFunction
    {
        size_t operator()(const TypeConformance& conformance) const
        {
            size_t hash = std::hash<std::string>()(conformance.typeName);
            hash = hash ^ std::hash<std::string>()(conformance.interfaceName);
            return hash;
        }
    };
};

class TypeConformanceList : public std::map<TypeConformance, uint32_t>
{
public:
    /**
     * Adds a type conformance. If the type conformance exists, it will be replaced.
     * @param[in] typeName The name of the implementation type.
     * @param[in] interfaceName The name of the interface type.
     * @param[in] id Optional. The id representing the implementation type for this interface. If it is -1, Slang will automatically
     * assign a unique Id for the type.
     * @return The updated list of type conformances.
     */
    TypeConformanceList& add(const std::string& typeName, const std::string& interfaceName, uint32_t id = -1)
    {
        (*this)[TypeConformance(typeName, interfaceName)] = id;
        return *this;
    }

    /**
     * Removes a type conformance. If the type conformance doesn't exist, the call will be silently ignored.
     * @param[in] typeName The name of the implementation type.
     * @param[in] interfaceName The name of the interface type.
     * @return The updated list of type conformances.
     */
    TypeConformanceList& remove(const std::string& typeName, const std::string& interfaceName)
    {
        (*this).erase(TypeConformance(typeName, interfaceName));
        return *this;
    }

    /**
     * Add a type conformance list to the current list
     */
    TypeConformanceList& add(const TypeConformanceList& cl)
    {
        for (const auto& p : cl)
            add(p.first.typeName, p.first.interfaceName, p.second);
        return *this;
    }

    /**
     * Remove a type conformance list from the current list
     */
    TypeConformanceList& remove(const TypeConformanceList& cl)
    {
        for (const auto& p : cl)
            remove(p.first.typeName, p.first.interfaceName);
        return *this;
    }

    TypeConformanceList() = default;
    TypeConformanceList(std::initializer_list<std::pair<const TypeConformance, uint32_t>> il) : std::map<TypeConformance, uint32_t>(il) {}
};

enum class SlangCompilerFlags
{
    None = 0x0,
    TreatWarningsAsErrors = 0x1,
    /// Enable dumping of intermediate artifacts during compilation.
    /// Note that if a shader is cached no artifacts are being produced.
    /// Delete the `.shadercache` directory in the build directory before dumping.
    DumpIntermediates = 0x2,
    FloatingPointModeFast = 0x4,
    FloatingPointModePrecise = 0x8,
    GenerateDebugInfo = 0x10,
    MatrixLayoutColumnMajor = 0x20, // Falcor is using row-major, use this only to compile stand-alone external shaders.
};
FALCOR_ENUM_CLASS_OPERATORS(SlangCompilerFlags);

/**
 * Description of a program to be created.
 * This includes the following:
 * - shader sources organized into shader modules (each module is compiled as a separate translation unit)
 * - entry points organized into entry point groups
 * - type conformances (global and per entry point group)
 * - compiler options (shader model, flags, etc.)
 */
struct ProgramDesc
{
    struct ShaderID
    {
        int32_t groupIndex = -1; ///< Entry point group index.
        bool isValid() const { return groupIndex >= 0; }
    };

    /// Represents a single piece of shader source code.
    /// This can either be a file or a string.
    struct ShaderSource
    {
        enum class Type
        {
            File,
            String
        };

        /// Type of the shader source.
        Type type{Type::File};

        /// Shader source file path.
        /// For Type::File this is the actual path to the file.
        /// For Type::String this is an optional virtual path used for diagnostics purposes.
        std::filesystem::path path;

        /// Shader source string if type == Type::String.
        std::string string;

        bool operator==(const ShaderSource& rhs) const { return type == rhs.type && path == rhs.path && string == rhs.string; }
    };

    /// Represents a single shader module made up from a list of sources (files/strings).
    /// A shader module corresponds to a single translation unit.
    struct ShaderModule
    {
        /// The name of the shader module.
        /// This is the name used by other modules to import this module.
        /// If left empty, a name based on a hash from the module sources will be generated.
        std::string name;

        /// List of shader sources.
        std::vector<ShaderSource> sources;

        ShaderModule() = default;
        explicit ShaderModule(std::string name_) : name(std::move(name_)) {}

        /// Create a shader module description containing a single file.
        static ShaderModule fromFile(std::filesystem::path path)
        {
            ShaderModule sm;
            sm.addFile(std::move(path));
            return sm;
        }

        /// Create a shader module description containing a single string.
        static ShaderModule fromString(std::string string, std::filesystem::path path = {}, std::string moduleName = {})
        {
            ShaderModule sm(std::move(moduleName));
            sm.addString(std::move(string), std::move(path));
            return sm;
        }

        /// Add a source file to the shader module.
        ShaderModule& addFile(std::filesystem::path path)
        {
            sources.push_back({ShaderSource::Type::File, std::move(path), {}});
            return *this;
        }

        /// Add a source string to the shader module.
        /// The path is optional and only used for diagnostic purposes.
        ShaderModule& addString(std::string string, std::filesystem::path path = {})
        {
            sources.push_back({ShaderSource::Type::String, std::move(path), std::move(string)});
            return *this;
        }

        bool operator==(const ShaderModule& rhs) const { return name == rhs.name && sources == rhs.sources; }
    };

    /// Represents a single entry point.
    struct EntryPoint
    {
        /// Shader type (compute, vertex, pixel, etc.).
        ShaderType type;
        /// The name of the entry point in the shader source.
        std::string name;
        /// The name of the entry point in the generated code.
        std::string exportName;

        /// Global linear entry point index. This is computed when creating the program.
        /// TODO we should look into eventally removing this.
        uint32_t globalIndex;
    };

    /// Represents a group of entry points.
    /// This is mostly used for grouping raytracing hitgroup entry points.
    struct EntryPointGroup
    {
        uint32_t shaderModuleIndex;
        TypeConformanceList typeConformances;
        std::vector<EntryPoint> entryPoints;

        /// Set the type conformances used for this entry point group.
        EntryPointGroup& setTypeConformances(TypeConformanceList conformances)
        {
            typeConformances = std::move(conformances);
            return *this;
        }

        /// Add type conformances used for this entry point group.
        EntryPointGroup& addTypeConformances(const TypeConformanceList& conformances)
        {
            typeConformances.add(conformances);
            return *this;
        }

        /// Add an entry point to the group.
        EntryPointGroup& addEntryPoint(ShaderType type, std::string_view name, std::string_view exportName = "")
        {
            if (exportName.empty())
                exportName = name;
            entryPoints.push_back({type, std::string(name), std::string(exportName)});
            return *this;
        }
    };

    using ShaderModuleList = std::vector<ShaderModule>;

    /// List of shader modules used by the program.
    ShaderModuleList shaderModules;

    /// List of entry point groups.
    std::vector<EntryPointGroup> entryPointGroups;

    /// Global type conformances.
    TypeConformanceList typeConformances;

    /// Shader model.
    /// If not specified, the most recent supported shader model is used.
    ShaderModel shaderModel{ShaderModel::Unknown};

    /// Compiler flags.
    SlangCompilerFlags compilerFlags{SlangCompilerFlags::None};

    /// List of compiler arguments (as set on the compiler command line).
    std::vector<std::string> compilerArguments;

    /// Max trace recursion depth (only used for raytracing programs).
    uint32_t maxTraceRecursionDepth = uint32_t(-1);

    /// Max ray payload size in bytes (only used for raytracing programs).
    uint32_t maxPayloadSize = uint32_t(-1);

    /// Max attribute size in bytes (only used for raytracing programs).
    uint32_t maxAttributeSize = getRaytracingMaxAttributeSize();

    /// Raytracing pipeline flags (only used for raytracing programs).
    RtPipelineFlags rtPipelineFlags = RtPipelineFlags::None;

    /// Add a new empty shader module description.
    /// @param[in] name Optional name of the shader module.
    /// @return Returns a reference to the newly created shader module for adding sources.
    ShaderModule& addShaderModule(std::string name = {})
    {
        shaderModules.push_back(ShaderModule(std::move(name)));
        return shaderModules.back();
    }

    /// Add an existing shader module description.
    ProgramDesc& addShaderModule(ShaderModule shaderModule)
    {
        shaderModules.push_back(std::move(shaderModule));
        return *this;
    }

    /// Add a list of existing shader module descriptions.
    ProgramDesc& addShaderModules(const ShaderModuleList& modules)
    {
        shaderModules.insert(shaderModules.end(), modules.begin(), modules.end());
        return *this;
    }

    /// Helper to add a shader module made from a single file.
    ProgramDesc& addShaderLibrary(std::filesystem::path path)
    {
        addShaderModule().addFile(std::move(path));
        return *this;
    }

    /// Add a new entry point group.
    /// If no `shaderModuleIndex` is not specified, the last shader module added to the program will be used.
    /// @param shaderModuleIndex The index of the shader module to use for this entry point group.
    /// @return Returns a reference to the newly created entry point group for adding entry points.
    EntryPointGroup& addEntryPointGroup(uint32_t shaderModuleIndex = uint32_t(-1))
    {
        if (shaderModules.empty())
            FALCOR_THROW("Can't add entry point group without a shader module");
        if (shaderModuleIndex == uint32_t(-1))
            shaderModuleIndex = uint32_t(shaderModules.size()) - 1;
        if (shaderModuleIndex >= shaderModules.size())
            FALCOR_THROW("Invalid shader module index");
        entryPointGroups.push_back({shaderModuleIndex});
        return entryPointGroups.back();
    }

    /// Helper for adding an entry point defined in the most recent added shader module.
    ProgramDesc& addEntryPoint(ShaderType shaderType, std::string_view name)
    {
        if (entryPointGroups.empty() || entryPointGroups.back().shaderModuleIndex != shaderModules.size() - 1)
            addEntryPointGroup();
        entryPointGroups.back().addEntryPoint(shaderType, name);
        return *this;
    }

    ProgramDesc& vsEntry(const std::string& name) { return addEntryPoint(ShaderType::Vertex, name); }
    ProgramDesc& hsEntry(const std::string& name) { return addEntryPoint(ShaderType::Hull, name); }
    ProgramDesc& dsEntry(const std::string& name) { return addEntryPoint(ShaderType::Domain, name); }
    ProgramDesc& gsEntry(const std::string& name) { return addEntryPoint(ShaderType::Geometry, name); }
    ProgramDesc& psEntry(const std::string& name) { return addEntryPoint(ShaderType::Pixel, name); }
    ProgramDesc& csEntry(const std::string& name) { return addEntryPoint(ShaderType::Compute, name); }

    /// Checks if the program description has at least one entry point of a given type.
    bool hasEntryPoint(ShaderType stage) const
    {
        for (const auto& group : entryPointGroups)
            for (const auto& entryPoint : group.entryPoints)
                if (entryPoint.type == stage)
                    return true;
        return false;
    }

    /// Set the global type conformances.
    ProgramDesc& setTypeConformances(TypeConformanceList conformances)
    {
        typeConformances = std::move(conformances);
        return *this;
    }

    /// Add global type conformances.
    ProgramDesc& addTypeConformances(const TypeConformanceList& conformances)
    {
        typeConformances.add(conformances);
        return *this;
    }

    /// Set the shader model.
    ProgramDesc& setShaderModel(ShaderModel shaderModel_)
    {
        shaderModel = shaderModel_;
        return *this;
    }

    /// Set the compiler flags.
    ProgramDesc& setCompilerFlags(SlangCompilerFlags flags)
    {
        compilerFlags = flags;
        return *this;
    }

    /// Set the compiler arguments (as set on the compiler command line).
    ProgramDesc& setCompilerArguments(std::vector<std::string> args)
    {
        compilerArguments = std::move(args);
        return *this;
    }

    /// Add compiler arguments (as set on the compiler command line).
    ProgramDesc& addCompilerArguments(const std::vector<std::string>& args)
    {
        compilerArguments.insert(compilerArguments.end(), args.begin(), args.end());
        return *this;
    }

    //
    // Compatibility functions
    //

    /**
     * Add a raygen shader.
     * @param[in] raygen Entry point for the raygen shader.
     * @param[in] typeConformances Optional list of type conformances for the raygen shader.
     * @param[in] entryPointNameSuffix Optional suffix added to the entry point names in the generated code.
     * @return Shader ID for raygen shader. This is used when building the binding table.
     */
    ShaderID addRayGen(
        const std::string& raygen,
        const TypeConformanceList& typeConformances_ = TypeConformanceList(),
        const std::string& entryPointNameSuffix = ""
    )
    {
        FALCOR_CHECK(!raygen.empty(), "'raygen' entry point name must not be empty");

        auto& group = addEntryPointGroup();
        group.setTypeConformances(typeConformances_);
        group.addEntryPoint(ShaderType::RayGeneration, raygen, raygen + entryPointNameSuffix);

        return ShaderID{int32_t(entryPointGroups.size() - 1)};
    }

    /**
     * Add a miss shader.
     * @param[in] miss Entry point for the miss shader.
     * @param[in] typeConformances Optional list of type conformances for the miss shader.
     * @param[in] entryPointNameSuffix Optional suffix added to the entry point names in the generated code.
     * @return Shader ID for miss shader. This is used when building the binding table.
     */
    ShaderID addMiss(
        const std::string& miss,
        const TypeConformanceList& typeConformances_ = TypeConformanceList(),
        const std::string& entryPointNameSuffix = ""
    )
    {
        FALCOR_CHECK(!miss.empty(), "'miss' entry point name must not be empty");

        auto& group = addEntryPointGroup();
        group.setTypeConformances(typeConformances_);
        group.addEntryPoint(ShaderType::Miss, miss, miss + entryPointNameSuffix);

        return ShaderID{int32_t(entryPointGroups.size() - 1)};
    }

    /**
     * Add a hit group.
     * A hit group consists of any combination of closest hit, any hit, and intersection shaders.
     * Note that a hit group that contains an intersection shader only be used with procedural geometry.
     * A hit group that does not contain an intersection shader can only be used with triangle geometry.
     * It is valid to create a hit group entirely without entry points. Geometry using it will act
     * as an occluder blocking miss shader exuection, but hits will not spawn any shader executions.
     * @param[in] closestHit Entry point for the closest hit shader.
     * @param[in] anyHit Entry point for the any hit shader.
     * @param[in] intersection Entry point for the intersection shader.
     * @param[in] typeConformances Optional list of type conformances for the hit group.
     * @param[in] entryPointNameSuffix Optional suffix added to the entry point names in the generated code.
     * @return Shader ID for hit group. This is used when building the binding table.
     */
    ShaderID addHitGroup(
        const std::string& closestHit,
        const std::string& anyHit = "",
        const std::string& intersection = "",
        const TypeConformanceList& typeConformances_ = TypeConformanceList(),
        const std::string& entryPointNameSuffix = ""
    )
    {
        FALCOR_CHECK(
            !(closestHit.empty() && anyHit.empty() && intersection.empty()),
            "At least one of 'closestHit', 'anyHit' or 'intersection' entry point names must not be empty"
        );

        auto& group = addEntryPointGroup();
        group.setTypeConformances(typeConformances_);
        if (!closestHit.empty())
            group.addEntryPoint(ShaderType::ClosestHit, closestHit, closestHit + entryPointNameSuffix);
        if (!anyHit.empty())
            group.addEntryPoint(ShaderType::AnyHit, anyHit, anyHit + entryPointNameSuffix);
        if (!intersection.empty())
            group.addEntryPoint(ShaderType::Intersection, intersection, intersection + entryPointNameSuffix);

        return ShaderID{int32_t(entryPointGroups.size() - 1)};
    }

    /**
     * Set the max recursion depth.
     * @param[in] maxDepth The maximum ray recursion depth (0 = raygen).
     */
    ProgramDesc& setMaxTraceRecursionDepth(uint32_t maxDepth)
    {
        maxTraceRecursionDepth = maxDepth;
        return *this;
    }

    /**
     * Set the max payload size.
     */
    ProgramDesc& setMaxPayloadSize(uint32_t maxPayloadSize_)
    {
        maxPayloadSize = maxPayloadSize_;
        return *this;
    }

    /**
     * Set the max attribute size.
     * @param[in] maxAttributeSize The maximum attribute size in bytes.
     */
    ProgramDesc& setMaxAttributeSize(uint32_t maxAttributeSize_)
    {
        maxAttributeSize = maxAttributeSize_;
        return *this;
    }

    /**
     * Set raytracing pipeline flags.
     * These flags are added to any TraceRay() call within this pipeline, and may be used to
     * optimize the pipeline for particular primitives types. Requires Tier 1.1 support.
     * @param[in] flags Pipeline flags.
     */
    ProgramDesc& setRtPipelineFlags(RtPipelineFlags flags)
    {
        rtPipelineFlags = flags;
        return *this;
    }

    void finalize();
};

/**
 * High-level abstraction of a program class.
 * This class manages different versions of the same program. Different versions means same shader files, different macro definitions.
 * This allows simple usage in case different macros are required - for example static vs. animated models.
 */
class FALCOR_API Program : public Object
{
    FALCOR_OBJECT(Program)
public:
    Program(ref<Device> pDevice, ProgramDesc desc, DefineList programDefines);
    virtual ~Program() override;

    /**
     * Create a new program.
     * Note that this call merely creates a program object. The actual compilation and link happens at a later time.
     * @param[in] pDevice GPU device.
     * @param[in] desc The program description.
     * @param[in] programDefines Optional list of macro definitions to set into the program.
     * @return A new object, or an exception is thrown if creation failed.
     */
    static ref<Program> create(ref<Device> pDevice, ProgramDesc desc, DefineList programDefines = {})
    {
        return make_ref<Program>(std::move(pDevice), std::move(desc), std::move(programDefines));
    }

    /**
     * Create a new compute program from file.
     * Note that this call merely creates a program object. The actual compilation and link happens at a later time.
     * @param[in] pDevice GPU device.
     * @param[in] path Compute program file path.
     * @param[in] csEntry Name of the entry point in the program.
     * @param[in] programDefines Optional list of macro definitions to set into the program.
     * @param[in] flags Optional program compilation flags.
     * @param[in] shaderModel Optional shader model.
     * @return A new object, or an exception is thrown if creation failed.
     */
    static ref<Program> createCompute(
        ref<Device> pDevice,
        const std::filesystem::path& path,
        const std::string& csEntry,
        DefineList programDefines = {},
        SlangCompilerFlags flags = SlangCompilerFlags::None,
        ShaderModel shaderModel = ShaderModel::Unknown
    )
    {
        ProgramDesc d;
        d.addShaderLibrary(path);
        if (shaderModel != ShaderModel::Unknown)
            d.setShaderModel(shaderModel);
        d.setCompilerFlags(flags);
        d.csEntry(csEntry);
        return create(std::move(pDevice), std::move(d), std::move(programDefines));
    }

    /**
     * Create a new graphics program from file.
     * @param[in] pDevice GPU device.
     * @param[in] path Graphics program file path.
     * @param[in] vsEntry Vertex shader entry point. If this string is empty (""), it will use a default vertex shader, which transforms and
     * outputs all default vertex attributes.
     * @param[in] psEntry Pixel shader entry point
     * @param[in] programDefines Optional list of macro definitions to set into the program.
     * @param[in] flags Optional program compilation flags.
     * @param[in] shaderModel Optional shader model.
     * @return A new object, or an exception is thrown if creation failed.
     */
    static ref<Program> createGraphics(
        ref<Device> pDevice,
        const std::filesystem::path& path,
        const std::string& vsEntry,
        const std::string& psEntry,
        DefineList programDefines = {},
        SlangCompilerFlags flags = SlangCompilerFlags::None,
        ShaderModel shaderModel = ShaderModel::Unknown
    )
    {
        ProgramDesc d;
        d.addShaderLibrary(path);
        if (shaderModel != ShaderModel::Unknown)
            d.setShaderModel(shaderModel);
        d.setCompilerFlags(flags);
        d.vsEntry(vsEntry).psEntry(psEntry);
        return create(std::move(pDevice), std::move(d), std::move(programDefines));
    }

    /**
     * Get the API handle of the active program.
     * @return The active program version, or an exception is thrown on failure.
     */
    const ref<const ProgramVersion>& getActiveVersion() const;

    /**
     * Adds a macro definition to the program. If the macro already exists, it will be replaced.
     * @param[in] name The name of define.
     * @param[in] value Optional. The value of the define string.
     * @return True if any macro definitions were modified.
     */
    bool addDefine(const std::string& name, const std::string& value = "");

    /**
     * Add a list of macro definitions to the program. If a macro already exists, it will be replaced.
     * @param[in] dl List of macro definitions to add.
     * @return True if any macro definitions were modified.
     */
    bool addDefines(const DefineList& dl);

    /**
     * Remove a macro definition from the program. If the definition doesn't exist, the function call will be silently ignored.
     * @param[in] name The name of define.
     * @return True if any macro definitions were modified.
     */
    bool removeDefine(const std::string& name);

    /**
     * Removes a list of macro definitions from the program. If a macro doesn't exist, it is silently ignored.
     * @param[in] dl List of macro definitions to remove.
     * @return True if any macro definitions were modified.
     */
    bool removeDefines(const DefineList& dl);

    /**
     * Removes all macro definitions that matches string comparison from the program.
     * @param[in] pos Position of the first character in macro name. If this is greater than the string length, the macro will be silently
     * kept.
     * @param[in] len Length of compared macro name (if the string is shorter, as many characters as possible). A value of string::npos
     * indicates all characters.
     * @param[in] str The comparing string that is matched against macro names.
     * @return True if any macro definitions were modified.
     */
    bool removeDefines(size_t pos, size_t len, const std::string& str);

    /**
     * Set the macro definition list of the active program version.
     * @param[in] dl List of macro definitions.
     * @return True if any macro definition was changed, false otherwise.
     */
    bool setDefines(const DefineList& dl);

    /// Get current macro definitions.
    const DefineList& getDefines() const { return mDefineList; }

    /**
     * Add a type conformance to the program.
     * @param[in] typeName The name of the implementation shader type.
     * @param[in] interfaceType The name of the interface type that `typeName` implements.
     * @param[in] id The ID representing the implementation type. If set to -1, Slang will automatically assign an ID for the type.
     * @return True if any type conformances were added to the program.
     */
    bool addTypeConformance(const std::string& typeName, const std::string interfaceType, uint32_t id);

    /**
     * Remove a type conformance from the program. If the type conformance doesn't exist, the function call will be silently ignored.
     * @param[in] typeName The name of the implementation shader type.
     * @param[in] interfaceType The name of the interface type that `typeName` implements.
     * @return True if any type conformances were modified.
     */
    bool removeTypeConformance(const std::string& typeName, const std::string interfaceType);

    /**
     * Set the type conformance list of the active program version.
     * @param[in] conformances List of type conformances.
     * @return True if any type conformance was changed, false otherwise.
     */
    bool setTypeConformances(const TypeConformanceList& conformances);

    /// Get current type conformances.
    const TypeConformanceList& getTypeConformances() const { return mTypeConformanceList; }

    /**
     * Get the macro definition list of the active program version.
     */
    const DefineList& getDefineList() const { return mDefineList; }

    const ProgramDesc& getDesc() const { return mDesc; }

    /**
     * Get the program reflection for the active program.
     * @return Program reflection object, or an exception is thrown on failure.
     */
    const ref<const ProgramReflection>& getReflector() const { return getActiveVersion()->getReflector(); }

    uint32_t getEntryPointGroupCount() const { return uint32_t(mDesc.entryPointGroups.size()); }
    uint32_t getGroupEntryPointCount(uint32_t groupIndex) const { return (uint32_t)mDesc.entryPointGroups[groupIndex].entryPoints.size(); }
    uint32_t getGroupEntryPointIndex(uint32_t groupIndex, uint32_t entryPointIndexInGroup) const
    {
        return mDesc.entryPointGroups[groupIndex].entryPoints[entryPointIndexInGroup].globalIndex;
    }

    void breakStrongReferenceToDevice();

    ref<RtStateObject> getRtso(RtProgramVars* pVars);

protected:
    friend class ProgramManager;
    friend class ProgramVersion;
    friend class ParameterBlockReflection;

    void validateEntryPoints() const;
    bool link() const;

    BreakableReference<Device> mpDevice;

    // The description used to create this program
    // TODO we should make this const again
    ProgramDesc mDesc;

    DefineList mDefineList;
    TypeConformanceList mTypeConformanceList;

    struct ProgramVersionKey
    {
        DefineList defineList;
        TypeConformanceList typeConformanceList;

        bool operator<(const ProgramVersionKey& rhs) const
        {
            return std::tie(defineList, typeConformanceList) < std::tie(rhs.defineList, rhs.typeConformanceList);
        }
    };

    // We are doing lazy compilation, so these are mutable
    mutable bool mLinkRequired = true;
    mutable std::map<ProgramVersionKey, ref<const ProgramVersion>> mProgramVersions;
    mutable ref<const ProgramVersion> mpActiveVersion;
    void markDirty() { mLinkRequired = true; }

    std::string getProgramDescString() const;

    using string_time_map = std::unordered_map<std::string, time_t>;
    mutable string_time_map mFileTimeMap;

    bool checkIfFilesChanged();
    void reset();

    using StateGraph = Falcor::StateGraph<ref<RtStateObject>, void*>;
    StateGraph mRtsoGraph;
};

} // namespace Falcor
