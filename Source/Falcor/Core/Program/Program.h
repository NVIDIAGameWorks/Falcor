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
#include "Core/API/ShaderType.h"
#include <filesystem>
#include <memory>
#include <string_view>
#include <string>
#include <map>
#include <unordered_map>
#include <vector>

namespace Falcor
{
/**
 * High-level abstraction of a program class.
 * This class manages different versions of the same program. Different versions means same shader files, different macro definitions.
 * This allows simple usage in case different macros are required - for example static vs. animated models.
 */
class FALCOR_API Program : public Object
{
    FALCOR_OBJECT(Program)
public:
    using ArgumentList = std::vector<std::string>;

    enum class CompilerFlags
    {
        None = 0x0,
        TreatWarningsAsErrors = 0x1,
        DumpIntermediates = 0x2,
        FloatingPointModeFast = 0x4,
        FloatingPointModePrecise = 0x8,
        GenerateDebugInfo = 0x10,
        MatrixLayoutColumnMajor = 0x20, // Falcor is using row-major, use this only to compile stand-alone external shaders.
    };

    /**
     * Representing a shader implementation of an interface.
     * When linked into a `ProgramVersion`, the specialized shader will contain
     * the implementation of the specified type in a dynamic dispatch function.
     */
    struct TypeConformance
    {
        std::string mTypeName;
        std::string mInterfaceName;
        TypeConformance() = default;
        TypeConformance(const std::string& typeName, const std::string& interfaceName) : mTypeName(typeName), mInterfaceName(interfaceName)
        {}
        bool operator<(const TypeConformance& other) const
        {
            return mTypeName < other.mTypeName || (mTypeName == other.mTypeName && mInterfaceName < other.mInterfaceName);
        }
        bool operator==(const TypeConformance& other) const
        {
            return mTypeName == other.mTypeName && mInterfaceName == other.mInterfaceName;
        }
        struct HashFunction
        {
            size_t operator()(const TypeConformance& conformance) const
            {
                size_t hash = std::hash<std::string>()(conformance.mTypeName);
                hash = hash ^ std::hash<std::string>()(conformance.mInterfaceName);
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
                add(p.first.mTypeName, p.first.mInterfaceName, p.second);
            return *this;
        }

        /**
         * Remove a type conformance list from the current list
         */
        TypeConformanceList& remove(const TypeConformanceList& cl)
        {
            for (const auto& p : cl)
                remove(p.first.mTypeName, p.first.mInterfaceName);
            return *this;
        }

        TypeConformanceList() = default;
        TypeConformanceList(std::initializer_list<std::pair<const TypeConformance, uint32_t>> il) : std::map<TypeConformance, uint32_t>(il)
        {}
    };

    /**
     * Shader module stored as a string or file.
     */
    struct FALCOR_API ShaderModule
    {
        enum class Type
        {
            String,
            File
        };

        ShaderModule(const std::filesystem::path& path, bool createTranslationUnit = true)
            : type(Type::File), filePath(path), createTranslationUnit(createTranslationUnit){};
        ShaderModule(
            const std::string_view str,
            const std::string_view moduleName,
            const std::string_view modulePath = "",
            bool createTranslationUnit = true
        )
            : type(Type::String), str(str), moduleName(moduleName), modulePath(modulePath), createTranslationUnit(createTranslationUnit){};

        Type type;
        std::filesystem::path filePath; ///< File path to shader source.
        std::string str;                ///< String of shader source.
        std::string moduleName; ///< Slang module name for module created from string. If not creating a new translation unit, this can be
                                ///< left empty.
        std::string modulePath; ///< Virtual file path to module created from string. This is just used for diagnostics purposesand can be
                                ///< left empty.
        bool createTranslationUnit; ///< Create new Slang translation unit for the module.
    };

    using ShaderModuleList = std::vector<ShaderModule>;

    /**
     * Description of a program to be created.
     */
    class FALCOR_API Desc
    {
    public:
        /**
         * Begin building a description, that initially has no source files or entry points.
         */
        Desc();

        /**
         * Begin building a description, based on a single path for source code.
         * This is equivalent to: `Desc().addShaderLibrary(path)`
         * @param[in] path Path to the source code.
         */
        explicit Desc(const std::filesystem::path& path);

        /**
         * Add a language specific (e.g. HLSL, GLSL) prelude to the shader.
         * @param[in] prelude Source string.
         */
        Desc& setLanguagePrelude(const std::string_view prelude);

        /**
         * Add a file of source code to use.
         * This also sets the given file as the "active" source for subsequent entry points.
         * @param[in] path Path to the source code.
         * @param[in] createTranslationUnit Whether a new Slang translation unit should be created, otherwise the source is added to the
         * previous translation unit.
         */
        Desc& addShaderLibrary(const std::filesystem::path& path, bool createTranslationUnit = true)
        {
            return addShaderModule(ShaderModule(path, createTranslationUnit));
        }

        /**
         * Add a string of source code to use.
         * This also sets the given string as the "active" source for subsequent entry points.
         * If `createTranslationUnit` is false, the source is directly visible to the previously added source.
         * If true, a new translation unit is created and the source has to be imported using the supplied `moduleName`.
         * Note that the source string has to be added *before* any source that imports it.
         * @param[in] shader Source code.
         * @param[in] moduleName Slang module name. If not creating a new translation unit, this can be left empty.
         * @param[in] modulePath Virtual file path to module created from string. This is just used for diagnostics purposes and can be left
         * empty.
         * @param[in] createTranslationUnit Whether a new Slang translation unit should be created, otherwise the source is added to the
         * previous translation unit.
         */
        Desc& addShaderString(
            const std::string_view shader,
            const std::string_view moduleName,
            const std::string_view modulePath = "",
            bool createTranslationUnit = true
        )
        {
            return addShaderModule(ShaderModule(shader, moduleName, modulePath, createTranslationUnit));
        }

        /**
         * Add a shader module.
         * This also sets the given module as "active" for subsequent entry points.
         * Note that the module has to be added *before* any module that imports it.
         */
        Desc& addShaderModule(const ShaderModule& module);

        /**
         * Add a list of shader modules.
         * Note that the modules have to be added *before* any module that imports them.
         */
        Desc& addShaderModules(const ShaderModuleList& modules);

        /**
         * Adds an entry point based on the "active" source.
         */
        Desc& entryPoint(ShaderType shaderType, const std::string& name);

        bool hasEntryPoint(ShaderType stage) const;

        Desc& vsEntry(const std::string& name) { return entryPoint(ShaderType::Vertex, name); }
        Desc& hsEntry(const std::string& name) { return entryPoint(ShaderType::Hull, name); }
        Desc& dsEntry(const std::string& name) { return entryPoint(ShaderType::Domain, name); }
        Desc& gsEntry(const std::string& name) { return entryPoint(ShaderType::Geometry, name); }
        Desc& psEntry(const std::string& name) { return entryPoint(ShaderType::Pixel, name); }
        Desc& csEntry(const std::string& name) { return entryPoint(ShaderType::Compute, name); }

        /**
         * Adds a list of type conformances.
         * The type conformances are linked into all shaders in the program.
         */
        Desc& addTypeConformances(const TypeConformanceList& typeConformances)
        {
            mTypeConformances.add(typeConformances);
            return *this;
        }

        /**
         * Set the shader model string.
         * This should be `6_0`, `6_1`, `6_2`, `6_3`, `6_4`, or `6_5`. The default is `6_3`.
         */
        Desc& setShaderModel(const std::string& sm);

        /**
         * Get the compiler flags.
         */
        CompilerFlags getCompilerFlags() const { return mShaderFlags; }

        /**
         * Set the compiler flags. Replaces any previously set flags.
         */
        Desc& setCompilerFlags(CompilerFlags flags)
        {
            mShaderFlags = flags;
            return *this;
        }

        /**
         * Get additional compiler arguments.
         */
        const ArgumentList& getCompilerArguments() const { return mCompilerArguments; }

        /**
         * Set additional compiler arguments. Replaces any previously set arguments.
         */
        Desc& setCompilerArguments(const ArgumentList& arguments)
        {
            mCompilerArguments = arguments;
            return *this;
        }

        /**
         * Add additional compiler arguments. Appended to previously set arguments.
         */
        Desc& addCompilerArguments(const ArgumentList& arguments)
        {
            mCompilerArguments.insert(mCompilerArguments.end(), arguments.begin(), arguments.end());
            return *this;
        }

    protected:
        friend class Program;
        friend class RtProgram;
        friend class ProgramManager;

        Desc& beginEntryPointGroup(const std::string& entryPointNameSuffix = "");
        Desc& addTypeConformancesToGroup(const TypeConformanceList& typeConformances);
        uint32_t declareEntryPoint(ShaderType type, const std::string& name);

        struct SourceEntryPoints
        {
            SourceEntryPoints(const ShaderModule& src) : source(src) {}
            ShaderModule::Type getType() const { return source.type; }

            ShaderModule source;               ///< Shader module source stored as a string or file.
            std::vector<uint32_t> entryPoints; ///< Indices into `mEntryPoints` for all entry points in the module.
        };

        struct EntryPointGroup
        {
            std::vector<uint32_t> entryPoints;    ///< Indices into `mEntryPoints` for all entry points in the group.
            TypeConformanceList typeConformances; ///< Type conformances linked into all shaders in the group.
            std::string nameSuffix;               ///< Suffix added to the entry point names by Slang's code generation.
        };

        struct EntryPoint
        {
            std::string name;       ///< Name of the entry point in the shader source.
            std::string exportName; ///< Name of the entry point in the generated code.
            ShaderType stage;
            int32_t sourceIndex;
            int32_t groupIndex; ///< Entry point group index.
        };

        std::vector<SourceEntryPoints> mSources;
        std::vector<EntryPointGroup> mGroups;
        std::vector<EntryPoint> mEntryPoints;
        TypeConformanceList mTypeConformances; ///< Type conformances linked into all shaders in the program.

        int32_t mActiveSource = -1; ///< Current source index.
        int32_t mActiveGroup = -1;  ///< Current entry point index.
        CompilerFlags mShaderFlags = CompilerFlags::None;
        ArgumentList mCompilerArguments;
        std::string mShaderModel = "6_3";
        std::string mLanguagePrelude;
    };

    virtual ~Program() = 0;

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

    /**
     * Get the macro definition list of the active program version.
     */
    const DefineList& getDefineList() const { return mDefineList; }

    /**
     * Get the program reflection for the active program.
     * @return Program reflection object, or an exception is thrown on failure.
     */
    const ref<const ProgramReflection>& getReflector() const { return getActiveVersion()->getReflector(); }

    uint32_t getEntryPointGroupCount() const { return uint32_t(mDesc.mGroups.size()); }
    uint32_t getGroupEntryPointCount(uint32_t groupIndex) const { return (uint32_t)mDesc.mGroups[groupIndex].entryPoints.size(); }
    uint32_t getGroupEntryPointIndex(uint32_t groupIndex, uint32_t entryPointIndexInGroup) const
    {
        return mDesc.mGroups[groupIndex].entryPoints[entryPointIndexInGroup];
    }

    void breakStrongReferenceToDevice();

protected:
    friend class ProgramManager;
    friend class ProgramVersion;
    friend class ParameterBlockReflection;

    Program(ref<Device> pDevice, const Desc& desc, const DefineList& programDefines);

    void validateEntryPoints() const;
    bool link() const;

    BreakableReference<Device> mpDevice;

    // The description used to create this program
    const Desc mDesc;

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
};

FALCOR_ENUM_CLASS_OPERATORS(Program::CompilerFlags);

} // namespace Falcor
