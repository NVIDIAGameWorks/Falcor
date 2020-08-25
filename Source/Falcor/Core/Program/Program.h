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
#pragma once
#include "Core/API/Shader.h"
#include "Core/Program/ShaderLibrary.h"
#include "Core/Program/ProgramVersion.h"

namespace Falcor
{
    /** High-level abstraction of a program class.
        This class manages different versions of the same program. Different versions means same shader files, different macro definitions.
        This allows simple usage in case different macros are required - for example static vs. animated models.
    */
    class dlldecl Program : public std::enable_shared_from_this<Program>
    {
    public:
        using SharedPtr = std::shared_ptr<Program>;
        using SharedConstPtr = std::shared_ptr<const Program>;

        using DefineList = Shader::DefineList;

        /** Description of a program to be created.
        */
        class dlldecl Desc
        {
        public:
            /** Begin building a description, that initially has no source files or entry points.
            */
            Desc();

            /** Begin building a description, based on a single path for source code.
                This is equivalent to: `Desc().sourceFile(path)`
            */
            explicit Desc(std::string const& filename);

            /** Add a file of source code to use.
                This also sets the given file as the "active" source for subsequent entry points.
            */
            Desc& addShaderLibrary(const std::string& path);

            /** Add a string of source code to use.
                This also sets the given string as the "active" source for subsequent entry points.
            */
            Desc& addShaderString(const std::string& shader);

            /** Adds an entry point based on the "active" source.
            */
            Desc& entryPoint(ShaderType shaderType, const std::string& name);

            Desc& vsEntry(const std::string& name) { return entryPoint(ShaderType::Vertex, name); }
            Desc& hsEntry(const std::string& name) { return entryPoint(ShaderType::Hull, name); }
            Desc& dsEntry(const std::string& name) { return entryPoint(ShaderType::Domain, name); }
            Desc& gsEntry(const std::string& name) { return entryPoint(ShaderType::Geometry, name); }
            Desc& psEntry(const std::string& name) { return entryPoint(ShaderType::Pixel, name); }
            Desc& csEntry(const std::string& name) { return entryPoint(ShaderType::Compute, name); }

            /** Enable/disable treat-warnings-as-error compilation flag
            */
            Desc& warningsAsErrors(bool enable) { enable ? mShaderFlags |= Shader::CompilerFlags::TreatWarningsAsErrors : mShaderFlags &= ~(Shader::CompilerFlags::TreatWarningsAsErrors); return *this; }

            /** Enable/disable pre-processed shader dump
            */
            Desc& dumpIntermediates(bool enable) { enable ? mShaderFlags |= Shader::CompilerFlags::DumpIntermediates : mShaderFlags &= ~(Shader::CompilerFlags::DumpIntermediates); return *this; }

            /** Set the shader model string. This depends on the API you are using.
                For DirectX it should be `4_0`, `4_1`, `5_0`, `5_1`, `6_0`, `6_1`, `6_2`, or `6_3`. The default is `6_0`. Shader model `6.x` will use dxcompiler, previous shader models use fxc.
                For Vulkan, it should be `400`, `410`, `420`, `430`, `440` or `450`. The default is `450`
            */
            Desc& setShaderModel(const std::string& sm);

            /** Get the compiler flags
            */
            Shader::CompilerFlags getCompilerFlags() const { return mShaderFlags; }

            /** Set the compiler flags. Replaces any previously set flags.
            */
            Desc& setCompilerFlags(Shader::CompilerFlags flags) { mShaderFlags = flags; return *this; }


            bool hasEntryPoint(ShaderType stage) const;

        protected:
            friend class Program;
            friend class GraphicsProgram;
            friend class RtProgram;

            Desc& beginEntryPointGroup();
            Desc& addDefaultVertexShaderIfNeeded();

            struct Source
            {
                enum class Type
                {
                    String,
                    File
                };

                Source(ShaderLibrary::SharedPtr pLib) : pLibrary(pLib), type(Type::File) {};
                Source(std::string s) : str(s), type(Type::String) {};

                Type type;
                ShaderLibrary::SharedPtr pLibrary;
                std::string str;

                uint32_t firstEntryPoint = 0;
                uint32_t entryPointCount = 0;
            };


            struct EntryPointGroup
            {
                uint32_t firstEntryPoint;
                uint32_t entryPointCount;
            };

            struct EntryPoint
            {
                std::string name;
                ShaderType stage;

                int32_t sourceIndex;
                int32_t groupIndex;
            };

            std::vector<Source> mSources;
            std::vector<EntryPointGroup> mGroups;
            std::vector<EntryPoint> mEntryPoints;

            int32_t mActiveSource = -1;
            int32_t mActiveGroup = -1;
            Shader::CompilerFlags mShaderFlags = Shader::CompilerFlags::None;
#ifdef FALCOR_VK
            std::string mShaderModel = "450";
#elif defined FALCOR_D3D12
            std::string mShaderModel = "6_2";
#endif
        };

        virtual ~Program() = 0;

        /** Get the API handle of the active program.
            \return The active program version, or an exception is thrown on failure.
        */
        const ProgramVersion::SharedConstPtr& getActiveVersion() const;

        /** Adds a macro definition to the program. If the macro already exists, it will be replaced.
            \param[in] name The name of define.
            \param[in] value Optional. The value of the define string.
            \return True if any macro definitions were modified.
        */
        bool addDefine(const std::string& name, const std::string& value = "");

        /** Add a list of macro definitions to the program. If a macro already exists, it will be replaced.
            \param[in] dl List of macro definitions to add.
            \return True if any macro definitions were modified.
        */
        bool addDefines(const DefineList& dl);

        /** Remove a macro definition from the program. If the definition doesn't exist, the function call will be silently ignored.
            \param[in] name The name of define.
            \return True if any macro definitions were modified.
        */
        bool removeDefine(const std::string& name);

        /** Removes a list of macro definitions from the program. If a macro doesn't exist, it is silently ignored.
            \param[in] dl List of macro definitions to remove.
            \return True if any macro definitions were modified.
        */
        bool removeDefines(const DefineList& dl);

        /** Removes all macro definitions that matches string comparison from the program.
            \param[in] pos Position of the first character in macro name. If this is greater than the string length, the macro will be silently kept.
            \param[in] len Length of compared macro name (if the string is shorter, as many characters as possible). A value of string::npos indicates all characters.
            \param[in] str The comparing string that is matched against macro names.
            \return True if any macro definitions were modified.
        */
        bool removeDefines(size_t pos, size_t len, const std::string& str);

        /** Set the macro definition list of the active program version.
            \param[in] dl List of macro definitions.
            \return True if any macro definition was changed, false otherwise.
        */
        bool setDefines(const DefineList& dl);

        /** Get the macro definition list of the active program version.
        */
        const DefineList& getDefineList() const { return mDefineList; }

        /** Reload and relink all programs.
            \param[in] forceReload Force reloading all programs.
            \return True if any program was reloaded, false otherwise.
        */
        static bool reloadAllPrograms(bool forceReload = false);

        /** Add a list of defines applied to all programs.
            \param[in] defineList List of macro definitions.
        */
        static void addGlobalDefines(const DefineList& defineList);

        /** Remove a list of defines applied to all programs.
            \param[in] defineList List of macro definitions.
        */
        static void removeGlobalDefines(const DefineList& defineList);

        /** Get the program reflection for the active program.
            \return Program reflection object, or an exception is thrown on failure.
        */
        const ProgramReflection::SharedPtr& getReflector() const { return getActiveVersion()->getReflector(); }

        uint32_t getEntryPointGroupCount() const { return uint32_t(mDesc.mGroups.size()); }
        uint32_t getGroupEntryPointCount(uint32_t groupIndex) const { return mDesc.mGroups[groupIndex].entryPointCount; }
        uint32_t getGroupEntryPointIndex(uint32_t groupIndex, uint32_t entryPointIndexInGroup) const
        {
            return mDesc.mGroups[groupIndex].firstEntryPoint + entryPointIndexInGroup;
        }

    protected:
        friend class ::Falcor::ProgramVersion;

        Program() = default;

        void init(Desc const& desc, DefineList const& programDefines);

        bool link() const;

        SlangCompileRequest* createSlangCompileRequest(
            DefineList  const& defineList) const;

        bool doSlangReflection(
            ProgramVersion const*                       pVersion,
            slang::IComponentType*                      pSlangGlobalScope,
            std::vector<ComPtr<slang::IComponentType>>  pSlangEntryPointGroups,
            ProgramReflection::SharedPtr&               pReflector,
            std::string&                                log) const;

        ProgramVersion::SharedPtr preprocessAndCreateProgramVersion(std::string& log) const;

        ProgramKernels::SharedPtr preprocessAndCreateProgramKernels(
            ProgramVersion const* pVersion,
            ProgramVars    const* pVars,
            std::string         & log) const;

        virtual EntryPointGroupKernels::SharedPtr createEntryPointGroupKernels(
            const std::vector<Shader::SharedPtr>& shaders,
            EntryPointGroupReflection::SharedPtr const& pReflector) const;

        // The description used to create this program
        Desc mDesc;

        DefineList mDefineList;

        // We are doing lazy compilation, so these are mutable
        mutable bool mLinkRequired = true;
        mutable std::map<DefineList, ProgramVersion::SharedConstPtr> mProgramVersions;
        mutable ProgramVersion::SharedConstPtr mpActiveVersion;
        void markDirty() { mLinkRequired = true; }

        std::string getProgramDescString() const;
        static std::vector<std::weak_ptr<Program>> sPrograms;

        using string_time_map = std::unordered_map<std::string, time_t>;
        mutable string_time_map mFileTimeMap;

        bool checkIfFilesChanged();
        void reset();
    };
}
