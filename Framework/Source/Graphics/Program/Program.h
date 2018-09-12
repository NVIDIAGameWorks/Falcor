/***************************************************************************
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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
#pragma once
#include <string>
#include <map>
#include <vector>
#include "Graphics/Program//ProgramVersion.h"

namespace Falcor
{
    class Shader;
    class RenderContext;
    class ShaderLibrary;

    /** Common interface for modifying the macro definitions of programs.
        This is a workaround for the fact that RtProgram is currently unrelated to Program.
    */
    class ProgramBase
    {
    public:
        using DefineList = Shader::DefineList;

        /** Adds a macro definition to the program. If the macro already exists, it will be replaced.
            \param[in] name The name of define.
            \param[in] value Optional. The value of the define string.
            \return True if any macro definitions were modified.
        */
        virtual bool addDefine(const std::string& name, const std::string& value = "") = 0;

        /** Add a list of macro definitions to the program. If a macro already exists, it will be replaced.
            \param[in] dl List of macro definitions to add.
            \return True if any macro definitions were modified.
        */
        virtual bool addDefines(const DefineList& dl) = 0;

        /** Remove a macro definition from the program. If the definition doesn't exist, the function call will be silently ignored.
            \param[in] name The name of define.
            \return True if any macro definitions were modified.
        */
        virtual bool removeDefine(const std::string& name) = 0;

        /** Removes a list of macro definitions from the program. If a macro doesn't exist, it is silently ignored.
            \param[in] dl List of macro definitions to remove.
            \return True if any macro definitions were modified.
        */
        virtual bool removeDefines(const DefineList& dl) = 0;

        /** Removes all macro definitions that matches string comparison from the program.
            \param[in] pos Position of the first character in macro name. If this is greater than the string length, the macro will be silently kept.
            \param[in] len Length of compared macro name (if the string is shorter, as many characters as possible). A value of string::npos indicates all characters.
            \param[in] str The comparing string that is matched against macro names.
            \return True if any macro definitions were modified.
        */
        virtual bool removeDefines(size_t pos, size_t len, const std::string& str) = 0;

        /** Set the macro definition list of the active program version. This replaces all previous macro definitions.
            \param[in] dl List of macro definitions.
            \return True if any macro definition was changed, false otherwise.
        */
        virtual bool setDefines(const DefineList& dl) = 0;

        /** Get the macro definition list of the active program version.
        */
        virtual const DefineList& getDefines() const = 0;
    };

    /** High-level abstraction of a program class.
        This class manages different versions of the same program. Different versions means same shader files, different macro definitions. This allows simple usage in case different macros are required - for example static vs. animated models.
    */
    class Program : public ProgramBase, public std::enable_shared_from_this<Program>
    {
    protected:
        static const uint32_t kShaderCount = (uint32_t)ShaderType::Count;

    public:
        using SharedPtr = std::shared_ptr<Program>;
        using SharedConstPtr = std::shared_ptr<const Program>;

        using DefineList = Shader::DefineList;

        /** Description of a program to be created.
        */
        class Desc
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

            /** Get the source library associated with a shader stage, or an empty library if one isn't bound to the shader
            */
            const std::shared_ptr<ShaderLibrary>& getShaderLibrary(ShaderType shaderType) const;

            /** Get the source string associated with a shader stage, or an empty string if one isn't bound to the shader
            */
            const std::string& getShaderString(ShaderType shaderType) const;

            /** Get the name of the shader entry point associated with a shader stage, or an empty string if no stage found
            */
            const std::string& getShaderEntryPoint(ShaderType shaderType) const;

            /** Enable/disable treat-warnings-as-error compilation flag
            */
            Desc& warningsAsErrors(bool enable) { enable ? mShaderFlags |= Shader::CompilerFlags::TreatWarningsAsErrors : mShaderFlags &= ~(Shader::CompilerFlags::TreatWarningsAsErrors); return *this; }

            /** Enable/disable pre-processed shader dump
            */
            Desc& dumpIntermediates(bool enable) { enable ? mShaderFlags |= Shader::CompilerFlags::DumpIntermediates : mShaderFlags &= ~(Shader::CompilerFlags::DumpIntermediates); return *this; }


            /** Set the shader model string. This depends on the API you are using.
            For DirectX it should be `4_0`, `4_1`, `5_0`, `5_1`, `6_0`, `6_1` or `6_2`. The default is `5_1`. Shader model `6.x` will use dxcompiler
            For Vulkan, it should be `400`, `410`, `420`, `430`, `440` or `450`. The default is `450`
            */
            Desc& setShaderModel(const std::string& sm);

            /** Get the compiler flags
            */
            Shader::CompilerFlags getCompilerFlags() const { return mShaderFlags; }

            /** Set the compiler flags. Replaces any previously set flags.
            */
            Desc& setCompilerFlags(Shader::CompilerFlags flags) { mShaderFlags = flags; return *this; }
        private:
            friend class Program;
            friend class GraphicsProgram;

            Desc& addDefaultVertexShaderIfNeeded();

            struct Source
            {
                enum class Type
                {
                    String,
                    File
                };

                Source(std::shared_ptr<ShaderLibrary> pLib) : pLibrary(pLib), type(Type::File) {};
                Source(std::string s) : str(s), type(Type::String) {};

                Type type;
                std::shared_ptr<ShaderLibrary> pLibrary;
                std::string str;
            };

            struct EntryPoint
            {
                std::string name;
                // The index of the shader module that this entry point will use, or `-1` to indicate that this entry point is disabled
                int index = -1;
                bool isValid() const { return index >= 0; }
            };

            std::vector<Source> mSources;
            EntryPoint mEntryPoints[kShaderCount];
            uint32_t mActiveSource = -1;
            Shader::CompilerFlags mShaderFlags = Shader::CompilerFlags::None;
#ifdef FALCOR_VK
            std::string mShaderModel = "450";
#elif defined FALCOR_D3D12
            std::string mShaderModel = "5_1";
#endif
        };

        virtual ~Program() = 0;

        /** Get the API handle of the active program
        */
        ProgramVersion::SharedConstPtr getActiveVersion() const;

        /** Adds a macro definition to the program. If the macro already exists, it will be replaced.
            \param[in] name The name of define.
            \param[in] value Optional. The value of the define string.
            \return True if any macro definitions were modified.
        */
        virtual bool addDefine(const std::string& name, const std::string& value = "") override;

        /** Add a list of macro definitions to the program. If a macro already exists, it will be replaced.
            \param[in] dl List of macro definitions to add.
            \return True if any macro definitions were modified.
        */
        virtual bool addDefines(const DefineList& dl) override;

        /** Remove a macro definition from the program. If the definition doesn't exist, the function call will be silently ignored.
            \param[in] name The name of define.
            \return True if any macro definitions were modified.
        */
        virtual bool removeDefine(const std::string& name) override;

        /** Removes a list of macro definitions from the program. If a macro doesn't exist, it is silently ignored.
            \param[in] dl List of macro definitions to remove.
            \return True if any macro definitions were modified.
        */
        virtual bool removeDefines(const DefineList& dl) override;

        /** Removes all macro definitions that matches string comparison from the program.
            \param[in] pos Position of the first character in macro name. If this is greater than the string length, the macro will be silently kept.
            \param[in] len Length of compared macro name (if the string is shorter, as many characters as possible). A value of string::npos indicates all characters.
            \param[in] str The comparing string that is matched against macro names.
            \return True if any macro definitions were modified.
        */
        virtual bool removeDefines(size_t pos, size_t len, const std::string& str) override;

        /** Sets the macro definition list of the active program version.
            \param[in] dl List of macro definitions.
            \return True if any macro definition was changed, false otherwise.
        */
        virtual bool setDefines(const DefineList& dl) override;

        /** Get the macro definition list of the active program version.
        */
        virtual const DefineList& getDefines() const override { return mDefineList; }

        /** Reload and relink all programs.
        */
        static void reloadAllPrograms();

        deprecate("3.2", "Use setDefines({}) instead")
        bool clearDefines();

        deprecate("3.2", "Use setDefines() instead")
        bool replaceAllDefines(const DefineList& dl);

        deprecate("3.2", "Use getDefines() instead")
        const DefineList& getActiveDefinesList() const { return mDefineList; }

        const ProgramReflection::SharedConstPtr getReflector() const { getActiveVersion(); return mActiveProgram.reflectors.pReflector; }
        const ProgramReflection::SharedConstPtr getLocalReflector() const { getActiveVersion(); return mActiveProgram.reflectors.pLocalReflector; }
        const ProgramReflection::SharedConstPtr getGlobalReflector() const { getActiveVersion(); return mActiveProgram.reflectors.pGlobalReflector; }

    protected:
        Program();

        void init(Desc const& desc, DefineList const& programDefines);

        struct ProgramReflectors
        {
            // Reflector for a particular version
            ProgramReflection::SharedPtr pReflector;
            ProgramReflection::SharedPtr pLocalReflector;
            ProgramReflection::SharedPtr pGlobalReflector;
        };

        struct VersionData
        {
            ProgramVersion::SharedConstPtr pVersion;
            ProgramReflectors reflectors;
        };

        bool link() const;
        VersionData preprocessAndCreateProgramVersion(std::string& log) const;
        virtual ProgramVersion::SharedPtr createProgramVersion(std::string& log, const Shader::Blob shaderBlob[kShaderCount], const ProgramReflectors& reflectors) const;

        // The description used to create this program
        Desc mDesc;

        DefineList mDefineList;

        // We are doing lazy compilation, so these are mutable
        mutable bool mLinkRequired = true;
        mutable std::map<const DefineList, VersionData> mProgramVersions;
        mutable VersionData mActiveProgram;

        std::string getProgramDescString() const;
        static std::vector<Program*> sPrograms;

        using string_time_map = std::unordered_map<std::string, time_t>;
        mutable string_time_map mFileTimeMap;

        bool checkIfFilesChanged();
        void reset();
    };
}