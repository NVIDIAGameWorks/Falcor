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
#include "Graphics/Program/ProgramVersion.h"

struct SlangCompileRequest;

namespace Falcor
{
    class Shader;
    class RenderContext;
    class ShaderLibrary;

    /** Reason why a shader is being compiled.
    */
    enum class CompilePurpose
    {
        ReflectionOnly, ///< Compiling just to get reflection information.
        CodeGen         ///< Compiling to generate executable code.
    };

    /** High-level abstraction of a program class.
    This class manages different versions of the same program. Different versions means same shader files, different macro definitions. This allows simple usage in case different macros are required - for example static vs. animated models.
    */
    class Program : public std::enable_shared_from_this<Program>
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

        /** Clear the macro definition list
            \return True if any macro definitions were modified.
        */
        bool clearDefines();
    
        /** Update define list
            \return True if any macro definitions were modified.
        */
        bool replaceAllDefines(const DefineList& dl);

        /** Get the macro definition string of the active program version
        */
        const DefineList& getActiveDefinesList() const { return mDefineList; }

        /** Reload and relink all programs.
        */
        static void reloadAllPrograms();

        const ProgramReflection::SharedConstPtr getReflector() const { getActiveVersion(); return mActiveProgram->getReflector(); }
        const ProgramReflection::SharedConstPtr getLocalReflector() const { getActiveVersion(); return mActiveProgram->getLocalReflector(); }
        const ProgramReflection::SharedConstPtr getGlobalReflector() const { getActiveVersion(); return mActiveProgram->getGlobalReflector(); }

        /** Get a reflector suitable for creating a parmaeter block the type with the given `name`.
        */
        ParameterBlockReflection::SharedConstPtr getParameterBlockReflectorForType(std::string const& name) const
        {
            return getActiveVersion()->getParameterBlockReflectorForType(name);
        }

    protected:
        friend class ProgramVersion;

        Program();

        void init(Desc const& desc, DefineList const& programDefines);

        /*
        struct VersionData
        {
            ProgramVersion::SharedPtr pVersion;
            ProgramReflectors reflectors;
        };
        */

        bool link() const;
        SlangCompileRequest* createSlangCompileRequest(
            DefineList const&               defines,
            CompilePurpose                  purpose,
            const std::vector<std::string>& typeArgs,
            int*                            outSlangTarget = nullptr) const;
        int Program::doSlangCompilation(SlangCompileRequest* slangRequest, std::string& log) const;
        ProgramVersion::SharedPtr preprocessAndCreateProgramVersion(std::string& log) const;
        ProgramKernels::SharedPtr preprocessAndCreateProgramKernels(
            ProgramVersion const* pVersion,
            ProgramVars    const* pVars,
            std::string         & log) const;
        virtual ProgramKernels::SharedPtr createProgramKernels(std::string& log, const Shader::Blob shaderBlob[kShaderCount], const ProgramReflectors& reflectors) const;
        
        // The description used to create this program
        Desc mDesc;

        DefineList mDefineList;

        // We are doing lazy compilation, so these are mutable
        mutable bool mLinkRequired = true;
        mutable std::map<const DefineList, ProgramVersion::SharedPtr> mProgramVersions;
        mutable ProgramVersion::SharedPtr mActiveProgram;

        std::string getProgramDescString() const;
        static std::vector<Program*> sPrograms;

        using string_time_map = std::unordered_map<std::string, time_t>;
        mutable string_time_map mFileTimeMap;

        bool checkIfFilesChanged();
        void reset();
    };
}