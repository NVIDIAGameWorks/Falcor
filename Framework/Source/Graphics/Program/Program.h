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
            explicit Desc(std::string const& path);

            /** Add a file of course code to use.
            This also sets the given file as the "active" source for subsequent entry points.
            */
            Desc& sourceFile(std::string const& path);

            /** Add a string of course code to use.
            This also sets the given file as the "active" source for subsequent entry points.
            */
            Desc& sourceString(std::string const& code);

            /** Adds an entry point based on the "active" source.
            */
            Desc& entryPoint(ShaderType shaderType, std::string const& name = "main");

            inline Desc& vertexEntryPoint()     { return entryPoint(ShaderType::Vertex); }
            inline Desc& hullEntryPoint()       { return entryPoint(ShaderType::Hull); }
            inline Desc& domainEntryPoint()     { return entryPoint(ShaderType::Domain); }
            inline Desc& geometryEntryPoint()   { return entryPoint(ShaderType::Geometry); }

            inline Desc& fragmentEntryPoint()   { return entryPoint(ShaderType::Pixel); }
            inline Desc& pixelEntryPoint()      { return entryPoint(ShaderType::Pixel); }

            inline Desc& computeEntryPoint()    { return entryPoint(ShaderType::Compute); }

            /** Convenience routine that optionally adds a source file and entry point.
            If `path` is empty, this function does nothing.
            Otherwise, is equivalent to:
            sourceFile(path).entryPoint(shaderType)
            */
            Desc& maybeSourceFile(std::string const& path, ShaderType shaderType);

            /** Convenience routine that optionally adds a source file and entry point.
            If `code` is empty, this function does nothing.
            Otherwise, is equivalent to:
            sourceString(code).entryPoint(shaderType)
            */
            Desc& maybeSourceString(std::string const& code, ShaderType shaderType);

            /** Add the default version shader if there is no vertex shader specified
            */
            Desc& addDefaultVertexShaderIfNeeded();

            /** Get the source string associated with a shader stage, or an empty string if no stage found
            */
            const std::string& getShaderSource(ShaderType shaderType) const;

            /** Get the name of the shader entry point associated with a shader stage, or an empty string if no stage found
            */
            const std::string& getShaderEntryPoint(ShaderType shaderType) const;

            /** Enable/disable treat-warnings-as-error compilation flag
            */
            void warningsAsErrors(bool enable) { enable ? shaderFlags |= Shader::CompilerFlags::TreatWarningsAsErrors : shaderFlags &= ~(Shader::CompilerFlags::TreatWarningsAsErrors); }

            /** Enable/disable pre-processed shader dump
            */
            void dumpIntermediates(bool enable) { enable ? shaderFlags |= Shader::CompilerFlags::DumpIntermediates : shaderFlags &= ~(Shader::CompilerFlags::DumpIntermediates); }

            /** Get the compiler flags
            */
            Shader::CompilerFlags getCompilerFlags() const { return shaderFlags; }
        private:
            friend class Program;
            friend class GraphicsProgram;

            /** A chunk of course code, either from a file or a string
            */
            struct Source
            {
                enum class Kind { File, String };

                /** The input path or source text
                If `kind` is `File`, this is the path to the file.
                If `kind` is `String`, this is the raw text.
                */
                std::string value;

                /** The kind of input source code
                */
                Kind kind;
            };

            typedef std::vector<Source> SourceList;

            /** An entry point to be compiled
            */
            struct EntryPoint
            {
                /** The name of the entry-point function
                */
                std::string name;

                /** The index of the source file/string that
                    this entry point will use, or `-1` to
                    indicate that this entry point is disabled.
                */
                int sourceIndex = -1;

                bool isValid() const { return sourceIndex >= 0; }
            };

            /** The input files/strings of source code that make up the program
            */
            SourceList mSources;

            /** The entry points that need to be compiled
            */
            EntryPoint mEntryPoints[kShaderCount];

            /** Index of the "active" source file/string.
            This is the file/string that will be used for subsequent entry points.
            By default, this is invalid, as there are no sources at first.
            */
            int activeSourceIndex = -1;

            /** The compiler flags to use when compiling shaders
            */
            Shader::CompilerFlags shaderFlags = Shader::CompilerFlags::None;
        };

        virtual ~Program() = 0;

        /** Get the API handle of the active program
        */
        ProgramVersion::SharedConstPtr getActiveVersion() const;

        /** Adds a macro definition to the program. If the macro already exists, it will be replaced.
            \param[in] name The name of define
            \param[in] value Optional. The value of the define string
        */
        void addDefine(const std::string& name, const std::string& value = "");

        /** Remove a macro definition from the program. If the definition doesn't exist, the function call will be silently ignored.
            \param[in] name The name of define
        */
        void removeDefine(const std::string& name);

        /** Clear the macro definition list
        */
        void clearDefines() { mDefineList.clear(); }
    
        /** Get the macro definition string of the active program version
        */
        const DefineList& getActiveDefinesList() const { return mDefineList; }

        /** Reload and relink all programs.
        */
        static void reloadAllPrograms();

        /** Update define list
        */
        void replaceAllDefines(const DefineList& dl) { mDefineList = dl; }

    protected:
        Program();

        void init(Desc const& desc, DefineList const& programDefines);

        bool link() const;
        ProgramVersion::SharedPtr preprocessAndCreateProgramVersion(std::string& log) const;
        virtual ProgramVersion::SharedPtr createProgramVersion(std::string& log, const Shader::Blob shaderBlob[kShaderCount]) const;

        // The description used to create this program
        Desc mDesc;

        // Shader strings after being preprocessed for a particular version

        // Reflector for a particular version
        mutable ProgramReflection::SharedPtr mPreprocessedReflector;

        DefineList mDefineList;

        // We are doing lazy compilation, so these are mutable
        mutable bool mLinkRequired = true;
        mutable std::map<const DefineList, ProgramVersion::SharedConstPtr> mProgramVersions;
        mutable ProgramVersion::SharedConstPtr mpActiveProgram = nullptr;

        std::string getProgramDescString() const;
        static std::vector<Program*> sPrograms;

        using string_time_map = std::unordered_map<std::string, time_t>;
        mutable string_time_map mFileTimeMap;

        bool checkIfFilesChanged();
        void reset();
    };
}