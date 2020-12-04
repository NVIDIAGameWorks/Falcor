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
#include "Core/Platform/OS.h"
#include "Utils/Scripting/Dictionary.h"

namespace Falcor
{
    /** Helper class to write Python script code including:
        - calling functions
        - calling member functions
        - getting/setting properties

        Arguments are automatically converted from C++ types to Python code using `repr()`.
    */
    class ScriptWriter
    {
    public:
        struct VariableName
        {
            std::string name;
            explicit VariableName(const std::string& name) : name(name) {}
        };

        static std::string makeFunc(const std::string& func)
        {
            return func + "()\n";
        }

        template<typename T>
        static std::string getArgString(const T& arg)
        {
            return ScriptBindings::repr(arg);
        }

        template<>
        static std::string getArgString(const Dictionary& dictionary)
        {
            return dictionary.toString();
        }

        template<>
        static std::string getArgString(const VariableName& varName)
        {
            return varName.name;
        }

        template<typename Arg, typename...Args>
        static std::string makeFunc(const std::string& func, Arg first, Args...args)
        {
            std::string s = func + "(" + getArgString(first);
            int32_t dummy[] = { 0, (s += ", " + getArgString(args), 0)... };
            s += ")\n";
            return s;
        }

        static std::string makeMemberFunc(const std::string& var, const std::string& func)
        {
            return std::string(var) + "." + makeFunc(func);
        }

        template<typename Arg, typename...Args>
        static std::string makeMemberFunc(const std::string& var, const std::string& func, Arg first, Args...args)
        {
            std::string s(var);
            s += std::string(".") + makeFunc(func, first, args...);
            return s;
        }

        static std::string makeGetProperty(const std::string& var, const std::string& property)
        {
            return var + "." + property + "\n";
        }

        template<typename Arg>
        static std::string makeSetProperty(const std::string& var, const std::string& property, Arg arg)
        {
            return var + "." + property + " = " + getArgString(arg) + "\n";
        }

        static std::string getFilenameString(const std::string& s, bool stripDataDirs = true)
        {
            std::string filename = stripDataDirs ? stripDataDirectories(s) : s;
            std::replace(filename.begin(), filename.end(), '\\', '/');
            return filename;
        }
    };
}
