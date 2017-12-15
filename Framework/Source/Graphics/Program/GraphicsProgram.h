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
#include "Program.h"

namespace Falcor
{
    /** Graphics program. See ComputeProgram to manage compute shaders.
    */
    class GraphicsProgram : public Program, inherit_shared_from_this<Program, GraphicsProgram>
    {
    public:
        using SharedPtr = std::shared_ptr<GraphicsProgram>;
        using SharedConstPtr = std::shared_ptr<const GraphicsProgram>;

        ~GraphicsProgram() = default;

        /** Create a new program object.
            \param[in] desc Description of the source files and entry points to use.
            \return A new object, or nullptr if creation failed.

            Note that this call merely creates a program object. The actual compilation and link happens when calling Program#getActiveVersion().
        */
        static SharedPtr create(Desc const& desc, const Program::DefineList& programDefines = DefineList());

        /** Create a new program object.
            \param[in] vertexFile Vertex shader filename. If this string is empty (""), it will use a default vertex shader which transforms and outputs all default vertex attributes.
            \param[in] fragmentFile Fragment shader filename.
            \param[in] programDefines A list of macro definitions to set into the shaders. The macro definitions will be assigned to all the shaders.
            \return A new object, or nullptr if creation failed.

            Note that this call merely creates a program object. The actual compilation and link happens when calling Program#getActiveVersion().
        */
        static SharedPtr createFromFile(const std::string& vertexFile, const std::string& fragmentFile, const DefineList& programDefines = DefineList());

        /** Create a new program object.
            \param[in] vertexFile Vertex shader string. If this string is empty (""), it will use a default vertex shader which transforms and outputs all default vertex attributes.
            \param[in] fragmentFile Fragment shader string.
            \param[in] programDefines A list of macro definitions to set into the shaders. The macro definitions will be assigned to all the shaders.

            \return A new object, or nullptr if creation failed.
            Note that this call merely creates a program object. The actual compilation and link happens when calling Program#getActiveVersion().
        */
        static SharedPtr createFromString(const std::string& vertexShader, const std::string& fragmentShader, const DefineList& programDefines = DefineList());

        /** Create a new program object.
            \param[in] vertexFile Vertex shader filename. If this string is empty (""), it will use a default vertex shader which transforms and outputs all default vertex attributes.
            \param[in] fragmentFile Fragment shader filename.
            \param[in] geometryFile Geometry shader filename.
            \param[in] hullFile Hull shader filename.
            \param[in] domainFile Domain shader filename.
            \param[in] programDefines A list of macro definitions to set into the shaders.

            \return A new object, or nullptr if creation failed.

            Note that this call merely creates a program object. The actual compilation and link happens when calling Program#getActiveVersion().
        */
        static SharedPtr createFromFile(const std::string& vertexFile, const std::string& fragmentFile, const std::string& geometryFile, const std::string& hullFile, const std::string& domainFile, const DefineList& programDefines = DefineList());

        /** Create a new program object.
            \param[in] vertexShader Vertex shader string. If this string is empty (""), it will use a default vertex shader which transforms and outputs all default vertex attributes.
            \param[in] fragmentShader Fragment shader string.
            \param[in] geometryShader Geometry shader string.
            \param[in] hullShader Hull shader string.
            \param[in] domainShader Domain shader string.
            \param[in] programDefines A list of macro definitions to set into the shaders.
            \return A new object, or nullptr if creation failed.

            Note that this call merely creates a program object. The actual compilation and link happens when calling Program#getActiveVersion().
        */
        static SharedPtr createFromString(const std::string& vertexShader, const std::string& fragmentShader, const std::string& geometryShader, const std::string& hullShader, const std::string& domainShader, const DefineList& programDefines = DefineList());

    private:
        GraphicsProgram() = default;
    };
}