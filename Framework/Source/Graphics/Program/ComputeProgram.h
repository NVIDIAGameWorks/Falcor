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
    /** Compute program.
    */
    class ComputeProgram : public Program, inherit_shared_from_this<Program, ComputeProgram>
    {
    public:
        using SharedPtr = std::shared_ptr<ComputeProgram>;
        using SharedConstPtr = std::shared_ptr<const ComputeProgram>;
        ~ComputeProgram() = default;

        /** Create a new program object.
            \param[in] filename Compute shader filename. Can also include a full path or relative path from a data directory.
            \param[in] programDefines A list of macro definitions to set into the shader
            \return A new object, or nullptr if creation failed.

            Note that this call merely creates a program object. The actual compilation and link happens when calling Program#getActiveVersion().
        */
        static SharedPtr createFromFile(const std::string& filename, const DefineList& programDefines = DefineList());

        /** Create a new program object.
            \param[in] filename Compute shader string.
            \param[in] programDefines A list of macro definitions to set into the shader

            \return A new object, or nullptr if creation failed.
            Note that this call merely creates a program object. The actual compilation and link happens when calling Program#getActiveVersion().
        */
        static SharedPtr createFromString(const std::string& filename, const DefineList& programDefines = DefineList());

    private:
        ComputeProgram() = default;
    };
}