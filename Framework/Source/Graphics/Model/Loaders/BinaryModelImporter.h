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
#include "Utils/BinaryFileStream.h"
#include "glm/vec3.hpp"
#include "../Model.h"
#include "Graphics/Model/Loaders/ModelImporter.h"

namespace Falcor
{
    class Texture;

    class BinaryModelImporter : public ModelImporter
    {
    public:
        /** import a new model from internal binary format
            \param[in] filename Model's filename. Loader will look for it in the data directories.
            \param[in] flags Flags controlling model creation
            returns nullptr if loading failed, otherwise a new Model object
        */
        static bool import(Model& model, const std::string& filename, Model::LoadFlags flags);

    private:
        BinaryModelImporter(const std::string& fullpath);
        bool importModel(Model& model, Model::LoadFlags flags);

        std::string mModelName;
        BinaryFileStream mStream;

        struct TangentSpace
        {
            glm::vec3 tangent;
            glm::vec3 bitangent;
        };

        static const uint32_t kInvalidOffset = uint32_t(-1);
    };
}
