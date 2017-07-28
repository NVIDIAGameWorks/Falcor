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
#include <map>
#include <vector>
#include "Graphics/Model/Mesh.h"

namespace Falcor
{
    class Model;
    class Mesh;
    class Vao;
    class Texture;

    class BinaryModelExporter
    {
    public:
        /** Export a model into a binary file
            \param[in] filename Model's filename or full path
            \param[in] pModel The model to export
        */
        static void exportToFile(const std::string& filename, const Model* pModel);

    private:
        BinaryModelExporter(const std::string& filename, const Model* pModel);
        const Model* mpModel = nullptr;
        BinaryFileStream mStream;
        const std::string& mFilename;

        bool writeHeader();
        bool writeTextures();
        bool writeMeshes();
        bool writeCommonMeshData(const Mesh::SharedPtr& pMesh, uint32_t submeshCount);
        bool writeSubmesh(const Mesh::SharedPtr& pMesh);
        bool writeInstances();

        bool writeMaterialTexture(uint32_t& texID, const Texture::SharedPtr& pTexture);
        
        bool exportBinaryImage(const Texture* pTexture);

        void error(const std::string& Msg);
        void warning(const std::string& Msg);

        bool prepareSubmeshes();
        std::map<const Vao*, std::vector<uint32_t>> mMeshes; // Maps to meshID in model
        std::map<const Texture*, int32_t> mTextureHash;
        uint32_t mInstanceCount = 0; // Not the same as Model::Instance count. Model keeps the total instance count, while the binary format has a concept of meshes and submeshes, and the instance count there is the mesh instance count.
    };
}
