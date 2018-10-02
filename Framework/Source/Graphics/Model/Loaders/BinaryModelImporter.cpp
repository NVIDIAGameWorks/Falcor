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
#include "Framework.h"
#include "BinaryModelImporter.h"
#include "BinaryModelSpec.h"
#include "../Model.h"
#include "../Mesh.h"
#include "Utils/Platform/OS.h"
#include "API/VertexLayout.h"
#include "Data/VertexAttrib.h"
#include "API/Buffer.h"
#include "BinaryImage.hpp"
#include "API/Formats.h"
#include "API/Texture.h"
#include "Graphics/Material/Material.h"
#include "API/Device.h"
#include <numeric>
#include <cstring>

namespace Falcor
{
    struct TextureData
    {
        uint32_t width  = 0;
        uint32_t height = 0;
        ResourceFormat format = ResourceFormat::Unknown;
        std::vector<uint8_t> data;
        std::string name;
    };

    bool isSpecialFloat(float f)
    {
        uint32_t d = *(uint32_t*)&f;
        // Check the exponent
        d = (d >> 23) & 0xff;
        return d == 0xff;
    }

    static vec3 projectNormalToBitangent(const vec3& normal)
    {
        vec3 bitangent;
        if (abs(normal.x) > abs(normal.y))
        {
            bitangent = vec3(normal.z, 0.f, -normal.x) / length(vec2(normal.x, normal.z));
        }
        else
        {
            bitangent = vec3(0.f, normal.z, -normal.y) / length(vec2(normal.y, normal.z));
        }
        return normalize(bitangent);
    }

    static bool isInvalidVec(const vec3& v)
    {
        return isSpecialFloat(v.x) || isSpecialFloat(v.y) || isSpecialFloat(v.z);
    }

    template<typename posType>
    void generateSubmeshTangentData(
        const std::vector<uint32_t>& indices,
        uint32_t vertexCount,
        const posType* vertexPosData,
        const glm::vec3* vertexNormalData,
        const glm::vec2* texCrdData,
        uint32_t texCrdCount,
        glm::vec3* bitangentData)
    {
        std::memset(bitangentData, 0, vertexCount * sizeof(vec3));

        // calculate the tangent and bitangent for every face
        size_t primCount = indices.size() / 3;
        for(size_t primID = 0; primID < primCount; primID++)
        {
            struct Data
            {
                posType position;
                glm::vec3 normal;
                glm::vec2 uv;
            };
            Data V[3];

            // Get the data
            for(uint32_t i = 0; i < 3; i++)
            {
                uint32_t index = indices[primID * 3 + i];
                V[i].position = vertexPosData[index];
                V[i].normal = vertexNormalData[index];
                V[i].uv = texCrdData ? texCrdData[index * texCrdCount] : vec2(0);
            }

            // Position delta
            posType posDelta[2];
            posDelta[0] = V[1].position - V[0].position;
            posDelta[1] = V[2].position - V[0].position;

            // Texture offset
            glm::vec2 s = V[1].uv - V[0].uv;
            glm::vec2 t = V[2].uv - V[0].uv;

            glm::vec3 tangent;
            glm::vec3 bitangent;

            // when t1, t2, t3 in same position in UV space, just use default UV direction.
            if((s == glm::vec2(0, 0)) || (t == glm::vec2(0, 0)))
            {
                const glm::vec3 &normal = V[0].normal;
                bitangent = projectNormalToBitangent(normal);
                tangent = cross(bitangent, normal);
            }
            else
            {
                float dirCorrection = 1.0f / (s.x * t.y - s.y * t.x);

                // tangent points in the direction where to positive X axis of the texture coord's would point in model space
                // bitangent's points along the positive Y axis of the texture coord's, respectively
                tangent   = (posDelta[0] * t.y - posDelta[1] * t.x) * dirCorrection;
                bitangent = (posDelta[1] * s.x - posDelta[0] * s.y) * dirCorrection;
            }

            // store for every vertex of that face
            for(uint32_t i = 0; i < 3; i++)
            {
                // project tangent and bitangent into the plane formed by the vertex' normal
                glm::vec3 localTangent = tangent - V[i].normal * (glm::dot(tangent, V[i].normal));
                localTangent = glm::normalize(localTangent);
                glm::vec3 localBitangent = bitangent - V[i].normal * (glm::dot(bitangent, V[i].normal));
                localBitangent = glm::normalize(localBitangent);
                localBitangent = localBitangent - localTangent * (glm::dot(localBitangent, localTangent));
                localBitangent = glm::normalize(localBitangent);

                if (isInvalidVec(bitangent) == false)
                {
                    // and write it into the mesh
                    uint32_t index = indices[primID * 3 + i];
                    bitangentData[index] += normalize(localBitangent);
                }
            }
        }

        for (uint32_t v = 0; v < vertexCount; v++)
        {
            bitangentData[v] = normalize(bitangentData[v]);
            if (isInvalidVec(bitangentData[v]))
            {
                bitangentData[v] = projectNormalToBitangent(vertexNormalData[v]);
            }
        }
    }

    static void setTexture(Material* pMaterial, Texture::SharedPtr pTexture, TextureType texType, const std::string& modelName)
    {
        switch(texType)
        {
        case TextureType_Diffuse:
            pMaterial->setBaseColorTexture(pTexture);
            break;
        case TextureType_Normal:
            pMaterial->setNormalMap(pTexture);
            break;
        case TextureType_Specular:
            pMaterial->setSpecularTexture(pTexture);
            break;
        case TextureType_Displacement:
            pMaterial->setHeightMap(pTexture);
            break;
        default:
            logWarning("Texture of Type " + std::to_string(texType) + " is not supported by the material system (model " + modelName + ")");
        }

    }

    static const std::string getSemanticName(AttribType type)
    {
        switch(type)
        {
        case AttribType_Position:
            return VERTEX_POSITION_NAME;
        case AttribType_Normal:
            return VERTEX_NORMAL_NAME;
        case AttribType_Color:
            return VERTEX_DIFFUSE_COLOR_NAME;
        case AttribType_TexCoord:
            return VERTEX_TEXCOORD_NAME;
        case AttribType_Bitangent:
            return VERTEX_BITANGENT_NAME;
        case AttribType_Tangent:
            return "unused";
        default:
            should_not_get_here();
            return "unused";
        }
    }

    static ResourceFormat getFalcorFormat(AttribFormat format, uint32_t components)
    {
        switch(format)
        {
        case AttribFormat_U8:
            switch(components)
            {
            case 1:
                return ResourceFormat::R8Unorm;
            case 2:
                return ResourceFormat::RG8Unorm;
            case 3:
                return ResourceFormat::RGBA8Unorm;
            case 4:
                return ResourceFormat::RGBA8Unorm;
            }
            break;
        case AttribFormat_S32:
            switch(components)
            {
            case 1:
                return ResourceFormat::R32Int;
            case 2:
                return ResourceFormat::RG32Int;
            case 3:
                return ResourceFormat::RGB32Int;
            case 4:
                return ResourceFormat::RGBA32Int;
            }
            break;
        case AttribFormat_F32:
            switch(components)
            {
            case 1:
                return ResourceFormat::R32Float;
            case 2:
                return ResourceFormat::RG32Float;
            case 3:
                return ResourceFormat::RGB32Float;
            case 4:
                return ResourceFormat::RGBA32Float;
            }
            break;
        }
        should_not_get_here();
        return ResourceFormat::Unknown;
    }

    static const uint32_t kUnusedShaderElement = -1;
    static uint32_t getShaderLocation(AttribType type)
    {
        switch(type)
        {
        case AttribType_Position:
            return VERTEX_POSITION_LOC;
        case AttribType_Normal:
            return VERTEX_NORMAL_LOC;
        case AttribType_Color:
            return VERTEX_DIFFUSE_COLOR_LOC;
        case AttribType_TexCoord:
            return VERTEX_TEXCOORD_LOC;
        case AttribType_Bitangent:
            return VERTEX_BITANGENT_LOC;
        case AttribType_Tangent:
            return kUnusedShaderElement;
        default:
            should_not_get_here();
            return kUnusedShaderElement;
        }
    }

    static uint32_t getFormatByteSize(AttribFormat format)
    {
        switch(format)
        {
        case AttribFormat_U8:
            return 1;
        case AttribFormat_S32:
        case AttribFormat_F32:
            return 4;
        default:
            should_not_get_here();
            return 0;
        }
    }

    static ResourceFormat getTextureFormat(FW::ImageFormat::ID formatId)
    {
        // Note: If you make changes here, make sure to update GetFormatFromMapType()
        switch(formatId)
        {
        case FW::ImageFormat::R8_G8_B8:
        case FW::ImageFormat::R8_G8_B8_A8:
            return ResourceFormat::RGBA8Unorm;
        case FW::ImageFormat::A8:
            return ResourceFormat::Alpha8Unorm;
        case FW::ImageFormat::XBGR_8888:
        case FW::ImageFormat::ABGR_8888:
            return ResourceFormat::RGBA8Unorm;
        case FW::ImageFormat::RGB_565:
            return ResourceFormat::R5G6B5Unorm;
        case FW::ImageFormat::RGBA_5551:
            return ResourceFormat::RGB5A1Unorm;
        case FW::ImageFormat::RGB_Vec3f:
            return ResourceFormat::RGB32Float;
        case FW::ImageFormat::RGBA_Vec4f:
            return ResourceFormat::RGBA32Float;
        case FW::ImageFormat::A_F32:
            return ResourceFormat::Alpha32Float;

        case FW::ImageFormat::BGRA_8888:
            return ResourceFormat::BGRA8Unorm;
        case FW::ImageFormat::BGR_888:
            return ResourceFormat::BGRA8Unorm;
        case FW::ImageFormat::RG_88:
            return ResourceFormat::RG8Unorm;
        case FW::ImageFormat::R8:
            return ResourceFormat::R8Unorm;

        case FW::ImageFormat::S3TC_DXT1:
            return ResourceFormat::BC1Unorm;
        case FW::ImageFormat::S3TC_DXT3:
            return ResourceFormat::BC2Unorm;
        case FW::ImageFormat::S3TC_DXT5:
            return ResourceFormat::BC3Unorm;
        case FW::ImageFormat::RGTC_R:
            return ResourceFormat::BC4Unorm;
        case FW::ImageFormat::RGTC_RG:
            return ResourceFormat::BC5Unorm;

        default:
            should_not_get_here();
            return ResourceFormat::Unknown;
        }
    }

    std::string readString(BinaryFileStream& stream)
    {
        int32_t length;
        stream >> length;
        std::vector<char> charVec(length + 1);
        stream.read(&charVec[0], length);
        charVec[length] = 0;
        return std::string(charVec.data());
    }

    bool loadBinaryTextureData(BinaryFileStream& stream, const std::string& modelName, TextureData& data)
    {
        // ImageHeader.
        char tag[9];
        stream.read(tag, 8);
        tag[8] = '\0';
        if(std::string(tag) != "BinImage")
        {
            std::string msg = "Error when loading model " + modelName + ".\nBinary image header corrupted.";
            logError(msg);
            return false;
        }

        int32_t version;
        stream >> version;

        if(version < 1 || version > 2)        
        {
            std::string msg = "Error when loading model " + modelName + ".\nUnsupported binary image version.";
            logError(msg);
            return false;
        }

        int32_t bpp, numChannels;
        stream >> data.width >> data.height >> bpp >> numChannels;
        if(data.width < 0 || data.height < 0 || bpp < 0 || numChannels < 0)
        {
            std::string msg = "Error when loading model " + modelName + ".\nCorrupt binary image version.";
            logError(msg);
            return false;
        }

        int32_t dataSize = -1;
        int32_t formatId = FW::ImageFormat::ID_Max;
        FW::ImageFormat format;
        if(version >= 2)
        {
            stream >> formatId >> dataSize;
            if(formatId < 0 || formatId >= FW::ImageFormat::ID_Generic || dataSize < 0)
            {
                std::string msg = "Error when loading model " + modelName + ".\nCorrupt binary image data (unsupported image format).";
                logError(msg);
                return false;
            }
            format = FW::ImageFormat(FW::ImageFormat::ID(formatId));
        }

        // Array of ImageChannel.
        for(int i = 0; i < numChannels; i++)
        {
            int32_t ctype, cformat;
            FW::ImageFormat::Channel c;
            stream >> ctype >> cformat >> c.wordOfs >> c.wordSize >> c.fieldOfs >> c.fieldSize;
            if(ctype < 0 || cformat < 0 || cformat >= FW::ImageFormat::ChannelFormat_Max ||
                c.wordOfs < 0 || (c.wordSize != 1 && c.wordSize != 2 && c.wordSize != 4) ||
                c.fieldOfs < 0 || c.fieldSize <= 0 || c.fieldOfs + c.fieldSize > c.wordSize * 8 ||
                (cformat == FW::ImageFormat::ChannelFormat_Float && c.fieldSize != 32))
            {
                std::string msg = "Error when loading model " + modelName + ".\nCorrupt binary image data (unsupported floating point format).";
                logError(msg);
                return false;
            }

            c.Type = (FW::ImageFormat::ChannelType)ctype;
            c.format = (FW::ImageFormat::ChannelFormat)cformat;
            format.addChannel(c);
        }

        if(bpp != format.getBPP())
        {
            std::string msg = "Error when loading model " + modelName + ".\nCorrupt binary image data (bits/pixel do not match with format).";
            logError(msg);
            return false;
        }

        // Format
        if(formatId == FW::ImageFormat::ID_Max)
            formatId = format.getID();
        data.format = getTextureFormat(FW::ImageFormat::ID(formatId));

        // Image data.
        const int32_t texelCount = data.width * data.height;
        if(dataSize == -1)
        {
            dataSize = bpp * texelCount;
        }
        size_t storageSize = dataSize;
        if(bpp == 3)
            storageSize = 4 * texelCount;

        data.data.resize(storageSize);
        stream.read(data.data.data(), dataSize);

        // Convert 3-channel 8-bits RGB formats to 4-channel RGBX by adding padding
        if(bpp == 3)
        {
            for(int32_t i=texelCount-1;i>=0;--i)
            {
                data.data[i * 4 + 0] = data.data[i * 3 + 0];
                data.data[i * 4 + 1] = data.data[i * 3 + 1];
                data.data[i * 4 + 2] = data.data[i * 3 + 2];
                data.data[i * 4 + 3] = 0xff;
            }
        }

        return true;
    }

    bool importTextures(std::vector<TextureData>& textures, uint32_t textureCount, BinaryFileStream& stream, const std::string& modelName)
    {
        textures.assign(textureCount, TextureData());

        bool success = true;
        for(uint32_t i = 0; i < textureCount; i++)
        {
            textures[i].name = readString(stream);
            if(loadBinaryTextureData(stream, modelName, textures[i]) == false)
            {
                success = false;
                break;
            }
        }

        // Flush upload heap after every material so we don't accumulate a ton of memory usage when loading a model with a lot of textures
        gpDevice->flushAndSync();
        return success;
    }

    BinaryModelImporter::BinaryModelImporter(const std::string& fullpath) : mModelName(fullpath), mStream(fullpath.c_str(), BinaryFileStream::Mode::Read)
    {
    }

    bool BinaryModelImporter::import(Model& model, const std::string& filename, Model::LoadFlags flags)
    {
        std::string fullpath;
        if(findFileInDataDirectories(filename, fullpath) == false)
        {
            logError(std::string("Can't find model file ") + filename);
            return false;
        }

        BinaryModelImporter loader(fullpath);
        return loader.importModel(model, flags);
    }

    static bool checkVersion(const std::string& formatID, uint32_t version, const std::string& modelName)
    {
        if(std::string(formatID) == "BinScene")
        {
            if(version < 6 || version > 8)
            {
                std::string Msg = "Error when loading model " + modelName + ".\nUnsupported binary scene version " + std::to_string(version);
                logError(Msg);
                return false;
            }
        }
        else if(std::string(formatID) == "BinMesh ")
        {
            if(version < 1 || version > 5)
            {
                std::string Msg = "Error when loading model " + modelName + ".\nUnsupported binary scene version " + std::to_string(version);
                logError(Msg);
                return false;
            }
        }
        else
        {
            std::string Msg = "Error when loading model " + modelName + ".\nNot a binary scene file!";
            logError(Msg);
            return false;
        }
        return true;
    }
    
    ResourceFormat getFormatFromMapType(bool requestSrgb, ResourceFormat originalFormat, TextureType texType)
    {
        if(requestSrgb == false)
        {
            return originalFormat;
        }

        switch(texType)
        {
        case TextureType_Diffuse:
        case TextureType_Specular:
        case TextureType_Environment:
            return srgbToLinearFormat(originalFormat);
        default:
            return originalFormat;
        }
    }
    
    bool BinaryModelImporter::importModel(Model& model, Model::LoadFlags flags)
    {
        // Format ID and version.
        char formatID[9];
        mStream.read(formatID, 8);
        formatID[8] = '\0';

        uint32_t version;
        mStream >> version;

        // Check if the version matches
        if(checkVersion(formatID, version, mModelName) == false)
        {
            return false;
        }

        int numTextureSlots;
        int numAttributesType = AttribType_AORadius + 1;

        switch(version)
        {
        case 1:     numTextureSlots = 0; break;
        case 2:     numTextureSlots = TextureType_Alpha + 1; break;
        case 3:     numTextureSlots = TextureType_Displacement + 1; break;
        case 4:     numTextureSlots = TextureType_Environment + 1; break;
        case 5:     numTextureSlots = TextureType_Specular + 1; break;
        case 6:     numTextureSlots = TextureType_Specular + 1; break;
        case 7:     numTextureSlots = TextureType_Glossiness + 1; break;
        case 8:     numTextureSlots = TextureType_Glossiness + 1; numAttributesType = AttribType_Max; break;
        default:
            should_not_get_here();
            return false;
        }


        // File header
        int32_t numTextures = 0;
        int32_t numMeshes = 0;
        int32_t numInstances = 0;
        int32_t numAttribs_v5 = 0;
        int32_t numVertices_v5 = 0;
        int32_t numSubmeshes_v5 = 0;

        if(version >= 6)
        {
            mStream >> numTextures >> numMeshes >> numInstances;
        }
        else
        {
            numMeshes = 1;
            numInstances = 1;
            mStream >> numAttribs_v5 >> numVertices_v5 >> numSubmeshes_v5;
            if(version >= 2)
            {
                mStream >> numTextures;
            }
        }

        if(numTextures < 0 || numMeshes < 0 || numInstances < 0)
        {
            std::string msg = "Error when loading model " + mModelName + ".\nFile is corrupted.";
            logError(msg);
            return false;
        }

        // create objects
        bool shouldGenerateTangents = is_set(flags, Model::LoadFlags::DontGenerateTangentSpace) == false;

        std::vector<TextureData> texData;

        if(version >= 6)
        {
            importTextures(texData, numTextures, mStream, mModelName);
        }

        // This file format has a concept of sub-meshes, which Falcor model doesn't have - Falcor creates a new mesh for each sub-mesh
        // When creating instances of meshes, it means we need to translate the original mesh index to all it's submeshes Falcor IDs. This is what the next 2 variables are for.
        std::vector<std::vector<uint32_t>> meshToSubmeshesID(numMeshes);

        // This importer loads mesh/submesh data before instance data, so the meshes are cached here.
        std::vector<Mesh::SharedPtr> falcorMeshCache;
        
        struct TexSignature
        {
            const uint8_t* pData;
            ResourceFormat format;
            bool operator<(const TexSignature& other) const 
            { 
                if(pData < other.pData) return true;
                if(pData == other.pData) return format < other.format;
                return false;
            }
            bool operator==(const TexSignature& other) const { return pData == other.pData || format == other.format; }
        };
        std::map<TexSignature, Texture::SharedPtr> textures;
        bool loadTexAsSrgb = !is_set(flags, Model::LoadFlags::AssumeLinearSpaceTextures);

        // Load the meshes
        for(int meshIdx = 0; meshIdx < numMeshes; meshIdx++)
        {
            // Mesh header
            int32_t numAttribs = 0;
            int32_t numVertices = 0;
            int32_t numSubmeshes = 0;

            if(version >= 6)
            {
                mStream >> numAttribs >> numVertices >> numSubmeshes;
            }
            else
            {
                numAttribs = numAttribs_v5;
                numVertices = numVertices_v5;
                numSubmeshes = numSubmeshes_v5;
            }

            if(numAttribs < 0 || numVertices < 0 || numSubmeshes < 0)
            {
                std::string Msg = "Error when loading model " + mModelName + ".\nCorrupted data.!";
                logError(Msg);
                return false;
            }

            Vao::BufferVec pVBs;
            VertexLayout::SharedPtr pLayout = VertexLayout::create();
            
            struct BufferData
            {
                std::vector<uint8_t> vec;
                bool shouldSkip = false;
                uint32_t elementSize = 0;
            };

            std::vector<BufferData> buffers;
            pVBs.resize(numAttribs);
            buffers.resize(numAttribs);

            const uint32_t kInvalidBufferIndex = (uint32_t)-1;
            uint32_t positionBufferIndex = kInvalidBufferIndex;
            uint32_t normalBufferIndex = kInvalidBufferIndex;
            uint32_t bitangentBufferIndex = kInvalidBufferIndex;
            uint32_t texCoordBufferIndex = kInvalidBufferIndex;

            for(int i = 0; i < numAttribs; i++)
            {
                VertexBufferLayout::SharedPtr pBufferLayout = VertexBufferLayout::create();
                pLayout->addBufferLayout(i, pBufferLayout);
                int32_t type, format, length;
                mStream >> type >> format >> length;

                if(type < 0 || type >= numAttributesType || format < 0 || format >= AttribFormat::AttribFormat_Max || length < 1 || length > 4)
                {
                    std::string msg = "Error when loading model " + mModelName + ".\nCorrupted data.!";
                    logError(msg);
                    return false;
                }
                else
                {
                    const std::string falcorName = getSemanticName(AttribType(type));
                    ResourceFormat falcorFormat = getFalcorFormat(AttribFormat(format), length);
                    uint32_t shaderLocation = getShaderLocation(AttribType(type));

                    switch (shaderLocation)
                    {
                    case VERTEX_POSITION_LOC:
                        positionBufferIndex = i;
                        assert(falcorFormat == ResourceFormat::RGB32Float || falcorFormat == ResourceFormat::RGBA32Float);
                        break;
                    case VERTEX_NORMAL_LOC:
                        normalBufferIndex = i;
                        assert(falcorFormat == ResourceFormat::RGB32Float);
                        break;
                    case VERTEX_BITANGENT_LOC:
                        bitangentBufferIndex = i;
                        assert(falcorFormat == ResourceFormat::RGB32Float);
                        break;
                    case VERTEX_TEXCOORD_LOC:
                        texCoordBufferIndex = i;
                        break;
                    }

                    buffers[i].elementSize = getFormatBytesPerBlock(falcorFormat);
                    if(shaderLocation != kUnusedShaderElement)
                    {
                        pBufferLayout->addElement(falcorName, 0, falcorFormat, 1, shaderLocation);
                        buffers[i].vec.resize(buffers[i].elementSize * numVertices);
                    }
                    else
                    {
                        buffers[i].shouldSkip = true;
                    }
                }
            }

            
            // Check if we need to generate tangents  
            bool genTangentForMesh = false;
            if(shouldGenerateTangents && (bitangentBufferIndex == kInvalidBufferIndex))
            {
                if(normalBufferIndex == kInvalidBufferIndex)
                {
                    logWarning("Can't generate tangent space for mesh " + std::to_string(meshIdx) + " when loading model " + mModelName + ".\nMesh doesn't contain normals coordinates\n");
                    genTangentForMesh = false;
                }
                else
                {
                    // Set the offsets
                    genTangentForMesh = true;
                    bitangentBufferIndex = (uint32_t)pVBs.size();
                    pVBs.resize(bitangentBufferIndex + 1);
                    buffers.resize(bitangentBufferIndex + 1);
                   
                    auto pBitangentLayout = VertexBufferLayout::create();
                    pLayout->addBufferLayout(bitangentBufferIndex, pBitangentLayout);
                    pBitangentLayout->addElement(VERTEX_BITANGENT_NAME, 0, ResourceFormat::RGB32Float, 1, VERTEX_BITANGENT_LOC);
                    buffers[bitangentBufferIndex].vec.resize(sizeof(glm::vec3) * numVertices);
                }
            }
            

            // Read the data, one vertex at a time
            for(int32_t i = 0; i < numVertices; i++)
            {
                for (int32_t attributes = 0; attributes < numAttribs; ++attributes)
                {
                    if (buffers[attributes].shouldSkip)
                    {
                        mStream.skip(buffers[attributes].elementSize);
                    }
                    else
                    {
                        uint32_t stride = pLayout->getBufferLayout(attributes)->getStride();
                        uint8_t* pDest = buffers[attributes].vec.data() + stride * i;
                        mStream.read(pDest, stride);
                    }
                }
            }

            Buffer::BindFlags vbBindFlags = Buffer::BindFlags::Vertex;
            if (is_set(flags, Model::LoadFlags::BuffersAsShaderResource))
            {
                vbBindFlags |= Buffer::BindFlags::ShaderResource;
            }

            for (int32_t i = 0; i < numAttribs; ++i)
            {
                if(buffers[i].shouldSkip == false)
                {
                    pVBs[i] = Buffer::create(buffers[i].vec.size(), vbBindFlags, Buffer::CpuAccess::None, buffers[i].vec.data());
                }
            }

            if(version <= 5)
            {
                importTextures(texData, numTextures, mStream, mModelName);
                textures.clear();
            }

            // Array of Submesh.
            // Falcor doesn't have a concept of submeshes, just create a new mesh for each submesh
            for(int submesh = 0; submesh < numSubmeshes; submesh++)
            {
                // create the material
                Material::SharedPtr pMaterial = Material::create("");

                glm::vec3 ambient;
                glm::vec4 diffuse;
                glm::vec3 specular;
                float glossiness;

                mStream >> ambient >> diffuse >> specular >> glossiness;
                diffuse.w = 1 - diffuse.w;
                pMaterial->setBaseColor(diffuse);
                pMaterial->setSpecularParams(vec4(specular, glossiness));

                if(version >= 3)
                {
                    float displacementCoeff;
                    float displacementBias;
                    mStream >> displacementCoeff >> displacementBias;
                    pMaterial->setHeightScaleOffset(displacementCoeff, displacementBias);
                }

                for(int i = 0; i < numTextureSlots; i++)
                {
                    int32_t texID;
                    mStream >> texID;
                    if(texID < -1 || texID >= numTextures)
                    {
                        std::string msg = "Error when loading model " + mModelName + ".\nCorrupt binary mesh data!";
                        logError(msg);
                        return false;
                    }
                    else if(texID != -1)
                    {
                        // Load the texture
                        TexSignature texSig;
                        texSig.format = getFormatFromMapType(loadTexAsSrgb, texData[texID].format, TextureType(i));
                        texSig.pData = texData[texID].data.data();
                        // Check if we already created a matching texture
                        auto existingTex = textures.find(texSig);
                        if(existingTex != textures.end())
                        {
                            setTexture(pMaterial.get(), existingTex->second, TextureType(i), mModelName);
                        }
                        else
                        {
                            auto pTexture = Texture::create2D(texData[texID].width, texData[texID].height, texSig.format, 1, Texture::kMaxPossible, texSig.pData);
                            pTexture->setSourceFilename(texData[texID].name);
                            textures[texSig] = pTexture;
                            setTexture(pMaterial.get(), pTexture, TextureType(i), mModelName);
                        }
                    }
                }

                // Create material and check if it already exists
                pMaterial = checkForExistingMaterial(pMaterial);

                int32_t numTriangles;
                mStream >> numTriangles;
                if(numTriangles < 0)
                {
                    std::string Msg = "Error when loading model " + mModelName + ".\nMesh has negative number of triangles!";
                    logError(Msg);
                    return false;
                }

                // create the index buffer
                uint32_t numIndices = numTriangles * 3;
                std::vector<uint32_t> indices(numIndices);
                uint32_t ibSize = 3 * numTriangles * sizeof(uint32_t);
                mStream.read(&indices[0], ibSize);


                Buffer::BindFlags ibBindFlags = Buffer::BindFlags::Index;
                if (is_set(flags, Model::LoadFlags::BuffersAsShaderResource))
                {
                    ibBindFlags |= Buffer::BindFlags::ShaderResource;
                }
                auto pIB = Buffer::create(ibSize, ibBindFlags, Buffer::CpuAccess::None, indices.data());

                // Generate tangent space data if needed
                if(genTangentForMesh)
                {
                    uint32_t texCrdCount = 0;
                    glm::vec2* texCrd = nullptr;
                    if(texCoordBufferIndex != kInvalidBufferIndex)
                    {
                        texCrdCount = pLayout->getBufferLayout(texCoordBufferIndex)->getStride() / sizeof(glm::vec2);
                        texCrd = (glm::vec2*)buffers[texCoordBufferIndex].vec.data();
                    }

                    ResourceFormat posFormat = pLayout->getBufferLayout(positionBufferIndex)->getElementFormat(0);

                    if (posFormat == ResourceFormat::RGB32Float)
                    {
                        generateSubmeshTangentData<glm::vec3>(indices, numVertices, (glm::vec3*)buffers[positionBufferIndex].vec.data(), (glm::vec3*)buffers[normalBufferIndex].vec.data(), texCrd, texCrdCount, (glm::vec3*)buffers[bitangentBufferIndex].vec.data());
                    }
                    else if (posFormat == ResourceFormat::RGBA32Float)
                    {
                        generateSubmeshTangentData<glm::vec4>(indices, numVertices, (glm::vec4*)buffers[positionBufferIndex].vec.data(), (glm::vec3*)buffers[normalBufferIndex].vec.data(), texCrd, texCrdCount, (glm::vec3*)buffers[bitangentBufferIndex].vec.data());
                    }

                    pVBs[bitangentBufferIndex] = Buffer::create(buffers[bitangentBufferIndex].vec.size(), Buffer::BindFlags::Vertex, Buffer::CpuAccess::None, buffers[bitangentBufferIndex].vec.data());
                }
                

                // Calculate the bounding-box
                glm::vec3 max, min;
                for(uint32_t i = 0; i < numIndices; i++)
                {
                    uint32_t vertexID = indices[i];
                    uint8_t* pVertex = (pLayout->getBufferLayout(positionBufferIndex)->getStride() * vertexID) + buffers[positionBufferIndex].vec.data();

                    float* pPosition = (float*)pVertex;

                    glm::vec3 xyz(pPosition[0], pPosition[1], pPosition[2]);
                    min = glm::min(min, xyz);
                    max = glm::max(max, xyz);
                }

                BoundingBox box = BoundingBox::fromMinMax(min, max);

                // create the mesh
                auto pMesh = Mesh::create(pVBs, numVertices, pIB, numIndices, pLayout, Vao::Topology::TriangleList, pMaterial, box, false);

                if (version >= 6)
                {
                    falcorMeshCache.push_back(pMesh);
                    meshToSubmeshesID[meshIdx].push_back((uint32_t)(falcorMeshCache.size() - 1));
                }
                else
                {
                    model.addMeshInstance(pMesh, glm::mat4());
                }
            }
        }

        if(version >= 6)
        {
            for(int32_t instanceID = 0; instanceID < numInstances; instanceID++)
            {
                int32_t meshIdx = 0;
                int32_t enabled = 1;
                glm::mat4 transformation;

                mStream >> meshIdx >> enabled >> transformation;
                //m_Stream >> inst.name >> inst.metadata;
                readString(mStream);   // Name
                readString(mStream);   // Meta-data

                if(enabled)
                {
                    for(uint32_t i : meshToSubmeshesID[meshIdx])
                    {
                        model.addMeshInstance(falcorMeshCache[i], transformation);
                    }
                }
            }
        }
        
        return true;
    }
}
