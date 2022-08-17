/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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
#include "SDFGrid.h"
#include "NormalizedDenseSDFGrid/NDSDFGrid.h"
#include "SparseVoxelSet/SDFSVS.h"
#include "SparseBrickSet/SDFSBS.h"
#include "SparseVoxelOctree/SDFSVO.h"
#include "Core/Assert.h"
#include "Core/Errors.h"
#include "Core/API/Device.h"
#include "Core/API/RenderContext.h"
#include "Utils/Logger.h"
#include "Utils/Math/Common.h"
#include "Utils/Scripting/ScriptBindings.h"
#include <rapidjson/rapidjson.h>
#include <rapidjson/document.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/error/en.h>
#include <random>
#include <fstream>

namespace Falcor
{
    namespace
    {
        const std::string kEvaluateSDFPrimitivesShaderName = "Scene/SDFs/EvaluateSDFPrimitives.cs.slang";

        const char kPrimitiveShapeTypeJSONKey[] = "shape_type";
        const char kPrimitiveShapeDataJSONKey[] = "shape_data";
        const char kPrimitiveShapeBlobbingJSONKey[] = "shape_blobbing";

        const char kPrimitiveOperationTypeJSONKey[] = "operation_type";
        const char kPrimitiveOperationSmoothingJSONKey[] = "operation_smoothing";

        const char kPrimitiveTranslationJSONKey[] = "translation";
        const char kPrimitiveInvRotationScaleJSONKey[] = "inv_rot_scale";

        void serializeUint(const char* pKey, uint32_t value, rapidjson::PrettyWriter<rapidjson::StringBuffer>& jsonWriter)
        {
            jsonWriter.String(pKey);
            jsonWriter.Uint(value);
        };

        void serializeFloat(const char* pKey, float value, rapidjson::PrettyWriter<rapidjson::StringBuffer>& jsonWriter)
        {
            jsonWriter.String(pKey);
            jsonWriter.Double((double)value);
        };

        void serializeFloat3(const char* pKey, const float3& value, rapidjson::PrettyWriter<rapidjson::StringBuffer>& jsonWriter)
        {
            jsonWriter.String(pKey);
            jsonWriter.StartArray();
            jsonWriter.Double((double)value.x);
            jsonWriter.Double((double)value.y);
            jsonWriter.Double((double)value.z);
            jsonWriter.EndArray();
        };

        void serializeFloat3x3(const char* pKey, const rmcv::mat3& value, rapidjson::PrettyWriter<rapidjson::StringBuffer>& jsonWriter)
        {
            // matrices stored col-major for backwards compatibility
            rmcv::mat3 m = value.getTranspose();
            jsonWriter.String(pKey);
            jsonWriter.StartArray();
            for(int i = 0; i < 9; ++i)
                jsonWriter.Double((double)m.data()[i]);
            jsonWriter.EndArray();
        };

        bool deserializeUint(const char* pKey, const rapidjson::Value& jsonPrimitive, uint32_t& value)
        {
            const auto jsonMember = jsonPrimitive.FindMember(pKey);

            if (jsonMember == jsonPrimitive.MemberEnd())
            {
                logWarning("JSON member '{}' could not be found!", pKey);
                return false;
            }

            if (!jsonMember->value.IsUint())
            {
                logWarning("JSON member '{}' is not of type uint!", pKey);
                return false;
            }

            value = jsonMember->value.GetUint();
            return true;
        };

        bool deserializeFloat(const char* pKey, const rapidjson::Value& jsonPrimitive, float& value)
        {
            const auto jsonMember = jsonPrimitive.FindMember(pKey);

            if (jsonMember == jsonPrimitive.MemberEnd())
            {
                logWarning("JSON member '{}' could not be found!", pKey);
                return false;
            }

            if (!jsonMember->value.IsNumber())
            {
                logWarning("JSON member '{}' is not a number!", pKey);
                return false;
            }

            value = jsonMember->value.GetFloat();
            return true;
        };

        bool deserializeFloat3(const char* pKey, const rapidjson::Value& jsonPrimitive, float3& value)
        {
            const auto jsonMember = jsonPrimitive.FindMember(pKey);

            if (jsonMember == jsonPrimitive.MemberEnd())
            {
                logWarning("JSON member '{}' could not be found!", pKey);
                return false;
            }

            if (!jsonMember->value.IsArray())
            {
                logWarning("JSON member '{}' is not of type float!", pKey);
                return false;
            }

            for (uint32_t i = 0; i < 3; i++)
            {
                const rapidjson::Value& jsonValue = jsonMember->value[i];

                if (!jsonValue.IsNumber())
                {
                    logWarning("JSON vector index {} of member '{}' is not a number!", i, pKey);
                    return false;
                }

                value[i] = jsonValue.GetFloat();
            }

            return true;
        };

        bool deserializeFloat3x3(const char* pKey, const rapidjson::Value& jsonPrimitive, rmcv::mat3& value)
        {
            const auto jsonMember = jsonPrimitive.FindMember(pKey);

            if (jsonMember == jsonPrimitive.MemberEnd())
            {
                logWarning("JSON member '{}' could not be found!", pKey);
                return false;
            }

            if (!jsonMember->value.IsArray())
            {
                logWarning("JSON member '{}' is not of type float!", pKey);
                return false;
            }

            for (uint32_t i = 0; i < 9; i++)
            {
                const rapidjson::Value& jsonValue = jsonMember->value[i];

                if (!jsonValue.IsNumber())
                {
                    logWarning("JSON matrix index '{}' of member is not a number!", i, pKey);
                    return false;
                }

                /// matrices stored col-major for backwards compatibility
                value[i % 3][i / 3] = jsonValue.GetFloat();
            }

            return true;
        };
    }

    uint32_t SDFGrid::setPrimitives(const std::vector<SDF3DPrimitive>& primitives, uint32_t gridWidth)
    {
        // All types except SBS need to have a gridWidth that is a power of 2.
        Type type = getType();
        if (type != Type::SparseBrickSet)
        {
            // TODO: Expand the grid to match a grid size that is a power of 2 instead of throwing an exception.
            checkArgument(isPowerOf2(gridWidth), "'gridWidth' ({}) must be a power of 2 for SDFGrid type of {}", gridWidth, getTypeName(type));
        }

        mGridWidth = gridWidth;
        mPrimitives.clear();
        mPrimitiveIDToIndex.clear();
        mNextPrimitiveID = 0;

        return addPrimitives(primitives);
    }

    uint32_t SDFGrid::addPrimitives(const std::vector<SDF3DPrimitive>& primitives)
    {
        // Copy primitives.
        uint32_t primitivesStartOffset = (uint32_t)mPrimitives.size();
        mPrimitives.reserve(mPrimitives.size() + primitives.size());
        mPrimitives.insert(mPrimitives.end(), primitives.begin(), primitives.end());

        // Assign indirection.
        mPrimitiveIDToIndex.reserve(mPrimitives.size());
        uint32_t basePrimitiveID = mNextPrimitiveID;

        for (uint32_t idx = primitivesStartOffset; idx < mPrimitives.size(); idx++)
        {
            mPrimitiveIDToIndex[mNextPrimitiveID++] = idx;
        }

        std::unordered_set<uint32_t> indexSet;
        for (const auto& [id, index] : mPrimitiveIDToIndex)
        {
            if (!indexSet.insert(index).second)
            {
                throw RuntimeError("Multiple copies of index {}!", index);
            }
        }

        mPrimitivesDirty = true;

        updatePrimitivesBuffer();

        return basePrimitiveID;
    }

    void SDFGrid::removePrimitives(const std::vector<uint32_t>& primitiveIDs)
    {
        for (uint32_t primitiveID : primitiveIDs)
        {
            auto idxIt = mPrimitiveIDToIndex.find(primitiveID);

            if (idxIt == mPrimitiveIDToIndex.end())
            {
                logWarning("Primitive with ID {} does not exist!", primitiveID);
                continue;
            }

            // Mark as dirty.
            mPrimitivesDirty = true;

            // Baked primitives cannot be removed.
            uint32_t idx = idxIt->second;
            if (idx < mBakedPrimitiveCount)
            {
                logWarning("Primitive with ID {} has been baked, cannot remove it!", primitiveID);
                continue;
            }

            // Erase the index from the indirection map.
            mPrimitiveIDToIndex.erase(idxIt);

            // Compactify the primitive list.
            mPrimitives.erase(mPrimitives.begin() + idx);

            if (idx < mPrimitives.size())
            {
                // Update larger IDs as the primitive list must be compact.
                for (auto& [id, index] : mPrimitiveIDToIndex)
                {
                    if (index > idx)
                    {
                        --index;
                    }
                }
            }
        }

        std::unordered_set<uint32_t> indexSet;
        for (const auto& [id, index] : mPrimitiveIDToIndex)
        {
            if (!indexSet.insert(index).second)
            {
                throw RuntimeError("Multiple copies of index {}!", index);
            }
        }

        updatePrimitivesBuffer();
    }

    void SDFGrid::updatePrimitives(const std::vector<std::pair<uint32_t, SDF3DPrimitive>>& primitives)
    {
        for (auto it = primitives.begin(); it != primitives.end(); it++)
        {
            uint32_t primitiveID = it->first;
            const SDF3DPrimitive& primitive = it->second;

            auto idxIt = mPrimitiveIDToIndex.find(primitiveID);
            if (idxIt == mPrimitiveIDToIndex.end())
            {
                logWarning("Primitive with ID {} does not exist!", primitiveID);
                continue;
            }

            // Mark as dirty.
            mPrimitivesDirty = true;

            // Update the primitive.
            mPrimitives[idxIt->second] = primitive;
        }

        updatePrimitivesBuffer();
    }

    void SDFGrid::setValues(const std::vector<float>& cornerValues, uint32_t gridWidth)
    {
        // All types except SBS need to have a gridWidth that is a power of 2.
        Type type = getType();
        if (type != Type::SparseBrickSet)
        {
            checkArgument(isPowerOf2(gridWidth), "'gridWidth' ({}) must be a power of 2 for SDFGrid type of {}", gridWidth, getTypeName(type));
        }

        mGridWidth = gridWidth;

        setValuesInternal(cornerValues);
    }

    bool SDFGrid::loadValuesFromFile(const std::filesystem::path& path)
    {
        std::filesystem::path fullPath;
        if (findFileInDataDirectories(path, fullPath))
        {
            std::ifstream file(fullPath, std::ios::in | std::ios::binary);

            if (file.is_open())
            {
                uint32_t gridWidth;
                file.read(reinterpret_cast<char*>(&gridWidth), sizeof(uint32_t));

                uint32_t totalValueCount = (gridWidth + 1) * (gridWidth + 1) * (gridWidth + 1);
                std::vector<float> cornerValues(totalValueCount, 0.0f);
                file.read(reinterpret_cast<char*>(cornerValues.data()), totalValueCount * sizeof(float));

                file.close();
                setValues(cornerValues, gridWidth);

                mInitializedWithPrimitives = false;
                return true;
            }
        }

        logWarning("SDFGrid::loadValuesFromFile() file '{}' could not be opened!", path);
        return false;
    }

    void SDFGrid::generateCheeseValues(uint32_t gridWidth, uint32_t seed)
    {
        const float kHalfCheeseExtent = 0.4f;
        const uint32_t kHoleCount = 32;
        float4 holes[kHoleCount];

        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        for (uint32_t s = 0; s < kHoleCount; s++)
        {
            float3 p = 2.0f * kHalfCheeseExtent * float3(dist(rng), dist(rng), dist(rng)) - float3(kHalfCheeseExtent);
            holes[s] = float4(p, dist(rng) * 0.2f + 0.01f);
        }

        uint32_t gridWidthInValues = 1 + gridWidth;
        uint32_t totalValueCount = gridWidthInValues * gridWidthInValues * gridWidthInValues;
        std::vector<float> cornerValues(totalValueCount, 0.0f);

        for (uint32_t z = 0; z < gridWidthInValues; z++)
        {
            for (uint32_t y = 0; y < gridWidthInValues; y++)
            {
                for (uint32_t x = 0; x < gridWidthInValues; x++)
                {
                    float3 pLocal = (float3(x, y, z) / float(gridWidth)) - 0.5f;
                    float sd;

                    // Create a Box.
                    {
                        float3 d = abs(pLocal) - float3(kHalfCheeseExtent);
                        float outsideDist = glm::length(float3(glm::max(d.x, 0.0f), glm::max(d.y, 0.0f), glm::max(d.z, 0.0f)));
                        float insideDist = glm::min(glm::max(glm::max(d.x, d.y), d.z), 0.0f);
                        sd = outsideDist + insideDist;
                    }

                    // Create holes.
                    for (uint32_t s = 0; s < kHoleCount; s++)
                    {
                        float4 holeData = holes[s];
                        sd = glm::max(sd, -(glm::length(pLocal - holeData.xyz) - holeData.w));
                    }

                    // We don't care about distance further away than the length of the diagonal of the unit cube where the SDF grid is defined.
                    cornerValues[x + gridWidthInValues * (y + gridWidthInValues * z)] = glm::clamp(sd, -glm::root_three<float>(), glm::root_three<float>());
                }
            }
        }

        setValues(cornerValues, gridWidth);
    }

    bool SDFGrid::writeValuesFromPrimitivesToFile(const std::filesystem::path& path, RenderContext* pRenderContext)
    {
        if (!pRenderContext) pRenderContext = gpDevice->getRenderContext();

        createEvaluatePrimitivesPass(false, mHasGridRepresentation);

        updatePrimitivesBuffer();

        uint32_t gridWidthInValues = mGridWidth + 1;
        uint32_t valueCount = gridWidthInValues * gridWidthInValues * gridWidthInValues;
        Buffer::SharedPtr pValuesBuffer = Buffer::createTyped<float>(valueCount);
        Buffer::SharedPtr pValuesStagingBuffer = Buffer::createTyped<float>(valueCount, Resource::BindFlags::None, Buffer::CpuAccess::Read);
        GpuFence::SharedPtr pFence = GpuFence::create();

        mpEvaluatePrimitivesPass["CB"]["gGridWidth"] = mGridWidth;
        mpEvaluatePrimitivesPass["CB"]["gPrimitiveCount"] = (uint32_t)mPrimitives.size() - mBakedPrimitiveCount;
        mpEvaluatePrimitivesPass["gPrimitives"] = mpPrimitivesBuffer;
        mpEvaluatePrimitivesPass["gOldValues"] = mHasGridRepresentation ? mpSDFGridTexture : nullptr;
        mpEvaluatePrimitivesPass["gValues"] = pValuesBuffer;
        mpEvaluatePrimitivesPass->execute(pRenderContext, uint3(gridWidthInValues));
        pRenderContext->copyResource(pValuesStagingBuffer.get(), pValuesBuffer.get());
        pRenderContext->flush(false);
        pFence->gpuSignal(pRenderContext->getLowLevelData()->getCommandQueue());
        pFence->syncCpu();
        const float* pValues = reinterpret_cast<const float*>(pValuesStagingBuffer->map(Buffer::MapType::Read));

        std::ofstream file(path, std::ios::out | std::ios::binary);

        if (file.is_open())
        {
            file.write(reinterpret_cast<const char*>(&mGridWidth), sizeof(uint32_t));
            file.write(reinterpret_cast<const char*>(pValues), valueCount * sizeof(float));
            file.close();
        }

        pValuesStagingBuffer->unmap();
        return true;
    }

    uint32_t SDFGrid::loadPrimitivesFromFile(const std::filesystem::path& path, uint32_t gridWidth, const std::filesystem::path& dir)
    {
        std::filesystem::path fullPath;
        if (dir.empty())
        {
            if (!findFileInDataDirectories(path, fullPath))
            {
                logWarning("File '{}' could not be found in data directories!", path);
                return 0;
            }
        }
        else
        {
            fullPath = dir / path;
        }

        std::string jsonData = readFile(fullPath);
        rapidjson::StringStream jsonStream(jsonData.c_str());

        rapidjson::Document jsonDocument;
        jsonDocument.ParseStream(jsonStream);

        if (jsonDocument.HasParseError())
        {
            size_t line;
            line = std::count(jsonData.begin(), jsonData.begin() + jsonDocument.GetErrorOffset(), '\n');
            logWarning("Error when deserializing SDF grid from '{}'. JSON Parse error in line {}: {}", fullPath, line, rapidjson::GetParseError_En(jsonDocument.GetParseError()));
            return 0;
        }

        if (!jsonDocument.IsArray())
        {
            logWarning("Error when deserializing SDF grid from '{}'. JSON document is not of array type!");
            return 0;
        }

        const auto& jsonPrimitivesArray = jsonDocument.GetArray();

        std::vector<SDF3DPrimitive> primitives;
        primitives.resize(jsonPrimitivesArray.Size());

        for (uint32_t i = 0; i < jsonPrimitivesArray.Size(); i++)
        {
            const rapidjson::Value& jsonPrimitive = jsonPrimitivesArray[i];
            SDF3DPrimitive& primitive = primitives[i];

            if (!deserializeUint(kPrimitiveShapeTypeJSONKey, jsonPrimitive, reinterpret_cast<uint32_t&>(primitive.shapeType))) return 0;
            if (!deserializeFloat3(kPrimitiveShapeDataJSONKey, jsonPrimitive, primitive.shapeData)) return 0;
            if (!deserializeFloat(kPrimitiveShapeBlobbingJSONKey, jsonPrimitive, primitive.shapeBlobbing)) return 0;

            if (!deserializeUint(kPrimitiveOperationTypeJSONKey, jsonPrimitive, reinterpret_cast<uint32_t&>(primitive.operationType))) return 0;
            if (!deserializeFloat(kPrimitiveOperationSmoothingJSONKey, jsonPrimitive, primitive.operationSmoothing)) return 0;

            if (!deserializeFloat3(kPrimitiveTranslationJSONKey, jsonPrimitive, primitive.translation)) return 0;
            if (!deserializeFloat3x3(kPrimitiveInvRotationScaleJSONKey, jsonPrimitive, primitive.invRotationScale)) return 0;
        }

        setPrimitives(primitives, gridWidth);

        mInitializedWithPrimitives = true;
        return (uint32_t)mPrimitives.size();
    }

    bool SDFGrid::writePrimitivesToFile(const std::filesystem::path& path)
    {
        std::ofstream file(path, std::ios::out | std::ios::trunc);

        if (!file.is_open())
        {
            logWarning("Failed to open SDF grid file '{}'.", path);
            return false;
        }

        rapidjson::StringBuffer jsonStringBuffer;
        rapidjson::PrettyWriter<rapidjson::StringBuffer> jsonWriter(jsonStringBuffer);

        // Serialize primitives into JSON array.
        jsonWriter.StartArray();
        for (const SDF3DPrimitive& primitive : mPrimitives)
        {
            jsonWriter.StartObject();
            {
                // Shape Type.
                serializeUint(kPrimitiveShapeTypeJSONKey, (uint32_t)primitive.shapeType, jsonWriter);

                // Shape Data.
                serializeFloat3(kPrimitiveShapeDataJSONKey, primitive.shapeData, jsonWriter);

                // Shape Blobbing.
                serializeFloat(kPrimitiveShapeBlobbingJSONKey, primitive.shapeBlobbing, jsonWriter);

                // Operation Type.
                serializeUint(kPrimitiveOperationTypeJSONKey, (uint32_t)primitive.operationType, jsonWriter);

                // Operation Smoothing.
                serializeFloat(kPrimitiveOperationSmoothingJSONKey, primitive.operationSmoothing, jsonWriter);

                // Translation.
                serializeFloat3(kPrimitiveTranslationJSONKey, primitive.translation, jsonWriter);

                // Inverse Rotation Scale.
                serializeFloat3x3(kPrimitiveInvRotationScaleJSONKey, primitive.invRotationScale, jsonWriter);
            }
            jsonWriter.EndObject();
        }
        jsonWriter.EndArray();

        // Write JSON string to file.
        file << jsonStringBuffer.GetString();
        file.close();
        return true;
    }

    const SDF3DPrimitive& SDFGrid::getPrimitive(uint32_t primitiveID) const
    {
        auto it = mPrimitiveIDToIndex.find(primitiveID);
        checkArgument(it != mPrimitiveIDToIndex.end(), "'primitiveID' ({}) is invalid.", primitiveID);
        return mPrimitives[it->second];
    }

    void SDFGrid::bakePrimitives(uint32_t batchSize)
    {
        // The baking is deferred, and occurs in the SDFSBS class.
        mBakedPrimitiveCount = std::min(mBakedPrimitiveCount + batchSize, (uint32_t)mPrimitives.size());

        // Tell the SDFSBS grid to bake the primitives when its update function is called.
        mBakePrimitives = true;
    }

    std::string SDFGrid::getTypeName(Type type)
    {
        switch (type)
        {
        case Type::NormalizedDenseGrid: return "NormalizedDenseGrid";
        case Type::SparseVoxelSet: return "SparseVoxelSet";
        case Type::SparseBrickSet: return "SparseBrickSet";
        case Type::SparseVoxelOctree: return "SparseVoxelOctree";
        default: FALCOR_UNREACHABLE(); return "";
        }
    }

    FALCOR_SCRIPT_BINDING(SDFGrid)
    {
        using namespace pybind11::literals;

        auto createSBS = [](const pybind11::kwargs& args)
        {
            uint32_t brickWidth = 7;
            uint32_t defaultGridWidth = 256;
            bool compressed = false;

            for (auto a : args)
            {
                auto key = a.first.cast<std::string>();
                const auto& value = a.second;

                bool isBool = pybind11::isinstance<pybind11::bool_>(value);
                bool isInt = pybind11::isinstance<pybind11::int_>(value);

                if (key == "brickWidth" && isInt)
                {
                    brickWidth = pybind11::cast<uint32_t>(value);
                }
                if (key == "defaultGridWidth" && isInt)
                {
                    defaultGridWidth = pybind11::cast<uint32_t>(value);
                }
                else if (key == "compressed" && isBool)
                {
                    compressed = pybind11::cast<bool>(value);
                }
            }
            return SDFGrid::SharedPtr(SDFSBS::create(brickWidth, compressed, defaultGridWidth));
        };

        pybind11::class_<SDFGrid, SDFGrid::SharedPtr> sdfGrid(m, "SDFGrid");
        sdfGrid.def_static("createNDGrid", [](float narrowBandThickness) { return SDFGrid::SharedPtr(NDSDFGrid::create(narrowBandThickness)); }, "narrowBandThickness"_a);
        sdfGrid.def_static("createSVS", [](){ return SDFGrid::SharedPtr(SDFSVS::create()); });
        sdfGrid.def_static("createSBS", createSBS);
        sdfGrid.def_static("createSVO", [](){ return SDFGrid::SharedPtr(SDFSVO::create()); });
        sdfGrid.def("loadValuesFromFile", &SDFGrid::loadValuesFromFile, "path"_a);
        sdfGrid.def("loadPrimitivesFromFile", &SDFGrid::loadPrimitivesFromFile, "path"_a, "gridWidth"_a, "dir"_a = "");
        sdfGrid.def("generateCheeseValues", &SDFGrid::generateCheeseValues, "gridWidth"_a, "seed"_a);
        sdfGrid.def_property("name", &SDFGrid::getName, &SDFGrid::setName);
    }

    void SDFGrid::createEvaluatePrimitivesPass(bool writeToTexture3D, bool mergeWithSDField)
    {
        if (!mpEvaluatePrimitivesPass)
        {
            Program::Desc desc;
            desc.addShaderLibrary(kEvaluateSDFPrimitivesShaderName).csEntry("main").setShaderModel("6_5");
            mpEvaluatePrimitivesPass = ComputePass::create(desc);
        }

        if (writeToTexture3D)
        {
            mpEvaluatePrimitivesPass->addDefine("_USE_SD_FIELD_3D_TEXTURE");
        }
        else
        {
            mpEvaluatePrimitivesPass->removeDefine("_USE_SD_FIELD_3D_TEXTURE");
        }

        if (mergeWithSDField)
        {
            mpEvaluatePrimitivesPass->addDefine("_MERGE_WITH_THE_SD_FIELD");
        }
        else
        {
            mpEvaluatePrimitivesPass->removeDefine("_MERGE_WITH_THE_SD_FIELD");
        }
    }

    void SDFGrid::updatePrimitivesBuffer()
    {
        if (mPrimitives.empty() || mPrimitives.size() <= mPrimitivesExcludedFromBuffer) return;

        uint32_t count = (uint32_t)mPrimitives.size() - mPrimitivesExcludedFromBuffer;
        void* pData = (void*)&mPrimitives[mPrimitivesExcludedFromBuffer];
        if (!mpPrimitivesBuffer || mpPrimitivesBuffer->getElementCount() < count)
        {
            mpPrimitivesBuffer = Buffer::createStructured(sizeof(SDF3DPrimitive), count, ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, pData, false);
        }
        else
        {
            mpPrimitivesBuffer->setBlob(pData, 0, count * sizeof(SDF3DPrimitive));
        }
    }
}
