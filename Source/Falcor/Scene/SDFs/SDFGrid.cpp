/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
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
#include "stdafx.h"
#include "SDFGrid.h"
#include "Scene/SDFs/NormalizedDenseSDFGrid/NDSDFGrid.h"
#include "Scene/SDFs/SparseVoxelSet/SDFSVS.h"
#include "Scene/SDFs/SparseBrickSet/SDFSBS.h"
#include "Scene/SDFs/SparseVoxelOctree/SDFSVO.h"
#include "rapidjson/rapidjson.h"
#include "rapidjson/document.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/error/en.h"

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

        void serializeFloat3x3(const char* pKey, const float3x3& value, rapidjson::PrettyWriter<rapidjson::StringBuffer>& jsonWriter)
        {
            jsonWriter.String(pKey);
            jsonWriter.StartArray();
            jsonWriter.Double((double)value[0][0]);
            jsonWriter.Double((double)value[0][1]);
            jsonWriter.Double((double)value[0][2]);
            jsonWriter.Double((double)value[1][0]);
            jsonWriter.Double((double)value[1][1]);
            jsonWriter.Double((double)value[1][2]);
            jsonWriter.Double((double)value[2][0]);
            jsonWriter.Double((double)value[2][1]);
            jsonWriter.Double((double)value[2][2]);
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

        bool deserializeFloat3x3(const char* pKey, const rapidjson::Value& jsonPrimitive, float3x3& value)
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

                value[i / 3][i % 3] = jsonValue.GetFloat();
            }

            return true;
        };
    }

    void SDFGrid::setPrimitives(const std::vector<SDF3DPrimitive>& primitives, uint32_t gridWidth)
    {
        checkArgument(isPowerOf2(gridWidth), "'gridWidth' ({}) must be a power of 2.", gridWidth);

        mOriginalGridWidth = gridWidth;
        mGridWidth = gridWidth;
        mPrimitives = primitives;
        updatePrimitivesBuffer();
    }

    void SDFGrid::setValues(const std::vector<float>& cornerValues, uint32_t gridWidth)
    {
        checkArgument(isPowerOf2(gridWidth), "'gridWidth' ({}) must be a power of 2.", gridWidth);

        mOriginalGridWidth = gridWidth;
        mGridWidth = gridWidth;

        setValuesInternal(cornerValues);
    }

    bool SDFGrid::loadValuesFromFile(const std::string& filename)
    {
        std::string filePath;
        if (findFileInDataDirectories(filename, filePath))
        {
            std::ifstream file(filePath, std::ios::in | std::ios::binary);

            if (file.is_open())
            {
                uint32_t gridWidth;
                file.read(reinterpret_cast<char*>(&gridWidth), sizeof(uint32_t));

                uint32_t totalValueCount = (gridWidth + 1) * (gridWidth + 1) * (gridWidth + 1);
                std::vector<float> cornerValues(totalValueCount, 0.0f);
                file.read(reinterpret_cast<char*>(cornerValues.data()), totalValueCount * sizeof(float));

                file.close();
                setValues(cornerValues, gridWidth);
                return true;
            }
        }

        logWarning("SDFGrid::loadValues() file '{}' could not be opened!", filename);
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

    bool SDFGrid::writeValuesFromPrimitivesToFile(const std::string& filePath, RenderContext* pRenderContext)
    {
        if (!pRenderContext) pRenderContext = gpDevice->getRenderContext();

        if (!mpEvaluatePrimitivesPass)
        {
            Program::Desc desc;
            desc.addShaderLibrary(kEvaluateSDFPrimitivesShaderName).csEntry("main").setShaderModel("6_5");
            mpEvaluatePrimitivesPass = ComputePass::create(desc);
        }

        if (!mpPrimitivesBuffer || mpPrimitivesBuffer->getElementCount() < (uint32_t)mPrimitives.size())
        {
            mpPrimitivesBuffer = Buffer::createStructured(sizeof(SDF3DPrimitive), (uint32_t)mPrimitives.size(), ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, mPrimitives.data(), false);
        }

        uint32_t gridWidthInValues = mOriginalGridWidth + 1;
        uint32_t valueCount = gridWidthInValues * gridWidthInValues * gridWidthInValues;
        Buffer::SharedPtr pValuesBuffer = Buffer::createTyped<float>(valueCount);
        Buffer::SharedPtr pValuesStagingBuffer = Buffer::createTyped<float>(valueCount, Resource::BindFlags::None, Buffer::CpuAccess::Read);
        GpuFence::SharedPtr pFence = GpuFence::create();

        mpEvaluatePrimitivesPass["CB"]["gGridWidth"] = mOriginalGridWidth;
        mpEvaluatePrimitivesPass["CB"]["gPrimitiveCount"] = (uint32_t)mPrimitives.size();
        mpEvaluatePrimitivesPass["gPrimitives"] = mpPrimitivesBuffer;
        mpEvaluatePrimitivesPass["gValues"] = pValuesBuffer;
        mpEvaluatePrimitivesPass->execute(pRenderContext, uint3(mOriginalGridWidth + 1));
        pRenderContext->copyResource(pValuesStagingBuffer.get(), pValuesBuffer.get());
        pRenderContext->flush(false);
        pFence->gpuSignal(pRenderContext->getLowLevelData()->getCommandQueue());
        pFence->syncCpu();
        const float* pValues = reinterpret_cast<const float*>(pValuesStagingBuffer->map(Buffer::MapType::Read));

        std::ofstream file(filePath, std::ios::out | std::ios::binary);

        if (file.is_open())
        {
            file.write(reinterpret_cast<const char*>(&mOriginalGridWidth), sizeof(uint32_t));
            file.write(reinterpret_cast<const char*>(pValues), valueCount * sizeof(float));
            file.close();
        }

        pValuesStagingBuffer->unmap();
        return true;
    }

    uint32_t SDFGrid::loadPrimitivesFromFile(const std::string& filename, uint32_t gridWidth, const std::string& dir)
    {
        std::string filePath;
        if (dir.empty())
        {
            if (!findFileInDataDirectories(filename, filePath))
            {
                logWarning("File '{}' could not be found in data directories!", filename);
                return 0;
            }
        }
        else
        {
            filePath = dir + filename;
        }

        std::string jsonData = readFile(filePath);
        rapidjson::StringStream jsonStream(jsonData.c_str());

        rapidjson::Document jsonDocument;
        jsonDocument.ParseStream(jsonStream);

        if (jsonDocument.HasParseError())
        {
            size_t line;
            line = std::count(jsonData.begin(), jsonData.begin() + jsonDocument.GetErrorOffset(), '\n');
            logWarning("Error when deserializing SDF grid from '{}'. JSON Parse error in line {}: {}", filePath, line, rapidjson::GetParseError_En(jsonDocument.GetParseError()));
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

        return (uint32_t)mPrimitives.size();
    }

    FALCOR_SCRIPT_BINDING(SDFGrid)
    {
        auto createSBS = [](const pybind11::kwargs& args)
        {
            uint32_t brickWidth = 7;
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
                else if (key == "compressed" && isBool)
                {
                    compressed = pybind11::cast<bool>(value);
                }
            }
            return SDFGrid::SharedPtr(SDFSBS::create(brickWidth, compressed));
        };

        pybind11::class_<SDFGrid, SDFGrid::SharedPtr> sdfGrid(m, "SDFGrid");
        sdfGrid.def_static("createNDGrid", [](float narrowBandThickness) { return SDFGrid::SharedPtr(NDSDFGrid::create(narrowBandThickness)); }, "narrowBandThickness"_a);
        sdfGrid.def_static("createSVS", [](){ return SDFGrid::SharedPtr(SDFSVS::create()); });
        sdfGrid.def_static("createSBS", createSBS);
        sdfGrid.def_static("createSVO", []() { return SDFGrid::SharedPtr(SDFSVO::create()); });
        sdfGrid.def("loadValuesFromFile", &SDFGrid::loadValuesFromFile, "filename"_a);
        sdfGrid.def("loadPrimitivesFromFile", &SDFGrid::loadPrimitivesFromFile, "filename"_a, "gridWidth"_a, "dir"_a = "");
        sdfGrid.def("generateCheeseValues", &SDFGrid::generateCheeseValues, "gridWidth"_a, "seed"_a);
        sdfGrid.def_property("name", &SDFGrid::getName, &SDFGrid::setName);
    }

    void SDFGrid::updatePrimitivesBuffer()
    {
        if (!mpPrimitivesBuffer || mpPrimitivesBuffer->getElementCount() < (uint32_t)mPrimitives.size())
        {
            mpPrimitivesBuffer = Buffer::createStructured(sizeof(SDF3DPrimitive), (uint32_t)mPrimitives.size(), ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, mPrimitives.data(), false);
        }
        else
        {
            mpPrimitivesBuffer->setBlob(mPrimitives.data(), 0, mPrimitives.size() * sizeof(SDF3DPrimitive));
        }
    }
}
