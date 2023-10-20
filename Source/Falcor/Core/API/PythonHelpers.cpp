/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
#include "PythonHelpers.h"

namespace Falcor
{

pybind11::dlpack::dtype dataTypeToDtype(DataType type)
{
    switch (type)
    {
    case DataType::int8:
        return pybind11::dlpack::dtype{(uint8_t)pybind11::dlpack::dtype_code::Int, (uint8_t)8, 1};
    case DataType::int16:
        return pybind11::dlpack::dtype{(uint8_t)pybind11::dlpack::dtype_code::Int, (uint8_t)16, 1};
    case DataType::int32:
        return pybind11::dlpack::dtype{(uint8_t)pybind11::dlpack::dtype_code::Int, (uint8_t)32, 1};
    case DataType::int64:
        return pybind11::dlpack::dtype{(uint8_t)pybind11::dlpack::dtype_code::Int, (uint8_t)64, 1};
    case DataType::uint8:
        return pybind11::dlpack::dtype{(uint8_t)pybind11::dlpack::dtype_code::UInt, (uint8_t)8, 1};
    case DataType::uint16:
        return pybind11::dlpack::dtype{(uint8_t)pybind11::dlpack::dtype_code::UInt, (uint8_t)16, 1};
    case DataType::uint32:
        return pybind11::dlpack::dtype{(uint8_t)pybind11::dlpack::dtype_code::UInt, (uint8_t)32, 1};
    case DataType::uint64:
        return pybind11::dlpack::dtype{(uint8_t)pybind11::dlpack::dtype_code::UInt, (uint8_t)64, 1};
    case DataType::float16:
        return pybind11::dlpack::dtype{(uint8_t)pybind11::dlpack::dtype_code::Float, (uint8_t)16, 1};
    case DataType::float32:
        return pybind11::dlpack::dtype{(uint8_t)pybind11::dlpack::dtype_code::Float, (uint8_t)32, 1};
    case DataType::float64:
        return pybind11::dlpack::dtype{(uint8_t)pybind11::dlpack::dtype_code::Float, (uint8_t)64, 1};
    }
    FALCOR_THROW("Unhandled data type.");
}

std::optional<pybind11::dlpack::dtype> resourceFormatToDtype(ResourceFormat format)
{
    // Unknown and compressed formats are not supported.
    if (format == ResourceFormat::Unknown || isCompressedFormat(format))
        return {};

    // Formats with different bits per channel are not supported.
    uint32_t channelCount = getFormatChannelCount(format);
    uint32_t channelBits = getNumChannelBits(format, 0);
    for (uint32_t i = 1; i < channelCount; ++i)
        if (getNumChannelBits(format, i) != channelBits)
            return {};

    // Only formats with 8, 16, 32, or 64 bits per channel are supported.
    if (channelBits != 8 && channelBits != 16 && channelBits != 32 && channelBits != 64)
        return {};

    switch (getFormatType(format))
    {
    case FormatType::Float:
        return pybind11::dlpack::dtype{(uint8_t)pybind11::dlpack::dtype_code::Float, (uint8_t)channelBits, 1};
    case FormatType::Uint:
        return pybind11::dlpack::dtype{(uint8_t)pybind11::dlpack::dtype_code::UInt, (uint8_t)channelBits, 1};
    case FormatType::Sint:
        return pybind11::dlpack::dtype{(uint8_t)pybind11::dlpack::dtype_code::Int, (uint8_t)channelBits, 1};
    }

    return {};
}

pybind11::dict defineListToPython(const DefineList& defines)
{
    pybind11::dict dict;
    for (const auto& [key, value] : defines)
        dict[key.c_str()] = value;
    return dict;
}

DefineList defineListFromPython(const pybind11::dict& dict)
{
    DefineList defines;
    for (const auto& [key, value] : dict)
    {
        if (!pybind11::isinstance<pybind11::str>(key))
            FALCOR_THROW("Define key must be a string.");
        auto keyStr = key.cast<std::string>();
        if (pybind11::isinstance<pybind11::str>(value))
            defines.add(keyStr, value.cast<std::string>());
        else if (pybind11::isinstance<pybind11::bool_>(value))
            defines.add(keyStr, value.cast<bool>() ? "1" : "0");
        else if (pybind11::isinstance<pybind11::int_>(value))
            defines.add(keyStr, std::to_string(value.cast<int64_t>()));
        else
            FALCOR_THROW("Define value for key '{}' must be a string, bool, or int.", keyStr);
    }
    return defines;
}

pybind11::dict typeConformanceListToPython(const TypeConformanceList& conformances)
{
    pybind11::dict dict;
    for (const auto& [key, value] : conformances)
        dict[pybind11::make_tuple(key.typeName, key.interfaceName)] = value;
    return dict;
}

TypeConformanceList typeConformanceListFromPython(const pybind11::dict& dict)
{
    TypeConformanceList conformances;
    for (const auto& [key, value] : dict)
    {
        auto [typeName, interfaceType] = key.cast<std::tuple<std::string, std::string>>();
        conformances.add(typeName, interfaceType, value.cast<uint32_t>());
    }
    return conformances;
}

ProgramDesc programDescFromPython(const pybind11::kwargs& kwargs)
{
    ProgramDesc desc;
    for (const auto& arg : kwargs)
    {
        std::string key = arg.first.cast<std::string>();
        const auto& value = arg.second;

        if (key == "file")
            desc.addShaderModule().addFile(value.cast<std::filesystem::path>());
        else if (key == "string")
            desc.addShaderModule().addString(value.cast<std::string>());
        else if (key == "cs_entry")
            desc.csEntry(value.cast<std::string>());
        else if (key == "type_conformances")
            desc.typeConformances = typeConformanceListFromPython(value.cast<pybind11::dict>());
        else if (key == "shader_model")
            desc.shaderModel = value.cast<ShaderModel>();
        else if (key == "compiler_flags")
            desc.compilerFlags = value.cast<SlangCompilerFlags>();
        else if (key == "compiler_arguments")
            desc.compilerArguments = value.cast<std::vector<std::string>>();
        else
            FALCOR_THROW("Unknown keyword argument '{}'.", key);
    }
    return desc;
}

} // namespace Falcor
