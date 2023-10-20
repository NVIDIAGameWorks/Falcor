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
#pragma once

#include "Core/API/Formats.h"
#include "Core/Program/Program.h"
#include "Utils/Scripting/ScriptBindings.h"
#include "Utils/Scripting/ndarray.h"

#include <optional>

namespace Falcor
{

inline size_t getDtypeByteSize(pybind11::dlpack::dtype dtype)
{
    return (dtype.bits * dtype.lanes) / 8;
}

template<typename... Args>
size_t getNdarraySize(const pybind11::ndarray<Args...>& array)
{
    size_t size = 1;
    for (size_t i = 0; i < array.ndim(); i++)
        size *= array.shape(i);
    return size;
}

template<typename... Args>
size_t getNdarrayByteSize(const pybind11::ndarray<Args...>& array)
{
    return getNdarraySize(array) * getDtypeByteSize(array.dtype());
}

template<typename... Args>
size_t isNdarrayContiguous(const pybind11::ndarray<Args...>& array)
{
    if (array.ndim() == 0)
        return false;
    size_t prod = 1;
    for (size_t i = array.ndim() - 1;;)
    {
        if (array.stride(i) != prod)
            return false;
        prod *= array.shape(i);
        if (i == 0)
            break;
        --i;
    }
    return true;
}

pybind11::dlpack::dtype dataTypeToDtype(DataType type);
std::optional<pybind11::dlpack::dtype> resourceFormatToDtype(ResourceFormat format);

pybind11::dict defineListToPython(const DefineList& defines);
DefineList defineListFromPython(const pybind11::dict& dict);

pybind11::dict typeConformanceListToPython(const TypeConformanceList& conformances);
TypeConformanceList typeConformanceListFromPython(const pybind11::dict& dict);

ProgramDesc programDescFromPython(const pybind11::kwargs& kwargs);

} // namespace Falcor
