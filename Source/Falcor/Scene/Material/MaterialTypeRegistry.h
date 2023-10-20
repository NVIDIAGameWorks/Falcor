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
#include "Core/Macros.h"
#include "MaterialTypes.slang"
#include "MaterialParamLayout.h"
#include <fmt/format.h>

namespace Falcor
{
/** Registers a new material type with the given name. Returns the existing material type if name is already registered.
    The type name will be used by the system for symbols in the generated shader code and must not contain whitespaces etc.
    The first returned MaterialType is one past `MaterialType::BuiltinCount`.
    This operation is thread safe.
    \param[in] typeName Material type name.
    \return Material type.
*/
FALCOR_API MaterialType registerMaterialType(std::string typeName);

/** Get the material type name for the given type.
    This operation is thread safe.
    \param[in] type Material type.
    \return Material type name.
*/
FALCOR_API std::string to_string(MaterialType type);

/** Returns the total number of registered material types. This includes the `MaterialType::BuiltinCount`.
    This operation is thread safe.
    \return Total number of registered material types.
*/
FALCOR_API size_t getMaterialTypeCount();

/** Return the material parameter layout of serialized material parameters (for differentiable materials).
*/
FALCOR_API MaterialParamLayout getMaterialParamLayout(MaterialType type);
}

template<>
struct fmt::formatter<Falcor::MaterialType>
{
    template<typename ParseContext>
    constexpr auto parse(ParseContext& ctx)
    {
        return ctx.begin();
    }

    template<typename FormatContext>
    auto format(Falcor::MaterialType materialType, FormatContext& ctx)
    {
        return fmt::format_to(ctx.out(), "{0}", Falcor::to_string(materialType));
    }
};
