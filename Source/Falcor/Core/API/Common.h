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

#include "Core/Enum.h"

namespace Falcor
{
enum class ComparisonFunc
{
    Disabled,     ///< Comparison is disabled
    Never,        ///< Comparison always fails
    Always,       ///< Comparison always succeeds
    Less,         ///< Passes if source is less than the destination
    Equal,        ///< Passes if source is equal to the destination
    NotEqual,     ///< Passes if source is not equal to the destination
    LessEqual,    ///< Passes if source is less than or equal to the destination
    Greater,      ///< Passes if source is greater than to the destination
    GreaterEqual, ///< Passes if source is greater than or equal to the destination
};

FALCOR_ENUM_INFO(
    ComparisonFunc,
    {
        {ComparisonFunc::Disabled, "Disabled"},
        {ComparisonFunc::Never, "Never"},
        {ComparisonFunc::Always, "Always"},
        {ComparisonFunc::Less, "Less"},
        {ComparisonFunc::Equal, "Equal"},
        {ComparisonFunc::NotEqual, "NotEqual"},
        {ComparisonFunc::LessEqual, "LessEqual"},
        {ComparisonFunc::Greater, "Greater"},
        {ComparisonFunc::GreaterEqual, "GreaterEqual"},
    }
);
FALCOR_ENUM_REGISTER(ComparisonFunc);

} // namespace Falcor
