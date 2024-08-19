/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
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

#include "Falcor.h"

namespace Falcor
{
namespace Mitsuba
{
std::map<std::string, float> kIORTable = {
    {"vacuum", 1.0f},
    {"helium", 1.000036f},
    {"hydrogen", 1.000132f},
    {"air", 1.000277f},
    {"carbon dioxide", 1.00045f},

    {"water", 1.3330f},
    {"acetone", 1.36f},
    {"ethanol", 1.361f},
    {"carbon tetrachloride", 1.461f},
    {"glycerol", 1.4729f},
    {"benzene", 1.501f},
    {"silicone oil", 1.52045f},
    {"bromine", 1.661f},

    {"water ice", 1.31f},
    {"fused quartz", 1.458f},
    {"pyrex", 1.470f},
    {"acrylic glass", 1.49f},
    {"polypropylene", 1.49f},
    {"bk7", 1.5046f},
    {"sodium chloride", 1.544f},
    {"amber", 1.55f},
    {"pet", 1.5750f},
    {"diamond", 2.419f},
};

float lookupIOR(std::string name)
{
    std::transform(name.begin(), name.end(), name.begin(), ::tolower);

    auto it = kIORTable.find(name);
    if (it != kIORTable.end())
        return it->second;

    std::string list;
    for (const auto& [key, value] : kIORTable)
    {
        list += key + "\n";
    }
    logWarning("'{}' is not a valid IOR name. Valid choises are:\n{}", name, list);

    return 0.f;
}

} // namespace Mitsuba

} // namespace Falcor
