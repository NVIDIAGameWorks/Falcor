/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
 **************************************************************************/
#pragma once
#define GLM_FORCE_CTOR_INIT
#define GLM_ENABLE_EXPERIMENTAL
#include "glm/glm.hpp"
#include "glm/gtx/compatibility.hpp"

namespace Falcor
{
    using namespace glm;

    using uint2 = glm::uvec2;
    using uint3 = glm::uvec3;
    using uint4 = glm::uvec4;

    using int2 = glm::ivec2;
    using int3 = glm::ivec3;
    using int4 = glm::ivec4;

    // We cannot use glm::to_string inline because including "glm/gtx/string_cast.hpp" together
    // with "using namespace glm" pollutes the namespace with many to_string functions.

    inline std::string to_string(const glm::vec2& v) { return "vec2(" + std::to_string(v.x) + "," + std::to_string(v.y) + ")"; }
    inline std::string to_string(const glm::vec3& v) { return "vec3(" + std::to_string(v.x) + "," + std::to_string(v.y) + "," + std::to_string(v.z) + ")"; }
    inline std::string to_string(const glm::vec4& v) { return "vec4(" + std::to_string(v.x) + "," + std::to_string(v.y) + "," + std::to_string(v.z) + "," + std::to_string(v.w) + ")"; }

    inline std::string to_string(const glm::uvec2& v) { return "uvec2(" + std::to_string(v.x) + "," + std::to_string(v.y) + ")"; }
    inline std::string to_string(const glm::uvec3& v) { return "uvec3(" + std::to_string(v.x) + "," + std::to_string(v.y) + "," + std::to_string(v.z) + ")"; }
    inline std::string to_string(const glm::uvec4& v) { return "uvec4(" + std::to_string(v.x) + "," + std::to_string(v.y) + "," + std::to_string(v.z) + "," + std::to_string(v.w) + ")"; }

    inline std::string to_string(const glm::ivec2& v) { return "ivec2(" + std::to_string(v.x) + "," + std::to_string(v.y) + ")"; }
    inline std::string to_string(const glm::ivec3& v) { return "ivec3(" + std::to_string(v.x) + "," + std::to_string(v.y) + "," + std::to_string(v.z) + ")"; }
    inline std::string to_string(const glm::ivec4& v) { return "ivec4(" + std::to_string(v.x) + "," + std::to_string(v.y) + "," + std::to_string(v.z) + "," + std::to_string(v.w) + ")"; }
}
