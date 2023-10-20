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
#include "Version.h"

#include "git_version.h"

#include <fmt/format.h>

namespace Falcor
{
const std::string& getVersionString()
{
    static std::string str{fmt::format("{}.{}", FALCOR_MAJOR_VERSION, FALCOR_MINOR_VERSION)};
    return str;
}

const std::string& getLongVersionString()
{
    auto gitVersionString = []()
    {
        if (GIT_VERSION_AVAILABLE)
        {
            return fmt::format(
                "commit: {}, branch: {}{}",
                GIT_VERSION_COMMIT,
                GIT_VERSION_BRANCH,
                GIT_VERSION_DIRTY ? ", contains uncommitted changes" : ""
            );
        }
        else
        {
            return std::string{"git version unknown, git shell not installed"};
        }
    };
    static std::string str{fmt::format("{}.{} ({})", FALCOR_MAJOR_VERSION, FALCOR_MINOR_VERSION, gitVersionString())};
    return str;
}
} // namespace Falcor
