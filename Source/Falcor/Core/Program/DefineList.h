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

#include <initializer_list>
#include <map>
#include <string>

namespace Falcor
{

class DefineList : public std::map<std::string, std::string>
{
public:
    /**
     * Adds a macro definition. If the macro already exists, it will be replaced.
     * @param[in] name The name of macro.
     * @param[in] value Optional. The value of the macro.
     * @return The updated list of macro definitions.
     */
    DefineList& add(const std::string& name, const std::string& val = "")
    {
        (*this)[name] = val;
        return *this;
    }

    /**
     * Removes a macro definition. If the macro doesn't exist, the call will be silently ignored.
     * @param[in] name The name of macro.
     * @return The updated list of macro definitions.
     */
    DefineList& remove(const std::string& name)
    {
        (*this).erase(name);
        return *this;
    }

    /**
     * Add a define list to the current list
     */
    DefineList& add(const DefineList& dl)
    {
        for (const auto& p : dl)
            add(p.first, p.second);
        return *this;
    }

    /**
     * Remove a define list from the current list
     */
    DefineList& remove(const DefineList& dl)
    {
        for (const auto& p : dl)
            remove(p.first);
        return *this;
    }

    DefineList() = default;
    DefineList(std::initializer_list<std::pair<const std::string, std::string>> il) : std::map<std::string, std::string>(il) {}
};
} // namespace Falcor
