/***************************************************************************
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
***************************************************************************/
#pragma once
#include "Graphics/Scene/Scene.h"
#include <string>

namespace Falcor
{
    class RenderPassSerializer
    {
    public:
        Scene::UserVariable getValue(const std::string& key) const
        {
            auto it = mData.find(key); 
            if (it == mData.end())
            {
                return Scene::UserVariable(0);
            }
            return it->second;
        }

        const Scene::UserVariable& getValue(size_t index) const
        {
            assert(index < getVariableCount());
            return std::next(mData.begin(), index)->second;
        }

        void setValue(const std::string& name, const Scene::UserVariable& data)
        {
            if (mData.find(name) == mData.end())
            {
                logWarning("Unable to find serialize variable.");
                return;
            }

            mData[name] = data;
        }

        void addVariable(const std::string& name, const Scene::UserVariable& data)
        {
            mData[name] = data;
        }

        template<typename T>
        void addVariable(const std::string& name, const T& defaultValue)
        {
            T temp = T(defaultValue);
            mData[name] = { temp };
        }

        template<typename T>
        void addVariable(const std::string& name)
        {
            T temp;
            mData[name] = { temp };
        }

        RenderPassSerializer()
        {
        }

        size_t getVariableCount() const
        {
            return mData.size();
        }

        const std::string& getVariableName(size_t index) const
        {
            assert(index < getVariableCount());
            return std::next(mData.begin(), index)->first;
        }

    private:
        std::unordered_map<std::string, Scene::UserVariable> mData;

    };
}
