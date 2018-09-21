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
#include "pybind11/pybind11.h"

namespace Falcor
{
    class Scripting
    {
        public:
            class Context
            {
            public:
                template<typename T>
                struct ObjectDesc
                {
                    ObjectDesc(const std::string& name_, const T& obj_) : name(name_), obj(obj_) {}
                    std::string name;
                    T obj;
                };

                template<typename T>
                std::vector<ObjectDesc<T>> getObjects()
                {
                    std::vector<ObjectDesc<T>> v;
                    for (const auto& l : mLocals)
                    {
                        try
                        {
                            v.push_back(ObjectDesc<T>(l.first.cast<std::string>(), l.second.cast<T>()));
                        }
                        catch (std::exception&) {}
                    }
                    return v;
                }

                template<typename T>
                void setObject(const std::string& name, T obj)
                {
                    mLocals[name.c_str()] = obj;
                }

                template<typename T>
                T getObject(const std::string& name) const
                {
                    return mLocals[name.c_str()].cast<T>();
                }
            private:
                friend class Scripting;
                pybind11::dict mLocals;
            };

            static bool start();
            static void shutdown();
            static bool runScript(const std::string& script, std::string& errorLog);
            static bool runScript(const std::string& script, std::string& errorLog, Context& context);
            Context getGlobalContext() const;
    private:
        static bool sRunning;
    };
}