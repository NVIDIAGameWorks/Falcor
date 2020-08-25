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

#include <unordered_map>
#include <any>

namespace Falcor
{
    class InternalDictionary
    {
    public:
        class Value
        {
        public:
            Value() = default;
            Value(std::any& value) : mValue(value) {};

            template<typename T>
            void operator=(const T& t) { mValue = t; }

            template<typename T>
            operator T() const { return std::any_cast<T>(mValue); }

        private:
            std::any mValue;
        };

        using Container = std::unordered_map<std::string, Value>;

        using SharedPtr = std::shared_ptr<InternalDictionary>;

        InternalDictionary() = default;
        InternalDictionary(const InternalDictionary& d) : mContainer(d.mContainer) {}

        /** Create a new dictionary.
            \return A new object, or throws an exception if creation failed.
        */
        static SharedPtr create() { return SharedPtr(new InternalDictionary); }

        Value& operator[](const std::string& key) { return mContainer[key]; }
        const Value& operator[](const std::string& key) const { return mContainer.at(key); }

        Container::const_iterator begin() const { return mContainer.begin(); }
        Container::const_iterator end() const { return mContainer.end(); }

        Container::iterator begin() { return mContainer.begin(); }
        Container::iterator end() { return mContainer.end(); }

        size_t size() const { return mContainer.size(); }

        /** Check if a key exists.
        */
        bool keyExists(const std::string& key) const
        {
            return mContainer.find(key) != mContainer.end();
        }

        /** Get value by key. Throws an exception if key does not exist.
        */
        template<typename T>
        T getValue(const std::string& key)
        {
            auto it = mContainer.find(key);
            if (it == mContainer.end()) throw std::exception(("Key '" + key + "' does not exist").c_str());
            return it->second;
        }

        /** Get value by key. Returns the specified default value if key does not exist.
        */
        template<typename T>
        T getValue(const std::string& key, const T& defaultValue)
        {
            auto it = mContainer.find(key);
            return it != mContainer.end() ? it->second : defaultValue;
        }

    private:
        Container mContainer;
    };
}
