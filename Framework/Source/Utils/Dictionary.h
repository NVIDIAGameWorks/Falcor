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

namespace Falcor
{
    class Dictionary
    {
    public:        
        class Value
        {
        public:
            enum class Type
            {
                Unknown,    // Indicates an invalid/uninitialized variable
                Int,
                Uint,
                Int64,
                Uint64,
                Float,
                Double,
                String,
                Vec2,
                Vec3,
                Vec4,
                Bool,
                Vector,
            };

            Value() { }
            Value(const uint32_t&     v) : u32(v), mType(Type::Uint) { }
            Value(const uint64_t&     v) : u64(v), mType(Type::Uint64) { }
            Value(const int32_t&      v) : i32(v), mType(Type::Int) { }
            Value(const int64_t&      v) : i64(v), mType(Type::Int64) { }
            Value(const float&        v) : d64(v), mType(Type::Float) { }
            Value(const double&        v) : d64(v), mType(Type::Double) { }
            Value(const glm::vec2&    v) : vec2(v), mType(Type::Vec2) { }
            Value(const glm::vec3&    v) : vec3(v), mType(Type::Vec3) { }
            Value(const glm::vec4&    v) : vec3(v), mType(Type::Vec4) { }
            Value(const std::string&  s) : str(s), mType(Type::String) { }
            Value(const char* c)         : str(c), mType(Type::String) { }
            Value(const std::vector<float>& v) : vector(v), mType(Type::Vector) {}


            Type getType() const { return mType; }
            int32_t asInt() const { checkType(Type::Int); return i32; }
            uint32_t asUint() const { checkType(Type::Uint); return u32; }
            int64_t asInt64() const { checkType(Type::Int64); return i64; }
            uint64_t asUint64() const { checkType(Type::Uint64); return u64; }
            float asFloat() const { checkType(Type::Float); return f; }
            double asDouble() const { checkType(Type::Double); return d64; }
            bool asBool() const { checkType(Type::Bool); return b; }
            const std::string& asString() const { checkType(Type::String); return str; }
            const glm::vec2& asVec2() const { checkType(Type::Vec2); return vec2; }
            const glm::vec3& asVec3() const { checkType(Type::Vec3); return vec3; }
            const glm::vec4& asVec4() const { checkType(Type::Vec4); return vec4; }
            const std::vector<float>& asFloatVec() const { checkType(Type::Vector); return vector; }
        private:
            Type mType = Type::Unknown;
            union
            {
                int32_t  i32;
                uint32_t u32;
                int64_t  i64;
                uint64_t u64;
                double   d64;
                bool     b;
                float    f;
            };
            std::string str;
            glm::vec2 vec2;
            glm::vec3 vec3;
            glm::vec4 vec4;
            std::vector<float> vector;

            void checkType(Type t) const
            {
                if (t != mType) throw(std::runtime_error("Dictionary::Value - type doesn't match"));
            }
        };

        using Container = std::unordered_map<std::string, Value>;
        using Iterator = Container::iterator;
        using ConstIterator = Container::const_iterator;

        Value& operator[](const std::string& name) { return mMap[name]; }

        const Value& operator[](const std::string& name) const { return mMap.at(name); }

        Iterator& begin() { return mMap.begin(); }
        Iterator& end() { return mMap.end(); }

        ConstIterator& begin() const { return mMap.begin(); }
        ConstIterator& end() const { return mMap.end(); }
    private:

        Container mMap;
    };
}