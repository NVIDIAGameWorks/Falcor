/***************************************************************************
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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
#include <string>
#include <vector>
#include <unordered_map>

namespace Falcor
{
    /** Parses command line arguments and stores them for look-up by the user.
    */
    class ArgList
    {
    public:
        class Arg
        {
        public:
            Arg(const std::string& s) : mValue(s) {}

            /** Attempts to return internal string as an Int, -1 if fail
            */
            int32_t asInt() const;

            /** Attempts to return internal string as an uint, -1(max unsigned) if fail
            */
            uint32_t asUint() const;

            /** Attempts to return internal string as a float, -1 if fail
            */
            float asFloat() const;

            /** Returns the internal string representing the argument value
            */
            std::string asString() const;
        private:
            std::string mValue;
        };

        /** Parses command line string
            \param commandLine the command line string
        */
        void parseCommandLine(const std::string& commandLine);

        /** Adds a key with no arguments to the list 
            \param key
        */
        void addArg(const std::string& key);

        /** Adds an arg to an existing key or creates an key for the arg
            \param key the key the arg is associated with
            \param arg the value
        */
        void addArg(const std::string& key, Arg arg); 

        /** Check if the key already exists within the map
            \param key the key to check for
        */
        bool argExists(const std::string& key) const;

        /** Return the values associated with an arg, or an empty vector if none
            \param key the key to get the values for 
        */
        std::vector<Arg> getValues(const std::string& key) const;

        /** Return the value associated with key, or asserts if key has no values or multiple values
            \param key the key to get the value for
        */
        const Arg& operator[](const std::string& key) const;
    private:
        std::unordered_map<std::string, std::vector<Arg>> mMap;
    };

}//namespace falcor