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
#include "stdafx.h"
#include "ArgList.h"
#include <sstream>

namespace Falcor
{
    static std::string readToken(std::stringstream& args)
    {
        std::string token;

        while(1)
        {
            std::string tmp;
            std::getline(args, tmp, ' ');
            token += tmp;
            // If there are odd number of '"', read some more
            if (std::count(token.begin(), token.end(), '"') % 2)
            {
                // Read until the next '"'
                std::string s;
                std::getline(args, s, '"');
                token += ' ' + s + '"';
                // If there is a space after the '"', we're done, otherwise keep reading
                if (args.eof() || args.peek() == ' ') return token;
            }
            else
            {
                return token;
            }
        }
    }

    void ArgList::parseCommandLine(const std::string& cmdLine)
    {
        std::stringstream args(cmdLine);
        std::string currentArg;
        while (!args.eof())
        {
            std::string token = readToken(args);

            size_t dashIndex = token.find('-');
            if (dashIndex == 0 && isalpha(token[1]))
            {
                currentArg = token.substr(1);
                addArg(currentArg);
            }
            else if(!token.empty() && token.find_first_not_of(' ') != std::string::npos)
            {
                addArg(currentArg, token);
            }
        }
    }

    void ArgList::addArg(const std::string& arg)
    {
        mMap.insert(std::make_pair(arg, std::vector<Arg>()));
    }

    void ArgList::addArg(const std::string& key, Arg arg)
    {
        mMap[key].push_back(arg);
    }

    bool ArgList::argExists(const std::string& arg) const
    {
        return mMap.find(arg) != mMap.end();
    }

    std::vector<ArgList::Arg> ArgList::getValues(const std::string& key) const
    {
        try 
        {
            return mMap.at(key);
        }
        catch(const std::out_of_range&)
        {
            return std::vector<ArgList::Arg>();
        }
    }

    const ArgList::Arg& ArgList::operator[](const std::string& key) const
    {
        assert(mMap.at(key).size() == 1);
        return mMap.at(key)[0];
    }

    int32_t ArgList::Arg::asInt() const
    {
        try
        {
            return std::stoi(mValue);
        }
        catch (const std::invalid_argument& e)
        {
            logWarning("Unable to convert " + mValue + " to int. Exception: " + e.what());
            return -1;
        }
        catch (const std::out_of_range& e)
        {
            logWarning("Unable to convert " + mValue + " to int. Exception: " + e.what());
            return -1;
        }
    }

    uint32_t ArgList::Arg::asUint() const
    {
        try
        {
            return std::stoul(mValue);
        }
        catch (const std::invalid_argument& e)
        {
            logWarning("Unable to convert " + mValue + " to unsigned. Exception: " + e.what());
            return -1;
        }
        catch (const std::out_of_range& e)
        {
            logWarning("Unable to convert " + mValue + " to unsigned. Exception: " + e.what());
            return -1;
        }
    }

    uint64_t ArgList::Arg::asUint64() const
    {
        try
        {
            return std::stoull(mValue);
        }
        catch (const std::invalid_argument& e)
        {
            logWarning("Unable to convert " + mValue + " to unsigned 64. Exception: " + e.what());
            return -1;
        }
        catch (const std::out_of_range& e)
        {
            logWarning("Unable to convert " + mValue + " to unsigned 64. Exception: " + e.what());
            return -1;
        }
    }

    float ArgList::Arg::asFloat() const
    {
        try
        {
            return std::stof(mValue);
        }
        catch (const std::invalid_argument& e)
        {
            logWarning("Unable to convert " + mValue + " to float. Exception: " + e.what());
            return -1;
        }
        catch (const std::out_of_range& e)
        {
            logWarning("Unable to convert " + mValue + " to float. Exception: " + e.what());
            return -1;
        }
    }

    std::string ArgList::Arg::asString() const
    {
        return mValue;
    }
}
