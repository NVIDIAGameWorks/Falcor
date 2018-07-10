#include "ArgList.h"
#include "Framework.h"
#include <sstream>
#include <ctype.h>

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
        catch (std::invalid_argument& e)
        {
            logWarning("Unable to convert " + mValue + " to int. Exception: " + e.what());
            return -1;
        }
        catch (std::out_of_range& e)
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
        catch (std::invalid_argument& e)
        {
            logWarning("Unable to convert " + mValue + " to unsigned. Exception: " + e.what());
            return -1;
        }
        catch (std::out_of_range& e)
        {
            logWarning("Unable to convert " + mValue + " to unsigned. Exception: " + e.what());
            return -1;
        }
    }

    float ArgList::Arg::asFloat() const
    {
        try
        {
            return std::stof(mValue);
        }
        catch (std::invalid_argument& e)
        {
            logWarning("Unable to convert " + mValue + " to float. Exception: " + e.what());
            return -1;
        }
        catch (std::out_of_range& e)
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