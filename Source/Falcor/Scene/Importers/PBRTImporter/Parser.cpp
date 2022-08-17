/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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

// This code is based on pbrt:
// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include "Parser.h"
#include "Helpers.h"
#include "Core/Assert.h"
#include "Core/Platform/OS.h"
#include "Utils/Logger.h"

#include <fast_float/fast_float.h>

#include <atomic>
#include <charconv>

namespace Falcor
{
    namespace pbrt
    {
        ParserTarget::~ParserTarget() {}

        std::string toString(std::string_view sv)
        {
            return std::string(sv);
        }

        std::string Token::toString() const
        {
            return fmt::format("[ Token token: {} loc: {} ]", token, loc.toString());
        }

        static char decodeEscaped(int ch, const FileLoc& loc)
        {
            switch (ch)
            {
            case EOF:
                throwError(loc, "Premature EOF after character escape '\\'");
            case 'b':
                return '\b';
            case 'f':
                return '\f';
            case 'n':
                return '\n';
            case 'r':
                return '\r';
            case 't':
                return '\t';
            case '\\':
                return '\\';
            case '\'':
                return '\'';
            case '\"':
                return '\"';
            default:
                throwError(loc, "Unexpected escaped character '%c'", ch);
            }
            return 0;
        }

        std::unique_ptr<Tokenizer> Tokenizer::createFromFile(const std::filesystem::path& path)
        {
            if (hasExtension(path, "gz"))
            {
                std::string str = decompressFile(path);
                return std::make_unique<Tokenizer>(std::move(str), path);
            }
            else
            {
                std::string str = readFile(path);
                return std::make_unique<Tokenizer>(std::move(str), path);
            }
        }

        std::unique_ptr<Tokenizer> Tokenizer::createFromString(std::string str)
        {
            return std::make_unique<Tokenizer>(std::move(str), "<string>");
        }

        Tokenizer::Tokenizer(std::string str, const std::filesystem::path& path)
            : mPath(path)
            , mContents(std::move(str))
        {
            auto pFilename = std::make_unique<std::string>(path.string());
            mLoc = FileLoc(*pFilename);
            getFilenames().push_back(std::move(pFilename));

            mPos = mContents.data();
            mEnd = mPos + mContents.size();
            if (isUTF16(mContents.data(), mContents.size())) throwError("File is encoded with UTF-16, which is not currently supported.");
        }

        bool Tokenizer::isUTF16(const void* ptr, size_t len) const
        {
            auto c = reinterpret_cast<const unsigned char*>(ptr);
            // https://en.wikipedia.org/wiki/Byte_order_mark
            return (len >= 2 && ((c[0] == 0xfe && c[1] == 0xff) || (c[0] == 0xff && c[1] == 0xfe)));
        }

        std::optional<Token> Tokenizer::next()
        {
            while (true)
            {
                const char* tokenStart = mPos;
                FileLoc startLoc = mLoc;

                int ch = getChar();
                if (ch == EOF)
                {
                    return {};
                }
                else if (ch == ' ' || ch == '\n' || ch == '\t' || ch == '\r')
                {
                    // Skip.
                }
                else if (ch == '"')
                {
                    // Scan to closing quote.
                    bool haveEscaped = false;
                    while ((ch = getChar()) != '"')
                    {
                        if (ch == EOF)
                        {
                            throwError(startLoc, "Premature EOF.");
                        }
                        else if (ch == '\n')
                        {
                            throwError(startLoc, "Unterminated string.");
                        }
                        else if (ch == '\\')
                        {
                            haveEscaped = true;
                            // Grab the next character.
                            if ((ch = getChar()) == EOF)
                            {
                                throwError(startLoc, "Premature EOF.");
                            }
                        }
                    }

                    if (!haveEscaped)
                    {
                        return Token({tokenStart, size_t(mPos - tokenStart)}, startLoc);
                    }
                    else
                    {
                        mEscaped.clear();
                        for (const char* p = tokenStart; p < mPos; ++p)
                        {
                            if (*p != '\\')
                            {
                                mEscaped.push_back(*p);
                            }
                            else
                            {
                                ++p;
                                FALCOR_ASSERT(p < mPos);
                                mEscaped.push_back(decodeEscaped(*p, startLoc));
                            }
                        }
                        return Token({mEscaped.data(), mEscaped.size()}, startLoc);
                    }
                }
                else if (ch == '[' || ch == ']')
                {
                    return Token({tokenStart, size_t(1)}, startLoc);
                }
                else if (ch == '#')
                {
                    // Comment: scan to EOL (or EOF).
                    while ((ch = getChar()) != EOF)
                    {
                        if (ch == '\n' || ch == '\r')
                        {
                            ungetChar();
                            break;
                        }
                    }

                    return Token({tokenStart, size_t(mPos - tokenStart)}, startLoc);
                }
                else
                {
                    // Regular statement or numeric token. Scan until we hit a space, opening quote, or bracket.
                    while ((ch = getChar()) != EOF)
                    {
                        if (ch == ' ' || ch == '\n' || ch == '\t' || ch == '\r' || ch == '"' || ch == '[' || ch == ']')
                        {
                            ungetChar();
                            break;
                        }
                    }
                    return Token({tokenStart, size_t(mPos - tokenStart)}, startLoc);
                }
            }
        }

        static int32_t parseInt(const Token& t)
        {
            auto begin = t.token.data();
            auto end = t.token.data() + t.token.size();
            // Skip '+' character, std::from_chars doesn't handle '+'.
            if (*begin == '+') begin++;
            int64_t value;
            auto result = std::from_chars(begin, end, value);
            if (result.ptr != end)
            {
                throwError(t.loc, "'{}': Expected a number.", t.token);
            }
            if (value < std::numeric_limits<int32_t>::lowest() || value > std::numeric_limits<int32_t>::max())
            {
                throwError(t.loc, "'{}': Numeric value cannot be represented as a 32-bit integer.", t.token);
            }
            return (int32_t)value;
        }

        static Float parseFloat(const Token& t)
        {
            // Fast path for a single digit.
            if (t.token.size() == 1)
            {
                if (!(t.token[0] >= '0' && t.token[0] <= '9'))
                {
                    throwError(t.loc, "'{}': Expected a number.", t.token);
                }
                return (Float)(t.token[0] - '0');
            }

            auto begin = t.token.data();
            auto end = t.token.data() + t.token.size();
            // Skip '+' character, std::from_chars (and fast_float::from_chars) doesn't handle '+'.
            if (*begin == '+') begin++;
            Float value;
            // Note: We currently use fast_float::from_chars because std::from_chars for float/double is not well supported yet.
            auto result = fast_float::from_chars(begin, end, value);
            if (result.ptr != end)
            {
                throwError(t.loc, "'{}': Expected a number.", t.token);
            }
            return value;
        }

        inline bool isQuotedString(std::string_view str)
        {
            return str.size() >= 2 && str[0] == '"' && str.back() == '"';
        }

        static std::string_view dequoteString(const Token& t)
        {
            if (!isQuotedString(t.token)) throwError(t.loc, "'{}' is not a quoted string.", t.token);
            std::string_view str = t.token;
            str.remove_prefix(1);
            str.remove_suffix(1);
            return str;
        }

        constexpr uint32_t TokenOptional = 0;
        constexpr uint32_t TokenRequired = 1;

        template <typename Next, typename Unget>
        static ParsedParameterVector parseParameters(Next nextToken, Unget ungetToken)
        {
            ParsedParameterVector parameterVector;

            while (true)
            {
                auto t = nextToken(TokenOptional);
                if (!t.has_value()) return parameterVector;

                if (!isQuotedString(t->token))
                {
                    ungetToken(*t);
                    return parameterVector;
                }

                ParsedParameter param(t->loc);

                std::string_view decl = dequoteString(*t);

                auto skipSpace = [&decl](std::string_view::const_iterator iter)
                {
                    while (iter != decl.end() && (*iter == ' ' || *iter == '\t')) ++iter;
                    return iter;
                };

                // Skip to the next whitespace character (or the end of the string).
                auto skipToSpace = [&decl](std::string_view::const_iterator iter)
                {
                    while (iter != decl.end() && *iter != ' ' && *iter != '\t') ++iter;
                    return iter;
                };

                auto typeBegin = skipSpace(decl.begin());
                if (typeBegin == decl.end()) throwError(t->loc, "Parameter '{}' doesn't have a type declaration.", decl);

                // Find end of type declaration.
                auto typeEnd = skipToSpace(typeBegin);
                param.type.assign(typeBegin, typeEnd);

                auto nameBegin = skipSpace(typeEnd);
                if (nameBegin == decl.end()) throwError(t->loc, "Unable to find parameter name from '{}'.", decl);

                auto nameEnd = skipToSpace(nameBegin);
                param.name.assign(nameBegin, nameEnd);

                enum ValType { Unknown, String, Bool, Float, Int } valType = Unknown;

                if (param.type == "integer") valType = Int;

                auto addVal = [&](const Token& t)
                {
                    if (isQuotedString(t.token))
                    {
                        switch (valType) {
                        case Unknown:
                            valType = String;
                            break;
                        case String:
                            break;
                        case Float:
                            throwError(t.loc, "'{}': Expected floating-point value", t.token);
                        case Int:
                            throwError(t.loc, "'{}': Expected integer value", t.token);
                        case Bool:
                            throwError(t.loc, "'{}': Expected Boolean value", t.token);
                        }

                        param.addString(dequoteString(t));
                    }
                    else if (t.token[0] == 't' && t.token == "true")
                    {
                        switch (valType) {
                        case Unknown:
                            valType = Bool;
                            break;
                        case String:
                            throwError(t.loc, "'{}': Expected string value", t.token);
                        case Float:
                            throwError(t.loc, "'{}': Expected floating-point value", t.token);
                        case Int:
                            throwError(t.loc, "'{}': Expected integer value", t.token);
                        case Bool:
                            break;
                        }

                        param.addBool(true);
                    }
                    else if (t.token[0] == 'f' && t.token == "false")
                    {
                        switch (valType) {
                        case Unknown:
                            valType = Bool;
                            break;
                        case String:
                            throwError(t.loc, "'{}': Expected string value", t.token);
                        case Float:
                            throwError(t.loc, "'{}': Expected floating-point value", t.token);
                        case Int:
                            throwError(t.loc, "'{}': Expected integer value", t.token);
                        case Bool:
                            break;
                        }

                        param.addBool(false);
                    }
                    else
                    {
                        switch (valType) {
                        case Unknown:
                            valType = Float;
                            break;
                        case String:
                            throwError(t.loc, "'{}': Expected string value", t.token);
                        case Float:
                            break;
                        case Int:
                            break;
                        case Bool:
                            throwError(t.loc, "'{}': Expected Boolean value", t.token);
                        }

                        if (valType == Int) param.addInt(parseInt(t));
                        else param.addFloat(parseFloat(t));
                    }
                };

                Token val = *nextToken(TokenRequired);

                if (val.token == "[")
                {
                    while (true)
                    {
                        val = *nextToken(TokenRequired);
                        if (val.token == "]") break;
                        addVal(val);
                    }
                }
                else
                {
                    addVal(val);
                }

                parameterVector.push_back(param);
            }

            return parameterVector;
        }

        void parse(ParserTarget& target, std::unique_ptr<Tokenizer> tokenizer)
        {
            static std::atomic<bool> warnedTransformBeginEndDeprecated{false};

            logInfo("PBRTImporter: Started parsing '{}'.", tokenizer->getPath().string());

            auto searchPath = tokenizer->getPath().parent_path();

            std::vector<std::unique_ptr<Tokenizer>> fileStack;
            fileStack.push_back(std::move(tokenizer));

            std::optional<Token> ungetToken;

            /** Helper function that handles the file stack, returning the next token from
                the file until reaching EOF, at which point it switches to the next file (if any).
            */
            std::function<std::optional<Token>(uint32_t flags)> nextToken;
            nextToken = [&](uint32_t flags) -> std::optional<Token>
            {
                if (ungetToken.has_value()) return std::exchange(ungetToken, {});

                if (fileStack.empty())
                {
                    if ((flags & TokenRequired) != 0)
                    {
                        throwError("Premature end of file.");
                    }
                    return {};
                }

                std::optional<Token> tok = fileStack.back()->next();

                if (!tok)
                {
                    // We've reached EOF in the current file. Anything more to parse?
                    logInfo("PBRTImporter: Finished parsing '{}'.", fileStack.back()->getPath().string());
                    fileStack.pop_back();
                    return nextToken(flags);
                }
                else if (tok->token[0] == '#')
                {
                    // Swallow comments.
                    return nextToken(flags);
                }
                else
                {
                    // Regular token.
                    return tok;
                }
            };

            auto unget = [&](Token t)
            {
                FALCOR_ASSERT(!ungetToken.has_value());
                ungetToken = t;
            };

            /** Helper function for pbrt API entrypoints that take a single string
                parameter and a ParameterVector (e.g. onShape()).
            */
            auto basicParamListEntrypoint = [&](void (ParserTarget::*apiFunc)(const std::string&, ParsedParameterVector, FileLoc), FileLoc loc)
            {
                Token t = *nextToken(TokenRequired);
                std::string_view dequoted = dequoteString(t);
                std::string n = toString(dequoted);
                ParsedParameterVector parameterVector = parseParameters(nextToken, unget);
                (target.*apiFunc)(n, std::move(parameterVector), loc);
            };

            auto syntaxError = [&](const Token& t)
            {
                if (t.token == "WorldEnd")
                {
                    throwError(t.loc, "Unknown directive: {}.\nThis looks like old (pre pbrt-v4) scene format which is not supported.", t.token);
                }
                else
                {
                    throwError(t.loc, "Unknown directive: {}", t.token);
                }
            };

            std::optional<Token> tok;

            while (true)
            {
                tok = nextToken(TokenOptional);
                if (!tok.has_value()) break;

                switch (tok->token[0])
                {
                case 'A':
                    if (tok->token == "AttributeBegin")
                    {
                        target.onAttributeBegin(tok->loc);
                    }
                    else if (tok->token == "AttributeEnd")
                    {
                        target.onAttributeEnd(tok->loc);
                    }
                    else if (tok->token == "Attribute")
                    {
                        basicParamListEntrypoint(&ParserTarget::onAttribute, tok->loc);
                    }
                    else if (tok->token == "ActiveTransform")
                    {
                        Token a = *nextToken(TokenRequired);
                        if (a.token == "All") target.onActiveTransformAll(tok->loc);
                        else if (a.token == "EndTime") target.onActiveTransformEndTime(tok->loc);
                        else if (a.token == "StartTime") target.onActiveTransformStartTime(tok->loc);
                        else syntaxError(*tok);
                    }
                    else if (tok->token == "AreaLightSource")
                    {
                        basicParamListEntrypoint(&ParserTarget::onAreaLightSource, tok->loc);
                    }
                    else if (tok->token == "Accelerator")
                    {
                        basicParamListEntrypoint(&ParserTarget::onAccelerator, tok->loc);
                    }
                    else
                    {
                        syntaxError(*tok);
                    }
                    break;

                case 'C':
                    if (tok->token == "ConcatTransform")
                    {
                        if (nextToken(TokenRequired)->token != "[") syntaxError(*tok);
                        Float m[16];
                        for (int i = 0; i < 16; ++i) m[i] = parseFloat(*nextToken(TokenRequired));
                        if (nextToken(TokenRequired)->token != "]") syntaxError(*tok);
                        target.onConcatTransform(m, tok->loc);
                    }
                    else if (tok->token == "CoordinateSystem")
                    {
                        std::string_view n = dequoteString(*nextToken(TokenRequired));
                        target.onCoordinateSystem(toString(n), tok->loc);
                    }
                    else if (tok->token == "CoordSysTransform")
                    {
                        std::string_view n = dequoteString(*nextToken(TokenRequired));
                        target.onCoordSysTransform(toString(n), tok->loc);
                    }
                    else if (tok->token == "ColorSpace")
                    {
                        std::string_view n = dequoteString(*nextToken(TokenRequired));
                        target.onColorSpace(toString(n), tok->loc);
                    }
                    else if (tok->token == "Camera")
                    {
                        basicParamListEntrypoint(&ParserTarget::onCamera, tok->loc);
                    }
                    else
                    {
                        syntaxError(*tok);
                    }
                    break;

                case 'F':
                    if (tok->token == "Film")
                    {
                        basicParamListEntrypoint(&ParserTarget::onFilm, tok->loc);
                    }
                    else
                    {
                        syntaxError(*tok);
                    }
                    break;

                case 'I':
                    if (tok->token == "Integrator")
                    {
                        basicParamListEntrypoint(&ParserTarget::onIntegrator, tok->loc);
                    }
                    else if (tok->token == "Include")
                    {
                        Token filenameToken = *nextToken(TokenRequired);
                        std::string filename = toString(dequoteString(filenameToken));
                        auto path = searchPath / filename;
                        std::unique_ptr<Tokenizer> includeTokenizer = Tokenizer::createFromFile(path);
                        logInfo("PBRTImporter: Started parsing '{}'.", includeTokenizer->getPath().string());
                        fileStack.push_back(std::move(includeTokenizer));
                    }
                    else if (tok->token == "Import")
                    {
                        throwError(tok->loc, "'Import' directive not supported yet.");
                    }
                    else if (tok->token == "Identity")
                    {
                        target.onIdentity(tok->loc);
                    }
                    else
                    {
                        syntaxError(*tok);
                    }
                    break;

                case 'L':
                    if (tok->token == "LightSource")
                    {
                        basicParamListEntrypoint(&ParserTarget::onLightSource, tok->loc);
                    }
                    else if (tok->token == "LookAt")
                    {
                        Float v[9];
                        for (int i = 0; i < 9; ++i)
                            v[i] = parseFloat(*nextToken(TokenRequired));
                        target.onLookAt(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8],
                                    tok->loc);
                    }
                    else
                    {
                        syntaxError(*tok);
                    }
                    break;

                case 'M':
                    if (tok->token == "MakeNamedMaterial")
                    {
                        basicParamListEntrypoint(&ParserTarget::onMakeNamedMaterial, tok->loc);
                    }
                    else if (tok->token == "MakeNamedMedium")
                    {
                        basicParamListEntrypoint(&ParserTarget::onMakeNamedMedium, tok->loc);
                    }
                    else if (tok->token == "Material")
                    {
                        basicParamListEntrypoint(&ParserTarget::onMaterial, tok->loc);
                    }
                    else if (tok->token == "MediumInterface")
                    {
                        std::string_view n = dequoteString(*nextToken(TokenRequired));
                        std::string names[2];
                        names[0] = toString(n);

                        // Check for optional second parameter.
                        std::optional<Token> second = nextToken(TokenOptional);
                        if (second.has_value()) {
                            if (isQuotedString(second->token))
                                names[1] = toString(dequoteString(*second));
                            else {
                                unget(*second);
                                names[1] = names[0];
                            }
                        } else
                            names[1] = names[0];

                        target.onMediumInterface(names[0], names[1], tok->loc);
                    }
                    else
                    {
                        syntaxError(*tok);
                    }
                    break;

                case 'N':
                    if (tok->token == "NamedMaterial")
                    {
                        std::string_view n = dequoteString(*nextToken(TokenRequired));
                        target.onNamedMaterial(toString(n), tok->loc);
                    }
                    else
                    {
                        syntaxError(*tok);
                    }
                    break;

                case 'O':
                    if (tok->token == "ObjectBegin")
                    {
                        std::string_view n = dequoteString(*nextToken(TokenRequired));
                        target.onObjectBegin(toString(n), tok->loc);
                    }
                    else if (tok->token == "ObjectEnd")
                    {
                        target.onObjectEnd(tok->loc);
                    }
                    else if (tok->token == "ObjectInstance")
                    {
                        std::string_view n = dequoteString(*nextToken(TokenRequired));
                        target.onObjectInstance(toString(n), tok->loc);
                    }
                    else if (tok->token == "Option")
                    {
                        std::string name = toString(dequoteString(*nextToken(TokenRequired)));
                        std::string value = toString(nextToken(TokenRequired)->token);
                        target.onOption(name, value, tok->loc);
                    }
                    else
                    {
                        syntaxError(*tok);
                    }
                    break;

                case 'P':
                    if (tok->token == "PixelFilter")
                    {
                        basicParamListEntrypoint(&ParserTarget::onPixelFilter, tok->loc);
                    }
                    else
                    {
                        syntaxError(*tok);
                    }
                    break;

                case 'R':
                    if (tok->token == "ReverseOrientation")
                    {
                        target.onReverseOrientation(tok->loc);
                    }
                    else if (tok->token == "Rotate")
                    {
                        Float v[4];
                        for (int i = 0; i < 4; ++i) v[i] = parseFloat(*nextToken(TokenRequired));
                        target.onRotate(v[0], v[1], v[2], v[3], tok->loc);
                    }
                    else
                    {
                        syntaxError(*tok);
                    }
                    break;

                case 'S':
                    if (tok->token == "Shape")
                    {
                        basicParamListEntrypoint(&ParserTarget::onShape, tok->loc);
                    }
                    else if (tok->token == "Sampler")
                    {
                        basicParamListEntrypoint(&ParserTarget::onSampler, tok->loc);
                    }
                    else if (tok->token == "Scale")
                    {
                        Float v[3];
                        for (int i = 0; i < 3; ++i) v[i] = parseFloat(*nextToken(TokenRequired));
                        target.onScale(v[0], v[1], v[2], tok->loc);
                    }
                    else
                    {
                        syntaxError(*tok);
                    }
                    break;

                case 'T':
                    if (tok->token == "TransformBegin")
                    {
                        if (!warnedTransformBeginEndDeprecated)
                        {
                            logWarning(tok->loc, "TransformBegin/End are deprecated and should be replaced with AttributeBegin/End.");
                            warnedTransformBeginEndDeprecated = true;
                        }
                        target.onAttributeBegin(tok->loc);
                    }
                    else if (tok->token == "TransformEnd")
                    {
                        target.onAttributeEnd(tok->loc);
                    }
                    else if (tok->token == "Transform")
                    {
                        if (nextToken(TokenRequired)->token != "[")
                            syntaxError(*tok);
                        Float m[16];
                        for (int i = 0; i < 16; ++i)
                            m[i] = parseFloat(*nextToken(TokenRequired));
                        if (nextToken(TokenRequired)->token != "]")
                            syntaxError(*tok);
                        target.onTransform(m, tok->loc);
                    }
                    else if (tok->token == "Translate")
                    {
                        Float v[3];
                        for (int i = 0; i < 3; ++i)
                            v[i] = parseFloat(*nextToken(TokenRequired));
                        target.onTranslate(v[0], v[1], v[2], tok->loc);
                    }
                    else if (tok->token == "TransformTimes")
                    {
                        Float v[2];
                        for (int i = 0; i < 2; ++i)
                            v[i] = parseFloat(*nextToken(TokenRequired));
                        target.onTransformTimes(v[0], v[1], tok->loc);
                    }
                    else if (tok->token == "Texture")
                    {
                        std::string_view n = dequoteString(*nextToken(TokenRequired));
                        std::string name = toString(n);
                        n = dequoteString(*nextToken(TokenRequired));
                        std::string type = toString(n);

                        Token t = *nextToken(TokenRequired);
                        std::string_view dequoted = dequoteString(t);
                        std::string texName = toString(dequoted);
                        ParsedParameterVector params = parseParameters(nextToken, unget);
                        target.onTexture(name, type, texName, std::move(params), tok->loc);
                    }
                    else
                    {
                        syntaxError(*tok);
                    }
                    break;

                case 'W':
                    if (tok->token == "WorldBegin")
                    {
                        target.onWorldBegin(tok->loc);
                    }
                    else
                    {
                        syntaxError(*tok);
                    }
                    break;

                default:
                    syntaxError(*tok);
                }
            }
        }

        void parseFile(ParserTarget& target, const std::filesystem::path& path)
        {
            auto tokenizer = Tokenizer::createFromFile(path);
            parse(target, std::move(tokenizer));
            target.onEndOfFiles();
        }

        void parseString(ParserTarget& target, std::string str)
        {
            auto tokenizer = Tokenizer::createFromString(std::move(str));
            parse(target, std::move(tokenizer));
            target.onEndOfFiles();
        }
    }
}
