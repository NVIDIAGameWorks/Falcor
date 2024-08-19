/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
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
#include "Resolver.h"
#include "Utils/Math/MathHelpers.h"
#include "Utils/StringUtils.h"
#include "Utils/Math/ScalarMath.h"
#include "Utils/Math/VectorTypes.h"
#include "Utils/Math/VectorMath.h"
#include "Utils/Math/MatrixTypes.h"
#include "Utils/Math/MatrixMath.h"

#include <pugixml.hpp>

#include <set>
#include <map>
#include <variant>
#include <unordered_map>

namespace Falcor
{
namespace Mitsuba
{
struct Color3
{
    Color3() = default;
    Color3(float c) : r(c), g(c), b(c) {}
    Color3(float r, float g, float b) : r(r), g(g), b(b) {}
    operator float3() const { return float3(r, g, b); }
    operator float4() const { return float4(r, g, b, 1.f); }
    float r, g, b;
};

/** Represents a mitsuba object and it's properties.
 */
class Properties
{
public:
    enum class Type
    {
        Bool,
        Int,
        Float,
        String,
        Float3,
        Color3,
        Transform,
    };

    using VariantType = std::variant<bool, int64_t, float, std::string, float3, Color3, float4x4>;

    const std::map<std::string, VariantType>& getProperties() const { return mProperties; }
    const std::map<std::string, std::string>& getNamedReferences() const { return mNamedReferences; }

    void addNamedReference(const std::string& name, const std::string& id) { mNamedReferences.emplace(name, id); }

    bool hasNamedReference(const std::string& name) const { return mNamedReferences.find(name) != mNamedReferences.end(); }

    std::string getNamedReference(const std::string& name) const
    {
        auto it = mNamedReferences.find(name);
        if (it == mNamedReferences.end())
            FALCOR_THROW("Cannot find named reference '{}'.", name);
        return it->second;
    }

#define PROPERTY_ACCESSOR(_type_, _getter_, _setter_, _has_)                          \
    const _type_& _getter_(const std::string& name) const                             \
    {                                                                                 \
        return getter<_type_>(name);                                                  \
    }                                                                                 \
    const _type_& _getter_(const std::string& name, const _type_& defaultValue) const \
    {                                                                                 \
        return getter<_type_>(name, defaultValue);                                    \
    }                                                                                 \
    void _setter_(const std::string& name, const _type_& value)                       \
    {                                                                                 \
        setter<_type_>(name, value);                                                  \
    }                                                                                 \
    bool _has_(const std::string& name) const                                         \
    {                                                                                 \
        return has<_type_>(name);                                                     \
    }

    PROPERTY_ACCESSOR(bool, getBool, setBool, hasBool)
    PROPERTY_ACCESSOR(int64_t, getInt, setInt, hasInt)
    PROPERTY_ACCESSOR(float, getFloat, setFloat, hasFloat)
    PROPERTY_ACCESSOR(std::string, getString, setString, hasString)
    PROPERTY_ACCESSOR(float3, getFloat3, setFloat3, hasFloat3)
    PROPERTY_ACCESSOR(Color3, getColor3, setColor3, hasColor3)
    PROPERTY_ACCESSOR(float4x4, getTransform, setTransform, hasTransform)

#undef PROPERTY_ACCESSOR

private:
    template<typename T>
    const T& getter(const std::string& name) const
    {
        auto it = mProperties.find(name);
        if (it == mProperties.end())
        {
            FALCOR_THROW("Property '{}' not found.", name);
        }
        if (!std::holds_alternative<T>(it->second))
        {
            FALCOR_THROW("Property '{}' has invalid type.", name);
        }
        return std::get<T>(it->second);
    }

    template<typename T>
    const T& getter(const std::string& name, const T& defaultValue) const
    {
        auto it = mProperties.find(name);
        if (it == mProperties.end())
        {
            return defaultValue;
        }
        if (!std::holds_alternative<T>(it->second))
        {
            FALCOR_THROW("Property '{}' has invalid type.", name);
        }
        return std::get<T>(it->second);
    }

    template<typename T>
    void setter(const std::string& name, const T& value)
    {
        mProperties.emplace(name, value);
    }

    template<typename T>
    bool has(const std::string& name) const
    {
        auto it = mProperties.find(name);
        if (it == mProperties.end())
        {
            return false;
        }
        return std::holds_alternative<T>(it->second);
    }

    std::map<std::string, VariantType> mProperties;
    std::map<std::string, std::string> mNamedReferences;
};

/** Mitsuba scene tags.
 */
enum class Tag
{
    Boolean,
    Integer,
    Float,
    String,
    Point,
    Vector,
    Spectrum,
    RGB,
    Transform,
    Translate,
    Matrix,
    Rotate,
    Scale,
    LookAt,
    Object,
    NamedReference,
    Include,
    Alias,
    Default,
    Resource,
    Invalid
};

/** Mitsuba plugin classes.
 */
enum class Class
{
    Scene,
    Integrator,
    Sensor,
    Emitter,
    Sampler,
    Film,
    RFilter,
    Shape,
    BSDF,
    Texture,
    Medium
};

/** Map to lookup tag type from string.
 */
const std::unordered_map<std::string, Tag> kTags = {
    {"boolean", Tag::Boolean},
    {"integer", Tag::Integer},
    {"float", Tag::Float},
    {"string", Tag::String},
    {"point", Tag::Point},
    {"vector", Tag::Vector},
    {"transform", Tag::Transform},
    {"translate", Tag::Translate},
    {"matrix", Tag::Matrix},
    {"rotate", Tag::Rotate},
    {"scale", Tag::Scale},
    {"lookat", Tag::LookAt},
    {"ref", Tag::NamedReference},
    {"spectrum", Tag::Spectrum},
    {"rgb", Tag::RGB},
    {"include", Tag::Include},
    {"alias", Tag::Alias},
    {"default", Tag::Default},
    {"path", Tag::Resource},
    // Classes
    {"scene", Tag::Object},
    {"integrator", Tag::Object},
    {"sensor", Tag::Object},
    {"emitter", Tag::Object},
    {"sampler", Tag::Object},
    {"film", Tag::Object},
    {"rfilter", Tag::Object},
    {"shape", Tag::Object},
    {"bsdf", Tag::Object},
    {"texture", Tag::Object},
    {"medium", Tag::Object},
};

const std::unordered_map<std::string, Class> kClasses = {
    {"scene", Class::Scene},
    {"integrator", Class::Integrator},
    {"sensor", Class::Sensor},
    {"emitter", Class::Emitter},
    {"sampler", Class::Sampler},
    {"film", Class::Film},
    {"rfilter", Class::RFilter},
    {"shape", Class::Shape},
    {"bsdf", Class::BSDF},
    {"texture", Class::Texture},
    {"medium", Class::Medium},
};

/** Version helper.
 */
struct Version
{
    uint32_t major, minor, patch;

    Version() = default;

    Version(int major, int minor, int patch) : major(major), minor(minor), patch(patch) {}

    Version(const char* value)
    {
        if (std::sscanf(value, "%lu.%lu.%lu", &major, &minor, &patch) != 3)
        {
            FALCOR_THROW("Version string must have x.x.x format.");
        }
    }

    bool operator==(const Version& other) const { return std::tie(major, minor, patch) == std::tie(other.major, other.minor, other.patch); }

    bool operator!=(const Version& other) const { return std::tie(major, minor, patch) != std::tie(other.major, other.minor, other.patch); }

    bool operator<(const Version& other) const { return std::tie(major, minor, patch) < std::tie(other.major, other.minor, other.patch); }

    friend std::ostream& operator<<(std::ostream& os, const Version& v)
    {
        os << v.major << "." << v.minor << "." << v.patch;
        return os;
    }
};

struct XMLObject
{
    std::string id;
    Class cls;
    std::string type;
    Properties props;
    size_t location;
};

struct XMLContext
{
    std::unordered_map<std::string, XMLObject> instances;
    size_t idCounter = 0;
    float4x4 transform;
    Resolver resolver;

    std::string offset(size_t location) { return fmt::format("{}", location); }
};

struct XMLSource
{
    std::string id;
    const pugi::xml_document& doc;

    template<typename... Args>
    [[noreturn]] void throwError(const pugi::xml_node& node, const std::string& fmtString, Args&&... args)
    {
        FALCOR_THROW(fmtString, std::forward<Args>(args)...);
    }
};

/// Throws if non-whitespace characters are found after the given index.
void checkWhitespaceOnly(const std::string& s, size_t offset)
{
    for (size_t i = offset; i < s.size(); ++i)
    {
        if (!std::isspace(s[i]))
            FALCOR_THROW("Invalid whitespace");
    }
}

float parseFloat(const std::string& s)
{
    size_t offset = 0;
    float result = std::stof(s, &offset);
    checkWhitespaceOnly(s, offset);
    return result;
}

int64_t parseInt(const std::string& s)
{
    size_t offset = 0;
    int64_t result = std::stoll(s, &offset);
    checkWhitespaceOnly(s, offset);
    return result;
}

/** Helper to check if attributes are specified.
 */
void checkAttributes(XMLSource& src, const pugi::xml_node& node, std::set<std::string>&& attrs, bool expectAll = true)
{
    bool foundOne = false;
    for (auto attr : node.attributes())
    {
        auto it = attrs.find(attr.name());
        if (it == attrs.end())
        {
            src.throwError(node, "Unexpected attribute '{}' in node '{}'.", attr.name(), node.name());
        }
        attrs.erase(it);
        foundOne = true;
    }
    if (!attrs.empty() && (!foundOne || expectAll))
    {
        src.throwError(node, "Missing attribute '{}' in node '{}'.", *attrs.begin(), node.name());
    }
}

/// Helper function to split the 'value' attribute into X/Y/Z components
void expandValueToXYZ(XMLSource& src, pugi::xml_node& node)
{
    if (node.attribute("value"))
    {
        auto tokens = splitString(node.attribute("value").value(), ",");
        if (node.attribute("x") || node.attribute("y") || node.attribute("z"))
        {
            src.throwError(node, "Can't mix and match 'value' and 'x'/'y'/'z' attributes.");
        }
        if (tokens.size() == 1)
        {
            node.append_attribute("x") = tokens[0].c_str();
            node.append_attribute("y") = tokens[0].c_str();
            node.append_attribute("z") = tokens[0].c_str();
        }
        else if (tokens.size() == 3)
        {
            node.append_attribute("x") = tokens[0].c_str();
            node.append_attribute("y") = tokens[1].c_str();
            node.append_attribute("z") = tokens[2].c_str();
        }
        else
        {
            src.throwError(node, "'value' attribute must have exactly 1 or 3 elements.");
        }
        node.remove_attribute("value");
    }
}

float3 parseNamedVector(XMLSource& src, pugi::xml_node& node, const std::string& attrName)
{
    auto vecStr = node.attribute(attrName.c_str()).value();
    auto list = splitString(vecStr, ",");
    if (list.size() != 3)
        src.throwError(node, "'{}' attribute must have exactly 3 elements.", attrName);
    try
    {
        return float3(parseFloat(list[0]), parseFloat(list[1]), parseFloat(list[2]));
    }
    catch (...)
    {
        src.throwError(node, "Could not parse floating point values in '{}'.", vecStr);
    }
}

float3 parseVector(XMLSource& src, const pugi::xml_node& node, float defaultValue = 0.f)
{
    std::string value;
    try
    {
        float x = defaultValue, y = defaultValue, z = defaultValue;
        value = node.attribute("x").value();
        if (!value.empty())
            x = parseFloat(value);
        value = node.attribute("y").value();
        if (!value.empty())
            y = parseFloat(value);
        value = node.attribute("z").value();
        if (!value.empty())
            z = parseFloat(value);
        return float3(x, y, z);
    }
    catch (...)
    {
        src.throwError(node, "Could not parse floating point value '{}'.", value);
    }
}

Color3 parseRGB(XMLSource& src, const pugi::xml_node& node)
{
    auto value = node.attribute("value").value();
    auto tokens = splitString(value, ",");

    if (tokens.size() == 1)
    {
        tokens.push_back(tokens[0]);
        tokens.push_back(tokens[0]);
    }
    if (tokens.size() != 3)
    {
        src.throwError(node, "RGB value requires one or three components (got '{}').", value);
    }

    Color3 color;
    try
    {
        color = {parseFloat(tokens[0]), parseFloat(tokens[1]), parseFloat(tokens[2])};
    }
    catch (...)
    {
        src.throwError(node, "Could not parse RGB value '{}'.", value);
    }
    return color;
}

void upgradeTree(XMLSource& src, pugi::xml_node& node, const Version& version)
{
    if (version < Version(2, 0, 0))
    {
        // Upgrade all attribute names from camelCase to underscore_case
        for (pugi::xpath_node result : node.select_nodes("//*[@name]"))
        {
            pugi::xml_node n = result.node();
            if (std::strcmp(n.name(), "default") == 0)
                continue;
            pugi::xml_attribute attr = n.attribute("name");
            std::string name = attr.value();
            for (size_t i = 0; i < name.length() - 1; ++i)
            {
                if (std::islower(name[i]) && std::isupper(name[i + 1]))
                {
                    name = name.substr(0, i + 1) + std::string("_") + name.substr(i + 1);
                    i += 2;
                    while (i < name.length() && std::isupper(name[i]))
                    {
                        name[i] = std::tolower(name[i]);
                        ++i;
                    }
                }
            }
            attr.set_value(name.c_str());
        }

        for (pugi::xpath_node result : node.select_nodes("//lookAt"))
        {
            result.node().set_name("lookat");
        }

        // Automatically rename reserved identifiers.
        for (pugi::xpath_node result : node.select_nodes("//@id"))
        {
            pugi::xml_attribute attr = result.attribute();
            char const* val = attr.value();
            if (val && val[0] == '_')
            {
                std::string new_id = std::string("ID") + val + "__UPGR";
                // Log(Warn, "Changing identifier: \"%s\" -> \"%s\"", val, new_id.c_str()); TODO
                attr = new_id.c_str();
            }
        }

        // Changed parameters.
        for (pugi::xpath_node result : node.select_nodes("//bsdf[@type='diffuse']/*/@name[.='diffuse_reflectance']"))
        {
            result.attribute() = "reflectance";
        }

        // Update 'uoffset', 'voffset', 'uscale', 'vscale' to transform block
        for (pugi::xpath_node result :
             node.select_nodes("//node()[float[@name='uoffset' or @name='voffset' or @name='uscale' or @name='vscale']]"))
        {
            pugi::xml_node n = result.node();
            pugi::xml_node uoffset = n.select_node("float[@name='uoffset']").node();
            pugi::xml_node voffset = n.select_node("float[@name='voffset']").node();
            pugi::xml_node uscale = n.select_node("float[@name='uscale']").node();
            pugi::xml_node vscale = n.select_node("float[@name='vscale']").node();

            float2 offset(0.f);
            float2 scale(1.f);
            if (uoffset)
            {
                offset.x = parseFloat(uoffset.attribute("value").value());
                n.remove_child(uoffset);
            }
            if (voffset)
            {
                offset.y = parseFloat(voffset.attribute("value").value());
                n.remove_child(voffset);
            }
            if (uscale)
            {
                scale.x = parseFloat(uscale.attribute("value").value());
                n.remove_child(uscale);
            }
            if (vscale)
            {
                scale.y = parseFloat(vscale.attribute("value").value());
                n.remove_child(vscale);
            }

            pugi::xml_node trafo = n.append_child("transform");
            trafo.append_attribute("name") = "to_uv";

            if (any(offset != float2(0.f)))
            {
                pugi::xml_node element = trafo.append_child("translate");
                element.append_attribute("x") = std::to_string(offset.x).c_str();
                element.append_attribute("y") = std::to_string(offset.y).c_str();
            }

            if (any(scale != float2(1.f)))
            {
                pugi::xml_node element = trafo.append_child("scale");
                element.append_attribute("x") = std::to_string(scale.x).c_str();
                element.append_attribute("y") = std::to_string(scale.y).c_str();
            }
        }
    }

    // src.modified = true; TODO
}

std::pair<std::string, std::string> parseXML(
    XMLSource& src,
    XMLContext& ctx,
    pugi::xml_node& node,
    Tag parentTag,
    Properties& props,
    size_t& argCounter,
    uint32_t depth = 0,
    bool withinEmitter = false,
    bool withinSpectrum = false
)
{
    logDebug("Parsing tag {}", node.name());

    // Skip over comments.
    if (node.type() == pugi::node_comment || node.type() == pugi::node_declaration)
    {
        return std::make_pair("", "");
    }

    // Check for valid element.
    if (node.type() != pugi::node_element)
    {
        src.throwError(node, "Unexpected content.");
    }

    // Check for valid tag.
    auto it = kTags.find(node.name());
    if (it == kTags.end())
    {
        src.throwError(node, "Unexpected tag '{}'.", node.name());
    }
    const Tag tag = it->second;

    // Perform some safety checks to make sure that the XML tree really makes sense
    const bool hasParent = parentTag != Tag::Invalid;
    const bool parentIsObject = hasParent && parentTag == Tag::Object;
    const bool currentIsObject = tag == Tag::Object;
    const bool parentIsTransform = parentTag == Tag::Transform;
    const bool currentIsTransformOp =
        tag == Tag::Translate || tag == Tag::Rotate || tag == Tag::Scale || tag == Tag::LookAt || tag == Tag::Matrix;

    if (!hasParent && !currentIsObject)
    {
        src.throwError(node, "Root node '{}' must be an object.", node.name());
    }

    if (parentIsTransform != currentIsTransformOp)
    {
        if (parentIsTransform)
            src.throwError(node, "Transform nodes can only contain transform operations.");
        else
            src.throwError(node, "Transform operations can only occur in a transform node.");
    }

    if (hasParent && !parentIsObject && !(parentIsTransform && currentIsTransformOp))
    {
        src.throwError(node, "Node '{}' cannot occur as child of a property.", node.name());
    }

    // Check version.
    if (depth == 0)
    {
        if (!node.attribute("version"))
        {
            src.throwError(node, "Missing version attribute in root element '{}'.", node.name());
        }

        const Version version(node.attribute("version").value());

        // Upgrade XML tree to version 2.0.0.
        upgradeTree(src, node, version);

        // Remove version attribute, otherwise it will be detected as an unexpected attribute later.
        node.remove_attribute("version");
    }

    // Set type on scene node.
    if (std::string(node.name()) == "scene")
    {
        node.append_attribute("type") = "scene";
    }

    // Reset transform.
    if (tag == Tag::Transform)
    {
        // TODO reset transform
        // ctx.transform = Transform4f();
    }

    // Check for valid name and set generic name if none is set.
    if (node.attribute("name"))
    {
        std::string name = node.attribute("name").value();
        if (name.empty())
            src.throwError(node, "Node '{}' has empty name attribute.", node.name());
        else if (name[0] == '_')
            src.throwError(node, "Node '{}' has invalid name '{}' with leading underscores.", node.name(), name);
    }
    else if (currentIsObject || tag == Tag::NamedReference)
    {
        // To keep shape/bsdf/etc in the same order by padding leading zeros.
        node.append_attribute("name") = fmt::format("_arg_{:04d}", argCounter++).c_str();
    }

    // Check for valid id and set generic id if none is set.
    if (node.attribute("id"))
    {
        std::string id = node.attribute("id").value();
        if (id.empty())
            src.throwError(node, "Node '{}' has empty id attribute.", node.name());
        else if (id[0] == '_')
            src.throwError(node, "Node '{}' has invalid id '{}' with leading underscores.", node.name(), id);
    }
    else if (currentIsObject)
    {
        // TODO set id
        node.append_attribute("id") = fmt::format("_unnamed_{}", ctx.idCounter++).c_str();
    }

    switch (tag)
    {
    case Tag::Object:
    {
        checkAttributes(src, node, {"type", "id", "name"});
        std::string nodeName = node.name();
        std::string id = node.attribute("id").value();
        std::string name = node.attribute("name").value();
        std::string type = node.attribute("type").value();

        Properties nestedProps; // TODO (type);
        // props_nested.set_id(id);

        // Check if instance with this id already exists.
        {
            auto it2 = ctx.instances.find(id);
            if (it2 != ctx.instances.end())
            {
                src.throwError(node, "Node '{}' has duplicate id '{}' (previous was at {}).", nodeName, id, it2->second.location);
            }
        }

        // TODO
        // auto it2 = tag_class->find(class_key(node_name, ctx.variant));
        // if (it2 == tag_class->end())
        //     src.throw_error(node, "could not retrieve class object for "
        //                    "tag \"%s\" and variant \"%s\"", node_name, ctx.variant);

        size_t argCounterNested = 0;
        for (pugi::xml_node& child : node.children())
        {
            auto [nestedName, nestedID] =
                parseXML(src, ctx, child, tag, nestedProps, argCounter, depth + 1, nodeName == "emitter", nodeName == "spectrum");
            if (!nestedID.empty())
            {
                nestedProps.addNamedReference(nestedName, nestedID);
            }
        }

        auto cls = kClasses.at(nodeName);

        // Create instance.
        auto& inst = ctx.instances[id];
        inst.id = id;
        inst.cls = cls;
        inst.type = type;
        inst.props = nestedProps;
        // inst.class_ = it2->second;
        // inst.offset = src.offset;
        // inst.src_id = src.id;
        inst.location = node.offset_debug();
        return std::make_pair(name, id);
    }
    break;

    case Tag::NamedReference:
    {
        checkAttributes(src, node, {"name", "id"});
        auto name = node.attribute("name").value();
        auto id = node.attribute("id").value();
        return std::make_pair(name, id);
    }
    break;

    case Tag::Alias:
    {
        // TODO implement
        FALCOR_THROW("not implemented");
    }
    break;

    case Tag::Default:
    {
        // TODO implement
        FALCOR_THROW("not implemented");
    }
    break;

    case Tag::Resource:
    {
        // TODO implement
        FALCOR_THROW("not implemented");
    }
    break;

    case Tag::Include:
    {
        // TODO implement
        FALCOR_THROW("not implemented");
    }
    break;

    case Tag::String:
    {
        checkAttributes(src, node, {"name", "value"});

        std::string name = node.attribute("name").value();
        std::string value = node.attribute("value").value();
        if (name == "filename")
        {
            value = ctx.resolver.resolve(value).string();
        }

        props.setString(name, value);
    }
    break;

    case Tag::Float:
    {
        checkAttributes(src, node, {"name", "value"});
        std::string value = node.attribute("value").value();
        float floatValue;
        try
        {
            floatValue = parseFloat(value);
        }
        catch (...)
        {
            src.throwError(node, "Could not parse floating point value '{}'.", value);
        }
        props.setFloat(node.attribute("name").value(), floatValue);
    }
    break;

    case Tag::Integer:
    {
        checkAttributes(src, node, {"name", "value"});
        std::string value = node.attribute("value").value();
        int64_t valueInt;
        try
        {
            valueInt = parseInt(value);
        }
        catch (...)
        {
            src.throwError(node, "Could not parse integer value '{}'.", value);
        }
        props.setInt(node.attribute("name").value(), valueInt);
    }
    break;

    case Tag::Boolean:
    {
        checkAttributes(src, node, {"name", "value"});
        std::string value = node.attribute("value").value();
        std::transform(value.begin(), value.end(), value.begin(), ::tolower);
        bool boolValue = false;
        if (value == "true")
            boolValue = true;
        else if (value == "false")
            boolValue = false;
        else
            src.throwError(node, "Could not parse boolean value '{}', must be 'true' or 'false'.", value);
        props.setBool(node.attribute("name").value(), boolValue);
    }
    break;

    case Tag::Vector:
    {
        expandValueToXYZ(src, node);
        checkAttributes(src, node, {"name", "x", "y", "z"});
        props.setFloat3(node.attribute("name").value(), parseVector(src, node));
    }
    break;

    case Tag::Point:
    {
        expandValueToXYZ(src, node);
        checkAttributes(src, node, {"name", "x", "y", "z"});
        props.setFloat3(node.attribute("name").value(), parseVector(src, node));
    }
    break;

    case Tag::RGB:
    {
        checkAttributes(src, node, {"name", "value"});
        auto color = parseRGB(src, node);
        props.setColor3(node.attribute("name").value(), color);
        // if (!within_spectrum) {
        //     std::string name = node.attribute("name").value();
        //     ref<Object> obj = detail::create_texture_from_rgb(
        //         name, color, ctx.variant, within_emitter);
        //     props.set_object(name, obj);
        // } else {
        //     props.set_color("color", color);
        // }
    }
    break;

    case Tag::Spectrum:
    {
        // TODO: Support real multi-channel spectrum
        checkAttributes(src, node, {"name", "value"});
        auto color = parseRGB(src, node);
        props.setColor3(node.attribute("name").value(), color);
    }
    break;

    case Tag::Transform:
    {
        checkAttributes(src, node, {"name"});
        ctx.transform = float4x4::identity();
    }
    break;

    case Tag::Rotate:
    {
        expandValueToXYZ(src, node);
        checkAttributes(src, node, {"angle", "x", "y", "z"}, false);
        auto vec = parseVector(src, node);
        auto value = node.attribute("angle").value();
        float angle;
        try
        {
            angle = parseFloat(value);
        }
        catch (...)
        {
            src.throwError(node, "Could not parse floating point value '{}'.", value);
        }
        angle = angle / 180.f * M_PI; // Degree to radian
        ctx.transform = mul(math::matrixFromRotation(angle, vec), ctx.transform);
    }
    break;

    case Tag::Translate:
    {
        expandValueToXYZ(src, node);
        checkAttributes(src, node, {"x", "y", "z"}, false);
        auto vec = parseVector(src, node);
        ctx.transform = mul(math::matrixFromTranslation(vec), ctx.transform);
    }
    break;

    case Tag::Scale:
    {
        expandValueToXYZ(src, node);
        checkAttributes(src, node, {"x", "y", "z"}, false);
        auto vec = parseVector(src, node, 1.f);
        ctx.transform = mul(math::matrixFromScaling(vec), ctx.transform);
    }
    break;

    case Tag::LookAt:
    {
        if (!node.attribute("up"))
            node.append_attribute("up") = "0,0,0";
        checkAttributes(src, node, {"origin", "target", "up"});

        auto origin = parseNamedVector(src, node, "origin");
        auto target = parseNamedVector(src, node, "target");
        auto up = parseNamedVector(src, node, "up");

        if (length(up) == 0.f)
        {
            float3 tmp;
            buildFrame(normalize(target - origin), up, tmp);
        }

        auto vDir = normalize(target - origin);
        auto vRight = normalize(cross(normalize(up), vDir));
        auto vNewUp = cross(vDir, vRight);

        float4x4 lookAtMatrix;
        lookAtMatrix.setCol(0, float4(vRight, 0.f));
        lookAtMatrix.setCol(1, float4(vNewUp, 0.f));
        lookAtMatrix.setCol(2, float4(vDir, 0.f));
        lookAtMatrix.setCol(3, float4(origin, 1.f));

        ctx.transform = mul(lookAtMatrix, ctx.transform);
    }
    break;

    case Tag::Matrix:
    {
        checkAttributes(src, node, {"value"});
        auto tokens = splitString(node.attribute("value").value(), " ");
        if (tokens.size() != 16 && tokens.size() != 9)
        {
            src.throwError(node, "Matrix needs 9 or 16 values.");
        }

        float4x4 mat;
        if (tokens.size() == 16)
        {
            for (int i = 0; i < 4; ++i)
            {
                for (int j = 0; j < 4; ++j)
                {
                    try
                    {
                        mat[j][i] = parseFloat(tokens[i * 4 + j]);
                    }
                    catch (...)
                    {
                        src.throwError(node, "Could not parse floating point value '{}'.", tokens[i * 4 + j]);
                    }
                }
            }
        }
        else
        {
            float3x3 mat3;
            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    try
                    {
                        mat3[j][i] = parseFloat(tokens[i * 3 + j]);
                    }
                    catch (...)
                    {
                        src.throwError(node, "Could not parse floating point value '{}'.", tokens[i * 3 + j]);
                    }
                }
            }
            mat = float4x4(mat3);
        }
        ctx.transform = mul(mat, ctx.transform);
    }
    break;

    default:
        FALCOR_THROW("Unknown tag!");
    }

    for (pugi::xml_node& child : node.children())
    {
        parseXML(src, ctx, child, tag, props, argCounter, depth + 1);
    }

    if (tag == Tag::Transform)
    {
        props.setTransform(node.attribute("name").value(), ctx.transform);
    }

    return {"", ""};
}

} // namespace Mitsuba

} // namespace Falcor
