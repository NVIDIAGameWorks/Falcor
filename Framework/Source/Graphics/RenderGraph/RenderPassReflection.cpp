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
#include "Framework.h"
#include "RenderPassReflection.h"

namespace Falcor
{
    const ReflectionResourceType::SharedPtr RenderPassReflection::Field::kpTex2DType = ReflectionResourceType::create(ReflectionResourceType::Type::Texture, ReflectionResourceType::Dimensions::Texture2D);

    RenderPassReflection::Field::Field(const std::string& name, Type type) : mName(name), mType(type)
    {
        mBindFlags = Resource::BindFlags::None;
        if (is_set(type, Type::Input)) mBindFlags |= Resource::BindFlags::ShaderResource;
        if (is_set(type, Type::Output)) mBindFlags |= Resource::BindFlags::RenderTarget;
    }

    RenderPassReflection::Field::Field()
        : Field("", Type::None)
    {
    }

    bool RenderPassReflection::Field::isValid() const
    {
        return (mType != Type::None) && (mName.empty() == false);
    }

    RenderPassReflection::Field& RenderPassReflection::addField(const std::string& name, Field::Type type)
    {
        mFields.push_back(Field(name, type));
        return mFields.back();
    }

    RenderPassReflection::Field& RenderPassReflection::addInput(const std::string& name)
    {
        return addField(name, Field::Type::Input);
    }

    RenderPassReflection::Field& RenderPassReflection::addOutput(const std::string& name)
    {
        return addField(name, Field::Type::Output);
    }

    RenderPassReflection::Field& RenderPassReflection::addInputOutput(const std::string& name)
    {
        return addField(name, Field::Type::Input | Field::Type::Output);
    }

    RenderPassReflection::Field& RenderPassReflection::addInternal(const std::string& name)
    {
        return addField(name, Field::Type::Internal);
    }

    const RenderPassReflection::Field& RenderPassReflection::getField(const std::string& name, Field::Type type) const
    {
        for (const auto& field : mFields)
        {
            if (field.getName() == name && (is_set(field.getType(), type) || type == Field::Type::None))
            {
                return field;
            }
        }
        std::string error = "Can't find a field named `" + name + "` in RenderPassReflection";
        throw std::runtime_error(error.c_str());
    }

}