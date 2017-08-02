#include "GLSLDescs.h"

using namespace Falcor;

void GLSLDescs::UniformBufferData::addVariableToUniformBuffer(const GLSLDescs::GLSLResourcesDesc & rDesc)
{

}

void Falcor::GLSLDescs::UniformBufferData::setArrayDesc(const std::vector<uint32_t> & dimensions, bool isArray /*= false*/, bool isUnboundedArray /*= false*/)
{
    arrayDesc.dimensions = dimensions;
    arrayDesc.isArray = isArray;
}

void Falcor::GLSLDescs::UniformBufferData::setAttachmentDesc(const uint32_t newBindingIndex, const uint32_t newBindingSet /*= 0*/, bool isExplicitRegisterIndex /*= false*/, bool isExplicitRegisterSpace /*= true*/)
{
    attachmentDesc.registerType = "b";
    attachmentDesc.attachmentSubpoint = newBindingIndex;
    attachmentDesc.attachmentPoint = newBindingSet;
    attachmentDesc.isAttachmentSubpointExplicit = isExplicitRegisterIndex;
    attachmentDesc.isAttachmentPointExplicit = isExplicitRegisterSpace;

}

std::string Falcor::GLSLDescs::UniformBufferData::getUBVariable() const
{
    return resourceVariable;
}

bool Falcor::GLSLDescs::UniformBufferData::getIsGlobalUBType() const
{
    return isGlobalUCBType;
}

const Falcor::CommonShaderDescs::ArrayDesc & Falcor::GLSLDescs::UniformBufferData::viewArrayDesc() const
{
    return arrayDesc;
}

const Falcor::CommonShaderDescs::ResourceAttachmentDesc & Falcor::GLSLDescs::UniformBufferData::viewAttachmentDesc() const
{
    return attachmentDesc;
}

const Falcor::GLSLDescs::GLSLStructDesc & Falcor::GLSLDescs::UniformBufferData::viewStructDesc() const
{
    return structDesc;
}
