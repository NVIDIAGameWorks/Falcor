#include "HLSLDescs.h"

using namespace Falcor;

//  Add Variable to Constant Buffer.
void HLSLDescs::ConstantBufferData::addVariableToConstantBuffer(const HLSLDescs::HLSLResourcesDesc & rDesc)
{
    structDesc.structVariables.push_back(rDesc);
}


//  Set the Array Desc.
void HLSLDescs::ConstantBufferData::setArrayDesc(const std::vector<uint32_t> & dimensions, bool isArray /*= false*/, bool isUnboundedArray /*= false*/)
{
    arrayDesc.dimensions = dimensions;
    arrayDesc.isArray = isArray;
}

//  Set the Attachment Desc.
void HLSLDescs::ConstantBufferData::setAttachmentDesc(const uint32_t newRegisterIndex, const uint32_t newRegisterSpace /*= 0*/, bool isExplicitRegisterIndex /*= false*/, bool isExplicitRegisterSpace /*= true*/)
{
    attachmentDesc.registerType = "b";
    attachmentDesc.attachmentSubpoint = newRegisterIndex;
    attachmentDesc.attachmentPoint = newRegisterSpace;
    attachmentDesc.isAttachmentSubpointExplicit = isExplicitRegisterIndex;
    attachmentDesc.isAttachmentPointExplicit = isExplicitRegisterSpace;
}

//  
std::string HLSLDescs::ConstantBufferData::getCBVariable() const
{
    return resourceVariable;
}

bool HLSLDescs::ConstantBufferData::getIsGlobalCBType() const
{
    return isGlobalUCBType;
}

//  
const CommonShaderDescs::ArrayDesc & Falcor::HLSLDescs::ConstantBufferData::viewArrayDesc() const
{
    return arrayDesc;
}

//
const CommonShaderDescs::ResourceAttachmentDesc & Falcor::HLSLDescs::ConstantBufferData::viewAttachmentDesc() const
{
    return attachmentDesc;
}

const HLSLDescs::HLSLStructDesc & Falcor::HLSLDescs::ConstantBufferData::viewStructDesc() const
{
    return structDesc;
}

