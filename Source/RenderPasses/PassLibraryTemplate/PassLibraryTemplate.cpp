/***************************************************************************
 # Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
#include "PassLibraryTemplate.h"

// Don't remove this. it's required for hot-reload to function properly
extern "C" __declspec(dllexport) const char* getProjDir()
{
    return PROJECT_DIR;
}

extern "C" __declspec(dllexport) void getPasses(Falcor::RenderPassLibrary& lib)
{
    lib.registerClass("RenderPassTemplate", "Render Pass Template", RenderPassTemplate::create);
}

RenderPassTemplate::SharedPtr RenderPassTemplate::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    SharedPtr pPass = SharedPtr(new RenderPassTemplate);
    return pPass;
}

Dictionary RenderPassTemplate::getScriptingDictionary()
{
    return Dictionary();
}

RenderPassReflection RenderPassTemplate::reflect(const CompileData& compileData)
{
    // Define the required resources here
    RenderPassReflection reflector;
    //reflector.addOutput("dst");
    //reflector.addInput("src");
    return reflector;
}

void RenderPassTemplate::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    // renderData holds the requested resources
    // auto& pTexture = renderData["src"]->asTexture();
}

void RenderPassTemplate::renderUI(Gui::Widgets& widget)
{
}
