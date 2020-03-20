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
#include "FalcorTest.h"
#include "Testing/UnitTest.h"
#include <cstdio>
#include <string>
#include <vector>

static std::vector<std::string> librariesWithTests =
{
};

/** Global to hold return code.
    The instance of FalcorTest is destroyed before leaving Sample::run().
*/
static int sReturnCode = 1;

void FalcorTest::onLoad(RenderContext* pRenderContext)
{
    // Load all the DLLs so that they can register their tests.
    for (const auto& lib : librariesWithTests)
    {
        RenderPassLibrary::instance().loadLibrary(lib);
    }
}

void FalcorTest::onFrameRender(RenderContext* pRenderContext, const Fbo::SharedPtr& pTargetFbo)
{
    const char* kTestFilterSwitch = "test_filter";
    ArgList argList = gpFramework->getArgList();
    std::string testFilterRegex;
    if (argList.argExists(kTestFilterSwitch))
    {
        testFilterRegex = argList[kTestFilterSwitch].asString();
        std::cout << "No test_filter regex provided." << std::endl;
        sReturnCode = 1;
    }
    if (argList.argExists("h") || argList.argExists("help"))
    {
        std::cout << R"(usage: FalcorTest [-test_filter filter]
            Where, if |filter| is provided, only tests whose source filename or test name
            have |filter| as a substring are executed.
            )";
    }
    else
    {
        sReturnCode = runTests(std::cout, pRenderContext, testFilterRegex);
    }
    gpFramework->shutdown();
}

int main(int argc, char** argv)
{
    FalcorTest::UniquePtr pRenderer = std::make_unique<FalcorTest>();
    SampleConfig config;
    config.windowDesc.title = "FalcorTest";
    config.windowDesc.mode = Window::WindowMode::Minimized;
    config.windowDesc.resizableWindow = true;
    config.windowDesc.width = config.windowDesc.height = 2;
    Sample::run(config, pRenderer, argc, argv);
    return sReturnCode;
}
