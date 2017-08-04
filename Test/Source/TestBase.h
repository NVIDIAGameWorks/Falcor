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
#include "Falcor.h"

using namespace Falcor;

#define register_testing_func(x_) class x_ : public TestBase::TestFunction { public: TestData operator ()() override; };
#define testing_func(className_, functorName_) TestBase::TestData className_::functorName_::operator()()
#define test_pass() TestBase::TestData(TestBase::TestResult::Pass, mName);
#define test_fail(errorMessage_) TestBase::TestData(TestBase::TestResult::Fail, mName, errorMessage_);

class TestBase
{
public:
    enum class TestResult
    {
        Pass,
        Fail,
        Crash
    };

    struct TestData
    {
        TestData() : result(TestResult::Fail) {}
        TestData(TestResult r, std::string testName) : result(r), testName(testName) {}
        TestData(TestResult r, std::string testName, std::string err) :
            result(r), testName(testName), error(err) {}

        TestResult result;
        std::string testName;
        std::string error;
    };

    virtual ~TestBase();
    void init(bool initDevice = false);
    void run();

protected:
    TestBase();
    virtual void addTests() = 0;
    virtual void onInit() = 0;

    //Used to gen correct filename in base class method
    std::string mTestName;
    std::vector<TestData> mpTestResults;

    class TestFunction
    {
    public:
        TestFunction() {}
        virtual TestData operator()() { return TestData(); }
        std::string mName;
    };

    //This is templatized but it expects a type that derives from testfunction
    template <typename T>
    void addTestToList()
    {
        T* newTestFunctor = new T();
        std::string wholeName = std::string(typeid(*newTestFunctor).name());
        size_t colonIndex = wholeName.find(":");
        newTestFunctor->mName = wholeName.substr(colonIndex + 2, std::string::npos);

        mpTestList.push_back(newTestFunctor);
    }

    std::vector<TestFunction*> mpTestList;

private:
    class ResultSummary
    {
    public:
        ResultSummary() : total(0u), pass(0u), fail(0u), crash(0u) {}
        void addTestToSummary(TestData d)
        {
            ++total;
            if (d.result == TestResult::Pass) ++pass;
            else if (d.result == TestResult::Fail) ++fail;
            else ++crash;
        }

        uint32_t total;
        uint32_t pass;
        uint32_t fail;
        uint32_t crash;
    } mResultSummary;

    class DummyWindowCallbacks : public Window::ICallbacks
    {
        void renderFrame() override {}
        void handleWindowSizeChange() override {}
        void handleKeyboardEvent(const KeyboardEvent& keyEvent) override {}
        void handleMouseEvent(const MouseEvent& mouseEvent) override {}
    } mDummyCallbacks;

    std::vector<TestData> runTests();
    void GenerateXML(const std::vector<std::string>& xmlStrings);
    std::string XMLFromTestResult(const TestData& r);

    Window::SharedPtr mpWindow; //used for dummy device creation
};
