/***************************************************************************
# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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

#include "LiveTrainingDemo.h"

#if FALCOR_USE_PYTHON

#include "Python.h"
#include "pybind11/pybind11.h"
#include "pybind11/embed.h"
#include "pybind11/eval.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "pybind11/buffer_info.h"
#include "Utils/PythonEmbedding.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <streambuf>

// We use an arbitrarily brighten our light in this sample code.  For no good reason except it looks better
//    in the scenes we used.  No good reason for this particular constant, either (except it arose empirically  
//    out of some, now irrelevant, experimental tests we did at one point).
#define LIGHT_INTENSITY  3.882f

// Clarity.  Cleans pybind11 call notation a bit.
namespace py = pybind11;

void LiveTrainRenderer::onInitialize(RenderContext::SharedPtr)
{
    // Don't re-initialize if we already have.
    if (mIsInitialized) 
    {
        return;
    }

    // Seed our random number generator.
    mRng.seed((uint32)time(0));

    // Create a python interpreter with appropriate TensorFlow version imported
    if (!mPython)
    {
        // Create our embedded Python interpreter, import some common modules
        mPython = PythonEmbedding::create();
        mPython->importModule("os");               // think in Python: "import os"
        mPython->importModule("io");
        mPython->importModule("sys");
        mPython->importModule("math");
        mPython->importModule("importlib");
        mPython->importModule("contextlib");
        mPython->importModule("numpy", "np");     // think in Python: "import numpy as np"
        mPython->importModule("tensorflow", "tf");
        mPython->importModule("PIL");
        
        // Get a copy of the Python dictionary that stores global variables
        mGlobals = mPython->getGlobals();
    }

    // A texture to store the data we return from Python
    mPythonReturnTexture = Texture::create2D( 512, 512, ResourceFormat::RGBA8UnormSrgb ); 

    // Load out Python scripts
    reloadPythonScripts();

    // Run our Python-based initialization code
    doPythonInit();

    // Create a dropdown to swap between DNN sizes.
    mDNNSizeList.push_back( { 0, "Learn 128 x 128 image" } );
    mDNNSizeList.push_back( { 1, "Learn 256 x 256 image" } );
    mDNNSizeList.push_back( { 2, "Learn 512 x 512 image" } );

    // Resize our gui window so it's big enough
    mGuiSize = ivec2(300, 600);
    mIsInitialized = true;
}

void LiveTrainRenderer::doPythonInit()
{
    // Run our Python-based initialization code.  If we fail, get error messages and set appropriate flags
    if (!mPython->executeString(mPythonInit))
    {
        mHasFailure = true;
        mPythonInitialized = false;
        mTestResult = mPython->getError();
        return;
    }

    // Create our neural network model.  This is separately encapsulated, because we call it a couple times
    mPythonInitialized = ( executeStringAndSetFlags( mPythonCreateModel[mDNNSize] ) >= 0.0f );
}

void LiveTrainRenderer::doPythonTrain( Texture::SharedPtr fromTex )
{
    // Extract a our image data from the DX context.
    std::vector<uint8> textureData = gpDevice->getRenderContext()->readTextureSubresource(fromTex.get(), fromTex->getSubresourceIndex(0, 0));

    // Get our scene's light direction
    Light* pLight = mpScene->getScene()->getLight(0).get();
    float dirData[3] = { pLight->getData().worldDir.x, pLight->getData().worldDir.y, pLight->getData().worldDir.z };

    // Actually pass our light direction to Python (used as the input for this training run on the network)
    mGlobals["lightData"] = py::array_t<float>({ 3 }, { 4 }, dirData);  // data shape, data stride, raw data ptr (1D array of floats)

    // Pass Python information about our rendered image (used as the target/output for this training run on the network)
    mGlobals["imgW"] = 512;
    mGlobals["imgH"] = 512;
    mGlobals["imgData"] = py::array_t<unsigned char>( { 512, 512, 4 },                                                                       // Shape of the array
                                                      { 512 * 4 * sizeof(unsigned char), 4 * sizeof(unsigned char), sizeof(unsigned char) }, // data stride
                                                      textureData.data());  // raw buffer data (w*h*4 array of uchars)

    // If Python successfully transforms the input data into the NumPy arrays needed for the model training....
    mLastTrainTime = executeStringAndSetFlags( mPythonTrain );
    if (mLastTrainTime >= 0.0f)
    {
        mNumTrainingRuns++;
    }

    mDoTraining = false;
}

void LiveTrainRenderer::doPythonInference()
{
    // Send our light direction to Python
    DirectionalLight *dirLight = (DirectionalLight *)(mpScene->getScene()->getLight(0).get());
    float dirData[3] = { dirLight->getData().worldDir.x, dirLight->getData().worldDir.y, dirLight->getData().worldDir.z };
    mGlobals["inferLight"] = py::array_t<float>( { 3 }, { 4 }, dirData );
    
    // Predict the image given the above light direction
    mLastInferenceTime = executeStringAndSetFlags(mPythonInfer);
    if (mLastInferenceTime >= 0.0f)
    {
        // Readback the data.  Cast our Python output to an uchar array
        auto arr = mGlobals["infResult"].cast< py::array_t<unsigned char> >();
        py::buffer_info arr_info = arr.request();

        // Upload the Python uchar array into our texture (so we can render the result)
        gpDevice->getRenderContext()->updateTextureSubresource(mPythonReturnTexture.get(), mPythonReturnTexture->getSubresourceIndex(0, 0), arr_info.ptr);
    }

    mDoInference = false;
}

void LiveTrainRenderer::doRandomTrain()
{
    // Select a random light direction
    float phi = float(M_PI * randomFloat()); 
    float theta = float(2 * M_PI * randomFloat()); 
    vec3 rndDir = vec3(sinf(theta)*cosf(phi), sinf(theta)*sinf(phi), cosf(theta));

    // Pass it it, along with our arbitrarily brightened light.
    DirectionalLight *dirLight = (DirectionalLight *)(mpScene->getScene()->getLight(0).get());
    dirLight->setWorldDirection(rndDir);
    dirLight->setIntensity(vec3(LIGHT_INTENSITY, LIGHT_INTENSITY, LIGHT_INTENSITY));

    // Set flags to actually do training on next display refresh.
    mDoTraining = mDoInference = true;
    if (mTrainsLeft>0) mTrainsLeft--;
}

void LiveTrainRenderer::onDisplay(RenderContext::SharedPtr context, Fbo::SharedPtr targetFbo)
{
    context->clearFbo(targetFbo.get(), vec4(0.2f, 0.4f, 0.5f, 1), 1, 0);

    if (mpScene)
    {
        // Since, we're rendering into a differently sized window than our app window, reset aspect each frame,
        //     in case it get changed via a resize or other window message.
        mpScene->getScene()->getActiveCamera()->setAspectRatio(1.0f);

        // Continuously training?  Or set to train some batch of images?  Do it.
        if (mContinuousTrain || mTrainsLeft > 0)
        {
            doRandomTrain();
        }

        // Render our scene traditionally.
        mpState->setFbo(mpMainFbo);
        context->clearFbo(mpMainFbo.get(), glm::vec4(), 1, 0, FboAttachmentType::All);
        mpScene->update(mCurrentTime);
        lightingPass(context);
        msaaResolvePass(context);

        // If training, convert the data to a format Python can look at.  This is relatively slow & simplistic, 
        //     since we could probably use the original FBO (mpResolveFbo) directly, but in this simple example,  
        //     we just wanted to see if Python I/O transfers would work in simple cases.
        if (mDoTraining)
        {
            context->blit(mpResolveFbo->getColorTexture(0)->getSRV(), mpCaptureFbo->getRenderTargetView(0));
            doPythonTrain(mpCaptureFbo->getColorTexture(0));
        }

        // Use the current light direction to run inference from our current network.
        if (mDoInference)
        {
            doPythonInference();
        }

        // Blit our rendering and our Python return textures onto the screen
        context->blit(mpResolveFbo->getColorTexture(0)->getSRV(), targetFbo->getRenderTargetView(0), uvec4(-1), uvec4(504, 284, 1016, 796));
        context->blit(mPythonReturnTexture->getSRV(),             targetFbo->getRenderTargetView(0), uvec4(-1), uvec4(1204, 284, 1716, 796));
    }
}

void LiveTrainRenderer::onGuiRender()
{
    // If we haven't loaded a scene, there's not much we can do.
    if (!mpScene)
    {
        mpGui->addText(mPythonInitialized ? "Python Initialized: (Successfully)" : "Python Initialized: **Failure**");
        mpGui->addText("");
        mpGui->addText("Load a scene to enable training...");
    }

    // Now that we have a scene loaded, give more options.
    else
    {
        // Allow the user to update the current light position
        DirectionalLight *dirLight = (DirectionalLight *)(mpScene->getScene()->getLight(0).get());
        vec3 tmp = dirLight->getData().worldDir;
        if (mpGui->addDirectionWidget("Direction", tmp))
        {
            dirLight->setWorldDirection(tmp);
            mDoInference = true;
        }

        mpGui->addSeparator();

        if (mpGui->beginGroup("Training & Inference", true))
        {
            // Print some status messages
            mpGui->addText( mPythonInitialized ? "Python Initialized: (Successfully)" : "Python Initialized: **Failure**");  // Python available?
            addTextHelper("Trained example images: %d", int(mNumTrainingRuns));                                              // How many times trained?
            addTextHelper( mLastTrainTime > 0 ? "Last train cost: %.3f ms" : "", mLastTrainTime );                           // Last training cost (if available)?
            addTextHelper( mLastInferenceTime > 0 ? "Last inference cost: %.3f ms" : "", mLastInferenceTime );               // Last inference cost (if available)?
            mpGui->addText("");

            // Give some options for how to train...
            if (mpGui->addButton("(T) Train once"))
            {
                doRandomTrain();
            }
            if (mpGui->addButton("(B) Train batch of 128"))
            {
                mTrainsLeft = 128;
            }
            if (mpGui->addButton("(C) Continuous Train"))
            {
                mContinuousTrain = !mContinuousTrain;
            }

            mpGui->addText("");
            mpGui->addSeparator();

            // Show some other options.  These reset the training, so hide them in a closed group
            if (mpGui->beginGroup("Options (that reset training)"))
            {
                if (mpGui->addDropdown("Network Size", mDNNSizeList, mDNNSize) ||
                    mpGui->addButton("(R) Reset DNN Model"))
                {
                    mPythonInitialized = (executeStringAndSetFlags(mPythonCreateModel[mDNNSize]) >= 0.0f);
                    if (mPythonInitialized)
                    {
                        mDoInference = true;
                        mNumTrainingRuns = 0;
                    }
                }
                if (mpGui->addButton("Reload Python Scripts"))
                {
                    reloadPythonScripts();   // Reload Python
                    doPythonInit();          // Reinitialize all of Python
                    if (mPythonInitialized)  // If that worked, again reset training state.
                    {
                        mDoInference = true;
                        mNumTrainingRuns = 0;
                    }
                }
                mpGui->endGroup();
            }

            mpGui->addSeparator();
            mpGui->endGroup();
        }
    }

    // If we get a Python error, print out Python's error in a window rather than silently failing.
    if (mHasFailure)
    {
        mpGui->pushWindow("Python Error Message:" , 1000, 250, 500, 10 );
        mpGui->addText(mTestResult.c_str());
        mpGui->popWindow();
    }
}

bool LiveTrainRenderer::onKeyEvent(const KeyboardEvent& keyEvent)
{
    if (keyEvent.type != KeyboardEvent::Type::KeyPressed || !mpScene)
    {
        return false;
    }

    switch (keyEvent.key)
    {
    case KeyboardEvent::Key::T:
        doRandomTrain();
        break;
    case KeyboardEvent::Key::C:
        mContinuousTrain = !mContinuousTrain;
        break;
    case KeyboardEvent::Key::B:
        mTrainsLeft = 128;
        break;
    case KeyboardEvent::Key::R:
        // Recreate our model to reset it
        mPythonInitialized = (executeStringAndSetFlags(mPythonCreateModel[mDNNSize]) >= 0.0f);
        if (mPythonInitialized)
        {
            // Rerun inference, reset count of # of training runs
            mDoInference = true;
            mNumTrainingRuns = 0;
        }
        break;
    default:
        return false;
    }

    return true;
}

// Encapsulates calls to stringified Python code, plus timing the execution, setting any 
//     error flags, and grabbing an error message (if the Python code failed)
float LiveTrainRenderer::executeStringAndSetFlags(const std::string &pyCode)
{
    float curTime = -1.0f;
    mHasFailure = false;
    if (!mPython->executeString(pyCode))
    {
        mTestResult = mPython->getError();
        mHasFailure = true;
    }
    else
    {
        curTime = float(mPython->lastExecutionTime());
    }
    return curTime;
}

// Stringify the on-disk Python scripts to avoid reading from disk each frame.
void LiveTrainRenderer::reloadPythonScripts( void )
{
    // Load the initialization script
    std::ifstream initFile("Data/demo_init.py");
    initFile.seekg(0, std::ios::end);
    mPythonInit.reserve(initFile.tellg());
    initFile.seekg(0, std::ios::beg);
    mPythonInit.assign((std::istreambuf_iterator<char>(initFile)), std::istreambuf_iterator<char>());

    // Load the training script
    std::ifstream trainFile("Data/demo_train.py");
    trainFile.seekg(0, std::ios::end);
    mPythonTrain.reserve(trainFile.tellg());
    trainFile.seekg(0, std::ios::beg);
    mPythonTrain.assign((std::istreambuf_iterator<char>(trainFile)), std::istreambuf_iterator<char>());

    // Load the infer script
    std::ifstream inferFile("Data/demo_infer.py");
    inferFile.seekg(0, std::ios::end);
    mPythonInfer.reserve(inferFile.tellg());
    inferFile.seekg(0, std::ios::beg);
    mPythonInfer.assign((std::istreambuf_iterator<char>(inferFile)), std::istreambuf_iterator<char>());
}

bool LiveTrainRenderer::onMouseEvent(const MouseEvent& mouseEvent)
{
    // Prevents mouse motion from changing the camera.
    if (mouseEvent.type == MouseEvent::Type::LeftButtonDown ||
        mouseEvent.type == MouseEvent::Type::LeftButtonUp)
    {
        return true;
    }
    return false;
}

///////////////////////////////////////////////////////////////////////////////////
// Below here:  Mostly encapsulation of simple/basic Falcor rendering code
///////////////////////////////////////////////////////////////////////////////////

void LiveTrainRenderer::msaaResolvePass(RenderContext::SharedPtr context)
{
    context->blit(mpMainFbo->getColorTexture(0)->getSRV(), mpResolveFbo->getRenderTargetView(0));
    context->blit(mpMainFbo->getColorTexture(1)->getSRV(), mpResolveFbo->getRenderTargetView(1));
    context->blit(mpMainFbo->getDepthStencilTexture()->getSRV(), mpResolveFbo->getRenderTargetView(2));
}

void LiveTrainRenderer::lightingPass(RenderContext::SharedPtr context)
{
    mpState->setProgram(mLightingPass.pProgram);
    context->setGraphicsVars(mLightingPass.pVars);
    mpScene->renderScene(context.get());
}

void LiveTrainRenderer::initLightingPass()
{
    mLightingPass.pProgram = GraphicsProgram::createFromFile("RenderForLearning.vs.slang", "RenderForLearning.ps.slang");  
    mLightingPass.pProgram->addDefine("_LIGHT_COUNT", std::to_string(mpScene->getScene()->getLightCount()));
    initControls();
    mLightingPass.pVars = GraphicsVars::create(mLightingPass.pProgram->getActiveVersion()->getReflector());
}

void LiveTrainRenderer::applyLightingProgramControl(ControlID controlId)
{
    const ProgramControl control = mControls[controlId];
    if (control.define.size())
    {
        if (control.enabled)
        {
            mLightingPass.pProgram->addDefine(control.define, control.value);
        }
        else
        {
            mLightingPass.pProgram->removeDefine(control.define);
        }
    }
}

void LiveTrainRenderer::onResizeSwapChain(uint32_t newWidth, uint32_t newHeight)
{
    Fbo::Desc fboDesc;
    fboDesc.setColorTarget(0, ResourceFormat::RGBA8UnormSrgb);
    mpCaptureFbo = FboHelper::create2D(512, 512, fboDesc);

    fboDesc.setColorTarget(0, ResourceFormat::RGBA32Float).setColorTarget(1, ResourceFormat::RGBA8Unorm).setColorTarget(2, ResourceFormat::R32Float);
    mpResolveFbo = FboHelper::create2D(512, 512, fboDesc);

    fboDesc.setSampleCount(mSampleCount).setColorTarget(2, ResourceFormat::Unknown).setDepthStencilTarget(ResourceFormat::D32Float);
    mpMainFbo = FboHelper::create2D(512, 512, fboDesc);
}

void LiveTrainRenderer::onInitNewScene(SceneRenderer::SharedPtr pScene)
{
    mpScene = pScene;

    // Here's our arbitrary light intensity we're working with.
    DirectionalLight *dirLight = (DirectionalLight *)(mpScene->getScene()->getLight(0).get());
    dirLight->setIntensity(vec3(LIGHT_INTENSITY, LIGHT_INTENSITY, LIGHT_INTENSITY));
    
    initLightingPass();
}

void LiveTrainRenderer::initControls(void)
{
    mControls.resize(ControlID::Count);
    mControls[ControlID::SuperSampling] = { false, "INTERPOLATION_MODE", "sample" };
    mControls[ControlID::DisableSpecAA] = { false, "_MS_DISABLE_ROUGHNESS_FILTERING" };
    mControls[ControlID::EnableShadows] = { false, "_ENABLE_SHADOWS" };
    mControls[ControlID::EnableReflections] = { false, "_ENABLE_REFLECTIONS" };
    mControls[ControlID::EnableSSAO] = { false, "" };

    for (uint32_t i = 0; i < ControlID::Count; i++)
    {
        applyLightingProgramControl((ControlID)i);
    }
}


// Some dumb, printf-like helpers for adding text to the GUI.  These were hacked in quickly to hide some of my UI mess before releasing the code.

void LiveTrainRenderer::addTextHelper(const char *format, int intVal)
{
    char buf[512];
    if (format[0])
    {
        sprintf_s(buf, format, intVal);
        mpGui->addText(buf);
    }
    else mpGui->addText("");
}

void LiveTrainRenderer::addTextHelper(const char *format, float floatVal)
{
    char buf[512];
    if (format[0])
    {
        sprintf_s(buf, format, floatVal);
        mpGui->addText(buf);
    }
    else mpGui->addText("");
}

void LiveTrainRenderer::addTextHelper(const char *format, float floatVal1, float floatVal2)
{
    char buf[512];
    if (format[0])
    {
        sprintf_s(buf, format, floatVal1, floatVal2);
        mpGui->addText(buf);
    }
    else mpGui->addText("");
}

#endif
