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
#include "Framework.h"
#include "Sample.h"
#include <map>
#include <fstream>
#include "API/Window.h"
#include "Graphics/Program/Program.h"
#include "Utils/Platform/OS.h"
#include "API/FBO.h"
#include "VR/OpenVR/VRSystem.h"
#include "Utils/Platform/ProgressBar.h"
#include "Utils/StringUtils.h"
#include "Graphics/FboHelper.h"
#include <sstream>
#include <iomanip>

namespace Falcor
{
    void Sample::handleWindowSizeChange()
    {
        if (!gpDevice) return;
        // Tell the device to resize the swap chain
        mpBackBufferFBO = gpDevice->resizeSwapChain(mpWindow->getClientAreaWidth(), mpWindow->getClientAreaHeight());
        auto width = mpBackBufferFBO->getWidth();
        auto height = mpBackBufferFBO->getHeight();

        //Recopy back buffer to recreate target fbo 
        mpTargetFBO = FboHelper::create2D(width, height, mpBackBufferFBO->getDesc());
        mpDefaultPipelineState->setFbo(mpTargetFBO);

        // Tell the GUI the swap-chain size changed
        mpGui->onWindowResize(width, height);

        // Call the user callback
        mpRenderer->onResizeSwapChain(this, width, height);
    }

    void Sample::handleKeyboardEvent(const KeyboardEvent& keyEvent)
    {
        if (keyEvent.type == KeyboardEvent::Type::KeyPressed)
        {
            mPressedKeys.insert(keyEvent.key);
        }
        else if (keyEvent.type == KeyboardEvent::Type::KeyReleased)
        {
            mPressedKeys.erase(keyEvent.key);
        }

        if(gpDevice)
        {
            // Check if the GUI consumes it
            if (mpGui->onKeyboardEvent(keyEvent))
            {
                return;
            }

            // Checks if should toggle zoom
            mpPixelZoom->onKeyboardEvent(keyEvent);

            // Consume system messages first
            if (keyEvent.type == KeyboardEvent::Type::KeyPressed)
            {
                if (keyEvent.mods.isShiftDown && keyEvent.key == KeyboardEvent::Key::F12)
                {
                    initVideoCapture();
                }
                else if (!keyEvent.mods.isAltDown && !keyEvent.mods.isCtrlDown && !keyEvent.mods.isShiftDown)
                {
                    switch (keyEvent.key)
                    {
                    case KeyboardEvent::Key::F12:
                        mCaptureScreen = true;
                        break;
#if _PROFILING_ENABLED
                    case KeyboardEvent::Key::P:
                        gProfileEnabled = !gProfileEnabled;
                        break;
#endif
                    case KeyboardEvent::Key::V:
                        mVsyncOn = !mVsyncOn;
                        gpDevice->toggleVSync(mVsyncOn);
                        mFrameRate.resetClock();
                        break;
                    case KeyboardEvent::Key::F1:
                        toggleText(!mShowText);
                        break;
                    case KeyboardEvent::Key::F2:
                        toggleUI(!mShowUI);
                        break;
                    case KeyboardEvent::Key::F5:
                        Program::reloadAllPrograms();
                        mpRenderer->onDataReload(this);
                        break;
                    case KeyboardEvent::Key::Escape:
                        if (mVideoCapture.pVideoCapture)
                        {
                            endVideoCapture();
                        }
                        else
                        {
                            mpWindow->shutdown();
                        }
                        break;
                    case KeyboardEvent::Key::Equal:
                        mFreezeTime = !mFreezeTime;
                        break;
                    }
                }
            }
        }

        // If we got here, this is a user specific message
        mpRenderer->onKeyEvent(this, keyEvent);
    }

    void Sample::handleDroppedFile(const std::string& filename)
    {
        mpRenderer->onDroppedFile(this, filename);
    }

    void Sample::handleMouseEvent(const MouseEvent& mouseEvent)
    {
        if(gpDevice)
        {
            if (mpGui->onMouseEvent(mouseEvent)) return;
            if (mpPixelZoom->onMouseEvent(mouseEvent)) return;
        }
        mpRenderer->onMouseEvent(this, mouseEvent);
    }

    // Sample functions
    Sample::~Sample()
    {
        if (mVideoCapture.pVideoCapture)
        {
            endVideoCapture();
        }

        VRSystem::cleanup();

        mpGui.reset();
        mpDefaultPipelineState.reset();
        mpBackBufferFBO.reset();
        mpTargetFBO.reset();
        mpTextRenderer.reset();
        mpPixelZoom.reset();
        mGpuTimer.reset();
        mpRenderContext.reset();
        if(gpDevice) gpDevice->cleanup();
        gpDevice.reset();
    }

    void Sample::run(const SampleConfig& config, Renderer::UniquePtr& pRenderer)
    {
        Sample s(pRenderer);
        s.runInternal(config, config.argc, config.argv);
    }

    void Sample::runInternal(const SampleConfig& config, uint32_t argc, char** argv)
    {
        mTimeScale = config.timeScale;
        mFixedTimeDelta = config.fixedTimeDelta;
        mFreezeTime = config.freezeTimeOnStartup;
        mVsyncOn = config.deviceDesc.enableVsync;

        // Start the logger
        Logger::init();
        Logger::showBoxOnError(config.showMessageBoxOnError);

        // Create the window
        mpWindow = Window::create(config.windowDesc, this);
        if (mpWindow == nullptr)
        {
            logError("Failed to create device and window");
            return;
        }

        // Show the progress bar
        ProgressBar::MessageList msgList =
        {
            { "Initializing Falcor" },
            { "Takes a while, doesn't it?" },
            { "Don't get too bored now" },
            { "Getting there" },
            { "Loading. Seriously, loading" },
            { "Are we there yet?"},
            { "NI!"}
        };

        ProgressBar::SharedPtr pBar = ProgressBar::create(msgList);

        if(is_set(config.flags, SampleConfig::Flags::DoNotCreateDevice) == false)
        {
            Device::Desc d = config.deviceDesc;
            gpDevice = Device::create(mpWindow, config.deviceDesc);
            if (gpDevice == nullptr)
            {
                logError("Failed to create device");
                return;
            }

            if (config.deviceCreatedCallback != nullptr)
            {
                config.deviceCreatedCallback();
            }

            // Get the default objects before calling onLoad()
            mpBackBufferFBO = gpDevice->getSwapChainFbo();
            mpTargetFBO = FboHelper::create2D(mpBackBufferFBO->getWidth(), mpBackBufferFBO->getHeight(), mpBackBufferFBO->getDesc());
            mpDefaultPipelineState = GraphicsState::create();
            mpDefaultPipelineState->setFbo(mpTargetFBO);
            mpRenderContext = gpDevice->getRenderContext();
            mpRenderContext->setGraphicsState(mpDefaultPipelineState);

            // Init the UI
            initUI();
            mpPixelZoom = PixelZoom::create(mpTargetFBO.get());
        }
        else
        {
            mShowText = false;
            mShowUI = false;
        }

#ifdef _WIN32
        // Set the icon
        setWindowIcon("Framework\\Nvidia.ico", mpWindow->getApiHandle());

        if (argc == 0 || argv == nullptr)
        {
            mArgList.parseCommandLine(GetCommandLineA());
        }
        else
#endif
        {
            mArgList.parseCommandLine(concatCommandLine(argc, argv));
        }

        if (mEnableProfiling)
            mpRenderContext->enableStablePowerState();
        mGpuTimer = GpuTimer::create();
        
        // Load and run
        mpRenderer->onLoad(this, mpRenderContext);
        initializeTesting();
        pBar = nullptr;

        mFrameRate.resetClock();
        mpWindow->msgLoop();

        mpRenderer->onShutdown(this);
        mpRenderer.release();

        Logger::shutdown();
    }

    void Sample::calculateTime()
    {
        if (mFixedTimeDelta > 0.0f)
        {
            mCurrentTime += mFixedTimeDelta * mTimeScale;
        }
        else if (mFreezeTime == false)
        {
            float elapsedTime = mFrameRate.getLastFrameTime() * mTimeScale;
            mCurrentTime += elapsedTime;
        }
    }

    void Sample::setDefaultGuiSize(uint32_t width, uint32_t height)
    {
        mSampleGuiWidth = width;
        mSampleGuiHeight = height;
    }

    void Sample::renderGUI()
    {
        mpGui->beginFrame();

        constexpr char help[] =
            "  'F1'      - Show\\Hide text\n"
            "  'F2'      - Show\\Hide GUI\n"
            "  'F5'      - Reload shaders\n"
            "  'ESC'     - Quit\n"
            "  'V'       - Toggle VSync\n"
            "  'F12'     - Capture screenshot\n"
            "  'Shift+F12' - Video capture\n"
            "  '='       - Pause\\resume timer\n"
            "  'Z'       - Zoom in on a pixel\n"
            "  'MouseWheel' - Change level of zoom\n"
#if _PROFILING_ENABLED
            "  'P'       - Enable profiling\n";
#else
            ;
#endif

        mpGui->pushWindow("Falcor", mSampleGuiWidth, mSampleGuiHeight, 20, 40, false);
        mpGui->addText("Keyboard Shortcuts");
        mpGui->addTooltip(help, true);

        if (mpGui->beginGroup("Global Controls"))
        {
            mpGui->addFloatVar("Time", mCurrentTime, 0, FLT_MAX);
            mpGui->addFloatVar("Time Scale", mTimeScale, 0, FLT_MAX);

            if (mVideoCapture.pVideoCapture == nullptr)
            {
                mpGui->addFloatVar("Fixed Time Delta", mFixedTimeDelta, 0, FLT_MAX);
            }

            if (mpGui->addButton("Reset"))
            {
                mCurrentTime = 0.0f;
            }

            if (mpGui->addButton(mFreezeTime ? "Play" : "Pause", true))
            {
                mFreezeTime = !mFreezeTime;
            }

            if (mpGui->addButton("Stop", true))
            {
                mFreezeTime = true;
                mCurrentTime = 0.0f;
            }
            
            mCaptureScreen = mpGui->addButton("Screen Capture");
            if (mpGui->addButton("Video Capture", true))
            {
                initVideoCapture();
            }

            mpGui->endGroup();
        }

        mpRenderer->onGuiRender(this, mpGui.get());
        mpGui->popWindow();
        
        if (mVideoCapture.pUI)
        {
            mVideoCapture.pUI->render(mpGui.get());
        }

        mpGui->render(mpRenderContext.get(), mFrameRate.getLastFrameTime());
    }

    bool Sample::initializeTesting()
    {
        if (mArgList.argExists("test"))
        {
            mpSampleTest = SampleTest::create();
            mpSampleTest->initializeTests(this);
            mpRenderer->onInitializeTesting(this);
            return true;
        }
        else
        {
            return false;
        }
    }

    void Sample::beginTestFrame()
    { 
        if (mpSampleTest != nullptr) 
        { 
            mpSampleTest->beginTestFrame(this); 
            mpRenderer->onBeginTestFrame(mpSampleTest.get());
        } 
    }

    void Sample::endTestFrame()
    {
        if (mpSampleTest != nullptr)
        {
            mpSampleTest->endTestFrame(this);
            mpRenderer->onEndTestFrame(this, mpSampleTest.get());
        }
    }

    void Sample::renderFrame()
    {
        if (gpDevice && gpDevice->isWindowOccluded())
        {
            return;
        }

        mFrameRate.newFrame();
        beginTestFrame();
        {
            PROFILE(onFrameRender);
            // The swap-chain FBO might have changed between frames, so get it
            if(gpDevice)
            {
                mpBackBufferFBO = gpDevice->getSwapChainFbo();
                mpRenderContext = gpDevice->getRenderContext();
                // Bind the default state
                mpDefaultPipelineState->setFbo(mpTargetFBO);
                mpRenderContext->setGraphicsState(mpDefaultPipelineState);
            }
            calculateTime();
            mGpuTimer->begin();
            mpRenderer->onFrameRender(this, mpRenderContext, mpTargetFBO);
            mGpuTimer->end();
            mGpuElaspedTime = (float)mGpuTimer->getElapsedTime();
        }
        //blits the temp fbo given to user's renderer onto the backbuffer
        mpRenderContext->blit(mpTargetFBO->getColorTexture(0)->getSRV(), mpBackBufferFBO->getColorTexture(0)->getRTV());
        //Takes testing screenshots if desired (leaves out gui and fps text)
        endTestFrame();
        //Swaps back to backbuffer to render fps text and gui directly onto it
        mpDefaultPipelineState->setFbo(mpBackBufferFBO);
        mpRenderContext->setGraphicsState(mpDefaultPipelineState);
        {
            PROFILE(renderGUI);
            if (mShowUI)
            {
                renderGUI();
            }
        }

        renderText(getFpsMsg(), glm::vec2(10, 10));
        if(mpPixelZoom)
        {
            mpPixelZoom->render(mpRenderContext.get(), mpTargetFBO.get());
        }

        captureVideoFrame();
        printProfileData();
        if (mCaptureScreen)
        {
            captureScreen();
        }

        if(gpDevice)
        {
            PROFILE(present);
            gpDevice->present();
        }
    }

    std::string Sample::captureScreen(const std::string explicitFilename, const std::string explicitOutputDirectory)
    {
        mCaptureScreen = false;

        std::string filename = explicitFilename != "" ? explicitFilename : getExecutableName();
        std::string outputDirectory = explicitOutputDirectory != "" ? explicitOutputDirectory : getExecutableDirectory();

        std::string pngFile;
        if (findAvailableFilename(filename, outputDirectory, "png", pngFile))
        {
            Texture::SharedPtr pTexture;
            pTexture = gpDevice->getSwapChainFbo()->getColorTexture(0);
            pTexture->captureToFile(0, 0, pngFile);
        }
        else
        {
            logError("Could not find available filename when capturing screen");
            return "";
        }

         return pngFile;
    }

    void Sample::initUI()
    {
        mpGui = Gui::create(mpBackBufferFBO->getWidth(), mpBackBufferFBO->getHeight());
        mpTextRenderer = TextRenderer::create();
    }

    std::string Sample::getFpsMsg()
    {
        std::string s;
        if (mShowText)
        {
            std::stringstream strstr;
            float msPerFrame = mFrameRate.getAverageFrameTime();
            std::string msStr = std::to_string(msPerFrame);
            s = std::to_string(int(ceil(1000 / msPerFrame))) + " FPS (" + msStr.erase(msStr.size() - 4) + " ms/frame)";
            if (mVsyncOn) s += std::string(", VSync");
            std::string msGpuTimeStr = std::to_string(mGpuElaspedTime);
            s += std::string(" GPU time: ") + msGpuTimeStr.erase(msGpuTimeStr.size() - 4) + "ms";
        }
        return s;
    }

    void Sample::resizeSwapChain(uint32_t width, uint32_t height)
    {
        mpWindow->resize(width, height);
        mpPixelZoom->onResizeSwapChain(gpDevice->getSwapChainFbo().get());
    }

    bool Sample::isKeyPressed(const KeyboardEvent::Key& key)
    {
        return mPressedKeys.find(key) != mPressedKeys.cend();
    }

    void Sample::renderText(const std::string& msg, const glm::vec2& position, const glm::vec2 shadowOffset)
    {
        if (mShowText)
        {
            // Render outline first
            if (shadowOffset.x != 0.f || shadowOffset.y != 0)
            {
                const glm::vec3 oldColor = mpTextRenderer->getTextColor();
                mpTextRenderer->setTextColor(glm::vec3(0.f));   // Black outline 
                mpTextRenderer->begin(mpRenderContext, position + shadowOffset);
                mpTextRenderer->renderLine(msg);
                mpTextRenderer->end();
                mpTextRenderer->setTextColor(oldColor);
            }
            mpTextRenderer->begin(mpRenderContext, position);
            mpTextRenderer->renderLine(msg);
            mpTextRenderer->end();
        }
    }

    void Sample::printProfileData()
    {
#if _PROFILING_ENABLED
        if (gProfileEnabled)
        {
            std::string profileMsg;
            Profiler::endFrame(profileMsg);
            renderText(profileMsg, glm::vec2(10, 300));
        }
#endif
    }

    void Sample::initVideoCapture()
    {
        if (mVideoCapture.pUI == nullptr)
        {
            mVideoCapture.pUI = VideoEncoderUI::create(20, 300, 240, 220, [this]() {startVideoCapture(); }, [this]() {endVideoCapture(); });
        }
    }

    void Sample::startVideoCapture()
    {
        // Create the Capture Object and Framebuffer.
        VideoEncoder::Desc desc;
        desc.flipY = false;
        desc.codec = mVideoCapture.pUI->getCodec();
        desc.filename = mVideoCapture.pUI->getFilename();
        desc.format = mpBackBufferFBO->getColorTexture(0)->getFormat();
        desc.fps = mVideoCapture.pUI->getFPS();
        desc.height = mpBackBufferFBO->getHeight();
        desc.width = mpBackBufferFBO->getWidth();
        desc.bitrateMbps = mVideoCapture.pUI->getBitrate();
        desc.gopSize = mVideoCapture.pUI->getGopSize();

        mVideoCapture.pVideoCapture = VideoEncoder::create(desc);

        assert(mVideoCapture.pVideoCapture);
        mVideoCapture.pFrame = new uint8_t[desc.width*desc.height * 4];

        mVideoCapture.sampleTimeDelta = mFixedTimeDelta;
        mFixedTimeDelta = 1.0f / (float)desc.fps;

        if (mVideoCapture.pUI->useTimeRange())
        {
            if (mVideoCapture.pUI->getStartTime() > mVideoCapture.pUI->getEndTime())
            {
                mFixedTimeDelta = -mFixedTimeDelta;
            }
            mCurrentTime = mVideoCapture.pUI->getStartTime();
            if (!mVideoCapture.pUI->captureUI())
            {
                mShowUI = false;
            }
        }
    }
 
    void Sample::endVideoCapture()
    {
        if (mVideoCapture.pVideoCapture)
        {
            mVideoCapture.pVideoCapture->endCapture();
            mShowUI = true;
        }
        mVideoCapture.pUI = nullptr;
        mVideoCapture.pVideoCapture = nullptr;
        safe_delete_array(mVideoCapture.pFrame);
        mFixedTimeDelta = mVideoCapture.sampleTimeDelta;
    }

    void Sample::captureVideoFrame()
    {
        if (mVideoCapture.pVideoCapture)
        {
            mVideoCapture.pVideoCapture->appendFrame(mpRenderContext->readTextureSubresource(mpBackBufferFBO->getColorTexture(0).get(), 0).data());

            if (mVideoCapture.pUI->useTimeRange())
            {
                if (mFixedTimeDelta >= 0)
                {
                    if (mCurrentTime >= mVideoCapture.pUI->getEndTime())
                    {
                        endVideoCapture();
                    }
                }
                else if (mCurrentTime < mVideoCapture.pUI->getEndTime())
                {
                    endVideoCapture();
                }
            }
        }
    }
}