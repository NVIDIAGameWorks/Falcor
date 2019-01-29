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
#include "Experimental/RenderGraph/RenderPassLibrary.h"

namespace Falcor
{
    static std::string kMonospaceFont = "monospace";

    void Sample::handleWindowSizeChange()
    {
        if (!gpDevice) return;
        // Tell the device to resize the swap chain
        auto pBackBufferFBO = gpDevice->resizeSwapChain(mpWindow->getClientAreaWidth(), mpWindow->getClientAreaHeight());
        auto width = pBackBufferFBO->getWidth();
        auto height = pBackBufferFBO->getHeight();

        // Recreate target fbo
        auto pCurrentFbo = mpTargetFBO;
        mpTargetFBO = FboHelper::create2D(width, height, pBackBufferFBO->getDesc());
        if(mpDefaultPipelineState) mpDefaultPipelineState->setFbo(mpTargetFBO);
        gpDevice->getRenderContext()->blit(pCurrentFbo->getColorTexture(0)->getSRV(), mpTargetFBO->getRenderTargetView(0));
        
        // Tell the GUI the swap-chain size changed
        if(mpGui) mpGui->onWindowResize(width, height);

        // Resize the pixel zoom
        if(mpPixelZoom) mpPixelZoom->onResizeSwapChain(gpDevice->getSwapChainFbo().get());

        // Call the user callback
        if(mpRenderer) mpRenderer->onResizeSwapChain(this, width, height);
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
                else if (keyEvent.mods.isCtrlDown)
                {
                    switch (keyEvent.key)
                    {
                    case KeyboardEvent::Key::Pause:
                        mFreezeRendering = !mFreezeRendering;
                        break;
                    }
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
                        toggleUI((mShowUI == UIStatus::ShowAll));
                        break;
                    case KeyboardEvent::Key::F5:
                        Program::reloadAllPrograms();
                        if(mpRenderer) mpRenderer->onDataReload(this);
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
                    case KeyboardEvent::Key::Pause:
                        mFreezeTime = !mFreezeTime;
                        break;
                    }
                }
            }
        }

        // If we got here, this is a user specific message
        if(mpRenderer) mpRenderer->onKeyEvent(this, keyEvent);
    }

    void Sample::handleDroppedFile(const std::string& filename)
    {
        if(mpRenderer)
        {
            mpRenderer->onDroppedFile(this, filename);
        }
    }

    void Sample::handleMouseEvent(const MouseEvent& mouseEvent)
    {
        if(gpDevice)
        {
            if (mpGui->onMouseEvent(mouseEvent)) return;
            if (mpPixelZoom->onMouseEvent(mouseEvent)) return;
        }
        if(mpRenderer)
        {
            mpRenderer->onMouseEvent(this, mouseEvent);
        }
    }

    dlldecl void releaseSharedObjects();

    // Sample functions
    Sample::~Sample()
    {
        if (mVideoCapture.pVideoCapture)
        {
            endVideoCapture();
        }

        VRSystem::cleanup();

        RenderPassLibrary::instance().shutdown();
        Scripting::shutdown();
        mpGui.reset();
        mpDefaultPipelineState.reset();
        mpTargetFBO.reset();
        mpTextRenderer.reset();
        mpPixelZoom.reset();
        releaseSharedObjects();
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

        Scripting::start();

#if _LOG_ENABLED
        Logger::initialize();
        Logger::showBoxOnError(config.showMessageBoxOnError);
#endif

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
            
            // Get the default objects before calling onLoad()
            auto pBackBufferFBO = gpDevice->getSwapChainFbo();
            mpTargetFBO = FboHelper::create2D(pBackBufferFBO->getWidth(), pBackBufferFBO->getHeight(), pBackBufferFBO->getDesc());
            mpDefaultPipelineState = GraphicsState::create();
            mpDefaultPipelineState->setFbo(mpTargetFBO);
            auto pRenderContext = gpDevice->getRenderContext();
            pRenderContext->setGraphicsState(mpDefaultPipelineState);

            // Init the UI
            initUI();
            mpPixelZoom = PixelZoom::create(mpTargetFBO.get());
        }
        else
        {
            mShowText = false;
            mShowUI = UIStatus::HideAll;
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

        // set fixed time delta if it is provided in the command line
        if (mArgList.argExists("fixedtimedelta"))  mFixedTimeDelta = mArgList["fixedtimedelta"].asFloat();

        // Load and run
        mpRenderer->onLoad(this, getRenderContext());
        initializeTesting();
        pBar = nullptr;

        mFrameRate.resetClock();
        mpWindow->msgLoop();

        if (mTestingFrames.size()) { onTestShutdown(); }
        mpRenderer->onShutdown(this);
        if (gpDevice) gpDevice->flushAndSync();
        mpRenderer = nullptr;
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

    void Sample::setDefaultGuiPosition(uint32_t x, uint32_t y)
    {
        mSampleGuiPositionX = x;
        mSampleGuiPositionY = y;
    }

    void Sample::renderGUI()
    {
        if((mShowUI != UIStatus::HideAll) || gProfileEnabled)
        {
            mpGui->beginFrame();

            constexpr char help[] =
                "  'F1'      - Show\\Hide text\n"
                "  'F2'      - Show\\Hide GUI\n"
                "  'F5'      - Reload shaders\n"
                "  'ESC'     - Quit\n"
                "  'V'       - Toggle VSync\n"
                "  'F12'     - Capture screenshot\n"
                "  'Shift+F12'  - Video capture\n"
                "  'Pause'      - Pause\\resume timer\n"
                "  'Ctrl+Pause' - Pause\\resume the renderer\n"
                "  'Z'       - Zoom in on a pixel\n"
                "  'MouseWheel' - Change level of zoom\n"
#if _PROFILING_ENABLED
                "  'P'       - Enable profiling\n";
#else
                ;
#endif

            if(mShowUI == UIStatus::ShowAll)
            {
                mpGui->pushWindow("Falcor", mSampleGuiWidth, mSampleGuiHeight, mSampleGuiPositionX, mSampleGuiPositionY, false, true, true, false);
                mpGui->addText("Keyboard Shortcuts");
                mpGui->addTooltip(help, true);

                if (mpGui->beginGroup("Global Controls"))
                {
                    mpGui->addFloatVar("Time", mCurrentTime, 0, FLT_MAX);

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
                    if(mpGui->beginGroup("Time Controls"))
                    {
                        mpGui->addFloatVar("Scale", mTimeScale, 0, FLT_MAX);

                        if (mVideoCapture.pVideoCapture == nullptr)
                        {
                            mpGui->addFloatVar("Fixed Delta", mFixedTimeDelta, 0, FLT_MAX);
                        }
                        mpGui->endGroup();
                    }
                    mpGui->addSeparator();

                    if (mpGui->addButton(mFreezeRendering ? "Resume Rendering" : "Pause Rendering")) mFreezeRendering = !mFreezeRendering;
                    mpGui->addTooltip("Freeze the renderer and keep displaying The last rendered frame. The renderer will keep accepting mouse/keyboard/GUI messages. Changes in the UI will not be reflected in the displayed image until the renderer is unfrozen");

                    mpGui->addSeparator();

                    mCaptureScreen = mpGui->addButton("Screen Capture");
                    if (mpGui->addButton("Video Capture", true))
                    {
                        initVideoCapture();
                    }

                    mpGui->endGroup();
                }

                mpRenderer->onGuiRender(this, mpGui.get());
                mpGui->popWindow();

                if (mVideoCapture.displayUI && mVideoCapture.pUI)
                {
                    mVideoCapture.pUI->render(mpGui.get());
                }
            }

            if (mShowUI == UIStatus::HideGlobal)
            {
                mpRenderer->onGuiRender(this, mpGui.get());
            }

            if (gProfileEnabled)
            {
                uint32_t y = gpDevice->getSwapChainFbo()->getHeight() - 360;

                mpGui->setActiveFont(kMonospaceFont);
                mpGui->pushWindow("Profiler", 650, 350, 10, y);
                // Stop the timer
                Profiler::endEvent("renderGUI");
                mpGui->addText(Profiler::getEventsString().c_str());
                Profiler::startEvent("renderGUI");
                mpGui->popWindow();
                mpGui->setActiveFont("");
            }

            mpGui->render(getRenderContext(), mFrameRate.getLastFrameTime());
        }
    }

    bool Sample::initializeTesting()
    {
        bool testFrames = mArgList.argExists("testFrames");
        if (testFrames)
        {
            auto testFrames = mArgList.getValues("testFrames");
            for (auto frame : testFrames) { mTestingFrames.push_back(frame.asUint64()); }

            mpRenderer->onInitializeTesting(this);
        }
        if (mArgList.argExists("shutdownframe"))   mShutdownFrame = mArgList["shutdownframe"].asUint64();
        else if(!testFrames)  return false;

        return true;
    }

    void Sample::onTestFrame()
    {
        uint64_t currentFrameID = getFrameID();
        if (mCurrentTestingIndex < mTestingFrames.size() && currentFrameID == mTestingFrames[mCurrentTestingIndex])
        {
            mpRenderer->onTestFrame(this);
            mCurrentTestingIndex++;
        }
        else if (currentFrameID == mShutdownFrame) { shutdown(); }
    }

    void Sample::renderFrame()
    {
        if (gpDevice && gpDevice->isWindowOccluded())
        {
            return;
        }

        mFrameRate.newFrame();
        {
            PROFILE("onFrameRender");
            calculateTime();
            // The swap-chain FBO might have changed between frames, so get it
            if (!mFreezeRendering)
            {
                RenderContext* pRenderContext = nullptr;
                if (gpDevice)
                {
                    // Bind the default state
                    pRenderContext = gpDevice->getRenderContext();
                    mpDefaultPipelineState->setFbo(mpTargetFBO);
                    pRenderContext->setGraphicsState(mpDefaultPipelineState);
                }
                mpRenderer->onFrameRender(this, pRenderContext, mpTargetFBO);
            }
        }
        
        if (gpDevice)
        {
            // Copy the render-target
            const auto& pSwapChainFbo = gpDevice->getSwapChainFbo();
            RenderContext* pCtx = getRenderContext();
            getRenderContext()->copyResource(pSwapChainFbo->getColorTexture(0).get(), mpTargetFBO->getColorTexture(0).get());

            if (mTestingFrames.size())  onTestFrame();

            // Capture video frame before UI is rendered
            bool captureVideoUI = mVideoCapture.pUI && mVideoCapture.pUI->captureUI();  // Check capture mode here once only, as its value may change after renderGUI()
            if (!captureVideoUI)
            {
                captureVideoFrame();
            }

            //Swaps back to backbuffer to render fps text and gui directly onto it
            mpDefaultPipelineState->setFbo(pSwapChainFbo);
            pCtx->setGraphicsState(mpDefaultPipelineState);
            {
                PROFILE("renderGUI");
                renderGUI();
            }

            renderText(getFpsMsg(), glm::vec2(10, 10));
            if (mpPixelZoom)
            {
                mpPixelZoom->render(pCtx, pSwapChainFbo.get());
            }

#if _PROFILING_ENABLED
            Profiler::endFrame();
#endif
            // Capture video frame after UI is rendered
            if (captureVideoUI)
            {
                captureVideoFrame();
            }

            if (mCaptureScreen)
            {
                captureScreen();
            }

            {
                PROFILE("present");
                gpDevice->present();
            }
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
        float scaling = getDisplayScaleFactor();
        const auto& pSwapChainFbo = gpDevice->getSwapChainFbo();
        mpGui = Gui::create(uint32_t(pSwapChainFbo->getWidth()), uint32_t(pSwapChainFbo->getHeight()), scaling);
        mpGui->addFont(kMonospaceFont, "Framework/Fonts/consolab.ttf");
        mSampleGuiHeight = (uint32_t)(mSampleGuiHeight * scaling);
        mSampleGuiWidth = (uint32_t)(mSampleGuiWidth * scaling);
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
        }
        return s;
    }

    void Sample::resizeSwapChain(uint32_t width, uint32_t height)
    {
        mpWindow->resize(width, height);
    }

    bool Sample::isKeyPressed(const KeyboardEvent::Key& key)
    {
        return mPressedKeys.find(key) != mPressedKeys.cend();
    }

    void Sample::renderText(const std::string& msg, const glm::vec2& position, const glm::vec2 shadowOffset)
    {
        if (mShowText)
        {
            RenderContext* pCtx = getRenderContext();
            // Render outline first
            if (shadowOffset.x != 0.f || shadowOffset.y != 0)
            {
                const glm::vec3 oldColor = mpTextRenderer->getTextColor();
                mpTextRenderer->setTextColor(glm::vec3(0.f));   // Black outline 
                mpTextRenderer->begin(pCtx, position + shadowOffset);
                mpTextRenderer->renderLine(pCtx, msg);
                mpTextRenderer->end(pCtx);
                mpTextRenderer->setTextColor(oldColor);
            }
            mpTextRenderer->begin(pCtx, position);
            mpTextRenderer->renderLine(pCtx, msg);
            mpTextRenderer->end(pCtx);
        }
    }

    void Sample::initVideoCapture()
    {
        if (mVideoCapture.pUI == nullptr)
        {
            mVideoCapture.pUI = VideoEncoderUI::create(20, 300, 300, 280, [this]() {startVideoCapture(); }, [this]() {endVideoCapture(); });
        }
        mVideoCapture.displayUI = true;
    }

    void Sample::startVideoCapture()
    {
        // Create the Capture Object and Framebuffer.
        VideoEncoder::Desc desc;
        desc.flipY = false;
        desc.codec = mVideoCapture.pUI->getCodec();
        desc.filename = mVideoCapture.pUI->getFilename();
        const auto& pSwapChainFbo = gpDevice->getSwapChainFbo();
        desc.format = pSwapChainFbo->getColorTexture(0)->getFormat();
        desc.fps = mVideoCapture.pUI->getFPS();
        desc.height = pSwapChainFbo->getHeight();
        desc.width = pSwapChainFbo->getWidth();
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
        }

        mShouldResetRendering = mVideoCapture.pUI->resetOnFirstFrame();
    }
 
    void Sample::endVideoCapture()
    {
        if (mVideoCapture.pVideoCapture)
        {
            mVideoCapture.pVideoCapture->endCapture();
            mShowUI = UIStatus::ShowAll;
        }
        mVideoCapture.pUI->setCaptureState(false);
        mVideoCapture.displayUI = false;
        mVideoCapture.pVideoCapture = nullptr;
        safe_delete_array(mVideoCapture.pFrame);
        mFixedTimeDelta = mVideoCapture.sampleTimeDelta;
    }

    void Sample::captureVideoFrame()
    {
        if (mVideoCapture.pVideoCapture)
        {
            mVideoCapture.pVideoCapture->appendFrame(getRenderContext()->readTextureSubresource(gpDevice->getSwapChainFbo()->getColorTexture(0).get(), 0).data());

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

            mShouldResetRendering = false;
        }
    }
}
