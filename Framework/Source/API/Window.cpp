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
#include "API/Window.h"
#include "Utils/UserInput.h"
#include "Utils/Platform/OS.h"
#include <algorithm>
#include "API/Texture.h"
#include "API/FBO.h"
#include "Utils/StringUtils.h"

// Don't include GL/GLES headers
#define GLFW_INCLUDE_NONE

#ifdef _WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#include "glfw3.h"
#include "glfw3native.h"
#else // LINUX

// Replace the defines we undef'd in FalcorVK.h, because glfw will need them when it includes Xlib
#define None 0L
#define Bool int
#define Status int
#define Always 2

#define GLFW_EXPOSE_NATIVE_X11
#include "GLFW/glfw3.h"
#include "GLFW/glfw3native.h"
#endif

namespace Falcor
{
    class ApiCallbacks
    {
    public:
        static void windowSizeCallback(GLFWwindow* pGlfwWindow, int width, int height)
        {
            // We also get here in case the window was minimized, so we need to ignore it
            if (width == 0 || height == 0)
            {
                return;
            }

            Window* pWindow = (Window*)glfwGetWindowUserPointer(pGlfwWindow);
            if (pWindow != nullptr)
            {
                pWindow->resize(width, height); // Window callback is handled in here
            }
        }

        static void keyboardCallback(GLFWwindow* pGlfwWindow, int key, int scanCode, int action, int modifiers)
        {
            KeyboardEvent event;
            if (prepareKeyboardEvent(key, action, modifiers, event))
            {
                Window* pWindow = (Window*)glfwGetWindowUserPointer(pGlfwWindow);
                if (pWindow != nullptr)
                {
                    pWindow->mpCallbacks->handleKeyboardEvent(event);
                }
            }
        }

        static void charInputCallback(GLFWwindow* pGlfwWindow, uint32_t input)
        {
            KeyboardEvent event;
            event.type = KeyboardEvent::Type::Input;
            event.codepoint = input;

            Window* pWindow = (Window*)glfwGetWindowUserPointer(pGlfwWindow);
            if (pWindow != nullptr)
            {
                pWindow->mpCallbacks->handleKeyboardEvent(event);
            }
        }

        static void mouseMoveCallback(GLFWwindow* pGlfwWindow, double mouseX, double mouseY)
        {
            Window* pWindow = (Window*)glfwGetWindowUserPointer(pGlfwWindow);
            if (pWindow != nullptr)
            {
                // Prepare the mouse data
                MouseEvent event;
                event.type = MouseEvent::Type::Move;
                event.pos = calcMousePos(mouseX, mouseY, pWindow->getMouseScale());
                event.wheelDelta = glm::vec2(0, 0);

                pWindow->mpCallbacks->handleMouseEvent(event);
            }
        }

        static void mouseButtonCallback(GLFWwindow* pGlfwWindow, int button, int action, int modifiers)
        {
            MouseEvent event;
            // Prepare the mouse data
            switch (button)
            {
            case GLFW_MOUSE_BUTTON_LEFT:
                event.type = (action == GLFW_PRESS) ? MouseEvent::Type::LeftButtonDown : MouseEvent::Type::LeftButtonUp;
                break;
            case GLFW_MOUSE_BUTTON_MIDDLE:
                event.type = (action == GLFW_PRESS) ? MouseEvent::Type::MiddleButtonDown : MouseEvent::Type::MiddleButtonUp;
                break;
            case GLFW_MOUSE_BUTTON_RIGHT:
                event.type = (action == GLFW_PRESS) ? MouseEvent::Type::RightButtonDown : MouseEvent::Type::RightButtonUp;
                break;
            default:
                // Other keys are not supported
                break;
            }

            Window* pWindow = (Window*)glfwGetWindowUserPointer(pGlfwWindow);
            if (pWindow != nullptr)
            {
                // Modifiers
                event.mods = getInputModifiers(modifiers);
                double x, y;
                glfwGetCursorPos(pGlfwWindow, &x, &y);
                event.pos = calcMousePos(x, y, pWindow->getMouseScale());

                pWindow->mpCallbacks->handleMouseEvent(event);
            }
        }

        static void mouseWheelCallback(GLFWwindow* pGlfwWindow, double scrollX, double scrollY)
        {
            Window* pWindow = (Window*)glfwGetWindowUserPointer(pGlfwWindow);
            if (pWindow != nullptr)
            {
                MouseEvent event;
                event.type = MouseEvent::Type::Wheel;
                double x, y;
                glfwGetCursorPos(pGlfwWindow, &x, &y);
                event.pos = calcMousePos(x, y, pWindow->getMouseScale());
                event.wheelDelta = (glm::vec2(float(scrollX), float(scrollY)));

                pWindow->mpCallbacks->handleMouseEvent(event);
            }
        }

        static void errorCallback(int errorCode, const char* pDescription)
        {
            std::string errorMsg = std::to_string(errorCode) + " - " + std::string(pDescription) + "\n";
            logError(errorMsg.c_str());
        }

    private:

        static inline KeyboardEvent::Key glfwToFalcorKey(int glfwKey)
        {
            static_assert(GLFW_KEY_ESCAPE == 256, "GLFW_KEY_ESCAPE is expected to be 256");
            if (glfwKey < GLFW_KEY_ESCAPE)
            {
                // Printable keys are expected to have the same value
                return (KeyboardEvent::Key)glfwKey;
            }

            switch (glfwKey)
            {
            case GLFW_KEY_ESCAPE:
                return KeyboardEvent::Key::Escape;
            case GLFW_KEY_ENTER:
                return KeyboardEvent::Key::Enter;
            case GLFW_KEY_TAB:
                return KeyboardEvent::Key::Tab;
            case GLFW_KEY_BACKSPACE:
                return KeyboardEvent::Key::Backspace;
            case GLFW_KEY_INSERT:
                return KeyboardEvent::Key::Insert;
            case GLFW_KEY_DELETE:
                return KeyboardEvent::Key::Del;
            case GLFW_KEY_RIGHT:
                return KeyboardEvent::Key::Right;
            case GLFW_KEY_LEFT:
                return KeyboardEvent::Key::Left;
            case GLFW_KEY_DOWN:
                return KeyboardEvent::Key::Down;
            case GLFW_KEY_UP:
                return KeyboardEvent::Key::Up;
            case GLFW_KEY_PAGE_UP:
                return KeyboardEvent::Key::PageUp;
            case GLFW_KEY_PAGE_DOWN:
                return KeyboardEvent::Key::PageDown;
            case GLFW_KEY_HOME:
                return KeyboardEvent::Key::Home;
            case GLFW_KEY_END:
                return KeyboardEvent::Key::End;
            case GLFW_KEY_CAPS_LOCK:
                return KeyboardEvent::Key::CapsLock;
            case GLFW_KEY_SCROLL_LOCK:
                return KeyboardEvent::Key::ScrollLock;
            case GLFW_KEY_NUM_LOCK:
                return KeyboardEvent::Key::NumLock;
            case GLFW_KEY_PRINT_SCREEN:
                return KeyboardEvent::Key::PrintScreen;
            case GLFW_KEY_PAUSE:
                return KeyboardEvent::Key::Pause;
            case GLFW_KEY_F1:
                return KeyboardEvent::Key::F1;
            case GLFW_KEY_F2:
                return KeyboardEvent::Key::F2;
            case GLFW_KEY_F3:
                return KeyboardEvent::Key::F3;
            case GLFW_KEY_F4:
                return KeyboardEvent::Key::F4;
            case GLFW_KEY_F5:
                return KeyboardEvent::Key::F5;
            case GLFW_KEY_F6:
                return KeyboardEvent::Key::F6;
            case GLFW_KEY_F7:
                return KeyboardEvent::Key::F7;
            case GLFW_KEY_F8:
                return KeyboardEvent::Key::F8;
            case GLFW_KEY_F9:
                return KeyboardEvent::Key::F9;
            case GLFW_KEY_F10:
                return KeyboardEvent::Key::F10;
            case GLFW_KEY_F11:
                return KeyboardEvent::Key::F11;
            case GLFW_KEY_F12:
                return KeyboardEvent::Key::F12;
            case GLFW_KEY_KP_0:
                return KeyboardEvent::Key::Keypad0;
            case GLFW_KEY_KP_1:
                return KeyboardEvent::Key::Keypad1;
            case GLFW_KEY_KP_2:
                return KeyboardEvent::Key::Keypad2;
            case GLFW_KEY_KP_3:
                return KeyboardEvent::Key::Keypad3;
            case GLFW_KEY_KP_4:
                return KeyboardEvent::Key::Keypad4;
            case GLFW_KEY_KP_5:
                return KeyboardEvent::Key::Keypad5;
            case GLFW_KEY_KP_6:
                return KeyboardEvent::Key::Keypad6;
            case GLFW_KEY_KP_7:
                return KeyboardEvent::Key::Keypad7;
            case GLFW_KEY_KP_8:
                return KeyboardEvent::Key::Keypad8;
            case GLFW_KEY_KP_9:
                return KeyboardEvent::Key::Keypad9;
            case GLFW_KEY_KP_DECIMAL:
                return KeyboardEvent::Key::KeypadDel;
            case GLFW_KEY_KP_DIVIDE:
                return KeyboardEvent::Key::KeypadDivide;
            case GLFW_KEY_KP_MULTIPLY:
                return KeyboardEvent::Key::KeypadMultiply;
            case GLFW_KEY_KP_SUBTRACT:
                return KeyboardEvent::Key::KeypadSubtract;
            case GLFW_KEY_KP_ADD:
                return KeyboardEvent::Key::KeypadAdd;
            case GLFW_KEY_KP_ENTER:
                return KeyboardEvent::Key::KeypadEnter;
            case GLFW_KEY_KP_EQUAL:
                return KeyboardEvent::Key::KeypadEqual;
            case GLFW_KEY_LEFT_SHIFT:
                return KeyboardEvent::Key::LeftShift;
            case GLFW_KEY_LEFT_CONTROL:
                return KeyboardEvent::Key::LeftControl;
            case GLFW_KEY_LEFT_ALT:
                return KeyboardEvent::Key::LeftAlt;
            case GLFW_KEY_LEFT_SUPER:
                return KeyboardEvent::Key::LeftSuper;
            case GLFW_KEY_RIGHT_SHIFT:
                return KeyboardEvent::Key::RightShift;
            case GLFW_KEY_RIGHT_CONTROL:
                return KeyboardEvent::Key::RightControl;
            case GLFW_KEY_RIGHT_ALT:
                return KeyboardEvent::Key::RightAlt;
            case GLFW_KEY_RIGHT_SUPER:
                return KeyboardEvent::Key::RightSuper;
            case GLFW_KEY_MENU:
                return KeyboardEvent::Key::Menu;
            default:
                should_not_get_here();
                return (KeyboardEvent::Key)0;
            }
        }

        static inline InputModifiers getInputModifiers(int mask)
        {
            InputModifiers mods;
            mods.isAltDown = (mask & GLFW_MOD_ALT) != 0;
            mods.isCtrlDown = (mask & GLFW_MOD_CONTROL) != 0;
            mods.isShiftDown = (mask & GLFW_MOD_SHIFT) != 0;
            return mods;
        }

        static inline glm::vec2 calcMousePos(double xPos, double yPos, const glm::vec2& mouseScale)
        {
            glm::vec2 pos = glm::vec2(float(xPos), float(yPos));
            pos *= mouseScale;
            return pos;
        }

        static inline bool prepareKeyboardEvent(int key, int action, int modifiers, KeyboardEvent& event)
        {
            if (action == GLFW_REPEAT || key == GLFW_KEY_UNKNOWN)
            {
                return false;
            }

            event.type = (action == GLFW_RELEASE ? KeyboardEvent::Type::KeyReleased : KeyboardEvent::Type::KeyPressed);
            event.key = glfwToFalcorKey(key);
            event.mods = getInputModifiers(modifiers);
            return true;
        }
    };

    Window::Window(ICallbacks* pCallbacks, uint32_t width, uint32_t height)
        : mpCallbacks(pCallbacks)
        , mWidth(width)
        , mHeight(height)
        , mMouseScale(1.0f / (float)width, 1.0f / (float)height)
    {
    }

    Window::~Window()
    {
        glfwDestroyWindow(mpGLFWWindow);
        glfwTerminate();
    }

    void Window::shutdown()
    {
        glfwSetWindowShouldClose(mpGLFWWindow, 1);
    }

    Window::SharedPtr Window::create(const Desc& desc, ICallbacks* pCallbacks)
    {
        // Set error callback
        glfwSetErrorCallback(ApiCallbacks::errorCallback);

        // Init GLFW
        if (glfwInit() == GLFW_FALSE)
        {
            logError("GLFW initialization failed");
            return nullptr;
        }

        SharedPtr pWindow = SharedPtr(new Window(pCallbacks, desc.width, desc.height));

        // Create the window
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        GLFWmonitor* mon = desc.fullScreen ? glfwGetPrimaryMonitor() :nullptr;
        GLFWwindow* pGLFWWindow = glfwCreateWindow(desc.width, desc.height, desc.title.c_str(), mon, nullptr);

        if (pGLFWWindow == nullptr)
        {
            logError("Window creation failed!");
            return nullptr;
        }

        // Init handles
        pWindow->mpGLFWWindow = pGLFWWindow;
#ifdef _WIN32
        pWindow->mApiHandle = glfwGetWin32Window(pGLFWWindow);
        assert(pWindow->mApiHandle);
#else
        pWindow->mApiHandle.pDisplay = glfwGetX11Display();
        pWindow->mApiHandle.window = glfwGetX11Window(pGLFWWindow);
        assert(pWindow->mApiHandle.pDisplay != nullptr);
#endif

        glfwSetWindowUserPointer(pGLFWWindow, pWindow.get());

        // Set callbacks
        glfwSetWindowSizeCallback(pGLFWWindow, ApiCallbacks::windowSizeCallback);
        glfwSetKeyCallback(pGLFWWindow, ApiCallbacks::keyboardCallback);
        glfwSetMouseButtonCallback(pGLFWWindow, ApiCallbacks::mouseButtonCallback);
        glfwSetCursorPosCallback(pGLFWWindow, ApiCallbacks::mouseMoveCallback);
        glfwSetScrollCallback(pGLFWWindow, ApiCallbacks::mouseWheelCallback);
        glfwSetCharCallback(pGLFWWindow, ApiCallbacks::charInputCallback);

        return pWindow;
    }

    void Window::resize(uint32_t width, uint32_t height)
    {
        glfwSetWindowSize(mpGLFWWindow, width, height);

        mWidth = width;
        mHeight = height;

        mMouseScale.x = 1 / float(width);
        mMouseScale.y = 1 / float(height);

        mpCallbacks->handleWindowSizeChange();
    }

    void Window::msgLoop()
    {
        // Samples often rely on a size change event as part of initialization
        // This would have happened from a WM_SIZE message when calling ShowWindow on Win32
        mpCallbacks->handleWindowSizeChange();

        glfwShowWindow(mpGLFWWindow);
        glfwFocusWindow(mpGLFWWindow);

        while (glfwWindowShouldClose(mpGLFWWindow) == false)
        {
            glfwPollEvents();
            mpCallbacks->renderFrame();
        }
    }

    void Window::setWindowTitle(const std::string& title)
    {
        glfwSetWindowTitle(mpGLFWWindow, title.c_str());
    }

    void Window::pollForEvents()
    {
        glfwPollEvents();
    }
}
