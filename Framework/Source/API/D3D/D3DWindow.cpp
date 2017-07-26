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
#include "Utils/OS.h"
#include <algorithm>
#include "API/texture.h"
#include "API/FBO.h"
#include <Initguid.h>
#include <Windowsx.h>
#include "Utils/StringUtils.h"

// #VKTODO This probably makes more sense as "WindowsWindow" rather than specifically D3D

namespace Falcor
{
	class ApiCallbacks
    {
    public:
        static LRESULT CALLBACK msgProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
        {
            Window* pWindow;

            if(msg == WM_CREATE)
            {
                CREATESTRUCT* pCreateStruct = (CREATESTRUCT*)lParam;
                SetWindowLongPtr(hwnd, GWLP_USERDATA, (LONG_PTR)pCreateStruct->lpCreateParams);
                return DefWindowProc(hwnd, msg, wParam, lParam);
            }
            else
            {
				pWindow = (Window*)GetWindowLongPtr(hwnd, GWLP_USERDATA);
                switch(msg)
                {
                case WM_CLOSE:
                    DestroyWindow(hwnd);
                    return 0;
                case WM_DESTROY:
                    PostQuitMessage(0);
                    return 0;
                case WM_SIZE:
                    if(wParam != SIZE_MINIMIZED)
                    {
                        resizeWindow(pWindow);
                    }
                    break;
                case WM_KEYDOWN:
                case WM_KEYUP:
                    dispatchKeyboardEvent(pWindow, wParam, msg == WM_KEYDOWN);
                    return 0;
                case WM_MOUSEMOVE:
                case WM_LBUTTONDOWN:
                case WM_LBUTTONUP:
                case WM_MBUTTONDOWN:
                case WM_MBUTTONUP:
                case WM_RBUTTONDOWN:
                case WM_RBUTTONUP:
                case WM_MOUSEWHEEL:
                    dispatchMouseEvent(pWindow, msg, wParam, lParam);
                    return 0;
                }
                return DefWindowProc(hwnd, msg, wParam, lParam);
            }
        }

    private:
        static void resizeWindow(Window* pWindow)
        {
            RECT r;
            GetClientRect(pWindow->getApiHandle(), &r);
            uint32_t width = r.right - r.left;
            uint32_t height = r.bottom - r.top;
			pWindow->resize(width, height);
		}

        static KeyboardEvent::Key translateKeyCode(WPARAM keyCode)
        {
            switch(keyCode)
            {
            case VK_TAB:
                return KeyboardEvent::Key::Tab;
            case VK_RETURN:
                return KeyboardEvent::Key::Enter;
            case VK_BACK:
                return KeyboardEvent::Key::Backspace;
            case VK_PAUSE:
            case VK_CANCEL:
                return KeyboardEvent::Key::Pause;
            case VK_ESCAPE:
                return KeyboardEvent::Key::Escape;
            case VK_DECIMAL:
                return KeyboardEvent::Key::KeypadDel;
            case VK_DIVIDE:
                return KeyboardEvent::Key::KeypadDivide;
            case VK_MULTIPLY:
                return KeyboardEvent::Key::KeypadMultiply;
            case VK_NUMPAD0:
                return KeyboardEvent::Key::Keypad0;
            case VK_NUMPAD1:
                return KeyboardEvent::Key::Keypad1;
            case VK_NUMPAD2:
                return KeyboardEvent::Key::Keypad2;
            case VK_NUMPAD3:
                return KeyboardEvent::Key::Keypad3;
            case VK_NUMPAD4:
                return KeyboardEvent::Key::Keypad4;
            case VK_NUMPAD5:
                return KeyboardEvent::Key::Keypad5;
            case VK_NUMPAD6:
                return KeyboardEvent::Key::Keypad6;
            case VK_NUMPAD7:
                return KeyboardEvent::Key::Keypad7;
            case VK_NUMPAD8:
                return KeyboardEvent::Key::Keypad8;
            case VK_NUMPAD9:
                return KeyboardEvent::Key::Keypad9;
            case VK_SUBTRACT:
                return KeyboardEvent::Key::KeypadSubtract;
            case VK_CAPITAL:
                return KeyboardEvent::Key::CapsLock;
            case VK_DELETE:
                return KeyboardEvent::Key::Del;
            case VK_DOWN:
                return KeyboardEvent::Key::Down;
            case VK_UP:
                return KeyboardEvent::Key::Up;
            case VK_LEFT:
                return KeyboardEvent::Key::Left;
            case VK_RIGHT:
                return KeyboardEvent::Key::Right;
            case VK_F1:
                return KeyboardEvent::Key::F1;
            case VK_F2:
                return KeyboardEvent::Key::F2;
            case VK_F3:
                return KeyboardEvent::Key::F3;
            case VK_F4:
                return KeyboardEvent::Key::F4;
            case VK_F5:
                return KeyboardEvent::Key::F5;
            case VK_F6:
                return KeyboardEvent::Key::F6;
            case VK_F7:
                return KeyboardEvent::Key::F7;
            case VK_F8:
                return KeyboardEvent::Key::F8;
            case VK_F9:
                return KeyboardEvent::Key::F9;
            case VK_F10:
                return KeyboardEvent::Key::F10;
            case VK_F11:
                return KeyboardEvent::Key::F11;
            case VK_F12:
                return KeyboardEvent::Key::F12;
            case VK_END:
                return KeyboardEvent::Key::End;
            case VK_HOME:
                return KeyboardEvent::Key::Home;
            case VK_INSERT:
                return KeyboardEvent::Key::Insert;
            case VK_LCONTROL:
                return KeyboardEvent::Key::LeftControl;
            case VK_LMENU:
                return KeyboardEvent::Key::LeftAlt;
            case VK_LSHIFT:
                return KeyboardEvent::Key::LeftShift;
            case VK_LWIN:
                return KeyboardEvent::Key::LeftSuper;
            case VK_NUMLOCK:
                return KeyboardEvent::Key::NumLock;
            case VK_RCONTROL:
                return KeyboardEvent::Key::RightControl;
            case VK_RMENU:
                return KeyboardEvent::Key::RightAlt;
            case VK_RSHIFT:
                return KeyboardEvent::Key::RightShift;
            case VK_RWIN:
                return KeyboardEvent::Key::RightSuper;
            case VK_SCROLL:
                return KeyboardEvent::Key::ScrollLock;
            case VK_SNAPSHOT:
                return KeyboardEvent::Key::PrintScreen;
            case VK_OEM_MINUS:
                return KeyboardEvent::Key::Minus;
            case VK_OEM_PERIOD:
                return KeyboardEvent::Key::Period;
            case VK_OEM_1:
                return KeyboardEvent::Key::Semicolon;
            case VK_OEM_PLUS:
                return KeyboardEvent::Key::Equal;
            case VK_OEM_COMMA:
                return KeyboardEvent::Key::Comma;
            case VK_OEM_2:
                return KeyboardEvent::Key::Slash;
            case VK_OEM_3:
                return KeyboardEvent::Key::GraveAccent;
            default:
                // ASCII code
                return (KeyboardEvent::Key)keyCode;
            }
        }

        static InputModifiers getInputModifiers()
        {
            InputModifiers mods;
            mods.isShiftDown = (GetKeyState(VK_SHIFT) & 0x8000) != 0;
            mods.isCtrlDown = (GetKeyState(VK_CONTROL) & 0x8000) != 0;
            mods.isAltDown = (GetKeyState(VK_MENU) & 0x8000) != 0;
            return mods;
        }

        static void dispatchKeyboardEvent(const Window* pWindow, WPARAM keyCode, bool isKeyDown)
        {
            KeyboardEvent keyEvent;
            keyEvent.type = isKeyDown ? KeyboardEvent::Type::KeyPressed : KeyboardEvent::Type::KeyReleased;
            keyEvent.key = translateKeyCode(keyCode);
            keyEvent.mods = getInputModifiers();
            keyEvent.asciiChar = 0;

            // Update the state table for some keys
            BYTE keyboardState[256];
            GetKeyboardState(keyboardState);
            char c[2];
            if (ToAscii((UINT)keyCode, 0, keyboardState, (WORD*)c, 0) != 0)
            {
                keyEvent.asciiChar = c[0];
            }

            pWindow->mpCallbacks->handleKeyboardEvent(keyEvent);
        }

        static void dispatchMouseEvent(const Window* pWindow, UINT Msg, WPARAM wParam, LPARAM lParam)
        {
            MouseEvent mouseEvent;
            POINT pos = { GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam) };
            switch(Msg)
            {
            case WM_MOUSEMOVE:
                mouseEvent.type = MouseEvent::Type::Move;
                break;
            case WM_LBUTTONDOWN:
                mouseEvent.type = MouseEvent::Type::LeftButtonDown;
                break;
            case WM_LBUTTONUP:
                mouseEvent.type = MouseEvent::Type::LeftButtonUp;
                break;
            case WM_MBUTTONDOWN:
                mouseEvent.type = MouseEvent::Type::MiddleButtonDown;
                break;
            case WM_MBUTTONUP:
                mouseEvent.type = MouseEvent::Type::MiddleButtonUp;
                break;
            case WM_RBUTTONDOWN:
                mouseEvent.type = MouseEvent::Type::RightButtonDown;
                break;
            case WM_RBUTTONUP:
                mouseEvent.type = MouseEvent::Type::RightButtonUp;
                break;
            case WM_MOUSEWHEEL:
                mouseEvent.type = MouseEvent::Type::Wheel;
                mouseEvent.wheelDelta.y = ((float)GET_WHEEL_DELTA_WPARAM(wParam)) / WHEEL_DELTA;
                ScreenToClient(pWindow->getApiHandle(), &pos);
                break;
            case WM_MOUSEHWHEEL:
                mouseEvent.type = MouseEvent::Type::Wheel;
                mouseEvent.wheelDelta.x = ((float)GET_WHEEL_DELTA_WPARAM(wParam)) / WHEEL_DELTA;
                ScreenToClient(pWindow->getApiHandle(), &pos);
            default:
                should_not_get_here();
            }

            mouseEvent.pos = glm::vec2((float)pos.x, (float)pos.y);
            mouseEvent.pos *= pWindow->getMouseScale();
            mouseEvent.mods = getInputModifiers();

            pWindow->mpCallbacks->handleMouseEvent(mouseEvent);
        }
    };

	static HWND createWindow(const Window::Desc& desc, void* pUserData)
    {
        const WCHAR* className = L"FalcorWindowClass";
        DWORD winStyle = WS_OVERLAPPED | WS_CAPTION |  WS_SYSMENU;
        if(desc.resizableWindow == true)
        {
            winStyle = winStyle | WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX;
        }

        // Register the window class
        WNDCLASS wc = {};
        wc.lpfnWndProc = &ApiCallbacks::msgProc;
        wc.hInstance = GetModuleHandle(nullptr);
        wc.lpszClassName = className;
        wc.hIcon = nullptr;
        wc.hCursor = LoadCursor(NULL, IDC_ARROW);

        if(RegisterClass(&wc) == 0)
        {
            logErrorAndExit("RegisterClass() failed");
            return nullptr;
        }

        // Window size we have is for client area, calculate actual window size
        RECT r{0, 0, (LONG)desc.width, (LONG)desc.height};
        AdjustWindowRect(&r, winStyle, false);

        int windowWidth = r.right - r.left;
        int windowHeight = r.bottom - r.top;

        // create the window
        std::wstring wTitle = string_2_wstring(desc.title);
        HWND hWnd = CreateWindowEx(0, className, wTitle.c_str(), winStyle, CW_USEDEFAULT, CW_USEDEFAULT, windowWidth, windowHeight, nullptr, nullptr, wc.hInstance, pUserData);
        if(hWnd == nullptr)
        {
            logErrorAndExit("CreateWindowEx() failed");
            return nullptr;
        }

        // It might be tempting to call ShowWindow() here, but this fires a WM_SIZE message, which if you look at our MsgProc()
        // calls some device functions. That's a race condition, since the device wasn't initialized yet 
        return hWnd;
    }

    Window::Window(ICallbacks* pCallbacks, uint32_t width, uint32_t height) : mpCallbacks(pCallbacks), mWidth(width), mHeight(height)
    {

    }

    Window::~Window()
    {
        if(mApiHandle)
        {
			DestroyWindow(mApiHandle);
		}
    }

    void Window::shutdown()
    {
        PostQuitMessage(0);
    }

    Window::SharedPtr Window::create(const Desc& desc, ICallbacks* pCallbacks)
    {
		SharedPtr pWindow = SharedPtr(new Window(pCallbacks, desc.width, desc.height));
        
        // create the window
		pWindow->mApiHandle = createWindow(desc, pWindow.get());
        if(pWindow->mApiHandle == nullptr)
        {
            return false;
        }

		return pWindow;
    }

    void Window::resize(uint32_t width, uint32_t height)
    {
        // Resize the window
        RECT r = { 0, 0, (LONG)width, (LONG)height };
        DWORD style = GetWindowLong(mApiHandle, GWL_STYLE);
        mWidth = width;
        mHeight = height;

        // The next call will dispatch a WM_SIZE message which will take care of the framebuffer size change
        AdjustWindowRect(&r, style, false);
        uint32_t adjWidth = r.right - r.left;
        uint32_t adjHeight = r.bottom - r.top;

        // #HACK temporarily removing d3d_call so Window implementation links/works for Vulkan
        /*d3d_call(*/SetWindowPos(mApiHandle, nullptr, 0, 0, adjWidth , adjHeight, SWP_NOACTIVATE | SWP_NOOWNERZORDER | SWP_NOMOVE | SWP_NOZORDER)/*)*/;

        mMouseScale.x = 1 / float(width);
        mMouseScale.y = 1 / float(height);

        mpCallbacks->handleWindowSizeChange();
    }

	void Window::msgLoop()
    {
        // Show the window
        ShowWindow(mApiHandle, SW_SHOWNORMAL);
        SetForegroundWindow(mApiHandle);

        MSG msg;
        while(1) 
        {
            if(PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
            {
                if(msg.message == WM_QUIT)
                {
                    break;
                }
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            }
            else
            {
				mpCallbacks->renderFrame();
            }
        }
    }

    void Window::setWindowTitle(std::string title)
    {
		    }

    void Window::pollForEvents()
    {
    }
}
