/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
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
#pragma once
#include "InputTypes.h"
#include "Core/Macros.h"
#include <bitset>

namespace Falcor
{
    /** A class that hold the state of the current inputs such as Keys and mouse events. This class can be fetched from the gpFramework pointer in the Sample class.
        Like this:
            cosnt InputState& inputState = gpFramework->getInputState()
        The class does not tell Falcor that it has handled the input when the user calls the functions.
        It is up to the user to tell Falcor that the input event was handled by returning true in the input event callbacks.
    */
    class Sample;
    class FALCOR_API InputState
    {
    public:
        bool isMouseMoving() const { return mMouseMoving; }

        /** Check if the specified key is currently held down.
            \param[in] key The specified key.
            \return True if currently held down, else false.
        */
        bool isKeyDown(Input::Key key) const { return mCurrentKeyState[(size_t)key]; }

        /** Check if the specified key was just pressed down.
            \param[in] key The key to check.
            \return True if the specified key was just pressed down, else false.
        */
        bool isKeyPressed(Input::Key key) const { return mCurrentKeyState[(size_t)key] == true && mPreviousKeyState[(uint32_t)key] == false; }

        /** Check if the specified key was just released up.
            \param[in] key The key to check.
            \return True if the specified key was just released up, else false.
        */
        bool isKeyReleased(Input::Key key) const { return mCurrentKeyState[(size_t)key] == false && mPreviousKeyState[(uint32_t)key] == true; }

        /** Check if the specified mouse button is currently held down.
            \param[in] mb The specified mouse button.
            \return True if currently held down, else false.
        */
        bool isMouseButtonDown(Input::MouseButton mb) const { return mCurrentMouseState[(size_t)mb]; }

        /** Check if the specified mouse button was just pressed down.
            \param[in] mb The mouse button to check.
            \return True if the specified mouse button was just pressed down, else false.
        */
        bool isMouseButtonClicked(Input::MouseButton mb) const { return mCurrentMouseState[(size_t)mb] == true && mPreviousMouseState[(uint32_t)mb] == false; }

        /** Check if the specified mouse button was just released up.
            \param[in] mb The mouse button to check.
            \return True if the specified mouse button was just released up, else false.
        */
        bool isMouseButtonReleased(Input::MouseButton mb) const { return mCurrentMouseState[(size_t)mb] == false && mPreviousMouseState[(uint32_t)mb] == true; }

        /** Check if the specified modifier is currently held down.
            \param[in] mod The specified modifier.
            \return True if currently held down, else false.
        */
        bool isModifierDown(Input::Modifier mod) const;

        /** Check if the specified modifier key was just pressed down.
            \param[in] mod The key to check.
            \return True if the specified modifier key was just pressed down, else false.
        */
        bool isModifierPressed(Input::Modifier mod) const;

        /** Check if the specified modifier key was just released up.
            \param[in] mod The key to check.
            \return True if the specified modifier key was just released up, else false.
        */
        bool isModifierReleased(Input::Modifier mod) const;

    private:
        static const size_t kKeyCount = (size_t)Input::Key::Count;
        static const size_t kMouseButtonCount = (size_t)Input::MouseButton::Count;
        using KeyStates = std::bitset<kKeyCount>;
        using MouseState = std::bitset<kMouseButtonCount>;

        /** Processes the key event and set the internal state.
            \param[in] keyEvent The event to register.
        */
        void onKeyEvent(const KeyboardEvent& keyEvent);

        /** Processes the mouse event and set the internal state.
            \param[in] mouseEvent The event to register.
        */
        void onMouseEvent(const MouseEvent& mouseEvent);

        /** Prepare the states for the next frame.
        */
        void endFrame();

        /** Helper function to get the modifier state from the key states.
            \param[in] states The key states to fetch from.
            \param[in] mod The modifier to check for.
            \return True if the modifier state was active, else false.
        */
        bool getModifierState(const KeyStates& states, Input::Modifier mod) const;

        // The state (up/down) of the keys and mouse buttons for the current and previous frame, default initialized to zero (up).
        KeyStates mCurrentKeyState;
        KeyStates mPreviousKeyState;
        MouseState mCurrentMouseState;
        MouseState mPreviousMouseState;

        bool mMouseMoving = false;

        friend class Sample;
    };
}
