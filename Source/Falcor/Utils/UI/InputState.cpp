/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
#include "InputState.h"
#include "Core/Error.h"

namespace Falcor
{
bool InputState::isModifierDown(Input::Modifier mod) const
{
    return getModifierState(mCurrentKeyState, mod);
}

bool InputState::isModifierPressed(Input::Modifier mod) const
{
    return getModifierState(mCurrentKeyState, mod) == true && getModifierState(mPreviousKeyState, mod) == false;
}

bool InputState::isModifierReleased(Input::Modifier mod) const
{
    return getModifierState(mCurrentKeyState, mod) == false && getModifierState(mPreviousKeyState, mod) == true;
}

void InputState::onKeyEvent(const KeyboardEvent& keyEvent)
{
    // Update the stored key state.
    if (keyEvent.type == KeyboardEvent::Type::KeyPressed || keyEvent.type == KeyboardEvent::Type::KeyReleased)
    {
        mCurrentKeyState[(size_t)keyEvent.key] = keyEvent.type == KeyboardEvent::Type::KeyPressed;
    }
}

void InputState::onMouseEvent(const MouseEvent& mouseEvent)
{
    // Update the stored mouse state.
    if (mouseEvent.type == MouseEvent::Type::ButtonDown || mouseEvent.type == MouseEvent::Type::ButtonUp)
    {
        mCurrentMouseState[(size_t)mouseEvent.button] = mouseEvent.type == MouseEvent::Type::ButtonDown;
    }
    else if (mouseEvent.type == MouseEvent::Type::Move)
    {
        mMouseMoving = true;
    }
}

void InputState::endFrame()
{
    mPreviousKeyState = mCurrentKeyState;
    mPreviousMouseState = mCurrentMouseState;

    mMouseMoving = false;
}

bool InputState::getModifierState(const KeyStates& states, Input::Modifier mod) const
{
    switch (mod)
    {
    case Input::Modifier::Shift:
        return states[(size_t)Input::Key::LeftShift] || states[(size_t)Input::Key::RightShift];
    case Input::Modifier::Ctrl:
        return states[(size_t)Input::Key::LeftControl] || states[(size_t)Input::Key::RightControl];
    case Input::Modifier::Alt:
        return states[(size_t)Input::Key::LeftAlt] || states[(size_t)Input::Key::RightAlt];
    default:
        FALCOR_UNREACHABLE();
        return false;
    }
}
} // namespace Falcor
