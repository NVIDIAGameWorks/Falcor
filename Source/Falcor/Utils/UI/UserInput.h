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

namespace Falcor
{
    /** Input modifiers used with some events
    */
    struct InputModifiers
    {
        bool isCtrlDown  = false;   ///< Any of the control keys are pressed
        bool isShiftDown = false;   ///< Any of the shift keys are pressed
        bool isAltDown   = false;   ///< Any of the alt keys are pressed
    };

    /** Abstracts mouse messages
    */
    struct MouseEvent
    {
        /** Message Type
        */
        enum class Type
        {
            LeftButtonDown,         ///< Left mouse button was pressed
            LeftButtonUp,           ///< Left mouse button was released
            MiddleButtonDown,       ///< Middle mouse button was pressed
            MiddleButtonUp,         ///< Middle mouse button was released
            RightButtonDown,        ///< Right mouse button was pressed
            RightButtonUp,          ///< Right mouse button was released
            Move,                   ///< Mouse cursor position changed
            Wheel                   ///< Mouse wheel was scrolled
        };

        Type type;              ///< Event Type.
        float2 pos;             ///< Normalized coordinates x,y in range [0, 1]. (0,0) is the top-left corner of the window.
        float2 screenPos;       ///< Screen-space coordinates in range [0, clientSize]. (0,0) is the top-left corner of the window.
        float2 wheelDelta;      ///< If the current event is CMouseEvent#Type#Wheel, the change in wheel scroll. Otherwise zero.
        InputModifiers mods;    ///< Keyboard modifiers. Only valid if the event Type is one the button events
    };

    struct KeyboardEvent
    {
        /** Keyboard event Type
        */
        enum class Type
        {
            KeyPressed,     ///< Key was pressed.
            KeyReleased,    ///< Key was released.
            Input           ///< Character input
        };

        /** Use this enum to find out which key was pressed. Alpha-numeric keys use their uppercase ASCII code, so you can use that as well.
        */
        enum class Key : uint32_t
        {
            // ASCII values. Do not change them.
            Space           = ' ',
            Apostrophe      = '\'',
            Comma           = ',',
            Minus           = '-',
            Period          = '.',
            Slash           = '/',
            Key0            = '0',
            Key1            = '1',
            Key2            = '2',
            Key3            = '3',
            Key4            = '4',
            Key5            = '5',
            Key6            = '6',
            Key7            = '7',
            Key8            = '8',
            Key9            = '9',
            Semicolon       = ';',
            Equal           = '=',
            A               = 'A',
            B               = 'B',
            C               = 'C',
            D               = 'D',
            E               = 'E',
            F               = 'F',
            G               = 'G',
            H               = 'H',
            I               = 'I',
            J               = 'J',
            K               = 'K',
            L               = 'L',
            M               = 'M',
            N               = 'N',
            O               = 'O',
            P               = 'P',
            Q               = 'Q',
            R               = 'R',
            S               = 'S',
            T               = 'T',
            U               = 'U',
            V               = 'V',
            W               = 'W',
            X               = 'X',
            Y               = 'Y',
            Z               = 'Z',
            LeftBracket     = '[',
            Backslash       = '\\',
            RightBracket    = ']',
            GraveAccent     = '`',

            // Special keys
            Escape          ,
            Tab             ,
            Enter           ,
            Backspace       ,
            Insert          ,
            Del             ,
            Right           ,
            Left            ,
            Down            ,
            Up              ,
            PageUp          ,
            PageDown        ,
            Home            ,
            End             ,
            CapsLock        ,
            ScrollLock      ,
            NumLock         ,
            PrintScreen     ,
            Pause           ,
            F1              ,
            F2              ,
            F3              ,
            F4              ,
            F5              ,
            F6              ,
            F7              ,
            F8              ,
            F9              ,
            F10             ,
            F11             ,
            F12             ,
            Keypad0         ,
            Keypad1         ,
            Keypad2         ,
            Keypad3         ,
            Keypad4         ,
            Keypad5         ,
            Keypad6         ,
            Keypad7         ,
            Keypad8         ,
            Keypad9         ,
            KeypadDel       ,
            KeypadDivide    ,
            KeypadMultiply  ,
            KeypadSubtract  ,
            KeypadAdd       ,
            KeypadEnter     ,
            KeypadEqual     ,
            LeftShift       ,
            LeftControl     ,
            LeftAlt         ,
            LeftSuper       , // Windows key on windows
            RightShift      ,
            RightControl    ,
            RightAlt        ,
            RightSuper      , // Windows key on windows
            Menu            ,
        };

        Type type;              ///< The event type
        Key  key;               ///< The last key that was pressed/released
        InputModifiers mods;    ///< Keyboard modifiers
        uint32_t codepoint = 0; ///< UTF-32 codepoint from GLFW for Input event types
    };
}
