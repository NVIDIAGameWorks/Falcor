'''
Module for printing colored text to a VT100 terminal using ANSI escape codes.
'''

import sys

COLOR_BEGIN = {
    'gray': '\33[90m',
    'red': '\33[91m',
    'green': '\33[92m',
    'yellow': '\33[93m',
    'blue': '\33[94m',
    'magenta': '\33[95m'
}

COLOR_END = '\033[0m'

def colored(text, color, stream=sys.stdout):
    '''
    Returns the given text wrapped in ANSI escape color codes if stream is attached to terminal.
    '''
    if stream.isatty() and color in COLOR_BEGIN:
        return COLOR_BEGIN[color] + text + COLOR_END
    return text

def test():
    '''
    Print all the available colors.
    '''
    for color in COLOR_BEGIN.keys():
        print(colored(f'This is {color}', color))


# Enable VT100 support for windows terminal
try:
    import ctypes
    import ctypes.wintypes

    kernel32 = ctypes.windll.kernel32
    STD_OUTPUT_HANDLE = -11
    STD_ERROR_HANDLE = -12
    ENABLE_VIRTUAL_TERMINAL_PROCESSING = 4

    def enable_virtual_terminal(handle):
        console = kernel32.GetStdHandle(handle)
        mode = ctypes.wintypes.DWORD()
        mode = kernel32.GetConsoleMode(console, ctypes.byref(mode))
        kernel32.SetConsoleMode(console, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING)

    enable_virtual_terminal(STD_OUTPUT_HANDLE)
    enable_virtual_terminal(STD_ERROR_HANDLE)
except:
    pass