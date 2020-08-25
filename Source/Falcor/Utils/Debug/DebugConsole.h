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
#include <fstream>
#include <iostream>

namespace Falcor
{
    /** Opens a console window and redirects std::cout, std::cerr, and std::cin there.
        Upon destruction of the object, the console is closed and the streams are restored to the previous state.
    */
    class DebugConsole
    {
    public:
        /** Opens a console window. The destructor closes it again.
            \param[in] waitForKey If true, the console waits for a key press before closing.
        */
        DebugConsole(bool waitForKey = true)
            : mWaitForKey(waitForKey)
        {
            // Open console window
            AllocConsole();

            // Redirect cout/cerr/cin streams to our console window
            mPrevCout = std::cout.rdbuf();
            mCout.open("CONOUT$");
            std::cout.rdbuf(mCout.rdbuf());

            mPrevCerr = std::cerr.rdbuf();
            mCerr.open("CONERR$");
            std::cerr.rdbuf(mCerr.rdbuf());

            mPrevCin = std::cin.rdbuf();
            mCin.open("CONIN$");
            std::cin.rdbuf(mCin.rdbuf());

            // Redirect stdout for printf() to our console
            //freopen_s(&mFp, "CONOUT$", "w", stdout);
            //std::cout.clear();
        }

        virtual ~DebugConsole()
        {
            flush();
            if (mWaitForKey)
            {
                pause();
            }

            // Restore the streams
            std::cin.rdbuf(mPrevCin);
            std::cerr.rdbuf(mPrevCerr);
            std::cout.rdbuf(mPrevCout);

            // Restore stdout to default
            //freopen("OUT", "w", stdout);
            //fclose(mFp);

            // Close console window
            FreeConsole();
        }

        void pause() const
        {
            std::cout << "Press any key to continue..." << std::endl;
            flush();
            char c = std::cin.get();
        }

        void flush() const
        {
            std::cout.flush();
            std::cerr.flush();
        }

    private:
        std::ofstream mCout;
        std::ofstream mCerr;
        std::ifstream mCin;
        std::streambuf* mPrevCout;
        std::streambuf* mPrevCerr;
        std::streambuf* mPrevCin;
        //FILE* mFp = nullptr;

        bool mWaitForKey = true;
    };
}
