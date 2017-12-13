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
#pragma once
#include <fstream>

namespace Falcor
{
    /** Helper class to manage file I/O with binary files
    */
    class BinaryFileStream
    {
    public:

        /** Mode to open file as
        */
        enum class Mode
        {
            Read    = 0x1,  ///< Open file for reading
            Write   = 0x2,  ///< Open file for writing
            ReadWrite = 0x3 ///< Open file for both reading and writing
        };

        /** Default constructor.
        */
        BinaryFileStream() {};

        /** Constructor that opens a file
            \param[in] filename Name of file to open or create
            \param[in] mode Mode to open file as
        */
        BinaryFileStream(const std::string& filename, Mode mode = Mode::ReadWrite)
        {
            open(filename, mode);
        }

        /** Destructor
        */
        ~BinaryFileStream()
        {
            close();
        }

        /** Opens a file stream. Fails if a file is already open.
            \param[in] filename Name of file to open or create
            \param[in] mode Mode to open file as
        */
        void open(const std::string& filename, Mode mode = Mode::ReadWrite)
        {
            std::ios::openmode iosMode = std::ios::binary;
            iosMode |= ((mode == Mode::Read) || (mode == Mode::ReadWrite)) ? std::ios::in : (std::ios::openmode)0;
            iosMode |= ((mode == Mode::Write) || (mode == Mode::ReadWrite))? std::ios::out : (std::ios::openmode)0;
            mStream.open(filename.c_str(), iosMode);
            mFilename = filename;
        }

        /** Close the file stream.
        */
        void close()
        {
            mStream.close();
        }

        /** Skip data in an input stream. Advances file stream without reading.
            \param[in] count Bytes to skip
        */
        void skip(uint32_t count)
        {
            mStream.ignore(count);
        }

        /** Deletes the managed file.
        */
        void remove()
        {
            if(mStream.is_open())
            {
                close();
            }
            std::remove(mFilename.c_str());
        }

        /** Calculates amount of remaining data in the file.
            \return Number of bytes remaining in the stream
        */
        uint32_t getRemainingStreamSize()
        {	
            std::streamoff currentPos = mStream.tellg();
            mStream.seekg(0, mStream.end);
            std::streamoff length = mStream.tellg();
            mStream.seekg(currentPos);
            return (uint32_t)(length - currentPos); 
        }

        /** Checks for validity of the stream
            \return Returns true if no errors have been encountered and the end of the stream has not been reached
        */
        bool isGood() { return mStream.good(); }

        /** Checks for stream errors.
            \return Returns true if an error has occurred while reading or writing data.
        */
        bool isBad()  { return mStream.bad(); }

        /** Checks for stream errors.
            \return Returns true if any error has occurred while reading the file.
        */
        bool isFail() { return mStream.fail(); }

        /** Checks if the end of file has been reached.
            \return Returns true if stream has reached the end of the file
        */
        bool isEof() { return mStream.eof(); }

        /** Reads data from the file stream
            \param[out] pData Pointer to a buffer to copy/read data into
            \param[in] count Number of bytes to read
        */
        BinaryFileStream& read(void* pData, size_t count) { mStream.read((char*)pData, count); return *this; }

        /** Writes data to the file stream
            \param[in] pData Pointer to buffer containing data to write to the stream
            \param[in] count Number of bytes to write
        */
        BinaryFileStream& write(const void* pData, size_t count) { mStream.write((char*)pData, count); return *this; }

        // Operator overloads

        /** Extracts a single value from the stream
            \param[out] val Reference of value to extract into
        */
        template<typename T>
        BinaryFileStream& operator>>(T& val) { return read(&val, sizeof(T)); }

        /** Writes a value into the file stream
            \param[in] val Value to write
        */
        template<typename T>
        BinaryFileStream& operator<<(const T& val) { return write(&val, sizeof(T)); }

    private:
        std::fstream mStream;
        std::string mFilename;
    };
}