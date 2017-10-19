/***************************************************************************
# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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

// Please note:  This is an early, hacked version of Python embedding that we're distributing
//               due to requests from multiple folks who heard our SIGGRAPH 2017 presentation
//               using a demo built on this code.  Not promised to be production ready!
//
// See the README.txt in the Falcor Sample "LearningWithEmbeddedPython" for more info on how
//     to use this file and what steps are needed to set up Python.

#pragma once

#include "FalcorConfig.h"

#if FALCOR_USE_PYTHON

#include <map>

/** Include to handle C++/Python type conversions.
        Used internally to simplify Python embedding. (Library is BSD licensed)
        Grab from here: https://github.com/pybind/pybind11
*/
#include "pybind11/pybind11.h"
#include "pybind11/embed.h"
#include "pybind11/eval.h"

/** If PYTHON_USE_SIMPLE_SHARING is defined, use a simplistic and fragile approach that
        allows usage of multiple PythonEmbedding class instantiations simultaneously (but
        all using a single Python interpreter and shared global namespace).  If not
        defined, the use of multiple PythonEmbedding objects simultaneously will lead to
        unknown/undefined results (likely random crashes).
    TODO: Sharing Python embeddings clearly needs some more work!
*/
//#define PYTHON_USE_SIMPLE_SHARING

/** Class for easily embedding Python into C++ code.
    IMPORTANT NOTES on current usage assumptions:
      1) The Python embedding used assumes it is the *only* embedded interpreter.
          -> Using multiple class instantiations may cause undefined behavior.
          -> See 'PYTHON_USE_SIMPLE_SHARING' for a possible, very fragile solution
      2) Class designed to initialize once, use for life of program, destroy once.
          -> Multiple creation/deletion cycles not extensively tested
      3) Anyways...  deletion of Python interpreter fails and is currently disabled.
          -> For now, treat this class as a create-once resource.
*/
class PythonEmbedding : public std::enable_shared_from_this<PythonEmbedding>
{
public:
    using SharedPtr = std::shared_ptr<PythonEmbedding>;
    using SharedConstPtr = std::shared_ptr<const PythonEmbedding>;

    /** Constructors & destructors
    */
    PythonEmbedding(bool redirectStdout = true);
    virtual ~PythonEmbedding();
    static SharedPtr create(bool redirectStdout = true) { return std::make_shared<PythonEmbedding>(redirectStdout); }

    /** Methods to globally import Python modules for later use. Note this can be done directly in Python,
        however doing it with these methods allows you to easily query module version information, pay
        module startup costs at an appropriate time, etc.
          -> Python: "import numpy"                      -> importModule( "numpy" );
          -> Python: "import numpy as np"                -> importModule( "numpy", "np" );
          -> Python: "from hashlib import md5"           -> fromModuleImport( "hashlib", "md5" )
          -> Python: "from hashlib import md5 as myname" -> fromModuleImport( "hashlib", "md5", "myname" )
        \return true on success, false on failure.
    */
    bool importModule(const char *moduleName, const char *importAs = nullptr);
    bool fromModuleImport(const char *moduleName, const char *toImport, const char *importAs = nullptr);

    /** Accessors for variables in the global namespace.
          -> These can be assigned directly, if you want to send data to Python (e.g., this->operator[]("myVar") = 45; )
             This has pybind11 do the conversion behind your back.  It usually works pretty well, for simple types.
             You can also explicitly convert to Python/pybind11 object types and pass those on the right hand side,
             for more complex data types.
          -> These can be used to access Python variables (e.g., int( this->operator[]("myVar").cast<pybind11::int_>() ); )
             Casting to C++ this way is, as you may notice, a lot more painful.  Oh well.  One day I hope to clean that up.
          -> If the variable <globalVarName> does not exist, casting it via pybind11's cast<T>() will throw an exception.
             Before casting, you can/should check if a variable exists with: doesGlobalVarExist(globalVarName)
          -> These accessors are equivalent to calling:  getGlobals()[globalVarName]
    */
    pybind11::detail::item_accessor operator[] (const char* globalVarName);
    pybind11::detail::item_accessor operator[] (const std::string &globalVarName);

    /** Check if a global variable exists
    */
    bool doesGlobalVarExist(const char *globalVarName);
    bool doesGlobalVarExist(const std::string &globalVarName);

    /** Return the entire dictionary of Python's global variables
    */
    pybind11::dict getGlobals(void);

    /** Return the last Python error encountered during executeFile() or executeString()
    */
    std::string getError(void);

    /** If redirecting stdout and/or stderr is enabled, the appropriate functions will get you a string
        with the outputs of stdout/stderr from your *last* fall to executeFile() or executeString().
    */
    std::string getStdout(void);
    std::string getStderr(void);

    /** Determine if stdout/stderr will be redirected when executing Python via this class.
         -> Note that the settings are independent for stdout and stderr (you can redirect one, but not the other)
         -> Note:  method defaults parameters are different than class defaults.  (redirectStderr() implies "make it so")
         -> Redirecting outputs introduces overheads of around 0.1 ms per execute() call, for a few lines of output.
    */
    void redirectStdout(bool redirect = true);
    void redirectStderr(bool redirect = true);

    /** Returns a string representing the version of Python.
    */
    std::string getPythonVersion(void);

    /** Returns a string representing the version number of the specified module... if it has successfully been loaded.
           -> The string returned is that stored in attrName (e.g., "numpy.__version__" in Python returns NumPy's version)
           -> Returns the string "<not loaded>" if a module is not loaded
           -> Returns "<unknown>" if a module is loaded but the specified attrName does not exist
    */
    std::string getModuleVersion(const char *moduleName, const char *attrName = "__version__");

    /** Methods to execute Python from a specified file/string in the global namespace
          -> Success returns true.  Failure returns false, errors available via getPythonErrors()
          -> Multiline strings need to be specified with a C++ 11 raw string literal, e.g., R"( ... )"
             in order to allow explicit newlines and tabs to be passed to the Python interpreter for
             correct understanding of code indentation.
    */
    bool executeFile(const char *scriptFile);
    bool executeFile(const std::string &scriptFile);
    bool executeString(const char *scriptCode);
    bool executeString(const std::string &scriptCode);

    /** Methods to execute Python from a specified file/string in a new, local namespace
          -> Important note:  Unless you really know what you're doing, this is *rarely*
             what you want to do.  It screws up variable scoping in a way you likely will
             not expect.  This behaves _exactly_ like calling exec() from within Python,
             so read this: https://docs.python.org/3/library/functions.html?highlight=exec#exec
          -> Note in that URL, the description of these functions:
             "code will be executed as if [the whole script] were embedded in a class definition"
          -> Success returns true.  Failure returns false, errors available via getPythonErrors()
    */
    bool executeFile(const char *scriptFile, pybind11::dict localNamespace);
    bool executeFile(const std::string &scriptFile, pybind11::dict localNamespace);
    bool executeString(const char *scriptCode, pybind11::dict localNamespace);
    bool executeString(const std::string &scriptCode, pybind11::dict localNamespace);

    /** Returns the cost of the last executeFile() or executeString() call in milliseconds
          (to the precision allowed by std::chrono::high_resolution_clock).  If the last
          execution threw an exception, the return value is -1.
          -> if totalTime is true, returns the cost of the execute*() call
          -> it totalTime is false, returns the cost of executing just the specified Python
             code, without setup overheads introduced by this class.
    */
    double lastExecutionTime( bool totalTime = true );

private:
    // Internal member variables

    pybind11::scoped_interpreter *mpInterp     = nullptr;  // The C++ embedded Python interpreter
    pybind11::module              mMainModule;             // Stores the Python module __main__
    pybind11::dict                mGlobals;                // A dictionary of the global Python variables
    std::string                   mPythonVersion;          // A string representation of the embedded Python's version
    std::string                   mLastPythonError = "";   // A string storing the last Python error (from an exception)

    bool                          mRedirectStdout;         // Redirect stdout of executions to mLastPythonStdout? (initialized in constructor)
    std::string                   mLastPythonStdout = "";  // A string storing the last execution's stdout output (if requested)
    bool                          mRedirectStderr = false; // Redirect stderr of executions to mLastPythonStderr?
    std::string                   mLastPythonStderr = "";  // A string storing the last execution's stderr output (if requested)
    bool                          mRedirectValid = false;  // An internal status check; was redirection successful?

    double                        mLastCostTotal = -1;     // The cost of the last executeFile()/executeString() call
    double                        mLastCostPython = -1;    // The cost of the Python code from the last executeFile()/executeString() call

    /** Stores a list of modules we've tried to load.
    */
    std::map< std::string,std::pair<pybind11::handle,std::string> > mModuleList;

    /** When executing a new command/file, clear existing error, I/o, etc. state
    */
    void clearPythonState(void);

    /** Internal method to ensure our interpreter has the necessary imports for redirecting stdout/stderr
    */
    void checkRedirectionImports(void);

    /** Internal methods to start or stop redirecting stdout and/or stderr to strings
    */
    void startRedirectingOutputs(void);
    void stopRedirectingOutputs(bool getOutputs = false);
    void getRedirectedOutputs(bool clearBuffers = true);

    /** A common execution routine that handles all execute*() routines above to keep exection code localized in one spot
    */
    bool commonExecRoutine(const pybind11::str &input, bool asString, bool useLocals = false, pybind11::object locals = class pybind11::object());

};

#endif