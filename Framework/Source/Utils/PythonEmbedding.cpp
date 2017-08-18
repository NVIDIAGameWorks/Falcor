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

#include "FalcorConfig.h"

#if FALCOR_USE_PYTHON

#include "Python.h"
#include "PythonEmbedding.h"
#include <ctime>
#include <chrono>

// Requires use of pybind11.  See README.txt in sample "LearningWithEmbeddedPython" for more info
namespace py = pybind11;
using namespace py::literals;

// If using this fragine sharing scheme, create a global variable for the interpreter that everyone can reuse
#ifdef PYTHON_USE_SIMPLE_SHARING
namespace {
    py::scoped_interpreter* gPython = nullptr;
};
#endif


PythonEmbedding::PythonEmbedding(bool redirectStdout) :
    mRedirectStdout(redirectStdout)
{
    bool doInit = true;

#ifdef PYTHON_USE_SIMPLE_SHARING
    // Create a new interpreter embedding
    if (!gPython)
    {
        gPython = new py::scoped_interpreter();
    }
    else
    {
        doInit = false;
    }
    mpInterp = gPython;
#else
    // Create a new interpreter embedding
    if (!mpInterp)
    {
        mpInterp = new py::scoped_interpreter();
    }
    else
    {
        doInit = false;
    }
#endif

    // Create a __main__ module for Python to run in
    mMainModule = py::module::import("__main__");

    // Grab a pointer to the dictionary used to store Python's global variables
    mGlobals = py::reinterpret_borrow<py::dict>(mMainModule.attr("__dict__").ptr());

    // We need to call PySys_SetArgvEx() to avoid errors when querying/reflecting on the Python interpreter.
    //    In theory, all this does is set the relative path to a script that is being run.  Since an embedded
    //    interpreter has no script running on startup, this is NULL.  This looks like a bunch of no-ops, but
    //    it's vital to avoid crazy embedded Python errors.
    if (doInit)
    {
        size_t size = 0;
        wchar_t* argv = Py_DecodeLocale("", &size);
        PySys_SetArgvEx(0, &argv, 0); // 3rd zero essentially for using Python as ongoing interpreter
        PyMem_RawFree(argv);
    }

    // Get a stringified representation of the version of Python embedded.
    //     -> Grab the value stored in 'sys.version'
    //     -> This is often a huge string with compiler name and build time; grab just until the first space.
    mPythonVersion = std::string(py::handle(py::module::import("sys").attr("version").ptr()).cast<py::str>());
    size_t offset = mPythonVersion.find_first_of(' ');
    mPythonVersion = mPythonVersion.substr(0, offset);

    // Insert this main module into our list of modules
    mModuleList["__main__"] = std::make_pair(mMainModule,std::string("__main__"));

    // When redirecting output, there's some setup that needs to happen.  Since the default behavior of this
    //    class is to start redirecting stdout without any user intervention, we need to actually call the
    //    hooks to start redirecting inside the constructor.  Do that here.  It checks to make sure all
    //    relevant state has been initialized appropriately.
    if (mRedirectStdout || mRedirectStderr)
    {
        PythonEmbedding::redirectStdout(mRedirectStdout);  // Whoops.  Method hidden by constructor param! :-/
    }

}

PythonEmbedding::~PythonEmbedding()
{
    // TODO: Need to figure out how to shutdown appropriately
    /*
    if (mpInterp)
    {
        delete mpInterp;
        mpInterp = nullptr;
    }
    */
}

std::string PythonEmbedding::getPythonVersion(void)
{
    return mPythonVersion;
}

std::string PythonEmbedding::getModuleVersion(const char* moduleName, const char* attrName)
{
    // Check.  Have we loaded the specified module?
    std::string strName = std::string(moduleName);
    auto found = mModuleList.find(strName);

    // Wasn't found in our list of loaded modules
    if (found == mModuleList.end())
    {
        return std::string("<not loaded>");
    }

    // The handle/name pair for our found module
    auto foundPair = found->second;
    py::handle foundHandle = foundPair.first;

    // We tried to load the module....  and failed
    if (foundHandle.ptr() == nullptr)
    {
        return std::string("<not loaded>");
    }

    // Get the version number
    PyObject* pVerAttrib = PyObject_GetAttrString(foundHandle.ptr(), attrName);

    // The specified AttrString was invalid, so we don't know the version number!
    if (!pVerAttrib)
    {
        return std::string("<unknown>");
    }

    // We got a version ID!!
    return std::string(py::handle(pVerAttrib).cast<py::str>());
}

bool PythonEmbedding::importModule(const char* moduleName, const char* importAs)
{
    bool success = false;

    // Check.  Have we already loaded this module?
    std::string strName = std::string(moduleName);
    auto found = mModuleList.find(strName);

    // We found an existing entry for this module
    if (found != mModuleList.end())
    {
        // Check to see if the stored py::handle is non-null.  If so, we've already imported the module.
        auto second = found->second;
        if (second.first.ptr() != nullptr)
        {
            return true;
        }

        // If we found the module, but hadn't imported successfully, might as well try again...
    }

    // Import the module
    PyObject* pLibModule = PyImport_ImportModule(moduleName);
    const char* importName = (importAs == nullptr) ? moduleName : importAs;
    if (pLibModule)
    {
        int error = PyModule_AddObject(mMainModule.ptr(), importName, pLibModule);
        if (!error)
        {
            success = true;
        }
    }

    // Insert this module into our list of modules we've loaded, to avoid reloading in the future.
    mModuleList[strName] = std::make_pair(py::handle(success ? pLibModule : nullptr),
                                          std::string(importName));

    return success;
}

bool PythonEmbedding::fromModuleImport(const char* moduleName, const char* toImport, const char* importAs)
{
    bool success = false;

    // Check.  Have we already loaded this object?  (Could be a module, could be a class, could be a function)
    std::string strName = std::string(moduleName) + std::string(".") + std::string(toImport);
    auto found = mModuleList.find(strName);

    // We found an existing entry for this object
    if (found != mModuleList.end())
    {
        // Check to see if the stored py::handle is non-null.  If so, we've already imported the module.
        auto second = found->second;
        if (second.first.ptr() != nullptr)
        {
            return true;
        }

        // If we found it, but havn't imported successfully, might as well try again...
    }

    // Import the requested object
    py::str tmpName = py::str(toImport);
    const char* importName = (importAs == nullptr) ? toImport : importAs;
    PyObject* pSubModules = PyList_New(0);
    PyList_Append(pSubModules, tmpName.ptr() );
    PyObject* pLibImports = PyImport_ImportModuleEx(moduleName, NULL, NULL, pSubModules);
    PyObject* pObjToImport = NULL;
    if (pLibImports)
    {
        pObjToImport = PyObject_GetAttr(pLibImports, tmpName.ptr());
        if (pObjToImport)
        {
            int error = PyModule_AddObject(mMainModule.ptr(), importName, pObjToImport );
            if (!error)
            {
                success = true;
            }
        }
    }

    // Insert this object into our list of things we've loaded, to avoid reloading in the future.
    mModuleList[strName] = std::make_pair(py::handle(success ? pObjToImport : nullptr),
                                          std::string(importName));

    return success;
}

pybind11::detail::item_accessor PythonEmbedding::operator[] (const char* globalVarName)
{
    return mGlobals[globalVarName];
}

pybind11::detail::item_accessor PythonEmbedding::operator[] (const std::string &globalVarName)
{
    return this->operator[](globalVarName.c_str());
}

pybind11::dict PythonEmbedding::getGlobals(void)
{
    return mGlobals;
}

bool PythonEmbedding::doesGlobalVarExist(const char* globalVarName)
{
    return mGlobals.contains(py::str(std::string(globalVarName)));
}
bool PythonEmbedding::doesGlobalVarExist(const std::string &globalVarName)
{
    return mGlobals.contains(py::str(globalVarName));
}

bool PythonEmbedding::executeFile(const std::string &scriptFile)
{
    return executeFile(scriptFile.c_str());
}

std::string PythonEmbedding::getError(void)
{
    return mLastPythonError;
}

std::string PythonEmbedding::getStdout(void)
{
    return mLastPythonStdout;
}

std::string PythonEmbedding::getStderr(void)
{
    return mLastPythonStderr;
}

void PythonEmbedding::clearPythonState(void)
{
    PyErr_Clear();
    mLastPythonError = std::string("");
    mLastPythonStdout = std::string("");
    mLastPythonStderr = std::string("");
    mLastCostPython = -1.0;
    mLastCostTotal = -1.0;
}

bool PythonEmbedding::executeFile(const char* scriptFile)
{
    return commonExecRoutine(py::str(scriptFile), false);
}

bool PythonEmbedding::executeString(const char* scriptCode)
{
    if (!scriptCode) return false;
    if (scriptCode[0] != '\n')
    {
        return commonExecRoutine(py::str(scriptCode), true);
    }
    else
    {
        return commonExecRoutine(py::str(py::module::import("textwrap").attr("dedent")(scriptCode)), true);
    }
}

bool PythonEmbedding::executeString(const std::string &scriptCode)
{
    return commonExecRoutine(py::str(scriptCode.c_str()), true);
}

bool PythonEmbedding::executeFile(const char* scriptFile, pybind11::dict localNamespace)
{
    return commonExecRoutine(py::str(scriptFile), false, true, localNamespace);
}

bool PythonEmbedding::executeFile(const std::string &scriptFile, pybind11::dict localNamespace)
{
    return commonExecRoutine(py::str(scriptFile.c_str()), false, true, localNamespace);
}

bool PythonEmbedding::executeString(const char* scriptCode, pybind11::dict localNamespace)
{
    return commonExecRoutine(py::str(scriptCode), true, true, localNamespace);
}

bool PythonEmbedding::executeString(const std::string &scriptCode, pybind11::dict localNamespace)
{
    return commonExecRoutine(py::str(scriptCode.c_str()), true, true, localNamespace);
}

bool PythonEmbedding::commonExecRoutine(const pybind11::str &input, bool asString, bool useLocals, pybind11::object locals)
{
    bool success = false;
    clearPythonState();
    std::chrono::high_resolution_clock::time_point t0, t1, t2, t3;
    t0 = std::chrono::high_resolution_clock::now();

    try
    {
        t1 = std::chrono::high_resolution_clock::now();
        if (asString)     // Treat the input as Python command(s) to execute
        {
            py::exec(input, mGlobals, locals);
        }
        else              // Treat the input as a file.  Load and run the code in that file.
        {
            py::eval_file(input, mGlobals, locals);
        }
        t2 = std::chrono::high_resolution_clock::now();

        success = true;
    }
    catch (pybind11::error_already_set &e)
    {
        mLastPythonError = std::string(e.what());
    }
    catch (...)
    {
        mLastPythonError = std::string("<unhandled exception>");
    }

    getRedirectedOutputs();
    t3 = std::chrono::high_resolution_clock::now();

    // Compute costs for the execution.  If no success, leave as defaults (-1 ms)
    if (success)
    {
        std::chrono::duration<double, std::milli> spanTotal = t3 - t0;
        std::chrono::duration<double, std::milli> spanPython = t2 - t1;
        mLastCostTotal = double(spanTotal.count());
        mLastCostPython = double(spanPython.count());
    }

    return success;
}

double PythonEmbedding::lastExecutionTime(bool totalTime)
{
    return totalTime ? mLastCostTotal : mLastCostPython;
}

void PythonEmbedding::checkRedirectionImports(void)
{
    // Redirection requires the 'io' library (we use io.StringIO())
    auto found = mModuleList.find("io");
    if (found == mModuleList.end())
    {
        importModule("io");
    }

    // Redirection requires the 'sys' library (to access/change str.stdout)
    found = mModuleList.find("sys");
    if (found == mModuleList.end())
    {
        importModule("sys");
    }
}

void PythonEmbedding::redirectStdout(bool redirect)
{
    // If we're enabling redirection, ensure Python's io and sys modules are loaded.
    if (redirect)
    {
        checkRedirectionImports();
    }
    if (mRedirectValid)
    {
        stopRedirectingOutputs();
    }

    // Enable/disable redirection
    mRedirectStdout = redirect;
    if (redirect)
    {
        startRedirectingOutputs();
    }
}

void PythonEmbedding::redirectStderr(bool redirect)
{
    // If we're enabling redirection, ensure Python's io and sys modules are loaded.
    if (redirect)
    {
        checkRedirectionImports();
    }
    if (mRedirectValid)
    {
        stopRedirectingOutputs();
    }

    // Enable/disable redirection
    mRedirectStderr = redirect;
    if (redirect)
    {
        startRedirectingOutputs();
    }
}

// This is fairly simplistic and not necessarily entirely robust.
//     Create a StringIO() object; redirect stdout to that via a hardcoded
//     redirect (i.e., assigning sys.stdout to our new object).  A context
//     manager would be "nice" and handle Python exceptions cleanly, but
//     would cause embedded Python code to execute in a different scope; this
//     likely would cause all sorts of unintended results/errors.
//     This does pollute Python's global namespace somewhat, but I named the
//     resources quite explicitly to avoid duplicate names.
// I've attempted to make failure of this redirection have no externally visible
//     consequences... except that redirection will not occur.
void PythonEmbedding::startRedirectingOutputs(void)
{
    // We call this every time we execute; if we don't need it, return.
    if (!mRedirectStdout && !mRedirectStderr) return;

    PyErr_Clear();
    bool success = false;
    try {
        // Probably should combine into a single string with one py::exec() call...  Would save a bit of overhead.
        if (mRedirectStdout)
        {
            py::exec("_fPyEmb_redirectStdout = io.StringIO(); sys.stdout = _fPyEmb_redirectStdout", mGlobals);
        }
        if (mRedirectStderr)
        {
            py::exec("_fPyEmb_redirectStderr = io.StringIO(); sys.stderr = _fPyEmb_redirectStderr", mGlobals);
        }

        success = true;
    } catch (...) { ; }

    // If this function was successful, we've successfully redirected those outputs requested.
    //     -> While failure here does not guarantee we have no successful redirection currently ongoing,
    //        we will be unable to guarantee that everything guarded by mRedirectValid will execute
    //        without exception, so we need to be conservative.  Fortunately, stopping redirection
    //        shouldn't need protection by this flag.
    mRedirectValid = success;

    PyErr_Clear();
}

// In the model where start/stop redirecting outputs are called only upon
//      toggling the redirection, we need a method to grab the current stdio/stderr
//      streams to our internal class structures.  This code also optionally clears
//      the Python streams (the default behavior) so that repeated calls will not
//      return only the incremental, new outputs rather than a cumulative history.
void PythonEmbedding::getRedirectedOutputs(bool clearBuffers)
{
    // We call this every time we execute; if we don't need it, return.
    if (!mRedirectStdout && !mRedirectStderr)
    {
        return;
    }

    // If we want to redirect, but we failed to begin redirecting, we need to abort.
    if (!mRedirectValid)
    {
        return;
    }

    PyErr_Clear();
    bool success = false;
    try {
        std::string stdoutGrabber = clearBuffers ?
            std::string("_fPyEmb_redirectStdoutResult = _fPyEmb_redirectStdout.getvalue(); _fPyEmb_redirectStdout.seek(0); _fPyEmb_redirectStdout.truncate(0)") :
            std::string("_fPyEmb_redirectStdoutResult = _fPyEmb_redirectStdout.getvalue()");
        std::string stderrGrabber = clearBuffers ?
            std::string("_fPyEmb_redirectStderrResult = _fPyEmb_redirectStderr.getvalue(); _fPyEmb_redirectStderr.seek(0); _fPyEmb_redirectStderr.truncate(0)") :
            std::string("_fPyEmb_redirectStderrResult = _fPyEmb_redirectStderr.getvalue()");

        if (mRedirectStdout)
        {
            py::exec(py::str(stdoutGrabber), mGlobals);
        }
        if (mRedirectStderr)
        {
            py::exec(py::str(stderrGrabber), mGlobals);
        }

        success = true;
    } catch (...) { ; }

    if (success)
    {
        if (mRedirectStdout)
        {
            mLastPythonStdout = std::string(mGlobals["_fPyEmb_redirectStdoutResult"].cast<py::str>());
        }
        if (mRedirectStderr)
        {
            mLastPythonStderr = std::string(mGlobals["_fPyEmb_redirectStderrResult"].cast<py::str>());
        }
    }

    PyErr_Clear();
}

// This undoes what occurs in startRedirectingStdout() and (possibly) dumps any remaining
//     redirected data to a C++ std::string.  Again, if designed correctly, failure should
//     have no externally visible consequences, except a lack of redirection.
void PythonEmbedding::stopRedirectingOutputs(bool getOutputs)
{
    // If we have no valid redirection ongoing, we could simply return.  But in order
    //     to give consistent behavior under the maximum # of scenarios, let's execute
    //     our Python to reset stdout and stderr back to their default.  But because
    //     we may have no valid redirection, we can't guarantee the existance of the
    //     redirection variables (e.g., _fPyEmb_redirectStdout), so avoid getting outputs
    if (!mRedirectValid)
    {
        getOutputs = false;
    }

    // If we call stopRedirectingOutput(), treat everything afterwards as if it
    //     has invalid redirection (even if we return early below or catch an exception).
    mRedirectValid = false;

    // We call this every time we execute; if we don't need it, return.
    if (!mRedirectStdout && !mRedirectStderr)
    {
        return;
    }

    PyErr_Clear();

    bool success = false;
    try    {
        std::string stdoutGrabber = getOutputs ?
            std::string("sys.stdout = sys.__stdout__; _fPyEmb_redirectStdoutResult = _fPyEmb_redirectStdout.getvalue()") :
            std::string("sys.stdout = sys.__stdout__");
        std::string stderrGrabber = getOutputs ?
            std::string("sys.stderr = sys.__stderr__; _fPyEmb_redirectStderrResult = _fPyEmb_redirectStderr.getvalue()") :
            std::string("sys.stderr = sys.__stderr__");

        if (mRedirectStdout)
        {
            py::exec(py::str(stdoutGrabber), mGlobals);
        }
        if (mRedirectStderr)
        {
            py::exec(py::str(stderrGrabber), mGlobals);
        }

        success = true;
    } catch (...) {    ; }

    if (success && getOutputs)
    {
        if (mRedirectStdout)
        {
            mLastPythonStdout = std::string(mGlobals["_fPyEmb_redirectStdoutResult"].cast<py::str>());
        }
        if (mRedirectStderr)
        {
            mLastPythonStderr = std::string(mGlobals["_fPyEmb_redirectStderrResult"].cast<py::str>());
        }
    }

    // Clear errors after converting mGlobals[] back to strings, since conversions can throw PyErrors too!
    PyErr_Clear();
}

#endif