/***************************************************************************
Copyright (c) 2014, NVIDIA CORPORATION.  All rights reserved.
***************************************************************************/
#include <fstream>
#include <string>
#include <stdio.h>
#include <sstream>
#include <algorithm>
#include <Shlwapi.h>
#include <vector>

bool ReadFileToString(const char* Filename, std::string& Str)
{
    std::ifstream File(Filename, std::ios_base::in);
    if(File.fail())
    {
        printf("Failed to open input file %s", Filename);
        return false;
    }

    std::stringstream StrStream;
    StrStream << File.rdbuf();

    Str = StrStream.str();
    File.close();

    return true;
}

bool PatchGroup(std::string& PropSheet, const std::string& Group, const std::string& GroupValue)
{
    const std::string GroupStart = "<" + Group + ">";
    const std::string GroupEnd = "</" + Group + ">";
    auto Start = PropSheet.find(GroupStart);
    auto End = PropSheet.find(GroupEnd);
    if(Start == std::string::npos || End == std::string::npos)
    {
        printf("Can't find a \"%s\" section in the file. This is probably because someone deleted it for the property sheet. Revert the changes and try again.\n", GroupStart.c_str());
        return false;
    }

    if(Start >= End)
    {
        printf("The property sheet is corrupted. %s can't appear before %s\n", GroupStart.c_str(), GroupEnd.c_str());
        return false;
    }

    Start += GroupStart.size();
    PropSheet.replace(Start, End - Start, GroupValue);
    return true;
}

int main(int argc, char* argv[])
{
    const std::string FalcorProject("falcor.vcxproj");

    if(argc != 4)
    {
        printf("Usage:\nPatchFalcorPropertySheet <Solution file> <property sheet> <Backend [FALCOR_DX11, FALCOR_GL]>");
        return 1;
    }

    std::string ProjFile;
    if(ReadFileToString(argv[1], ProjFile) == false)
    {
        return 1;
    }
    std::transform(ProjFile.begin(), ProjFile.end(), ProjFile.begin(), ::tolower);

    size_t ProjLoc = ProjFile.find(FalcorProject);
    if(ProjLoc == std::string::npos)
    {
        printf("Can't find the falcor.vcxproj in the file. Are you sure that this is a valid falcor solution project?");
        return 1;
    }

    size_t Begin = ProjFile.find_last_of("\"", ProjLoc);
    if(Begin == std::string::npos)
    {
        printf("Found the project file, but it's not surrounded by quotation mark");
        return 1;
    }

    // Got the relative path.
    ProjFile = ProjFile.substr(Begin + 1, ProjLoc - Begin - 1);

    // Find the solution path
    std::vector<char> cVec(strlen(argv[1]) + 1, 0);
    char* C = cVec.data();
    memcpy(C, argv[1], strlen(argv[1]));

    if(PathRemoveFileSpecA(C) == FALSE)
    {
        printf("Unexpected error. Remove file spec failed");
        return 1;
    }

    std::string Fullpath = std::string(C) + '\\' + ProjFile + "\\..\\";

    // Get a relative path
    char relPath[MAX_PATH];
    PathRelativePathToA(relPath, C, FILE_ATTRIBUTE_DIRECTORY, Fullpath.c_str(), FILE_ATTRIBUTE_DIRECTORY);

    Fullpath = std::string("$(SolutionDir)\\") + std::string(relPath);
    std::string PropSheet;
    if(ReadFileToString(argv[2], PropSheet) == false)
    {
        return 1;
    }

    // I assume that there's already a FALCOR_PROJECT_DIR property value. If not, display an error
    if(PatchGroup(PropSheet, "FALCOR_PROJECT_DIR", Fullpath) == false)
        return -1;
    
    // Patch the backend
    std::string Backend(argv[3]);

    if(PatchGroup(PropSheet, "FALCOR_BACKEND", Backend) == false)
        return -1;

    // Output the property sheet. The file is usually read-only, need to modify it
    SetFileAttributesA(argv[2], FILE_ATTRIBUTE_NORMAL);
    std::ofstream PropFile(argv[2]);
    PropFile << PropSheet;
    PropFile.close();

    return 0;
}