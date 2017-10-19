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
    std::string newline = "\n";
    


    if(argc != 4)
    {
        printf("Usage:\nPatchFalcorPropertySheet <Falcor Core Directory> <Current Solution Directory> <Backend [FALCOR_D3D12, FALCOR_VK]>");
        return 1;
    }

    std::string FCD(argv[1]);
    std::string CSD(argv[2]);
    std::string FB(argv[3]);

    // Get a relative path from the Current Solution Directory to the Falcor Core Directory.
    char RelativePath[MAX_PATH];
    PathRelativePathToA(RelativePath, CSD.c_str(), FILE_ATTRIBUTE_DIRECTORY, FCD.c_str(), FILE_ATTRIBUTE_DIRECTORY);

    //  Construct the Solution Directory to Falcor Core Directory.
    std::string SolutionDirectoryToFalcorCoreDirectory = std::string("$(SolutionDir)\\") + std::string(RelativePath);

    //  Read in the Property Sheet.
    std::string PropsSheetPath = FCD + "\\Source\\Falcor.props";
    std::string PropsSheet;
    if(ReadFileToString(PropsSheetPath.c_str(), PropsSheet) == false)
    {
        return 1;
    }

    // Assume that there's already a FALCOR_CORE_DIRECTORY property value. If not, display an error.
    if (PatchGroup(PropsSheet, "FALCOR_CORE_DIRECTORY", SolutionDirectoryToFalcorCoreDirectory) == false)
    {
        return -1;
    } 
    
    //  Assume that there's already a FALCOR_BACKEND property value. If not, display an error.
    if (PatchGroup(PropsSheet, "FALCOR_BACKEND", FB) == false)
    {
        return -1;
    }

    // Output the property sheet. The file is usually read-only, need to modify it
    SetFileAttributesA(PropsSheetPath.c_str(), FILE_ATTRIBUTE_NORMAL);
    std::ofstream PropFile(PropsSheetPath);
    PropFile << PropsSheet;
    PropFile.close();

    return 0;
}