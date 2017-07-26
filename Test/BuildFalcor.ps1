[CmdletBinding()]


<#

.SYNOPSIS
Parameterized Build for Falcor.

.DESCRIPTION
Builds Falcor based on the Arguments provided.

.EXAMPLE
(./BuildFalcor.ps1 ../Falcor.sln released3d12)
(./BuildFalcor.ps1 ../Falcor.sln released3d12 clean)
(./BuildFalcor.ps1 ../Falcor.sln debugd3d12)
(./BuildFalcor.ps1 ../Falcor.sln debugd3d12 clean)
(./BuildFalcor.ps1 ../Falcor.sln releasevk)
(./BuildFalcor.ps1 ../Falcor.sln releasevk clean)
(./BuildFalcor.ps1 ../Falcor.sln debugvk)
(./BuildFalcor.ps1 ../Falcor.sln debugvk clean)

#>


param
(   
    [Parameter(Mandatory=$true, Position=0, HelpMessage = "Path to the Target Solution To Build")]
    [ValidateNotNullOrEmpty()]
    [string]$argTargetSolution,

    [Parameter(Mandatory=$true, Position=1, HelpMessage = "Target Build Configuration")]
    [ValidateNotNullOrEmpty()]
    [string]$argBuildConfiguration,

    [Parameter(Mandatory=$true, Position=2, HelpMessage = "Target Build Type - Clean / Build / Rebuild ")]
    [ValidateNotNullOrEmpty()]
    [string]$argBuildType,

    [Parameter(Mandatory=$false, Position=3, HelpMessage = "Target Build Verbosity ")]
    [string]$argBuildVerbosity = "normal"

)



$build_solution_target = $argTargetSolution
$build_verbosity = " /v:${argBuildVerbosity} "
$build_parallel = " /m "
$build_configuration = " /p:Configuration=${argBuildConfiguration}"
$build_target = "/target:${argBuildType}"

$msbuild = "C:\Program Files (x86)\MSBuild\14.0\Bin\msbuild.exe"
$args_escaper = "--%"

Write-Output "Starting Build with Configuration : ${configuration} ."

& $msbuild $args_escaper $build_solution_target $build_configuration $build_target $build_verbosity $build_parallel

if($?)
{
    Write-Output "Build Success!"
}
else
{
    throw "Build Failed!"
}

