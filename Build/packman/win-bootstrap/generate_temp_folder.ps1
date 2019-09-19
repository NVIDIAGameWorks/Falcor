param(
[Parameter(Mandatory=$true)][string]$parentPath=$null
)
[string] $name = [System.Guid]::NewGuid()
$out = Join-Path $parentPath $name
New-Item -ItemType Directory -Path ($out) | Out-Null
Write-Host $out
