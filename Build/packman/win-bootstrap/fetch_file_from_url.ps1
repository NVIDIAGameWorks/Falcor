param(
[Parameter(Mandatory=$true)][string]$sourceUrl=$null,
[Parameter(Mandatory=$true)][string]$output=$null
)
$source = $sourceUrl
$filename = $output

$req = [System.Net.httpwebrequest]::Create($source)
$req.cookiecontainer = New-Object System.net.CookieContainer

Write-Host "Connecting to $source ..."
$res = $req.GetResponse()

if($res.StatusCode -eq "OK") {
  Write-Host "Downloading ..."
  [int]$goal = $res.ContentLength
  $reader = $res.GetResponseStream()
  $writer = new-object System.IO.FileStream $fileName, "Create"
  [byte[]]$buffer = new-object byte[] 4096
  [int]$total = [int]$count = 0
  do
  {
    $count = $reader.Read($buffer, 0, $buffer.Length);
    $writer.Write($buffer, 0, $count);
    $total += $count
    if($goal -gt 0) {
        Write-Progress "Downloading $url" "Saving $total of $goal" -id 0 -percentComplete (($total/$goal)*100)
    } else {
        Write-Progress "Downloading $url" "Saving $total bytes..." -id 0
    }
  } while ($count -gt 0)
 
  $reader.Close()
  $writer.Flush()
  $writer.Close()
}

