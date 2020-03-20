param(
[Parameter(Mandatory=$true)][string]$sourceName=$null,
[string]$output="out.exe"
)
$source = "http://packman-bootstrap.s3.amazonaws.com/" + $sourceName
$filename = $output

$triesLeft = 3

do
{
    $triesLeft -= 1
    $req = [System.Net.httpwebrequest]::Create($source)
    $req.cookiecontainer = New-Object System.net.CookieContainer

    try
    {
        Write-Host "Connecting to S3 ..."
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
         
          $triesLeft = 0
        }
    }
    catch
    {
        Write-Host "Error connecting to S3!"
        Write-Host $_.Exception|format-list -force
    }
    finally
    {
        if ($reader)
        {
            $reader.Close()
        }
        if ($writer)
        {
            $writer.Flush()
            $writer.Close()
        }
    }
} while ($triesLeft -gt 0)

