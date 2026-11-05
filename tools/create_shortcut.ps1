$wsh = New-Object -ComObject WScript.Shell
$desktop = [System.IO.Path]::Combine($env:USERPROFILE, "Desktop")
$sc = $wsh.CreateShortcut((Join-Path $desktop "Check GitHub Updates.lnk"))
$sc.TargetPath = "G:\Strata-3060\Check-GitHub-Updates.bat"
$sc.WorkingDirectory = "G:\Strata-3060"
$sc.IconLocation = "C:\WINDOWS\System32\shell32.dll,238"
$sc.Description = "Check PolyStrata and am2strata for new GitHub commits, forks, and models"
$sc.Save()
Write-Host "Desktop shortcut created: Check GitHub Updates.lnk"
