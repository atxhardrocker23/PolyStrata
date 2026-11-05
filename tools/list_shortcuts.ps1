$sh = New-Object -ComObject WScript.Shell
Get-ChildItem "C:\Users\Kenny\Desktop\*.lnk" | ForEach-Object {
    $lnk = $sh.CreateShortcut($_.FullName)
    Write-Output ("NAME: " + $_.Name)
    Write-Output ("  TARGET: " + $lnk.TargetPath)
    Write-Output ("  ARGS:   " + $lnk.Arguments)
    Write-Output ""
}
