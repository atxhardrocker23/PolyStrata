$sh = New-Object -ComObject WScript.Shell
$desktop = [System.IO.Path]::Combine($env:USERPROFILE, "Desktop")

Write-Host "Updating desktop shortcuts on: $desktop" -ForegroundColor Cyan

# Define All Active Shortcuts
$shortcuts = @(
    # --- Server Controls ---
    @{
        Name = "Start am2strata.lnk"
        Target = "G:\Strata-3060\am2strata\run-am2strata.bat"
        WorkDir = "G:\Strata-3060\am2strata"
        Icon = "G:\Strata-3060\icons\am2strata.ico"
        Desc = "Start am2strata Native Server & Web Workspace on Port 8088"
    },
    @{
        Name = "Stop am2strata.lnk"
        Target = "G:\Strata-3060\am2strata\stop-am2strata.bat"
        WorkDir = "G:\Strata-3060\am2strata"
        Icon = "G:\Strata-3060\icons\stop_strata.ico"
        Desc = "Stop am2strata and release all GPU VRAM"
    },
    # --- LINE Messenger Gateway ---
    @{
        Name = "Start LINE Gateway.lnk"
        Target = "C:\WINDOWS\System32\WindowsPowerShell\v1.0\powershell.exe"
        Args = "-NoExit -ExecutionPolicy Bypass -File `"G:\Start-PolyStrata-LINE.ps1`""
        WorkDir = "G:\"
        Icon = "G:\Strata-3060\icons\line_gateway.ico"
        Desc = "Start Cloudflare HTTPS Tunnel and Hermes LINE Messenger Gateway"
    },
    @{
        Name = "Stop LINE Gateway.lnk"
        Target = "C:\WINDOWS\system32\cmd.exe"
        Args = "/c `"G:\Stop-PolyStrata-LINE.bat`""
        WorkDir = "G:\"
        Icon = "G:\Strata-3060\icons\stop_line.ico"
        Desc = "Stop LINE Messenger Gateway and Cloudflare Tunnel"
    },
    # --- Model Switcher ---
    @{
        Name = "Switch Model in VRAM.lnk"
        Target = "C:\WINDOWS\System32\WindowsPowerShell\v1.0\powershell.exe"
        Args = "-NoExit -ExecutionPolicy Bypass -File `"G:\Switch-PolyStrata-Model.ps1`""
        WorkDir = "G:\"
        Icon = "G:\Strata-3060\icons\polystrata.ico"
        Desc = "Interactive CLI to hot-swap active model in VRAM"
    },
    # --- Active Models ---
    @{
        Name = "Tiel Coder NanoPlus (am2strata).lnk"
        Target = "G:\Strata-3060\am2strata\launchers\start-tiel-coder-nanoplus.bat"
        WorkDir = "G:\Strata-3060\am2strata"
        Icon = "G:\Strata-3060\icons\tiel_nanoplus.ico"
        Desc = "Launch Tiel Coder 35B APEX-I NanoPlus (TurboQuant 480B/tok, 62+ tok/s)"
    },
    @{
        Name = "Qwen Flash Coder (am2strata).lnk"
        Target = "G:\Strata-3060\am2strata\launchers\start-qwen-flash.bat"
        WorkDir = "G:\Strata-3060\am2strata"
        Icon = "G:\Strata-3060\icons\qwen_flash_nanoplus.ico"
        Desc = "Launch Qwen Coder 35B / Tiel Coder APEX-I NanoPlus (TurboQuant)"
    },
    @{
        Name = "Groot 65B GyroM (am2strata).lnk"
        Target = "G:\Strata-3060\am2strata\launchers\start-groot-65b-gyrom.bat"
        WorkDir = "G:\Strata-3060\am2strata"
        Icon = "G:\Strata-3060\icons\groot_gyrom.ico"
        Desc = "Launch Groot 65B-A6B GyroM Turbo (TQ2_T Trellis MMQ Overlay)"
    },
    @{
        Name = "Flint 40B NanoPlus (am2strata).lnk"
        Target = "G:\Strata-3060\am2strata\launchers\start-flint-nanoplus.bat"
        WorkDir = "G:\Strata-3060\am2strata"
        Icon = "G:\Strata-3060\icons\flint_nanoplus.ico"
        Desc = "Launch Flint 40B APEX-I NanoPlus V2.1 (100% VRAM)"
    },
    @{
        Name = "Gemma 4 NanoPlus Abliterated (am2strata).lnk"
        Target = "G:\Strata-3060\am2strata\launchers\start-gemma4-nanoplus.bat"
        WorkDir = "G:\Strata-3060\am2strata"
        Icon = "G:\Strata-3060\icons\gemma4_abl.ico"
        Desc = "Launch Gemma 4 26B-A4B IT Abliterated NanoPlus (78+ tok/s, Vision)"
    },
    @{
        Name = "Huihui Qwen 3.8 27B NanoPlus (am2strata).lnk"
        Target = "G:\Strata-3060\am2strata\launchers\start-huihui-qwen38-nanoplus.bat"
        WorkDir = "G:\Strata-3060\am2strata"
        Icon = "G:\Strata-3060\icons\qwen_flash_nanoplus.ico"
        Desc = "Launch Huihui Qwen 3.8 27B Abliterated VAL-APEX-I NanoPlus V3"
    },
    @{
        Name = "LFM 2.5 NanoPlus (am2strata).lnk"
        Target = "G:\Strata-3060\am2strata\launchers\start-lfm-nanoplus.bat"
        WorkDir = "G:\Strata-3060\am2strata"
        Icon = "G:\Strata-3060\icons\lfm_nanoplus.ico"
        Desc = "Launch LFM 2.5 24B-A2B APEX-I NanoPlus"
    },
    @{
        Name = "Ornith NanoPlus (am2strata).lnk"
        Target = "G:\Strata-3060\am2strata\launchers\start-ornith-nanoplus.bat"
        WorkDir = "G:\Strata-3060\am2strata"
        Icon = "G:\Strata-3060\icons\ornith_nanoplus.ico"
        Desc = "Launch Ornith 1.5 35B-A3B APEX-I NanoPlus (Vision)"
    }
)

foreach ($s in $shortcuts) {
    $p = Join-Path $desktop $s.Name
    $lnk = $sh.CreateShortcut($p)
    $lnk.TargetPath = $s.Target
    if ($s.Args) { $lnk.Arguments = $s.Args }
    $lnk.WorkingDirectory = $s.WorkDir
    if ($s.Icon -and (Test-Path $s.Icon)) {
        $lnk.IconLocation = $s.Icon
    }
    $lnk.Description = $s.Desc
    $lnk.Save()
    Write-Host "Created/Updated: $($s.Name)" -ForegroundColor Green
}

Write-Host "`nAll Desktop shortcuts successfully updated!" -ForegroundColor Cyan
