param([string]$VivadoBin = 'C:\Xilinx\Vivado\2022.2\bin')
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$runRoot = Join-Path $repoRoot ('build/mac-faults-' + [guid]::NewGuid().ToString('N'))
$cases = @(
    @{ Name = 'baseline-mac'; Test = 'mac' },
    @{ Name = 'baseline-array'; Test = 'mac_array' },
    @{ Name = 'zero-extension'; Test = 'mac'; File = 'mac.sv';
       Old = '{{16{product[15]}}, product}'; New = "{16'b0, product}" },
    @{ Name = 'clear-priority'; Test = 'mac'; File = 'mac.sv';
       Old = "else if (clear)`n            acc <= 32'sd0;`n        else if (enable)`n            acc <= acc + extended_product;";
       New = "else if (enable)`n            acc <= acc + extended_product;`n        else if (clear)`n            acc <= 32'sd0;" },
    @{ Name = 'ignored-enable'; Test = 'mac'; File = 'mac.sv';
       Old = 'else if (enable)'; New = "else if (1'b1)" },
    @{ Name = 'shared-lane-enable'; Test = 'mac_array'; File = 'board/rtl/mac_array.sv';
       Old = '.enable(enable[lane])'; New = '.enable(enable[0])' }
)
foreach ($case in $cases) {
    $caseRoot = Join-Path $runRoot $case.Name
    $rtlDir = Join-Path $caseRoot 'board/rtl'
    New-Item -ItemType Directory -Path $rtlDir -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $repoRoot 'mac.sv') -Destination $caseRoot
    foreach ($file in @('mac_array.sv', 'tb_mac.sv', 'tb_mac_array.sv', 'run_mac_tests.ps1')) {
        Copy-Item -LiteralPath (Join-Path $PSScriptRoot $file) -Destination $rtlDir
    }
    if ($case.File) {
        $target = Join-Path $caseRoot $case.File
        $source = [IO.File]::ReadAllText($target).Replace("`r`n", "`n")
        if (-not $source.Contains($case.Old)) { throw "Mutation anchor missing: $($case.Name)" }
        $source = $source.Replace($case.Old, $case.New)
        [IO.File]::WriteAllText($target, $source)
    }
    $logPath = Join-Path $caseRoot 'harness-output.txt'
    # Expected child-process failures must not abort before their logs are checked.
    $ErrorActionPreference = 'Continue'
    try {
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $rtlDir 'run_mac_tests.ps1') -VivadoBin $VivadoBin -Test $case.Test *> $logPath
        $childExit = $LASTEXITCODE
    } finally { $ErrorActionPreference = 'Stop' }
    $log = [IO.File]::ReadAllText($logPath)
    if ($case.File) {
        # Compile/tool errors are not evidence that the scoreboard caught a fault.
        if ($childExit -eq 0 -or $log -notmatch 'Fatal: (check=|LANES=)' -or
            $log -match 'ERROR: \[(VRFC|XSIM)') {
            throw "Fault was not detected by a testbench mismatch: $($case.Name); $logPath"
        }
        Write-Output "DETECTED: $($case.Name)"
    } else {
        if ($childExit -ne 0 -or $log -notmatch 'PASS: MAC') {
            throw "Baseline failed: $($case.Name); $logPath"
        }
        Write-Output "BASELINE PASS: $($case.Name)"
    }
}
"PASS: two baselines and four injected faults; evidence: $runRoot" |
    Tee-Object -FilePath (Join-Path $runRoot 'summary.txt') | Write-Output
