param([string]$VivadoBin = 'C:\Xilinx\Vivado\2022.2\bin',
      [ValidateSet('mac', 'mac_array')][string]$Test = 'mac')
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$runDir = Join-Path $repoRoot ("build/$Test-" + [guid]::NewGuid().ToString('N'))
foreach ($tool in @('xvlog', 'xelab', 'xsim')) {
    if (-not (Test-Path (Join-Path $VivadoBin "$tool.bat"))) {
        throw "Missing $tool.bat in $VivadoBin"
    }
}
New-Item -ItemType Directory -Path $runDir | Out-Null
Push-Location $runDir
try {
    $sources = @((Join-Path $repoRoot 'mac.sv'), (Join-Path $PSScriptRoot "tb_$Test.sv"))
    if ($Test -eq 'mac_array') { $sources += Join-Path $PSScriptRoot 'mac_array.sv' }
    & (Join-Path $VivadoBin 'xvlog.bat') --sv @sources
    if ($LASTEXITCODE -ne 0) { throw 'xvlog failed' }
    & (Join-Path $VivadoBin 'xelab.bat') "tb_$Test" --snapshot mac_test
    if ($LASTEXITCODE -ne 0) { throw 'xelab failed' }
    $output = & (Join-Path $VivadoBin 'xsim.bat') mac_test --runall 2>&1
    $simExit = $LASTEXITCODE
    $output | Tee-Object -FilePath 'runner-output.txt' | Write-Output
    $passMarker = if ($Test -eq 'mac_array') { 'PASS: MAC array all configurations' } else { 'PASS: MAC [0-9]+ cycle checks' }
    if ($simExit -ne 0 -or -not ($output -match $passMarker)) {
        throw "MAC simulation failed; evidence: $runDir"
    }
    Write-Output "Evidence: $runDir"
} finally {
    Pop-Location
}
