param([string]$VivadoBin = 'C:\Xilinx\Vivado\2022.2\bin')
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$runDir = Join-Path $repoRoot ('build/mac-' + [guid]::NewGuid().ToString('N'))
foreach ($tool in @('xvlog', 'xelab', 'xsim')) {
    if (-not (Test-Path (Join-Path $VivadoBin "$tool.bat"))) {
        throw "Missing $tool.bat in $VivadoBin"
    }
}
New-Item -ItemType Directory -Path $runDir | Out-Null
Push-Location $runDir
try {
    & (Join-Path $VivadoBin 'xvlog.bat') --sv (Join-Path $repoRoot 'mac.sv') (Join-Path $PSScriptRoot 'tb_mac.sv')
    if ($LASTEXITCODE -ne 0) { throw 'xvlog failed' }
    & (Join-Path $VivadoBin 'xelab.bat') tb_mac --snapshot mac_test
    if ($LASTEXITCODE -ne 0) { throw 'xelab failed' }
    $output = & (Join-Path $VivadoBin 'xsim.bat') mac_test --runall 2>&1
    $simExit = $LASTEXITCODE
    $output | Tee-Object -FilePath 'runner-output.txt' | Write-Output
    if ($simExit -ne 0 -or -not ($output -match 'PASS: MAC')) {
        throw "MAC simulation failed; evidence: $runDir"
    }
    Write-Output "Evidence: $runDir"
} finally {
    Pop-Location
}
