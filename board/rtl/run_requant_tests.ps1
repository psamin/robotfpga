param([string]$VivadoBin = 'C:\Xilinx\Vivado\2022.2\bin')
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$runDir = Join-Path $repoRoot ('build/requant-' + [guid]::NewGuid().ToString('N'))
foreach ($tool in @('xvlog', 'xelab', 'xsim')) {
    if (-not (Test-Path (Join-Path $VivadoBin "$tool.bat"))) { throw "Missing $tool" }
}
New-Item -ItemType Directory -Path $runDir | Out-Null
Push-Location $runDir
try {
    & (Join-Path $VivadoBin 'xvlog.bat') --sv (Join-Path $PSScriptRoot 'requant.sv') (Join-Path $PSScriptRoot 'tb_requant.sv')
    if ($LASTEXITCODE -ne 0) { throw 'xvlog failed' }
    & (Join-Path $VivadoBin 'xelab.bat') tb_requant --snapshot requant_test
    if ($LASTEXITCODE -ne 0) { throw 'xelab failed' }
    $output = & (Join-Path $VivadoBin 'xsim.bat') requant_test --runall 2>&1
    $simExit = $LASTEXITCODE
    $output | Tee-Object -FilePath 'runner-output.txt' | Write-Output
    if ($simExit -ne 0 -or -not ($output -match 'PASS: requant [0-9]+ checks')) {
        throw "Requant simulation failed; evidence: $runDir"
    }
    Write-Output "Evidence: $runDir"
} finally { Pop-Location }
