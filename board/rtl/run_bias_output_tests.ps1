param([string]$VivadoBin = 'C:\Xilinx\Vivado\2022.2\bin')
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$runDir = Join-Path $repoRoot ('build/bias-output-' + [guid]::NewGuid().ToString('N'))
foreach ($tool in @('xvlog', 'xelab', 'xsim')) {
    if (-not (Test-Path (Join-Path $VivadoBin "$tool.bat"))) { throw "Missing $tool" }
}
New-Item -ItemType Directory -Path $runDir | Out-Null
Push-Location $runDir
try {
    $sources = @('requant.sv', 'bias_output.sv', 'tb_bias_output.sv') | ForEach-Object { Join-Path $PSScriptRoot $_ }
    & (Join-Path $VivadoBin 'xvlog.bat') --sv @sources
    if ($LASTEXITCODE -ne 0) { throw 'xvlog failed' }
    & (Join-Path $VivadoBin 'xelab.bat') tb_bias_output --snapshot bias_test
    if ($LASTEXITCODE -ne 0) { throw 'xelab failed' }
    $output = & (Join-Path $VivadoBin 'xsim.bat') bias_test --runall 2>&1
    $simExit = $LASTEXITCODE
    $output | Tee-Object -FilePath 'runner-output.txt' | Write-Output
    if ($simExit -ne 0 -or -not ($output -match 'PASS: bias output all configurations')) {
        throw "Bias output simulation failed; evidence: $runDir"
    }
    Write-Output "Evidence: $runDir"
} finally { Pop-Location }
