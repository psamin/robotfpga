param(
    [string]$VivadoBin = 'C:\Xilinx\Vivado\2022.2\bin',
    [ValidateSet('mac', 'mac_array')][string]$Top = 'mac',
    [ValidateRange(1, 1024)][int]$Lanes = 8,
    [ValidateRange(0.1, 1000.0)][double]$PeriodNs = 10
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$vivado = Join-Path $VivadoBin 'vivado.bat'
if (-not (Test-Path $vivado)) { throw "Missing Vivado: $vivado" }
$runDir = Join-Path $repoRoot ("build/$Top-synth-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $runDir | Out-Null
$period = $PeriodNs.ToString([Globalization.CultureInfo]::InvariantCulture)
Push-Location $runDir
try {
    & $vivado -mode batch -source (Join-Path $PSScriptRoot 'synth_mac.tcl') -tclargs $repoRoot $Top $Lanes $period
    if ($LASTEXITCODE -ne 0) { throw "Vivado failed; evidence: $runDir" }
    foreach ($artifact in @('PASS.txt', 'utilization.rpt', 'timing.rpt', 'synth.dcp')) {
        $path = Join-Path $runDir $artifact
        if (-not (Test-Path $path) -or (Get-Item $path).Length -eq 0) {
            throw "Missing synthesis evidence $artifact in $runDir"
        }
    }
    Get-Content 'PASS.txt'
    Write-Output "Evidence: $runDir"
} finally {
    Pop-Location
}
