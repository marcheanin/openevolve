$p = Get-CimInstance Win32_Process -Filter "Name='python.exe'"
function C($re, $not) {
    @($p | Where-Object {
        $_.CommandLine -and ($_.CommandLine -match $re) -and
        (-not $not -or ($_.CommandLine -notmatch $not))
    }).Count
}
$a = C 'run_s12_live_loop.*--seeds 42,44' '--dataset mnli'
$mx = C 'score_s11_scorer2.*gemma_mnli'
"S12=$a MATRIX=$mx"
