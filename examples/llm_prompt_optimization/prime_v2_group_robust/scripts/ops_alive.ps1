$p = Get-CimInstance Win32_Process -Filter "Name='python.exe'"
function C($re, $not) {
    @($p | Where-Object {
        $_.CommandLine -and ($_.CommandLine -match $re) -and
        (-not $not -or ($_.CommandLine -notmatch $not))
    }).Count
}
$sc = C 'score_s11_scorer2.*gpt4omini'
$a = C 'run_s12_live_loop.*--seeds 42,44' '--dataset mnli'
$b = C 'run_s12_live_loop.*--seeds 43 ' '--dataset mnli'
$m = C 'run_s12_live_loop.*--dataset mnli'
"SC=$sc A=$a B=$b M=$m"
