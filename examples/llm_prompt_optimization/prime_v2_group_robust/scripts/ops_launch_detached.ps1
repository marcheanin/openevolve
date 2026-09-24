param(
    [string[]]$Jobs = @('sc', 'a', 'b', 'm'),
    # Потолок расхода КЛЮЧА с момента старта процесса (не расхода самого процесса).
    # Ставить как (желаемый общий потолок) - (текущий расход ключа).
    [double]$Cap = 32
)

$root = 'c:\Users\march\things\mipt\AlphaEvolveProject\openevolve\examples\llm_prompt_optimization\prime_v2_group_robust'
$py = 'c:\Users\march\things\mipt\AlphaEvolveProject\.venv\Scripts\python.exe'
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'

$loop = 'scripts/run_s12_live_loop.py --methods ape,evoprompt_de,gepa --protocols soft_min,hard_min,global'
$defs = @{
    sc  = @{ args = "scripts/score_s11_scorer2.py --tag gpt4omini --model openai/gpt-4o-mini --provider OpenAI --max-spend $Cap --max-parallel 16"; log = 'results/S11_protocol_matrix/scorer2_gpt4omini/run.log'; stop = 'results/S11_protocol_matrix/scorer2_gpt4omini/STOP' }
    a   = @{ args = "$loop --seeds 42,44 --max-spend $Cap --max-parallel 4"; log = 'results/S12_live_loop/run_A.log'; stop = 'results/S12_live_loop/STOP' }
    b   = @{ args = "$loop --seeds 43 --max-spend $Cap --max-parallel 4"; log = 'results/S12_live_loop/run_B.log'; stop = 'results/S12_live_loop/STOP' }
    m   = @{ args = "$loop --dataset mnli --seeds 42,43,44 --max-spend $Cap --max-parallel 4"; log = 'results/S13_mnli_loop/run_C.log'; stop = 'results/S13_mnli_loop/STOP' }
    m42 = @{ args = "$loop --dataset mnli --seeds 42 --max-spend $Cap --max-parallel 4"; log = 'results/S13_mnli_loop/run_m42.log'; stop = 'results/S13_mnli_loop/STOP' }
    m43 = @{ args = "$loop --dataset mnli --seeds 43 --max-spend $Cap --max-parallel 4"; log = 'results/S13_mnli_loop/run_m43.log'; stop = 'results/S13_mnli_loop/STOP' }
    m44 = @{ args = "$loop --dataset mnli --seeds 44 --max-spend $Cap --max-parallel 4"; log = 'results/S13_mnli_loop/run_m44.log'; stop = 'results/S13_mnli_loop/STOP' }
}

foreach ($j in $Jobs) {
    $d = $defs[$j]
    if (-not $d) { "неизвестное задание: $j"; continue }
    $stop = Join-Path $root $d.stop
    if (Test-Path $stop) { Remove-Item $stop -Force }
    $inner = '"' + $py + '" ' + $d.args + ' >> "' + $d.log + '" 2>&1'
    $p = Start-Process -FilePath 'cmd.exe' -ArgumentList ('/c "' + $inner + '"') -WorkingDirectory $root -WindowStyle Hidden -PassThru
    "$j запущен: cmd pid $($p.Id), потолок $Cap"
}
