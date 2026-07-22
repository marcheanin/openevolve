param(
    [string]$ApiKey = ""
)

$ErrorActionPreference = "Stop"

function Resolve-ApiKey {
    param([string]$ExplicitKey)
    if ($ExplicitKey -and $ExplicitKey.Trim().Length -gt 0) {
        return $ExplicitKey.Trim()
    }
    if ($env:OPENROUTER_API_KEY -and $env:OPENROUTER_API_KEY.Trim().Length -gt 0) {
        return $env:OPENROUTER_API_KEY.Trim()
    }
    if ($env:OPENAI_API_KEY -and $env:OPENAI_API_KEY.Trim().Length -gt 0) {
        return $env:OPENAI_API_KEY.Trim()
    }
    return ""
}

function Format-Value {
    param($Value)
    if ($null -eq $Value -or "$Value" -eq "") {
        return "<not provided>"
    }
    return "$Value"
}

try {
    $key = Resolve-ApiKey -ExplicitKey $ApiKey
    if (-not $key) {
        throw "API key not found. Set OPENROUTER_API_KEY (preferred) or OPENAI_API_KEY, or pass -ApiKey."
    }

    $headers = @{ Authorization = "Bearer $key" }
    $resp = Invoke-RestMethod -Uri "https://openrouter.ai/api/v1/auth/key" -Headers $headers
    $d = $resp.data

    Write-Host "OpenRouter key info"
    Write-Host "------------------"
    Write-Host ("label           : {0}" -f (Format-Value $d.label))
    Write-Host ("is_free_tier    : {0}" -f (Format-Value $d.is_free_tier))
    Write-Host ("limit           : {0}" -f (Format-Value $d.limit))
    Write-Host ("limit_remaining : {0}" -f (Format-Value $d.limit_remaining))
    Write-Host ("usage           : {0}" -f (Format-Value $d.usage))
    Write-Host ("rate_limit      : {0}" -f (Format-Value $d.rate_limit))
    Write-Host ("limit_reset     : {0}" -f (Format-Value $d.limit_reset))

    Write-Host ""
    Write-Host "Raw JSON (pretty):"
    $resp | ConvertTo-Json -Depth 10
}
catch {
    Write-Error $_.Exception.Message
    exit 1
}
