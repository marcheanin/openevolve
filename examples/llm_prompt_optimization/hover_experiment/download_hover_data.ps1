Param(
    [string]$DataRoot = "$PSScriptRoot\data"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function New-DirIfMissing([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path | Out-Null
    }
}

function Download-IfMissing([string]$Uri, [string]$OutFile) {
    if (Test-Path -LiteralPath $OutFile) {
        Write-Host "Exists: $OutFile (skip)" -ForegroundColor DarkGray
        return
    }
    Write-Host "Downloading: $Uri -> $OutFile" -ForegroundColor Cyan
    Invoke-WebRequest -Uri $Uri -OutFile $OutFile -UseBasicParsing
    Write-Host "Saved: $OutFile" -ForegroundColor Green
}

Write-Host "Preparing HoVer data under: $DataRoot" -ForegroundColor Yellow

# Base dirs
New-DirIfMissing -Path $DataRoot
$hoverDir = Join-Path $DataRoot "hover"
New-DirIfMissing -Path $hoverDir

# Subdirs inside hover
foreach ($sub in @("doc_retrieval","sent_retrieval","claim_verification","tfidf_retrieved")) {
    New-DirIfMissing -Path (Join-Path $hoverDir $sub)
}

# Files
$baseUrl = "https://nlp.cs.unc.edu/data/hover"

# 1) Wikipedia DB (placed in data root, as in the original bash script)
$wikiDbUrl = "$baseUrl/wiki_wo_links.db"
$wikiDbOut = Join-Path $DataRoot "wiki_wo_links.db"
Download-IfMissing -Uri $wikiDbUrl -OutFile $wikiDbOut

# 2) TF-IDF retrieval jsons
$tfidfDir = Join-Path $hoverDir "tfidf_retrieved"
Download-IfMissing -Uri "$baseUrl/train_tfidf_doc_retrieval_results.json" -OutFile (Join-Path $tfidfDir "train_tfidf_doc_retrieval_results.json")
Download-IfMissing -Uri "$baseUrl/dev_tfidf_doc_retrieval_results.json"   -OutFile (Join-Path $tfidfDir "dev_tfidf_doc_retrieval_results.json")
Download-IfMissing -Uri "$baseUrl/test_tfidf_doc_retrieval_results.json"  -OutFile (Join-Path $tfidfDir "test_tfidf_doc_retrieval_results.json")

Write-Host "Done. Data prepared at: $DataRoot" -ForegroundColor Green


