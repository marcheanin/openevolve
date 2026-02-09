# PowerShell script to set UTF-8 encoding for console
# Run this before starting the evolution to avoid UnicodeEncodeError

# Set console output encoding to UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8

# Set PowerShell output encoding
$OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "UTF-8 encoding set for console. You can now run the evolution script." -ForegroundColor Green

