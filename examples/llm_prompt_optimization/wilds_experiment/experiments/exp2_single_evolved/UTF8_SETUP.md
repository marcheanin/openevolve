# UTF-8 Encoding Setup

To avoid `UnicodeEncodeError` in Windows PowerShell, run these commands before starting evolution:

```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
```

Or use the provided script:

```powershell
.\set_utf8.ps1
```

**Note:** These settings are only active for the current PowerShell session. You need to run them each time you open a new terminal.

