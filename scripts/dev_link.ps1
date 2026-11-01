# Links the add-on source folder into Anki's add-ons folder for development.
# A directory junction is used instead of a symlink: it needs no admin rights or developer mode.
# Restart Anki after running it; edits to the source then take effect on the next Anki restart.
$source = Join-Path $PSScriptRoot "..\anki_ai_assistant" | Resolve-Path
$target = Join-Path $env:APPDATA "Anki2\addons21\anki_ai_assistant"

if (Test-Path $target) {
    Write-Output "Already exists: $target"
    exit 0
}
New-Item -ItemType Junction -Path $target -Target $source | Out-Null
Write-Output "Linked $target -> $source"
