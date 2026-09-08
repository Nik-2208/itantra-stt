param(
    [string]$OutFile,
    [string]$Text,
    [string]$TextFile,
    [string]$VoiceToken
)

if ($TextFile -and (Test-Path $TextFile)) {
    $Text = [System.IO.File]::ReadAllText($TextFile, [System.Text.Encoding]::UTF8)
}

if (-not $Text) {
    exit 0
}

# 1. Clean and sanitize text for crystal clear pronunciation with native accent
# Convert Indic danda to period
$Text = $Text -replace '[\u0964\u0965]', '.'
# Remove brackets, quotes, formatting symbols, replacement character
$Text = $Text -replace '[\(\)\[\]\{\}\<\>\"\'\'\‘\’\“\”_~`#*+=/\\|]', ' '
# Remove any unassigned or non-printable control characters
$Text = $Text -replace '[\x00-\x1F\x7F\uFFFD]', ' '
# Normalize whitespace
$Text = $Text -replace '\s+', ' '
$Text = $Text.Trim()

if (-not $Text) {
    exit 0
}

Add-Type -AssemblyName System.Runtime.WindowsRuntime
[Windows.Media.SpeechSynthesis.SpeechSynthesizer, Windows.Media, ContentType = WindowsRuntime] | Out-Null

$synth = New-Object Windows.Media.SpeechSynthesis.SpeechSynthesizer
$voices = [Windows.Media.SpeechSynthesis.SpeechSynthesizer]::AllVoices

$selectedVoice = $null

if ($VoiceToken) {
    $selectedVoice = $voices | Where-Object { $_.Id -like "*$VoiceToken*" -or $_.DisplayName -like "*$VoiceToken*" } | Select-Object -First 1
}

if (-not $selectedVoice) {
    # Default to authentic Indian accent voice (Heera or Ravi en-IN)
    $selectedVoice = $voices | Where-Object { $_.DisplayName -like '*Heera*' -or $_.DisplayName -like '*Ravi*' -or $_.Language -like '*IN*' } | Select-Object -First 1
}

if ($selectedVoice) {
    $synth.Voice = $selectedVoice
}

$asyncOp = $synth.SynthesizeTextToStreamAsync($Text)
$asTaskGeneric = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { 
    $_.Name -eq 'AsTask' -and $_.IsGenericMethod -and $_.GetParameters().Count -eq 1 
} | Select-Object -First 1
$asTask = $asTaskGeneric.MakeGenericMethod([Windows.Media.SpeechSynthesis.SpeechSynthesisStream])
$netTask = $asTask.Invoke($null, @($asyncOp))
$netTask.Wait(15000)
$stream = $netTask.Result

$outFull = [System.IO.Path]::GetFullPath($OutFile)
$fileStream = [System.IO.File]::Create($outFull)
[System.IO.WindowsRuntimeStreamExtensions]::AsStreamForRead($stream).CopyTo($fileStream)
$fileStream.Close()
