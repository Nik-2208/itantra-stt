
[Windows.Media.SpeechSynthesis.SpeechSynthesizer, Windows.Media, ContentType = WindowsRuntime] | Out-Null
 = New-Object Windows.Media.SpeechSynthesis.SpeechSynthesizer
 = [Windows.Media.SpeechSynthesis.SpeechSynthesizer]::AllVoices
 =  | Where-Object { .Language -like '*IN*' } | Select-Object -First 1
if () {
    .Voice = 
    Write-Host 'Selected Indian Voice:' .DisplayName
} else {
    Write-Host 'No IN voice, using:' .Voice.DisplayName
}

 = .SynthesizeTextToStreamAsync('Please help me. Yahaan aag lagi hai, turant police ko bulaayein.').GetAwaiter().GetResult()
 = New-Object Windows.Storage.Streams.DataReader()
 = New-Object byte[] .Size
.LoadAsync(.Size).GetAwaiter().GetResult() | Out-Null
.ReadBytes()
[System.IO.File]::WriteAllBytes('C:\\Users\\Nikhilesh\\Desktop\\iTantra - Models\\itantra_receiver\\test_in_voice.wav', )
