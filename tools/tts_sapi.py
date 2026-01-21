from __future__ import annotations

import subprocess
from pathlib import Path


def sapi_speak_to_wav(text: str, output_path: str, voice_hint: str = "es-ES") -> tuple[bool, str]:
    if not text.strip():
        return False, "empty text"
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    escaped_text = text.replace('"', '\\"')
    ps = (
        "Add-Type -AssemblyName System.Speech\n"
        "$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer\n"
        "$voice = $null\n"
        "foreach ($v in $synth.GetInstalledVoices()) {\n"
        "  $name = $v.VoiceInfo.Name\n"
        "  $culture = $v.VoiceInfo.Culture.Name\n"
        f"  if ($culture -like \"*{voice_hint}*\" -or $name -like \"*Spanish*\" -or $name -like \"*es-ES*\") {{\n"
        "    $voice = $name\n"
        "    break\n"
        "  }\n"
        "}\n"
        "if ($voice) { $synth.SelectVoice($voice) }\n"
        f"$synth.SetOutputToWaveFile(\"{str(out)}\")\n"
        f"$synth.Speak(\"{escaped_text}\")\n"
        "$synth.Dispose()\n"
    )
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            check=True,
            capture_output=True,
            text=True,
        )
        return True, "ok"
    except Exception as exc:
        return False, str(exc)
