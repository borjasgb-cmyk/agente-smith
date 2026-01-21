from typing import Callable

import gradio as gr
from fish_speech.i18n import i18n
from tools.agent_supervisor import SUPERVISOR
from tools.audio_utils import (
    build_device_choices,
    list_devices,
    mic_sys_bleed_check,
    sys_quick_check,
)
from tools.n8n_client import emit_event, set_config, test_connection
from tools.voice_collection import (
    clear_dataset,
    data_paths,
    load_manifest_stats,
    start_collection,
    stop_collection,
)
from tools.voices import load_config, load_voices, save_config
from tools.webui.variables import HEADER_MD, TEXTBOX_PLACEHOLDER


def _format_invalid_voices(invalid_voices: list[dict]) -> str:
    if not invalid_voices:
        return "Todas las voces disponibles."
    lines = ["Voces no disponibles:"]
    for voice in invalid_voices:
        name = voice.get("name") or voice.get("id") or "<unknown>"
        reason = voice.get("reason", "motivo desconocido")
        if "not found" in reason or "ref_audio not found" in reason or "model_path not found" in reason:
            reason = "Not configured (missing files)"
        lines.append(f"- {name}: {reason}")
    return "\n".join(lines)


def _format_available_voices(valid_voices: list[dict]) -> str:
    if not valid_voices:
        return "No hay voces disponibles."
    lines = ["Voces disponibles:"]
    for voice in valid_voices:
        lines.append(f"- {voice['name']} ({voice['id']})")
    return "\n".join(lines)


def _format_collect_status(message: str) -> str:
    return f"Estado: {message}"


def _collection_info() -> str:
    count, seconds = load_manifest_stats()
    raw_dir, manifest = data_paths()
    return (
        f"Clips: {count} | Segundos: {int(seconds)}\n"
        f"Raw: {raw_dir}\n"
        f"Manifest: {manifest}"
    )


def _emit_collection_stats(send_events: bool) -> None:
    if not send_events:
        return
    count, seconds = load_manifest_stats()
    emit_event("collection_stats", {"clips": count, "seconds": int(seconds)})


def _refresh_collection_stats(last_count: int, last_seconds: int, send_events: bool):
    count, seconds = load_manifest_stats()
    seconds_int = int(seconds)
    if send_events and (count != last_count or seconds_int != last_seconds):
        emit_event("collection_stats", {"clips": count, "seconds": seconds_int})
    return _collection_info(), count, seconds_int


def _on_collect_toggle(enabled: bool, mic_idx: int | None, send_events: bool):
    if not enabled:
        ok, message = stop_collection()
        if send_events:
            emit_event("collection_stopped", {"message": message})
            _emit_collection_stats(send_events)
        return _format_collect_status(message), _collection_info()

    ok, message = start_collection(mic_idx)
    if send_events:
        emit_event("collection_started", {"mic_idx": mic_idx, "message": message})
        _emit_collection_stats(send_events)
    return _format_collect_status(message), _collection_info()


def _on_collect_clear(confirm: bool, send_events: bool):
    if not confirm:
        return _format_collect_status("Confirma la limpieza"), _collection_info()
    stop_collection()
    clear_dataset()
    if send_events:
        emit_event("collection_cleared", {})
        _emit_collection_stats(send_events)
    return _format_collect_status("Dataset limpio"), _collection_info()


def _start_agent(mic_idx, sys_idx, mode, takeover, device_label: str):
    message = SUPERVISOR.start(mic_idx, sys_idx, mode)
    emit_event("mode_changed", {"mode": mode})
    emit_event("takeover_changed", {"enabled": takeover})
    emit_event("agent_state", {"running": True, "message": message})
    return _render_status(message, takeover, device_label)


def _stop_agent(takeover, device_label: str):
    message = SUPERVISOR.stop()
    emit_event("agent_state", {"running": False, "message": message})
    return _render_status(message, takeover, device_label)


def _refresh_status(takeover, device_label: str):
    return _render_status(None, takeover, device_label)


def _set_takeover(enabled: bool, device_label: str):
    emit_event("takeover_changed", {"enabled": enabled})
    return enabled, *_render_status(None, enabled, device_label)


def _render_status(message: str | None, takeover: bool, device_label: str):
    status = SUPERVISOR.status()
    lines = status["lines"][-12:]
    mic_lines = status["mic_lines"][-5:]
    sys_lines = status["sys_lines"][-5:]
    warnings = status["warnings"][-5:]
    cfg = status["config"]

    running = "ON" if status["running"] else "OFF"
    header = f"Agente: {running} | PID: {status['pid'] or '-'} | Device: {device_label}"
    mode = cfg.get("mode", "-")
    mic_idx = cfg.get("mic_idx", "-")
    sys_idx = cfg.get("sys_idx", "-")
    flags = f"Modo: {mode} | MIC: {mic_idx} | SYS: {sys_idx}"
    state_line = f"Takeover: {'ON' if takeover else 'OFF'}"
    if message:
        header = f"{header} | {message}"

    status_text = "\n".join([header, flags, state_line])
    log_tail = "\n".join(lines) if lines else "(sin logs)"
    mic_text = "\n".join(mic_lines) if mic_lines else "(sin transcripciones MIC)"
    sys_text = "\n".join(sys_lines) if sys_lines else "(sin transcripciones SYS)"
    warn_text = "\n".join(warnings) if warnings else "(sin warnings)"

    return status_text, log_tail, mic_text, sys_text, warn_text


def _run_sys_check(sys_idx: int):
    if sys_idx is None:
        return "SYS_CHECK: sin dispositivo"
    result = sys_quick_check(sys_idx)
    base = f"SYS_CHECK rms={result['rms']:.6f} peak={result['peak']:.6f} verdict={result['verdict']}"
    if result["verdict"] == "SILENT":
        emit_event("warning", {"type": "sys_silent", "sys_idx": sys_idx})
        SUPERVISOR.add_warning(f"SYS silent on device {sys_idx}")
        guidance = (
            "SILENT: 1) Windows Volume Mixer: set Chrome/Aircall output to "
            "'Altavoces (3- VB-Audio Virtual Cable)' o 'CABLE In 16 Ch'. "
            "2) Control Panel > Sound > Recording > CABLE Output > Listen "
            "> 'Listen to this device' > Playback through Razer (BT)."
        )
        return f"{base}\n{guidance}"
    return base


def _run_bleed_check(mic_idx: int, sys_idx: int):
    if mic_idx is None or sys_idx is None:
        return "BLEED_CHECK: faltan dispositivos"
    result = mic_sys_bleed_check(mic_idx, sys_idx)
    base = (
        "BLEED_CHECK "
        f"mic_rms={result['mic_rms']:.6f} sys_rms={result['sys_rms']:.6f} "
        f"ratio={result['ratio']:.3f} corr={result['corr']:.3f} verdict={result['verdict']}"
    )
    if result["verdict"] == "BLEEDING":
        emit_event(
            "warning",
            {"type": "mic_sys_bleed", "mic_idx": mic_idx, "sys_idx": sys_idx},
        )
        SUPERVISOR.add_warning(f"BLEED detected (MIC {mic_idx} -> SYS {sys_idx})")
        guidance = (
            "BLEEDING: desactiva 'Listen to this device' en CABLE Output "
            "o evita rutear el micro al cable."
        )
        return f"{base}\n{guidance}"
    return base


def _apply_n8n_settings(webhook_url, base_url, api_key, open_url, send_events: bool):
    set_config(
        base_url=base_url,
        webhook_url=webhook_url,
        api_key=api_key,
        open_url=open_url,
        enabled=send_events,
    )
    save_config(
        {
            "n8n_base_url": base_url,
            "n8n_webhook_url": webhook_url,
            "n8n_api_key": api_key,
            "n8n_open_url": open_url,
            "n8n_send_events": send_events,
        }
    )
    return _format_n8n_status(webhook_url, base_url)


def _test_n8n_connection(webhook_url, base_url, api_key):
    set_config(webhook_url=webhook_url, base_url=base_url, api_key=api_key)
    ok, msg = test_connection()
    status = "OK" if ok else f"ERROR: {msg}"
    return f"n8n test: {status}"


def _format_n8n_status(webhook_url: str, base_url: str | None = None) -> str:
    if not webhook_url:
        if base_url:
            return "n8n: rest configured (webhook required for events)"
        return "n8n: not configured"
    return f"n8n: webhook ready ({webhook_url})"


def _format_open_link(url: str) -> str:
    if not url:
        return "Sin enlace configurado."
    return f"[Abrir workflow]({url})"


def _on_open_url_change(url: str) -> str:
    save_config({"n8n_open_url": url})
    return _format_open_link(url)


def _on_voice_change(voice_id: str | None, valid_ids: set[str], invalid_ids: set[str]):
    save_config({"voice_id": voice_id})
    emit_event("voice_selected", {"voice_id": voice_id})
    if voice_id in invalid_ids:
        return "Selected voice is not configured yet."
    if voice_id in valid_ids:
        return "Voice ready."
    return "No voice selected."


def _on_mode_change(mode: str):
    emit_event("mode_changed", {"mode": mode})


def build_app(inference_fct: Callable, theme: str = "light", device_label: str = "CPU") -> gr.Blocks:
    try:
        devices = list_devices()
    except Exception:
        devices = []
    mic_choices = build_device_choices(devices, "input")
    sys_choices = build_device_choices(devices, "input")
    mic_label_map = {idx: label for label, idx in mic_choices}
    default_mic = mic_choices[0][1] if mic_choices else None
    default_sys = sys_choices[0][1] if sys_choices else None

    valid_voices, invalid_voices = load_voices()
    voice_map = {voice["id"]: voice for voice in valid_voices}
    valid_ids = set(voice_map.keys())
    invalid_ids = {voice.get("id") for voice in invalid_voices if voice.get("id")}
    voice_choices = [
        (f'{voice["name"]} ({voice["id"]})', voice["id"]) for voice in valid_voices
    ]
    for voice in invalid_voices:
        vid = voice.get("id")
        name = voice.get("name") or vid
        if not vid:
            continue
        voice_choices.append((f"{name} ({vid}) [Not configured]", vid))
    config = load_config()
    default_voice = config.get("voice_id") if config.get("voice_id") in (valid_ids | invalid_ids) else None
    if default_voice in valid_ids:
        voice_status_text = "Voice ready."
    elif default_voice in invalid_ids:
        voice_status_text = "Selected voice is not configured yet."
    else:
        voice_status_text = "No voice selected."

    n8n_webhook = config.get("n8n_webhook_url") or "http://localhost:5678/webhook/Transcripcion"
    n8n_base = config.get("n8n_base_url", "")
    n8n_api_key = config.get("n8n_api_key", "")
    n8n_open = config.get("n8n_open_url", "")
    n8n_send = bool(config.get("n8n_send_events", False))
    set_config(
        base_url=n8n_base,
        webhook_url=n8n_webhook,
        api_key=n8n_api_key,
        open_url=n8n_open,
        enabled=n8n_send,
    )

    with gr.Blocks(theme=gr.themes.Base()) as app:
        gr.Markdown("# AgenteSmith Panel")
        gr.Markdown("Device: " + device_label)
        gr.Markdown(
            "Quick Start:\n"
            "1) Configura audio y prueba SYS en la pestana Listening.\n"
            "2) Inicia el agente en la pestana Agent.\n"
            "3) (Opcional) Configura n8n para recibir eventos."
        )
        quick_cmd = gr.Textbox(
            value=r"C:\Users\Usuario\fish-speech\fishspeech_env\Scripts\python.exe tools\run_webui.py --device cpu",
            label="Run",
            interactive=False,
        )
        copy_btn = gr.Button("Copy command")
        with gr.Accordion("First run checklist", open=False):
            gr.Markdown(
                "- En Windows Volume Mixer, routea Chrome/Aircall a VB-Audio Virtual Cable.\n"
                "- En Control Panel > Sound > Recording, revisa CABLE Output y 'Listen to this device'.\n"
                "- Si usas n8n, configura el webhook en la pestana n8n."
            )
        gr.Markdown(HEADER_MD)

        app.load(
            None,
            None,
            js="() => {const params = new URLSearchParams(window.location.search);if (!params.has('__theme')) {params.set('__theme', '%s');window.location.search = params.toString();}}"
            % theme,
        )

        with gr.Tabs():
            with gr.Tab(label="Agent"):
                with gr.Row():
                    start_btn = gr.Button("Start Agent", variant="primary")
                    stop_btn = gr.Button("Stop Agent")
                    refresh_btn = gr.Button("Refresh Status")

                mode = gr.Radio(
                    label="Listening Mode",
                    choices=["MIC only", "SYS only", "MIC+SYS"],
                    value="MIC+SYS",
                )
                status_panel = gr.Markdown("Agente: OFF")
                warnings_panel = gr.Markdown("(sin warnings)")

                with gr.Row():
                    take_over_btn = gr.Button("Take Over")
                    resume_btn = gr.Button("Resume")
                takeover_state = gr.State(False)

                with gr.Row():
                    mic_lines = gr.Textbox(label="Ultimas transcripciones (MIC)", lines=6)
                    sys_lines = gr.Textbox(label="Ultimas transcripciones (SYS)", lines=6)

                with gr.Accordion("Log tail", open=False):
                    log_tail = gr.Textbox(label="Log tail", lines=10)

            with gr.Tab(label="Listening"):
                mic_dev = gr.Dropdown(
                    label="MIC_DEV_INDEX",
                    choices=mic_choices,
                    value=default_mic,
                )
                sys_dev = gr.Dropdown(
                    label="SYS_DEV_INDEX",
                    choices=sys_choices,
                    value=default_sys,
                )
                gr.Markdown(
                    "SYS routing: Windows Volume Mixer -> Chrome/Aircall output to "
                    "'Altavoces (3- VB-Audio Virtual Cable)' o 'CABLE In 16 Ch'."
                )
                if not mic_choices or not sys_choices:
                    gr.Markdown("No se pudieron listar dispositivos de audio.")
                sys_check_btn = gr.Button("Run quick SYS check (3s)")
                sys_check_out = gr.Markdown("SYS_CHECK: pendiente")
                bleed_check_btn = gr.Button("Run MIC bleed check (speak 3s)")
                bleed_check_out = gr.Markdown("BLEED_CHECK: pendiente")

            with gr.Tab(label="Voices"):
                gr.Markdown(_format_available_voices(valid_voices))
                gr.Markdown(
                    "To enable a voice, place a clean 10-30s WAV at the ref_audio path."
                )
                voice_status = gr.Markdown(voice_status_text)
                voice_select = gr.Dropdown(
                    label="Voz",
                    choices=voice_choices,
                    value=default_voice,
                    interactive=bool(voice_choices),
                )
                with gr.Accordion("Not configured / invalid voices", open=False):
                    gr.Markdown(_format_invalid_voices(invalid_voices))

            with gr.Tab(label="n8n"):
                gr.Markdown("Webhook-only. Configura `N8N_WEBHOOK_URL` para recibir eventos.")
                n8n_send_toggle = gr.Checkbox(label="Send events to n8n", value=n8n_send)
                n8n_webhook_input = gr.Textbox(label="N8N_WEBHOOK_URL", value=n8n_webhook)
                with gr.Accordion("REST optional", open=False):
                    n8n_base_input = gr.Textbox(label="N8N_BASE_URL", value=n8n_base)
                    n8n_api_key_input = gr.Textbox(label="N8N_API_KEY", value=n8n_api_key, type="password")
                n8n_open_input = gr.Textbox(label="Open n8n workflow (optional)", value=n8n_open)
                n8n_open_link = gr.Markdown(_format_open_link(n8n_open))
                n8n_status = gr.Markdown(_format_n8n_status(n8n_webhook, n8n_base))
                n8n_test = gr.Button("Test webhook")

            with gr.Tab(label="Voice Collection"):
                gr.Markdown(
                    "Only record your own microphone with consent. Do not record third parties."
                )
                gr.Markdown(
                    "Recolecta clips del micro en segundo plano. "
                    "Activa el toggle y empieza a hablar."
                )
                collect_toggle = gr.Checkbox(
                    label="Collect samples (MIC)", value=False
                )
                collect_status = gr.Markdown(_format_collect_status("OFF"))
                collect_info = gr.Markdown(_collection_info())
                count_init, seconds_init = load_manifest_stats()
                collect_count_state = gr.State(int(count_init))
                collect_seconds_state = gr.State(int(seconds_init))
                initial_mic_label = f"MIC device: {mic_label_map.get(default_mic, '-')}"
                collect_mic_label = gr.Markdown(initial_mic_label)
                confirm_clear = gr.Checkbox(label="Confirmar limpieza", value=False)
                collect_clear = gr.Button("Limpiar dataset")

        with gr.Accordion("TTS Quick Test", open=False):
            with gr.Row():
                text = gr.Textbox(
                    label=i18n("Input Text"), placeholder=TEXTBOX_PLACEHOLDER, lines=4
                )
                reference_id = gr.Textbox(label=i18n("Reference ID"), lines=1)
                reference_audio = gr.Audio(label=i18n("Reference Audio"), type="filepath")
                reference_text = gr.Textbox(label=i18n("Reference Text"), lines=1)

            with gr.Row():
                max_new_tokens = gr.Slider(
                    label=i18n("Maximum tokens per batch, 0 means no limit"),
                    minimum=0,
                    maximum=2048,
                    value=0,
                    step=8,
                )
                chunk_length = gr.Slider(
                    label=i18n("Iterative Prompt Length, 0 means off"),
                    minimum=100,
                    maximum=400,
                    value=300,
                    step=8,
                )
                top_p = gr.Slider(
                    label="Top-P",
                    minimum=0.7,
                    maximum=0.95,
                    value=0.8,
                    step=0.01,
                )

            with gr.Row():
                repetition_penalty = gr.Slider(
                    label=i18n("Repetition Penalty"),
                    minimum=1,
                    maximum=1.2,
                    value=1.1,
                    step=0.01,
                )
                temperature = gr.Slider(
                    label="Temperature",
                    minimum=0.7,
                    maximum=1.0,
                    value=0.8,
                    step=0.01,
                )
                seed = gr.Number(
                    label="Seed",
                    info="0 means randomized inference, otherwise deterministic",
                    value=0,
                )
                use_memory_cache = gr.Radio(
                    label=i18n("Use Memory Cache"),
                    choices=["on", "off"],
                    value="on",
                )

            with gr.Row():
                generate = gr.Button(value=i18n("Generate"), variant="primary")
                error = gr.HTML(label=i18n("Error Message"))
                audio = gr.Audio(label=i18n("Generated Audio"), type="numpy")

        generate.click(
            inference_fct,
            [
                text,
                voice_select,
                reference_id,
                reference_audio,
                reference_text,
                max_new_tokens,
                chunk_length,
                top_p,
                repetition_penalty,
                temperature,
                seed,
                use_memory_cache,
            ],
            [audio, error],
            concurrency_limit=1,
        )

        start_btn.click(
            lambda mic, sys, m, t: _start_agent(mic, sys, m, t, device_label),
            inputs=[mic_dev, sys_dev, mode, takeover_state],
            outputs=[status_panel, log_tail, mic_lines, sys_lines, warnings_panel],
        )
        stop_btn.click(
            lambda t: _stop_agent(t, device_label),
            inputs=[takeover_state],
            outputs=[status_panel, log_tail, mic_lines, sys_lines, warnings_panel],
        )
        refresh_btn.click(
            lambda t: _refresh_status(t, device_label),
            inputs=[takeover_state],
            outputs=[status_panel, log_tail, mic_lines, sys_lines, warnings_panel],
        )

        sys_check_btn.click(
            _run_sys_check,
            inputs=[sys_dev],
            outputs=[sys_check_out],
        )
        bleed_check_btn.click(
            _run_bleed_check,
            inputs=[mic_dev, sys_dev],
            outputs=[bleed_check_out],
        )
        mic_dev.change(
            lambda idx: f"MIC device: {mic_label_map.get(idx, '-')}",
            inputs=[mic_dev],
            outputs=[collect_mic_label],
        )

        voice_select.change(
            lambda vid: _on_voice_change(vid, valid_ids, invalid_ids),
            inputs=[voice_select],
            outputs=[voice_status],
        )
        mode.change(_on_mode_change, inputs=[mode], outputs=[])
        take_over_btn.click(
            lambda: _set_takeover(True, device_label),
            inputs=[],
            outputs=[takeover_state, status_panel, log_tail, mic_lines, sys_lines, warnings_panel],
        )
        resume_btn.click(
            lambda: _set_takeover(False, device_label),
            inputs=[],
            outputs=[takeover_state, status_panel, log_tail, mic_lines, sys_lines, warnings_panel],
        )

        collect_toggle.change(
            _on_collect_toggle,
            inputs=[collect_toggle, mic_dev, n8n_send_toggle],
            outputs=[collect_status, collect_info],
        )
        collect_clear.click(
            _on_collect_clear,
            inputs=[confirm_clear, n8n_send_toggle],
            outputs=[collect_status, collect_info],
        )

        n8n_open_input.change(
            _on_open_url_change,
            inputs=[n8n_open_input],
            outputs=[n8n_open_link],
        )
        n8n_webhook_input.change(
            _apply_n8n_settings,
            inputs=[n8n_webhook_input, n8n_base_input, n8n_api_key_input, n8n_open_input, n8n_send_toggle],
            outputs=[n8n_status],
        )
        n8n_test.click(
            _test_n8n_connection,
            inputs=[n8n_webhook_input, n8n_base_input, n8n_api_key_input],
            outputs=[n8n_status],
        )
        n8n_send_toggle.change(
            _apply_n8n_settings,
            inputs=[n8n_webhook_input, n8n_base_input, n8n_api_key_input, n8n_open_input, n8n_send_toggle],
            outputs=[n8n_status],
        )

        copy_btn.click(
            fn=None,
            inputs=[quick_cmd],
            outputs=[],
            js="(txt)=>{navigator.clipboard.writeText(txt);}",
        )

        status_timer = gr.Timer(2.0)
        status_timer.tick(
            lambda t: _refresh_status(t, device_label),
            inputs=[takeover_state],
            outputs=[status_panel, log_tail, mic_lines, sys_lines, warnings_panel],
        )

        collect_timer = gr.Timer(3.0)
        collect_timer.tick(
            _refresh_collection_stats,
            inputs=[collect_count_state, collect_seconds_state, n8n_send_toggle],
            outputs=[collect_info, collect_count_state, collect_seconds_state],
        )

    return app
