from typing import Callable

import gradio as gr
import numpy as np

from fish_speech.i18n import i18n
from tools.agent_supervisor import SUPERVISOR
from tools.audio_utils import build_device_choices, list_devices, sys_quick_check
from tools.n8n_client import emit_event, set_config, test_connection
from tools.voice_collection import (
    can_accept_more,
    clear_dataset,
    data_paths,
    load_manifest_stats,
    save_clip,
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
        lines.append(f"- {name}: {reason}")
    return "\n".join(lines)


def _init_collect_state() -> dict:
    return {"buffer": None, "sample_rate": None}


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


def _on_collect_toggle(enabled: bool, state: dict | None, send_events: bool):
    if not enabled:
        if send_events:
            emit_event("collection_stopped", {})
        return _init_collect_state(), _format_collect_status("OFF"), _collection_info()
    if state is None:
        state = _init_collect_state()
    ok, message = can_accept_more()
    if send_events:
        emit_event("collection_started", {})
    return state, _format_collect_status(message), _collection_info()


def _sync_collect_toggle(enabled: bool, state: dict | None, send_events: bool):
    state, status, info = _on_collect_toggle(enabled, state, send_events)
    return enabled, enabled, state, status, info


def _on_mic_stream(audio, enabled: bool, transcript: str, state: dict | None, send_events: bool):
    if not enabled:
        return state or _init_collect_state(), _format_collect_status("OFF"), _collection_info()
    if audio is None:
        return state or _init_collect_state(), _format_collect_status("Sin audio"), _collection_info()

    sample_rate, data = audio
    if data is None:
        return state or _init_collect_state(), _format_collect_status("Sin audio"), _collection_info()

    if state is None:
        state = _init_collect_state()

    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 2:
        data = data.mean(axis=1)

    if state["sample_rate"] is None:
        state["sample_rate"] = int(sample_rate)
        state["buffer"] = data
    else:
        state["buffer"] = (
            data
            if state["buffer"] is None
            else np.concatenate([state["buffer"], data], axis=0)
        )

    clip_seconds = 5.0
    clip_len = int(clip_seconds * state["sample_rate"])
    saved = 0

    while state["buffer"] is not None and state["buffer"].shape[0] >= clip_len:
        ok, message = can_accept_more()
        if not ok:
            return state, _format_collect_status(message), _collection_info()

        clip = state["buffer"][:clip_len]
        state["buffer"] = state["buffer"][clip_len:]
        entry = save_clip(clip, state["sample_rate"], transcript or None)
        saved += 1
        if send_events:
            emit_event("collection_clip_saved", entry)

    ok, message = can_accept_more()
    if saved:
        return state, _format_collect_status(f"Guardados: {saved}. {message}"), _collection_info()
    return state, _format_collect_status(message), _collection_info()


def _on_collect_clear(state: dict | None):
    clear_dataset()
    return _init_collect_state(), _format_collect_status("Dataset limpio"), _collection_info()


def _start_agent(mic_idx, sys_idx, mode, takeover, autopilot):
    message = SUPERVISOR.start(mic_idx, sys_idx, mode)
    emit_event("mode_changed", {"mode": mode})
    emit_event("takeover_changed", {"enabled": takeover})
    emit_event("autopilot_changed", {"enabled": autopilot})
    return _render_status(message, takeover, autopilot)


def _stop_agent(takeover, autopilot):
    message = SUPERVISOR.stop()
    return _render_status(message, takeover, autopilot)


def _refresh_status(takeover, autopilot):
    return _render_status(None, takeover, autopilot)


def _render_status(message: str | None, takeover: bool, autopilot: bool):
    status = SUPERVISOR.status()
    lines = status["lines"][-12:]
    mic_lines = status["mic_lines"][-5:]
    sys_lines = status["sys_lines"][-5:]
    warnings = status["warnings"][-5:]
    cfg = status["config"]

    running = "🟢 ON" if status["running"] else "⚪ OFF"
    header = f"Agente: {running} | PID: {status['pid'] or '-'}"
    mode = cfg.get("mode", "-")
    mic_idx = cfg.get("mic_idx", "-")
    sys_idx = cfg.get("sys_idx", "-")
    flags = f"Modo: {mode} | MIC: {mic_idx} | SYS: {sys_idx}"
    state_line = f"Takeover: {'ON' if takeover else 'OFF'} | Autopilot: {'ON' if autopilot else 'OFF'}"
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
    return (
        f"SYS_CHECK rms={result['rms']:.6f} peak={result['peak']:.6f} verdict={result['verdict']}"
    )


def _apply_n8n_settings(base_url, webhook_url, api_key, open_url, send_events: bool):
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
    if send_events:
        emit_event("collection_status", {"enabled": send_events})
    return _format_n8n_status("Configuracion actualizada")


def _test_n8n_connection(base_url, webhook_url, api_key):
    set_config(base_url=base_url, webhook_url=webhook_url, api_key=api_key)
    ok, msg = test_connection()
    return _format_n8n_status(msg if ok else f"ERROR: {msg}")


def _format_n8n_status(message: str) -> str:
    return f"N8N: {message}"


def _format_open_link(url: str) -> str:
    if not url:
        return "Sin enlace configurado."
    return f"[Abrir workflow]({url})"


def _on_open_url_change(url: str) -> str:
    save_config({"n8n_open_url": url})
    return _format_open_link(url)


def _on_voice_change(voice_id: str | None):
    save_config({"voice_id": voice_id})
    emit_event("voice_changed", {"voice_id": voice_id})


def _on_mode_change(mode: str):
    emit_event("mode_changed", {"mode": mode})


def _on_toggle_change(name: str, enabled: bool):
    emit_event(name, {"enabled": enabled})


def build_app(inference_fct: Callable, theme: str = "light") -> gr.Blocks:
    try:
        devices = list_devices()
    except Exception:
        devices = []
    mic_choices = build_device_choices(devices, "input")
    sys_choices = build_device_choices(devices, "input")
    default_mic = mic_choices[0][1] if mic_choices else None
    default_sys = sys_choices[0][1] if sys_choices else None

    valid_voices, invalid_voices = load_voices()
    voice_map = {voice["id"]: voice for voice in valid_voices}
    voice_choices = [
        (f'{voice["name"]} ({voice["id"]})', voice["id"]) for voice in valid_voices
    ]
    config = load_config()
    default_voice = config.get("voice_id") if config.get("voice_id") in voice_map else None

    n8n_base = config.get("n8n_base_url", "")
    n8n_webhook = config.get("n8n_webhook_url", "")
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
        gr.Markdown("# Agente Smith Control Panel")
        gr.Markdown(
            "Quick Start:\n"
            "1) Inicia el agente (tab Agent)\n"
            "2) Revisa dispositivos y SYS check (tab Listening)\n"
            "3) Abre la URL local para usar el panel"
        )
        gr.Markdown("Comando rapido (CPU):")
        quick_cmd = gr.Textbox(
            value=r"C:\Users\Usuario\fish-speech\fishspeech_env\Scripts\python.exe tools\run_webui.py --device cpu",
            label="Run",
            interactive=False,
        )
        copy_btn = gr.Button("Copy command")
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
                takeover = gr.Checkbox(label="Takeover (mute agent / stop actions)", value=False)
                autopilot = gr.Checkbox(label="Autopilot (agent actions enabled)", value=False)
                collect_toggle_agent = gr.Checkbox(label="Send voice samples", value=False)

                status_panel = gr.Markdown("Agente: ⚪ OFF")
                warnings_panel = gr.Markdown("(sin warnings)")

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
                if not mic_choices or not sys_choices:
                    gr.Markdown("No se pudieron listar dispositivos de audio.")
                sys_check_btn = gr.Button("Run quick SYS check (3s)")
                sys_check_out = gr.Markdown("SYS_CHECK: pendiente")

            with gr.Tab(label="Voices"):
                voice_select = gr.Dropdown(
                    label="Voz",
                    choices=voice_choices,
                    value=default_voice,
                    interactive=bool(voice_choices),
                )
                with gr.Accordion("Voces no disponibles", open=False):
                    gr.Markdown(_format_invalid_voices(invalid_voices))

            with gr.Tab(label="Voice Collection"):
                gr.Markdown(
                    "Recolecta clips del micro en segundo plano. "
                    "Activa el toggle y empieza a grabar en el mic."
                )
                collect_toggle_main = gr.Checkbox(
                    label="Recolectar muestras (MIC)", value=False
                )
                collect_transcript = gr.Textbox(
                    label="Transcripcion (opcional)", lines=1, value=""
                )
                collect_status = gr.Markdown(_format_collect_status("OFF"))
                collect_info = gr.Markdown(_collection_info())
                mic_stream = gr.Audio(
                    label="Mic Stream",
                    sources=["microphone"],
                    type="numpy",
                    streaming=True,
                )
                collect_clear = gr.Button("Limpiar dataset")
                collect_state = gr.State(_init_collect_state())

            with gr.Tab(label="n8n"):
                gr.Markdown(
                    "Nota: para emitir eventos en tiempo real se requiere `N8N_WEBHOOK_URL`. "
                    "Si solo usas BASE_URL + API_KEY, el test funciona pero el envio de eventos requiere webhook."
                )
                n8n_send_toggle = gr.Checkbox(label="Send events to n8n", value=n8n_send)
                n8n_base_input = gr.Textbox(label="N8N_BASE_URL", value=n8n_base)
                n8n_webhook_input = gr.Textbox(label="N8N_WEBHOOK_URL", value=n8n_webhook)
                n8n_api_key_input = gr.Textbox(label="N8N_API_KEY", value=n8n_api_key, type="password")
                n8n_open_input = gr.Textbox(label="Open n8n workflow (optional)", value=n8n_open)
                n8n_open_link = gr.Markdown(_format_open_link(n8n_open))
                n8n_status = gr.Markdown(_format_n8n_status("Sin configurar"))
                n8n_test = gr.Button("Test connection")

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
                generate = gr.Button(value="\U0001f3a7 " + i18n("Generate"), variant="primary")
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
            _start_agent,
            inputs=[mic_dev, sys_dev, mode, takeover, autopilot],
            outputs=[status_panel, log_tail, mic_lines, sys_lines, warnings_panel],
        )
        stop_btn.click(
            _stop_agent,
            inputs=[takeover, autopilot],
            outputs=[status_panel, log_tail, mic_lines, sys_lines, warnings_panel],
        )
        refresh_btn.click(
            _refresh_status,
            inputs=[takeover, autopilot],
            outputs=[status_panel, log_tail, mic_lines, sys_lines, warnings_panel],
        )

        sys_check_btn.click(
            _run_sys_check,
            inputs=[sys_dev],
            outputs=[sys_check_out],
        )

        voice_select.change(
            _on_voice_change,
            inputs=[voice_select],
            outputs=[],
        )
        mode.change(_on_mode_change, inputs=[mode], outputs=[])
        takeover.change(lambda v: _on_toggle_change("takeover_changed", v), inputs=[takeover], outputs=[])
        autopilot.change(lambda v: _on_toggle_change("autopilot_changed", v), inputs=[autopilot], outputs=[])

        collect_toggle_agent.change(
            _sync_collect_toggle,
            inputs=[collect_toggle_agent, collect_state, n8n_send_toggle],
            outputs=[
                collect_toggle_agent,
                collect_toggle_main,
                collect_state,
                collect_status,
                collect_info,
            ],
        )
        collect_toggle_main.change(
            _sync_collect_toggle,
            inputs=[collect_toggle_main, collect_state, n8n_send_toggle],
            outputs=[
                collect_toggle_agent,
                collect_toggle_main,
                collect_state,
                collect_status,
                collect_info,
            ],
        )
        mic_stream.stream(
            _on_mic_stream,
            inputs=[mic_stream, collect_toggle_main, collect_transcript, collect_state, n8n_send_toggle],
            outputs=[collect_state, collect_status, collect_info],
        )
        collect_clear.click(
            _on_collect_clear,
            inputs=[collect_state],
            outputs=[collect_state, collect_status, collect_info],
        )

        n8n_open_input.change(
            _on_open_url_change,
            inputs=[n8n_open_input],
            outputs=[n8n_open_link],
        )
        n8n_test.click(
            _test_n8n_connection,
            inputs=[n8n_base_input, n8n_webhook_input, n8n_api_key_input],
            outputs=[n8n_status],
        )
        n8n_send_toggle.change(
            _apply_n8n_settings,
            inputs=[n8n_base_input, n8n_webhook_input, n8n_api_key_input, n8n_open_input, n8n_send_toggle],
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
            _refresh_status,
            inputs=[takeover, autopilot],
            outputs=[status_panel, log_tail, mic_lines, sys_lines, warnings_panel],
        )

    return app
