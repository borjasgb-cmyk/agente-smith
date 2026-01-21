from typing import Callable

import numpy as np

import gradio as gr

from fish_speech.i18n import i18n
from tools.voice_collection import can_accept_more, clear_dataset, save_clip
from tools.webui.variables import HEADER_MD, TEXTBOX_PLACEHOLDER


def _init_collect_state() -> dict:
    return {"buffer": None, "sample_rate": None}


def _format_collect_status(message: str) -> str:
    return f"Estado: {message}"


def _on_collect_toggle(enabled: bool, state: dict | None) -> tuple[dict, str]:
    if not enabled:
        return _init_collect_state(), _format_collect_status("OFF")
    if state is None:
        state = _init_collect_state()
    ok, message = can_accept_more()
    return state, _format_collect_status(message)


def _on_mic_stream(audio, enabled: bool, transcript: str, state: dict | None):
    if not enabled:
        return state or _init_collect_state(), _format_collect_status("OFF")
    if audio is None:
        return state or _init_collect_state(), _format_collect_status("Sin audio")

    sample_rate, data = audio
    if data is None:
        return state or _init_collect_state(), _format_collect_status("Sin audio")

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
            return state, _format_collect_status(message)

        clip = state["buffer"][:clip_len]
        state["buffer"] = state["buffer"][clip_len:]
        save_clip(clip, state["sample_rate"], transcript or None)
        saved += 1

    ok, message = can_accept_more()
    if saved:
        return state, _format_collect_status(f"Guardados: {saved}. {message}")
    return state, _format_collect_status(message)


def _on_collect_clear(state: dict | None):
    clear_dataset()
    return _init_collect_state(), _format_collect_status("Dataset limpio")


def build_app(inference_fct: Callable, theme: str = "light") -> gr.Blocks:
    with gr.Blocks(theme=gr.themes.Base()) as app:
        gr.Markdown(HEADER_MD)

        # Use light theme by default
        app.load(
            None,
            None,
            js="() => {const params = new URLSearchParams(window.location.search);if (!params.has('__theme')) {params.set('__theme', '%s');window.location.search = params.toString();}}"
            % theme,
        )

        # Inference
        with gr.Row():
            with gr.Column(scale=3):
                text = gr.Textbox(
                    label=i18n("Input Text"), placeholder=TEXTBOX_PLACEHOLDER, lines=10
                )

                with gr.Row():
                    with gr.Column():
                        with gr.Tab(label=i18n("Advanced Config")):
                            with gr.Row():
                                chunk_length = gr.Slider(
                                    label=i18n("Iterative Prompt Length, 0 means off"),
                                    minimum=100,
                                    maximum=400,
                                    value=300,
                                    step=8,
                                )

                                max_new_tokens = gr.Slider(
                                    label=i18n(
                                        "Maximum tokens per batch, 0 means no limit"
                                    ),
                                    minimum=0,
                                    maximum=2048,
                                    value=0,
                                    step=8,
                                )

                            with gr.Row():
                                top_p = gr.Slider(
                                    label="Top-P",
                                    minimum=0.7,
                                    maximum=0.95,
                                    value=0.8,
                                    step=0.01,
                                )

                                repetition_penalty = gr.Slider(
                                    label=i18n("Repetition Penalty"),
                                    minimum=1,
                                    maximum=1.2,
                                    value=1.1,
                                    step=0.01,
                                )

                            with gr.Row():
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

                        with gr.Tab(label=i18n("Reference Audio")):
                            with gr.Row():
                                gr.Markdown(
                                    i18n(
                                        "5 to 10 seconds of reference audio, useful for specifying speaker."
                                    )
                                )
                            with gr.Row():
                                reference_id = gr.Textbox(
                                    label=i18n("Reference ID"),
                                    placeholder="Leave empty to use uploaded references",
                                )

                            with gr.Row():
                                use_memory_cache = gr.Radio(
                                    label=i18n("Use Memory Cache"),
                                    choices=["on", "off"],
                                    value="on",
                                )

                            with gr.Row():
                                reference_audio = gr.Audio(
                                    label=i18n("Reference Audio"),
                                    type="filepath",
                                )
                            with gr.Row():
                                reference_text = gr.Textbox(
                                    label=i18n("Reference Text"),
                                    lines=1,
                                    placeholder="在一无所知中，梦里的一天结束了，一个新的「轮回」便会开始。",
                                    value="",
                                )

                        with gr.Tab(label="Voice Collection"):
                            gr.Markdown(
                                "Recolecta clips del micro en segundo plano. "
                                "Activa el toggle y empieza a grabar en el mic."
                            )
                            collect_toggle = gr.Checkbox(
                                label="Recolectar muestras (MIC)", value=False
                            )
                            collect_transcript = gr.Textbox(
                                label="Transcripcion (opcional)", lines=1, value=""
                            )
                            collect_status = gr.Markdown(_format_collect_status("OFF"))
                            mic_stream = gr.Audio(
                                label="Mic Stream",
                                sources=["microphone"],
                                type="numpy",
                                streaming=True,
                            )
                            collect_clear = gr.Button("Limpiar dataset")
                            collect_state = gr.State(_init_collect_state())

            with gr.Column(scale=3):
                with gr.Row():
                    error = gr.HTML(
                        label=i18n("Error Message"),
                        visible=True,
                    )
                with gr.Row():
                    audio = gr.Audio(
                        label=i18n("Generated Audio"),
                        type="numpy",
                        interactive=False,
                        visible=True,
                    )

                with gr.Row():
                    with gr.Column(scale=3):
                        generate = gr.Button(
                            value="\U0001f3a7 " + i18n("Generate"),
                            variant="primary",
                        )

        # Submit
        generate.click(
            inference_fct,
            [
                text,
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

        collect_toggle.change(
            _on_collect_toggle,
            inputs=[collect_toggle, collect_state],
            outputs=[collect_state, collect_status],
        )
        mic_stream.stream(
            _on_mic_stream,
            inputs=[mic_stream, collect_toggle, collect_transcript, collect_state],
            outputs=[collect_state, collect_status],
        )
        collect_clear.click(
            _on_collect_clear,
            inputs=[collect_state],
            outputs=[collect_state, collect_status],
        )

    return app
