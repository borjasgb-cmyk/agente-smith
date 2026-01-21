import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import pyrootutils
import torch
from loguru import logger

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeTTSRequest
from tools.voices import load_config, load_voices
from tools.webui import build_app
from tools.webui.inference import get_inference_wrapper_with_voices

# Make einx happy
os.environ["EINX_FILTER_TRACEBACK"] = "false"


EXPECTED_PYTHON = r"C:\Users\Usuario\fish-speech\fishspeech_env\Scripts\python.exe"


def _require_venv():
    exe = Path(sys.executable).resolve()
    expected = Path(EXPECTED_PYTHON).resolve()
    if exe != expected:
        print("ERROR: run_webui.py must be executed with the venv python.")
        print(f"Expected: {expected}")
        print(f"Current:  {exe}")
        print(
            f"Run: {expected} tools\\run_webui.py",
        )
        raise SystemExit(1)


def _checkpoint_paths() -> tuple[Path, Path]:
    env_dir = os.environ.get("FISH_CHECKPOINT_DIR")
    if env_dir:
        base = Path(env_dir)
        llama = base / "openaudio-s1-mini"
        decoder = base / "openaudio-s1-mini" / "codec.pth"
        return llama, decoder
    return (
        Path("checkpoints/openaudio-s1-mini"),
        Path("checkpoints/openaudio-s1-mini/codec.pth"),
    )


def _check_checkpoints(llama_path: Path, decoder_path: Path) -> None:
    if llama_path.exists() and decoder_path.exists():
        return
    print("ERROR: checkpoints not found.")
    print(f"Llama path:   {llama_path}")
    print(f"Decoder path: {decoder_path}")
    print("Set FISH_CHECKPOINT_DIR or create a symlink:")
    print(r"  mklink /D checkpoints C:\Users\Usuario\fish-speech\checkpoints")
    raise SystemExit(1)


def parse_args():
    parser = ArgumentParser()
    default_llama, default_decoder = _checkpoint_paths()
    parser.add_argument(
        "--llama-checkpoint-path",
        type=Path,
        default=default_llama,
    )
    parser.add_argument(
        "--decoder-checkpoint-path",
        type=Path,
        default=default_decoder,
    )
    parser.add_argument("--decoder-config-name", type=str, default="modded_dac_vq")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--max-gradio-length", type=int, default=0)
    parser.add_argument("--theme", type=str, default="light")
    parser.add_argument("--server-port", type=int, default=7862)

    return parser.parse_args()


if __name__ == "__main__":
    _require_venv()
    args = parse_args()
    _check_checkpoints(args.llama_checkpoint_path, args.decoder_checkpoint_path)
    args.precision = torch.half if args.half else torch.bfloat16

    if os.environ.get("FISH_SMOKE_UI") == "1":
        logger.info("Smoke mode enabled, skipping model load.")

        def _dummy_infer(*_args, **_kwargs):
            return None, "Smoke mode"

        print(
            f"* Running on local URL:  http://127.0.0.1:{args.server_port}",
            flush=True,
        )
        app = build_app(_dummy_infer, args.theme, args.device.upper())
        try:
            app.launch(server_port=args.server_port)
        except OSError as exc:
            msg = str(exc)
            if "address already in use" in msg.lower():
                print(f"ERROR: puerto {args.server_port} en uso.")
            else:
                print(f"ERROR: no se pudo iniciar la UI: {exc}")
            raise SystemExit(1)
        raise SystemExit(0)

    if args.device == "auto":
        if torch.backends.mps.is_available():
            args.device = "mps"
            logger.info("mps is available, running on mps.")
        elif torch.xpu.is_available():
            args.device = "xpu"
            logger.info("XPU is available, running on XPU.")
        elif torch.cuda.is_available():
            args.device = "cuda"
            logger.info("CUDA is available, running on CUDA.")
        else:
            logger.info("CUDA is not available, running on CPU.")
            args.device = "cpu"

    logger.info("Loading Llama model...")
    llama_queue = launch_thread_safe_queue(
        checkpoint_path=args.llama_checkpoint_path,
        device=args.device,
        precision=args.precision,
        compile=args.compile,
    )

    logger.info("Loading VQ-GAN model...")
    decoder_model = load_decoder_model(
        config_name=args.decoder_config_name,
        checkpoint_path=args.decoder_checkpoint_path,
        device=args.device,
    )

    logger.info("Decoder model loaded, warming up...")

    # Create the inference engine
    inference_engine = TTSInferenceEngine(
        llama_queue=llama_queue,
        decoder_model=decoder_model,
        compile=args.compile,
        precision=args.precision,
    )

    if os.environ.get("FISH_SKIP_WARMUP") != "1":
        # Dry run to check if the model is loaded correctly and avoid the first-time latency
        list(
            inference_engine.inference(
                ServeTTSRequest(
                    text="Hello world.",
                    references=[],
                    reference_id=None,
                    max_new_tokens=1024,
                    chunk_length=200,
                    top_p=0.7,
                    repetition_penalty=1.5,
                    temperature=0.7,
                    format="wav",
                )
            )
        )

    logger.info("Warming up done, launching the web UI...")

    # Get the inference function with the immutable arguments
    valid_voices, _invalid_voices = load_voices()
    voice_map = {voice["id"]: voice for voice in valid_voices}
    inference_fct = get_inference_wrapper_with_voices(inference_engine, voice_map)

    app = build_app(
        inference_fct,
        args.theme,
        args.device.upper(),
    )
    try:
        app.launch(server_port=args.server_port)
    except OSError as exc:
        msg = str(exc)
        if "address already in use" in msg.lower():
            print(f"ERROR: puerto {args.server_port} en uso.")
            print("Prueba con otro puerto, por ejemplo:")
            print(
                f"{EXPECTED_PYTHON} tools\\run_webui.py --device {args.device} --server-port 7863"
            )
        else:
            print(f"ERROR: no se pudo iniciar la UI: {exc}")
        raise SystemExit(1)
