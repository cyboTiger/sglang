import argparse
import base64
import hashlib
import os
from pathlib import Path

from sglang.multimodal_gen import DiffGenerator

from vbench_loader import load_prompt_or_image


MODEL_PRESETS = {
    "wan-1.3b": {
        "model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "model_dir": "wan2.1-1.3b",
        "force_profile_env": "SGLANG_WAN_FORCE_PROFILE_DIR",
        "wan_profile": True,
    },
    "wan-14b": {
        "model_id": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        "model_dir": "wan2.1-14b",
        "force_profile_env": "SGLANG_WAN_FORCE_PROFILE_DIR",
        "wan_profile": True,
    },
    "hunyuan": {
        "model_id": "hunyuanvideo-community/HunyuanVideo",
        "model_dir": "hunyuanvideo",
        "force_profile_env": None,
        "wan_profile": False,
    },
}

MODEL_ID_TO_PRESET = {
    spec["model_id"]: preset_name for preset_name, spec in MODEL_PRESETS.items()
}


def normalize_visible_devices(spec: str) -> str:
    devices = [part.strip() for part in spec.split(",") if part.strip()]
    if not devices:
        raise ValueError("CUDA_VISIBLE_DEVICES cannot be empty.")
    return ",".join(devices)


def get_parallel_degree(visible_devices: str) -> int:
    return len(normalize_visible_devices(visible_devices).split(","))


def to_short_id(text: str, length: int = 4) -> str:
    hash_bytes = hashlib.sha256(text.encode()).digest()
    b64 = base64.urlsafe_b64encode(hash_bytes).decode("ascii")[:-2]
    return b64[:length]


def build_parser(default_model: str = "wan-1.3b") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified inference/profile launcher for WAN and Hunyuan video generation"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=sorted(MODEL_PRESETS),
        default=default_model,
        help="Model preset to use.",
    )
    parser.add_argument(
        "--model-id",
        "--model_id",
        dest="model_id",
        type=str,
        default=None,
        help="Optional explicit Hugging Face model id. Defaults to the selected --model preset.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="t2v_wan_vbench.txt",
        help="Text prompt for video generation.",
    )
    parser.add_argument(
        "--negative_prompt",
        "--negative-prompt",
        dest="negative_prompt",
        type=str,
        default=None,
        help="Negative text prompt to avoid certain features.",
    )
    parser.add_argument(
        "--prompt_source",
        type=str,
        default="T2V_Wan_VBench",
        choices=["prompt", "T2V_Wan_VBench", "T2V_Xingyang_VBench"],
        help="Source of the prompt.",
    )
    parser.add_argument(
        "--prompt_idx", type=int, default=0, help="Index of the prompt."
    )

    parser.add_argument(
        "--height", type=int, default=720, help="Height of the generated video."
    )
    parser.add_argument(
        "--width", type=int, default=1280, help="Width of the generated video."
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=240,
        help="Number of frames in the generated video.",
    )
    parser.add_argument("--fps", type=int, default=24, help="Frames per second.")
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=30,
        help="Number of denoising steps in the generated video.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="/workspace/sglang/outputs",
        help="Output path passed through to generator.generate().",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for generation."
    )
    parser.add_argument(
        "--attention_backend",
        type=str,
        default="sparse_video_gen_2_attn",
        help="Attention backend to use for denoising.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        type=str,
        default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"),
        help="Value to set for CUDA_VISIBLE_DEVICES before generator init.",
    )

    parser.add_argument(
        "--first_layers_fp",
        type=float,
        default=0.03,
        help="Only works for best config. Leave the earliest layers in FP.",
    )
    parser.add_argument(
        "--first_times_fp",
        type=float,
        default=0.2,
        help="Only works for best config. Leave the earliest timesteps in FP.",
    )

    parser.add_argument(
        "--num_q_centroids",
        "--qc",
        type=int,
        default=300,
        help="Number of query centroids for KMEANS_BLOCK.",
    )
    parser.add_argument(
        "--num_k_centroids",
        "--kc",
        type=int,
        default=1000,
        help="Number of key centroids for KMEANS_BLOCK.",
    )
    parser.add_argument(
        "--top_p_kmeans",
        type=float,
        default=0.95,
        help="Top-p threshold for block selection in KMEANS_BLOCK.",
    )
    parser.add_argument(
        "--min_kc_ratio",
        type=float,
        default=0.10,
        help="At least this proportion of key blocks to keep per query block in KMEANS_BLOCK.",
    )
    parser.add_argument(
        "--kmeans_iter_init",
        type=int,
        default=50,
        help="Number of KMeans iterations for initialization in KMEANS_BLOCK.",
    )
    parser.add_argument(
        "--kmeans_iter_step",
        type=int,
        default=2,
        help="Number of KMeans iterations for other diffusion steps in KMEANS_BLOCK.",
    )
    parser.add_argument(
        "--svg2_load_balance",
        choices=["off", "equal", "unequal_asymm"],
        default="off",
        help="SP head load balancing strategy for SVG2. "
        "'off' = contiguous head split + plain symm a2a (baseline). "
        "'equal' = density-driven LPT with equal heads/rank + symm a2a. "
        "'unequal_asymm' = density-driven LPT (variable heads/rank) routed via "
        "asymm pull/push over torch symmetric memory.",
    )
    parser.add_argument(
        "--svg2_cost_model_path",
        type=str,
        default="/workspace/sglang/.cache/sgl_diffusion/svg2_cost_models/",
        help="Root directory for model-specific profiling artifacts.",
    )



    # parser.add_argument("--enable_load_balance", action="store_true")
    # parser.add_argument("--enable_inequal_head", action="store_true")
    # parser.add_argument(
    #     "--enable_asymm_a2a",
    #     "--enable-asymm-a2a",
    #     dest="enable_asymm_a2a",
    #     action="store_true",
    #     help="Enable SVG2 asymmetric all2all pull kernel via environment variable.",
    # )
    # parser.add_argument(
    #     "--asymm_a2a_num_sms",
    #     "--asymm-a2a-num-sms",
    #     dest="asymm_a2a_num_sms",
    #     type=int,
    #     default=None,
    #     help="Optional SM budget for the asymmetric all2all Triton kernel.",
    # )

    parser.add_argument("--enable_profile", action="store_true")
    parser.add_argument(
        "--profile_root",
        type=str,
        default="/workspace/sglang/profile",
        help="Root directory for model-specific profiling artifacts.",
    )
    parser.add_argument(
        "--log_root",
        type=str,
        default="/workspace/sglang/log",
        help="Root directory for model-specific attention timing logs.",
    )
    parser.add_argument(
        "--profile_log_path",
        type=str,
        default="log",
        help="Leaf directory name under the model-specific log root.",
    )
    return parser


def resolve_model_spec(args: argparse.Namespace) -> dict:
    preset_name = args.model
    spec = dict(MODEL_PRESETS[preset_name])

    if args.model_id is not None:
        spec["model_id"] = args.model_id
        spec["model_dir"] = MODEL_ID_TO_PRESET.get(args.model_id, preset_name)
        if args.model_id in MODEL_ID_TO_PRESET:
            model_dir = MODEL_PRESETS[MODEL_ID_TO_PRESET[args.model_id]]["model_dir"]
            spec["model_dir"] = model_dir
    return spec


def configure_env(args: argparse.Namespace, model_spec: dict, prompt: str) -> dict[str, str]:
    visible_devices = normalize_visible_devices(args.cuda_visible_devices)
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"

    if model_spec["wan_profile"]:
        os.environ["SGLANG_WAN_PROFILE_TIMESTEPS"] = "0,10,20,30,40,50"
        os.environ["SGLANG_WAN_PROFILE_LAYERS"] = "0,4,8,12,16"

    degree = get_parallel_degree(visible_devices)
    lb = "1" if args.svg2_load_balance != "off" else "0"
    # ineqh = "1" if args.svg2_load_balance == "unequal_asymm" else "0"
    asymm = "1" if args.svg2_load_balance == "unequal_asymm" else "0"
    qc = args.num_q_centroids
    kc = args.num_k_centroids
    top_p = args.top_p_kmeans
    min_kc_ratio = args.min_kc_ratio
    prompt_tag = f"prompt-{to_short_id(prompt)}-frame{args.num_frames}"
    feature_tag = f"lb{lb}-asymm{asymm}"
    sap_tag = (
        f"qc{qc}-kc{kc}-topp{top_p:.2f}-minkc{min_kc_ratio:.2f}"
    )

    profile_dir = (
        Path(args.profile_root)
        / model_spec["model_dir"]
        / args.attention_backend
        / prompt_tag
        / f"{degree}gpus"
        / feature_tag
        / sap_tag
    )
    log_dir = (
        Path(args.log_root)
        / model_spec["model_dir"]
        / args.attention_backend
        / prompt_tag
        / f"{degree}gpus"
        / feature_tag
        / sap_tag
        / args.profile_log_path
    )

    output_dir = (
        Path(args.output_path)
        / model_spec["model_dir"]
    )

    profile_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    force_profile_env = model_spec["force_profile_env"]
    if force_profile_env is not None and args.enable_profile:
        os.environ[force_profile_env] = str(profile_dir)
    os.environ["ENABLE_PROFILE"] = "1" if args.enable_profile else "0"
    os.environ["PROFILE_LOG_DIR"] = str(log_dir)

    with open(profile_dir / "prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt)

    for log_name in ("attn_time_per_device.log", "dit_block_time_per_device.log"):
        log_path = log_dir / log_name
        with open(log_path, "w", encoding="utf-8"):
            pass

    return {
        "profile_dir": str(profile_dir),
        "log_dir": str(log_dir),
        "output_dir": str(output_dir),
    }


def run(args: argparse.Namespace) -> None:
    model_spec = resolve_model_spec(args)
    model_id = model_spec["model_id"]
    args.svg2_cost_model_path = os.path.join(args.svg2_cost_model_path, f'{MODEL_ID_TO_PRESET[model_id]}.json')

    prompt, _ = load_prompt_or_image(
        args.prompt_source, args.prompt_idx, args.prompt, None
    )
    args.prompt = prompt

    paths = configure_env(args, model_spec, prompt)
    degree = get_parallel_degree(args.cuda_visible_devices)

    generator = DiffGenerator.from_pretrained(
        model_path=model_id,
        attention_backend=args.attention_backend,
        num_gpus=degree,
        sp_degree=degree,
        ulysses_degree=degree,
        svg2_num_q_centroids=args.num_q_centroids,
        svg2_num_k_centroids=args.num_k_centroids,
        svg2_top_p_kmeans=args.top_p_kmeans,
        svg2_min_kc_ratio=args.min_kc_ratio,
        svg2_kmeans_iter_init=args.kmeans_iter_init,
        svg2_kmeans_iter_step=args.kmeans_iter_step,
        svg2_first_layers_fp=args.first_layers_fp,
        svg2_first_times_fp=args.first_times_fp,
        svg2_load_balance=args.svg2_load_balance,
        svg2_cost_model_path=args.svg2_cost_model_path,
    )

    generator.generate(
        sampling_params_kwargs=dict(
            prompt=args.prompt,
            # negative_prompt=args.negative_prompt,
            output_path=str(paths["output_dir"]),
            num_frames=args.num_frames,
            fps=args.fps,
            num_inference_steps=args.num_inference_steps,
            # height=args.height,
            # width=args.width,
            seed=args.seed,
            save_output=True,
        )
    )

    print(f"Model preset: {args.model}")
    print(f"Model id: {model_id}")
    print(f"Profile artifacts: {paths['profile_dir']}")
    print(f"Attention logs: {paths['log_dir']}")


def main(default_model: str = "wan-1.3b") -> None:
    parser = build_parser(default_model=default_model)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
