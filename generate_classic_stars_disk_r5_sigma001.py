from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import run_deblurring_study as study
import run_manual_kernel_charts as manual


OUT_DIR = Path("outputs/figures/classic_stars_disk_r5_sigma001")
IMAGE_SIZE = (256, 256)
NOISE_SIGMA = 0.01

WIENER_GRID = [0.001, 0.003, 0.005, 0.01, 0.02]
RL_GRID = [20, 40, 60, 80, 100]
TV_GRID = [0.001, 0.003, 0.005, 0.008, 0.01]


def generate_classic_star_field(seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = np.zeros(IMAGE_SIZE, dtype=np.float32)

    num_stars = 40
    for _ in range(num_stars):
        y, x = rng.integers(5, 250, size=2)
        brightness = rng.uniform(0.7, 1.0)
        image[y, x] = brightness

        # Make some stars slightly larger than one pixel.
        if rng.random() > 0.5:
            image[y + 1, x] = max(image[y + 1, x], brightness * 0.8)
            image[y, x + 1] = max(image[y, x + 1], brightness * 0.8)

        if rng.random() > 0.75:
            image[y - 1, x] = max(image[y - 1, x], brightness * 0.5)
            image[y, x - 1] = max(image[y, x - 1], brightness * 0.5)

    return np.clip(image, 0.0, 1.0).astype(np.float32)


def best_restore(case: study.TripleCase) -> tuple[dict[str, float | int], dict[str, np.ndarray], dict[str, dict[str, float]]]:
    best_params: dict[str, float | int] = {}
    restored: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, float]] = {}

    best_score = None
    best_img = None
    best_metric = None
    best_param = None
    for param in WIENER_GRID:
        candidate = study.wiener_restore(case, float(param))
        metric = study.psnr_ssim_mae(case.image.array, candidate)
        score = (metric["ssim"], metric["psnr"])
        if best_score is None or score > best_score:
            best_score, best_img, best_metric, best_param = score, candidate, metric, param
    best_params["wiener"] = float(best_param)
    restored["wiener"] = best_img
    metrics["wiener"] = best_metric

    best_score = None
    best_img = None
    best_metric = None
    best_param = None
    for param in RL_GRID:
        candidate = study.richardson_lucy_restore(case, int(param))
        metric = study.psnr_ssim_mae(case.image.array, candidate)
        score = (metric["ssim"], metric["psnr"])
        if best_score is None or score > best_score:
            best_score, best_img, best_metric, best_param = score, candidate, metric, param
    best_params["richardson_lucy"] = int(best_param)
    restored["richardson_lucy"] = best_img
    metrics["richardson_lucy"] = best_metric

    best_score = None
    best_img = None
    best_metric = None
    best_param = None
    for param in TV_GRID:
        candidate = study.tv_montalto_restore(case, float(param))
        metric = study.psnr_ssim_mae(case.image.array, candidate)
        score = (metric["ssim"], metric["psnr"])
        if best_score is None or score > best_score:
            best_score, best_img, best_metric, best_param = score, candidate, metric, param
    best_params["tv_montalto"] = float(best_param)
    restored["tv_montalto"] = best_img
    metrics["tv_montalto"] = best_metric

    return best_params, restored, metrics


def save_figure(original: np.ndarray, blurred: np.ndarray, restored: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.6))
    panels = [
        ("Original", original),
        ("Blurred", blurred),
        ("Wiener", restored["wiener"]),
        ("Richardson-Lucy", restored["richardson_lucy"]),
        ("TV", restored["tv_montalto"]),
    ]
    for ax, (title, image) in zip(axes, panels):
        ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=18)
        ax.axis("off")
    fig.suptitle("Synthetic classic stars: disk_R5, sigma=0.01", fontsize=18)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    study.ensure_dirs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    original = generate_classic_star_field()
    kernel = next(k for k in manual.build_manual_kernels() if k.kernel_id == "disk_R5")

    blurred_clean = np.clip(
        study.blur_with_pyolimp(original, kernel.psf.shifted), 0.0, 1.0
    ).astype(np.float32)
    rng = np.random.default_rng(study.stable_seed("classic_stars", "disk_R5", NOISE_SIGMA))
    blurred_noisy = np.clip(
        blurred_clean + rng.normal(0.0, NOISE_SIGMA, size=blurred_clean.shape).astype(np.float32),
        0.0,
        1.0,
    ).astype(np.float32)

    image = study.PreparedImage(
        image_id="classic_stars",
        name="Synthetic classic stars",
        source_url="synthetic",
        page_url="synthetic",
        array=original,
    )
    case = study.TripleCase(
        case_id="classic_stars_disk_r5_sigma001",
        image=image,
        psf=kernel.psf,
        noise_sigma=NOISE_SIGMA,
        blur_clean=blurred_clean,
        blur_noisy=blurred_noisy,
        fft_blur_noisy=np.fft.fft2(blurred_noisy),
        triple_dir=OUT_DIR,
    )

    best_params, restored, metrics = best_restore(case)
    baseline_metrics = study.psnr_ssim_mae(original, blurred_noisy)

    study.save_npy_and_png(original, OUT_DIR / "original")
    study.save_npy_and_png(blurred_noisy, OUT_DIR / "blurred_noisy")
    study.save_npy_and_png(restored["wiener"], OUT_DIR / "wiener")
    study.save_npy_and_png(restored["richardson_lucy"], OUT_DIR / "richardson_lucy")
    study.save_npy_and_png(restored["tv_montalto"], OUT_DIR / "tv_montalto")
    study.save_npy_and_png(kernel.support, OUT_DIR / "kernel_support", cmap="inferno")
    study.save_npy_and_png(kernel.psf.centered, OUT_DIR / "kernel_embedded", cmap="inferno")
    save_figure(original, blurred_noisy, restored)

    metadata = {
        "image": "synthetic classic stars",
        "kernel_id": kernel.kernel_id,
        "noise_sigma": NOISE_SIGMA,
        "baseline_metrics": baseline_metrics,
        "best_params": best_params,
        "metrics": metrics,
    }
    (OUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
