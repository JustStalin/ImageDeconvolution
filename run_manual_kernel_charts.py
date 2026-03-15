from __future__ import annotations

import json
import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_deblurring_study import (
    FIGURES_DIR,
    NOISE_LEVELS,
    TABLES_DIR,
    TARGET_SIZE,
    PSFInfo,
    TripleCase,
    blur_with_pyolimp,
    ensure_dirs,
    prepare_sipi_images,
    psnr_ssim_mae,
    richardson_lucy_restore,
    stable_seed,
    tv_montalto_restore,
    wiener_restore,
)


KERNEL_FIGURE_DIR = FIGURES_DIR / "manual_kernels"
KERNEL_TABLE_DIR = TABLES_DIR / "manual_kernels"

METHOD_ORDER = ["blurred", "wiener", "richardson_lucy", "tv_montalto"]
METHOD_LABELS_RU = {
    "blurred": "Искажённое",
    "wiener": "Винер",
    "richardson_lucy": "РЛ",
    "tv_montalto": "TV",
}
METHOD_COLORS = {
    "blurred": "#e3e3e3",
    "wiener": "#8cb3df",
    "richardson_lucy": "#bebebe",
    "tv_montalto": "#3e6cad",
}
FIXED_PARAMS = {
    "wiener": {0.01: 0.01, 0.05: 0.1, 0.1: 0.2},
    "richardson_lucy": {0.01: 30, 0.05: 16, 0.1: 12},
    "tv_montalto": {0.01: 0.005, 0.05: 0.02, 0.1: 0.04},
}


@dataclass
class ManualKernel:
    kernel_id: str
    title: str
    support: np.ndarray
    psf: PSFInfo


def ensure_kernel_dirs() -> None:
    ensure_dirs()
    KERNEL_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    KERNEL_TABLE_DIR.mkdir(parents=True, exist_ok=True)


def gaussian_kernel(sigma: float) -> np.ndarray:
    radius = int(math.ceil(3 * sigma))
    axis = np.arange(-radius, radius + 1, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


def motion_kernel(length: int) -> np.ndarray:
    size = length + 2
    kernel = np.zeros((size, size), dtype=np.float32)
    row = size // 2
    start = (size - length) // 2
    kernel[row, start : start + length] = 1.0
    kernel /= kernel.sum()
    return kernel


def disk_kernel(radius: int) -> np.ndarray:
    axis = np.arange(-radius, radius + 1, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    kernel = (xx**2 + yy**2 <= radius**2).astype(np.float32)
    kernel /= kernel.sum()
    return kernel


def embed_kernel(support: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    canvas = np.zeros(image_size, dtype=np.float32)
    h, w = support.shape
    top = image_size[0] // 2 - h // 2
    left = image_size[1] // 2 - w // 2
    canvas[top : top + h, left : left + w] = support
    canvas /= canvas.sum()
    return canvas


def make_psf_info(kernel_id: str, title: str, support: np.ndarray) -> ManualKernel:
    centered = embed_kernel(support, TARGET_SIZE)
    shifted = np.fft.fftshift(centered).astype(np.float32)
    otf = np.fft.fft2(shifted)
    psf = PSFInfo(
        psf_id=kernel_id,
        psf_type="manual",
        description=title,
        params={},
        shifted=shifted,
        centered=centered,
        otf=otf,
        otf_conj=np.conj(otf),
        abs_otf_sq=np.abs(otf) ** 2,
    )
    return ManualKernel(kernel_id=kernel_id, title=title, support=support, psf=psf)


def build_manual_kernels() -> list[ManualKernel]:
    return [
        make_psf_info("gaussian_sigma1", "gaussian_σ1.0", gaussian_kernel(1.0)),
        make_psf_info("gaussian_sigma3", "gaussian_σ3.0", gaussian_kernel(3.0)),
        make_psf_info("motion_L10", "motion_L10", motion_kernel(10)),
        make_psf_info("disk_R5", "disk_R5", disk_kernel(5)),
    ]


def build_cases(images, kernels: list[ManualKernel]) -> list[TripleCase]:
    cases: list[TripleCase] = []
    for image in images:
        for kernel in kernels:
            blur_clean = np.clip(
                blur_with_pyolimp(image.array, kernel.psf.shifted), 0.0, 1.0
            ).astype(np.float32)
            for sigma in NOISE_LEVELS:
                rng = np.random.default_rng(
                    stable_seed(image.image_id, kernel.kernel_id, sigma)
                )
                blur_noisy = np.clip(
                    blur_clean
                    + rng.normal(0.0, sigma, size=blur_clean.shape).astype(np.float32),
                    0.0,
                    1.0,
                ).astype(np.float32)
                cases.append(
                    TripleCase(
                        case_id=f"{image.image_id}_{kernel.kernel_id}_{sigma:.2f}",
                        image=image,
                        psf=kernel.psf,
                        noise_sigma=sigma,
                        blur_clean=blur_clean,
                        blur_noisy=blur_noisy,
                        fft_blur_noisy=np.fft.fft2(blur_noisy),
                        triple_dir=KERNEL_FIGURE_DIR,
                    )
                )
    return cases


def evaluate_cases(
    cases: list[TripleCase],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for case in cases:
        baseline = psnr_ssim_mae(case.image.array, case.blur_noisy)
        rows.append(
            {
                "case_id": case.case_id,
                "kernel_id": case.psf.psf_id,
                "noise_sigma": case.noise_sigma,
                "method": "blurred",
                "param": np.nan,
                **baseline,
            }
        )
        for method in ["wiener", "richardson_lucy", "tv_montalto"]:
            param = FIXED_PARAMS[method][case.noise_sigma]
            if method == "wiener":
                restored = wiener_restore(case, float(param))
            elif method == "richardson_lucy":
                restored = richardson_lucy_restore(case, int(param))
            else:
                restored = tv_montalto_restore(case, float(param))
            metrics = psnr_ssim_mae(case.image.array, restored)
            rows.append(
                {
                    "case_id": case.case_id,
                    "kernel_id": case.psf.psf_id,
                    "noise_sigma": case.noise_sigma,
                    "method": method,
                    "param": param,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def save_kernel_gallery(kernels: list[ManualKernel]) -> None:
    fig, axes = plt.subplots(1, len(kernels), figsize=(14.5, 4.2))
    fig.suptitle("Сгенерированные ФРТ (нормировка: сумма = 1)", fontsize=15)
    for ax, kernel in zip(axes, kernels):
        image = ax.imshow(kernel.support, cmap="inferno")
        ax.set_title(kernel.title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(KERNEL_FIGURE_DIR / "generated_kernels.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_kernel_comparison_chart(kernel: ManualKernel, mean_df: pd.DataFrame) -> None:
    subset = (
        mean_df[mean_df["kernel_id"] == kernel.kernel_id]
        .copy()
        .sort_values(["noise_sigma", "method"])
    )
    noise_levels = sorted(subset["noise_sigma"].unique())
    noise_labels = [f"σ={sigma:.2f}" for sigma in noise_levels]
    x = np.arange(len(noise_levels))
    width = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))
    fig.suptitle(
        f"Сравнение всех трёх методов для ядра {kernel.title}",
        fontsize=15,
        fontweight="bold",
    )
    for idx, method in enumerate(METHOD_ORDER):
        method_subset = (
            subset[subset["method"] == method].sort_values("noise_sigma").reset_index(drop=True)
        )
        offset = (idx - 1.5) * width
        axes[0].bar(
            x + offset,
            method_subset["psnr"],
            width=width,
            color=METHOD_COLORS[method],
            edgecolor="#4a4a4a",
            linewidth=1.0,
            label=METHOD_LABELS_RU[method],
        )
        axes[1].bar(
            x + offset,
            method_subset["ssim"],
            width=width,
            color=METHOD_COLORS[method],
            edgecolor="#4a4a4a",
            linewidth=1.0,
            label=METHOD_LABELS_RU[method],
        )

    axes[0].set_title("Средний PSNR по уровням шума", fontsize=11)
    axes[0].set_xlabel("Уровень шума σ", fontsize=12)
    axes[0].set_ylabel("PSNR (dB)", fontsize=12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(noise_labels)
    axes[0].grid(axis="y", alpha=0.4)
    axes[0].legend(loc="upper right")

    axes[1].set_title("Средний SSIM по уровням шума", fontsize=11)
    axes[1].set_xlabel("Уровень шума σ", fontsize=12)
    axes[1].set_ylabel("SSIM", fontsize=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(noise_labels)
    axes[1].grid(axis="y", alpha=0.4)
    axes[1].legend(loc="upper right")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(
        KERNEL_FIGURE_DIR / f"comparison_{kernel.kernel_id}.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    ensure_kernel_dirs()
    images = prepare_sipi_images()
    kernels = build_manual_kernels()
    save_kernel_gallery(kernels)
    cases = build_cases(images, kernels)
    results_df = evaluate_cases(cases)
    mean_by_kernel_noise = (
        results_df.groupby(["kernel_id", "noise_sigma", "method"], as_index=False)[
            ["psnr", "ssim", "mae"]
        ]
        .mean()
        .sort_values(["kernel_id", "noise_sigma", "method"])
        .reset_index(drop=True)
    )

    for kernel in kernels:
        save_kernel_comparison_chart(kernel, mean_by_kernel_noise)

    mean_by_kernel_noise.to_csv(
        KERNEL_TABLE_DIR / "mean_by_kernel_noise.csv", index=False, encoding="utf-8-sig"
    )
    params_rows = [
        {
            "method": method,
            "noise_sigma": sigma,
            "best_param": value,
        }
        for method, sigma_map in FIXED_PARAMS.items()
        for sigma, value in sigma_map.items()
    ]
    params_df = pd.DataFrame(params_rows).sort_values(["method", "noise_sigma"])
    params_df.to_csv(
        KERNEL_TABLE_DIR / "fixed_params_by_noise.csv", index=False, encoding="utf-8-sig"
    )

    metadata = {
        "kernels": [kernel.kernel_id for kernel in kernels],
        "num_images": len(images),
        "num_cases": len(cases),
        "params_policy": "fixed_by_noise_from_global_study",
        "figure_dir": str(KERNEL_FIGURE_DIR.resolve()),
        "table_dir": str(KERNEL_TABLE_DIR.resolve()),
    }
    (KERNEL_TABLE_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8-sig"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
