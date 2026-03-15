from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_deblurring_study as study
import run_manual_kernel_charts as manual


OUT_DIR = Path("outputs/figures/tv_showcases")


def collect_cases() -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    images = study.prepare_sipi_images()
    kernels = manual.build_manual_kernels()
    params = manual.FIXED_PARAMS
    rows: list[dict[str, object]] = []
    payloads: dict[str, dict[str, object]] = {}

    for image in images:
        for kernel in kernels:
            blur_clean = np.clip(
                study.blur_with_pyolimp(image.array, kernel.psf.shifted), 0.0, 1.0
            ).astype(np.float32)
            for sigma in study.NOISE_LEVELS:
                rng = np.random.default_rng(
                    study.stable_seed(image.image_id, kernel.kernel_id, sigma)
                )
                blur_noisy = np.clip(
                    blur_clean
                    + rng.normal(0.0, sigma, size=blur_clean.shape).astype(np.float32),
                    0.0,
                    1.0,
                ).astype(np.float32)
                case_id = f"{image.name}_{kernel.kernel_id}_{sigma:.2f}"
                case = study.TripleCase(
                    case_id=case_id,
                    image=image,
                    psf=kernel.psf,
                    noise_sigma=sigma,
                    blur_clean=blur_clean,
                    blur_noisy=blur_noisy,
                    fft_blur_noisy=np.fft.fft2(blur_noisy),
                    triple_dir=OUT_DIR,
                )
                restored = {
                    "wiener": study.wiener_restore(case, params["wiener"][sigma]),
                    "richardson_lucy": study.richardson_lucy_restore(
                        case, params["richardson_lucy"][sigma]
                    ),
                    "tv_montalto": study.tv_montalto_restore(
                        case, params["tv_montalto"][sigma]
                    ),
                }
                metrics = {
                    method: study.psnr_ssim_mae(image.array, arr)
                    for method, arr in restored.items()
                }
                ranking = sorted(
                    metrics.items(),
                    key=lambda kv: (kv[1]["ssim"], kv[1]["psnr"]),
                    reverse=True,
                )
                rows.append(
                    {
                        "case_id": case_id,
                        "image_name": image.name,
                        "kernel_id": kernel.kernel_id,
                        "noise_sigma": sigma,
                        "best_method": ranking[0][0],
                        "best_ssim": ranking[0][1]["ssim"],
                        "margin": ranking[0][1]["ssim"] - ranking[1][1]["ssim"],
                        "ranking": " > ".join(
                            f"{m}:{metrics[m]['ssim']:.4f}" for m, _ in ranking
                        ),
                    }
                )
                payloads[case_id] = {
                    "case": case,
                    "kernel": kernel,
                    "restored": restored,
                    "metrics": metrics,
                }

    df = pd.DataFrame(rows)
    df = df[df["best_method"] == "tv_montalto"].sort_values(
        ["margin", "best_ssim"], ascending=False
    )
    return df.reset_index(drop=True), payloads


def save_case_figure(case_id: str, payload: dict[str, object], rank: int) -> None:
    case = payload["case"]
    kernel = payload["kernel"]
    restored = payload["restored"]
    metrics = payload["metrics"]

    case_dir = OUT_DIR / f"{rank:02d}_{case_id}"
    case_dir.mkdir(parents=True, exist_ok=True)

    study.save_npy_and_png(case.image.array, case_dir / "original")
    study.save_npy_and_png(case.blur_noisy, case_dir / "blurred_noisy")
    study.save_npy_and_png(restored["wiener"], case_dir / "wiener")
    study.save_npy_and_png(restored["richardson_lucy"], case_dir / "richardson_lucy")
    study.save_npy_and_png(restored["tv_montalto"], case_dir / "tv_montalto")
    study.save_npy_and_png(kernel.support, case_dir / "kernel_support", cmap="inferno")

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.6))
    panels = [
        ("Original", case.image.array),
        ("Blurred", case.blur_noisy),
        ("Wiener", restored["wiener"]),
        ("Richardson-Lucy", restored["richardson_lucy"]),
        ("TV", restored["tv_montalto"]),
    ]
    for ax, (title, image) in zip(axes, panels):
        ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=17)
        ax.axis("off")
    fig.suptitle(
        f"TV showcase: {case.image.name}, {kernel.kernel_id}, sigma={case.noise_sigma:.2f}",
        fontsize=17,
    )
    fig.tight_layout()
    fig.savefig(case_dir / "comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "case_id": case_id,
        "image_name": case.image.name,
        "kernel_id": kernel.kernel_id,
        "noise_sigma": case.noise_sigma,
        "fixed_params": {
            "wiener": manual.FIXED_PARAMS["wiener"][case.noise_sigma],
            "richardson_lucy": manual.FIXED_PARAMS["richardson_lucy"][case.noise_sigma],
            "tv_montalto": manual.FIXED_PARAMS["tv_montalto"][case.noise_sigma],
        },
        "metrics": metrics,
    }
    (case_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8-sig"
    )


def save_overview(df: pd.DataFrame) -> None:
    top = df.head(3)
    fig, axes = plt.subplots(len(top), 2, figsize=(12, 10))
    if len(top) == 1:
        axes = np.array([axes])
    for row_idx, (_, row) in enumerate(top.iterrows()):
        case_dir = OUT_DIR / f"{row_idx + 1:02d}_{row['case_id']}"
        original = plt.imread(case_dir / "original.png")
        comparison = plt.imread(case_dir / "comparison.png")
        axes[row_idx, 0].imshow(original)
        axes[row_idx, 0].set_title(
            f"{row['image_name']}: {row['kernel_id']}, sigma={row['noise_sigma']:.2f}"
        )
        axes[row_idx, 0].axis("off")
        axes[row_idx, 1].imshow(comparison)
        axes[row_idx, 1].set_title(f"TV margin={row['margin']:.4f}\n{row['ranking']}")
        axes[row_idx, 1].axis("off")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "overview_top3.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    study.ensure_dirs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df, payloads = collect_cases()
    df.to_csv(OUT_DIR / "tv_showcases.csv", index=False, encoding="utf-8-sig")

    for idx, row in df.iterrows():
        save_case_figure(row["case_id"], payloads[row["case_id"]], idx + 1)

    save_overview(df)

    summary = {
        "num_tv_wins": int(len(df)),
        "best_cases": df.head(6).to_dict(orient="records"),
        "output_dir": str(OUT_DIR.resolve()),
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
