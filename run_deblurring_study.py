from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from olimp.processing import fft_conv
from olimp.simulate.psf_gauss import PSFGauss
from olimp.simulate.psf_sca import PSFSCA
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.restoration import denoise_tv_chambolle


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
RAW_DIR = OUTPUT_DIR / "raw"
SIPI_DIR = RAW_DIR / "sipi"
INTERNET_DIR = RAW_DIR / "internet"
PREPARED_DIR = OUTPUT_DIR / "prepared"
TRIPLES_DIR = PREPARED_DIR / "triples"
RESULTS_DIR = OUTPUT_DIR / "results"
RESTORED_DIR = RESULTS_DIR / "restored"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"

TARGET_SIZE = (256, 256)
RNG_SEED = 20260314
EPS = 1e-6

SIPI_IMAGES: list[dict[str, str]] = [
    {
        "id": "4.2.03",
        "name": "mandrill",
        "description": "Mandrill (a.k.a. Baboon)",
        "url": "https://sipi.usc.edu/database/misc/4.2.03.tiff",
        "page": "https://sipi.usc.edu/database/database.php?volume=misc",
    },
    {
        "id": "4.2.06",
        "name": "sailboat",
        "description": "Sailboat on lake",
        "url": "https://sipi.usc.edu/database/misc/4.2.06.tiff",
        "page": "https://sipi.usc.edu/database/database.php?volume=misc",
    },
    {
        "id": "4.2.07",
        "name": "peppers",
        "description": "Peppers",
        "url": "https://sipi.usc.edu/database/misc/4.2.07.tiff",
        "page": "https://sipi.usc.edu/database/database.php?volume=misc",
    },
]

INTERNET_IMAGES: list[dict[str, Any]] = [
    {
        "id": "landscape_unsplash",
        "name": "Landscape",
        "url": "https://upload.wikimedia.org/wikipedia/commons/4/4f/Landscape_%28Unsplash%29.jpg",
        "page": "https://commons.wikimedia.org/wiki/File:Landscape_(Unsplash).jpg",
        "crop": (0.05, 0.12, 0.95, 0.9),
    },
    {
        "id": "glass_facade_unsplash",
        "name": "Regular glass facade",
        "url": "https://upload.wikimedia.org/wikipedia/commons/b/b4/Regular_glass_facade_%28Unsplash%29.jpg",
        "page": "https://commons.wikimedia.org/wiki/File:Regular_glass_facade_(Unsplash).jpg",
        "crop": (0.05, 0.1, 0.82, 0.9),
    },
    {
        "id": "stars_constellations",
        "name": "Stars constellations",
        "url": "https://upload.wikimedia.org/wikipedia/commons/3/36/Stars_constellations.jpg",
        "page": "https://commons.wikimedia.org/wiki/File:Stars_constellations.jpg",
        "crop": (0.0, 0.0, 0.48, 1.0),
    },
]

INTERNET_EXAMPLE_CASES = [
    {
        "target_method": "wiener",
        "image_id": "stars_constellations",
        "psf_id": "sca_strong",
        "noise_sigma": 0.1,
    },
    {
        "target_method": "richardson_lucy",
        "image_id": "glass_facade_unsplash",
        "psf_id": "sca_mild",
        "noise_sigma": 0.01,
    },
    {
        "target_method": "tv_montalto",
        "image_id": "landscape_unsplash",
        "psf_id": "gauss_iso_mild",
        "noise_sigma": 0.01,
    },
]

NOISE_LEVELS = [0.01, 0.05, 0.1]

WIENER_BALANCES = {
    0.01: [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    0.05: [1e-3, 5e-3, 1e-2, 5e-2, 1e-1],
    0.1: [5e-3, 1e-2, 5e-2, 1e-1, 2e-1],
}
RL_ITERATIONS = {
    0.01: [6, 10, 18, 30],
    0.05: [4, 6, 10, 16],
    0.1: [3, 5, 8, 12],
}
TV_WEIGHTS = {
    0.01: [0.005, 0.01, 0.02, 0.04],
    0.05: [0.02, 0.04, 0.08, 0.12],
    0.1: [0.04, 0.08, 0.12, 0.18],
}
TV_OUTER_ITERS = 35

PSF_SPECS: list[dict[str, Any]] = [
    {
        "id": "gauss_iso_mild",
        "type": "gauss",
        "description": "Gaussian isotropic, mild blur",
        "params": {"theta": 0.0, "sigma_x": 2.2, "sigma_y": 2.2},
    },
    {
        "id": "gauss_aniso_rot",
        "type": "gauss",
        "description": "Gaussian anisotropic, rotated",
        "params": {"theta": math.radians(32.0), "sigma_x": 5.0, "sigma_y": 1.6},
    },
    {
        "id": "sca_mild",
        "type": "sca",
        "description": "SCA optic blur, mild",
        "params": {
            "sphere_dpt": -0.9,
            "cylinder_dpt": -0.35,
            "angle_rad": math.radians(20.0),
            "pupil_diameter_mm": 4.0,
            "am2px": 0.001,
        },
    },
    {
        "id": "sca_strong",
        "type": "sca",
        "description": "SCA optic blur, strong",
        "params": {
            "sphere_dpt": -1.8,
            "cylinder_dpt": -1.0,
            "angle_rad": math.radians(55.0),
            "pupil_diameter_mm": 5.0,
            "am2px": 0.001,
        },
    },
]

METHOD_LABELS = {
    "blurred": "Blurred+noise",
    "wiener": "Wiener",
    "richardson_lucy": "Richardson-Lucy",
    "tv_montalto": "TV (Montalto/FISTA)",
}


@dataclass
class PreparedImage:
    image_id: str
    name: str
    source_url: str
    page_url: str
    array: np.ndarray


@dataclass
class PSFInfo:
    psf_id: str
    psf_type: str
    description: str
    params: dict[str, Any]
    shifted: np.ndarray
    centered: np.ndarray
    otf: np.ndarray
    otf_conj: np.ndarray
    abs_otf_sq: np.ndarray


@dataclass
class TripleCase:
    case_id: str
    image: PreparedImage
    psf: PSFInfo
    noise_sigma: float
    blur_clean: np.ndarray
    blur_noisy: np.ndarray
    fft_blur_noisy: np.ndarray
    triple_dir: Path


def ensure_dirs() -> None:
    for path in [
        OUTPUT_DIR,
        RAW_DIR,
        SIPI_DIR,
        INTERNET_DIR,
        PREPARED_DIR,
        TRIPLES_DIR,
        RESULTS_DIR,
        RESTORED_DIR,
        TABLES_DIR,
        FIGURES_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def download_file(url: str, destination: Path) -> Path:
    if destination.exists():
        return destination
    response = requests.get(
        url,
        timeout=60,
        headers={"User-Agent": "Mozilla/5.0 (compatible; deblurring-study/1.0)"},
    )
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def to_grayscale_float32(image_path: Path, size: tuple[int, int]) -> np.ndarray:
    image = Image.open(image_path).convert("RGB").resize(
        size, Image.Resampling.BICUBIC
    )
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    gray = (
        0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    ).astype(np.float32)
    return np.clip(gray, 0.0, 1.0)


def crop_image_fraction(
    image: Image.Image, crop_box: tuple[float, float, float, float]
) -> Image.Image:
    width, height = image.size
    left = int(width * crop_box[0])
    top = int(height * crop_box[1])
    right = int(width * crop_box[2])
    bottom = int(height * crop_box[3])
    return image.crop((left, top, right, bottom))


def prepare_internet_image(record: dict[str, Any]) -> PreparedImage:
    file_name = Path(urlparse(record["url"]).path).name
    raw_path = download_file(record["url"], INTERNET_DIR / file_name)
    image = Image.open(raw_path).convert("RGB")
    image = crop_image_fraction(image, record["crop"]).resize(
        TARGET_SIZE, Image.Resampling.BICUBIC
    )
    temp_path = INTERNET_DIR / f"{record['id']}_cropped.png"
    image.save(temp_path)
    gray = to_grayscale_float32(temp_path, TARGET_SIZE)
    return PreparedImage(
        image_id=record["id"],
        name=record["name"],
        source_url=record["url"],
        page_url=record["page"],
        array=gray,
    )


def save_array_preview(array: np.ndarray, path: Path, cmap: str = "gray") -> None:
    plt.figure(figsize=(4, 4))
    plt.imshow(array, cmap=cmap, vmin=0.0, vmax=1.0)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_npy_and_png(array: np.ndarray, base_path: Path, cmap: str = "gray") -> None:
    np.save(base_path.with_suffix(".npy"), array.astype(np.float32))
    save_array_preview(array, base_path.with_suffix(".png"), cmap=cmap)


def stable_seed(*parts: Any) -> int:
    text = "|".join(map(str, parts))
    return RNG_SEED + sum((idx + 1) * ord(ch) for idx, ch in enumerate(text))


def prepare_sipi_images() -> list[PreparedImage]:
    prepared: list[PreparedImage] = []
    for record in SIPI_IMAGES:
        file_path = download_file(record["url"], SIPI_DIR / f"{record['id']}.tiff")
        gray = to_grayscale_float32(file_path, TARGET_SIZE)
        prepared.append(
            PreparedImage(
                image_id=record["id"],
                name=record["name"],
                source_url=record["url"],
                page_url=record["page"],
                array=gray,
            )
        )
        save_npy_and_png(gray, PREPARED_DIR / f"source_{record['name']}")
    return prepared


def generate_psfs(size: tuple[int, int]) -> list[PSFInfo]:
    height, width = size
    gauss = PSFGauss(width=width, height=height)
    sca = PSFSCA(width=width, height=height)
    psfs: list[PSFInfo] = []
    for spec in PSF_SPECS:
        if spec["type"] == "gauss":
            centered = gauss(
                center_x=width / 2.0,
                center_y=height / 2.0,
                theta=spec["params"]["theta"],
                sigma_x=spec["params"]["sigma_x"],
                sigma_y=spec["params"]["sigma_y"],
            ).numpy()
        else:
            centered = sca(**spec["params"]).numpy()
        centered = centered.astype(np.float32)
        centered /= centered.sum()
        shifted = np.fft.fftshift(centered).astype(np.float32)
        otf = np.fft.fft2(shifted)
        psfs.append(
            PSFInfo(
                psf_id=spec["id"],
                psf_type=spec["type"],
                description=spec["description"],
                params=spec["params"],
                shifted=shifted,
                centered=centered,
                otf=otf,
                otf_conj=np.conj(otf),
                abs_otf_sq=np.abs(otf) ** 2,
            )
        )
        save_npy_and_png(centered, PREPARED_DIR / f"psf_centered_{spec['id']}")
        save_npy_and_png(shifted, PREPARED_DIR / f"psf_shifted_{spec['id']}")
    return psfs


def blur_with_pyolimp(image: np.ndarray, shifted_psf: np.ndarray) -> np.ndarray:
    image_t = torch.from_numpy(image.astype(np.float32))
    psf_t = torch.from_numpy(shifted_psf.astype(np.float32))
    return fft_conv(image_t, psf_t).cpu().numpy().astype(np.float32)


def build_triples(images: list[PreparedImage], psfs: list[PSFInfo]) -> list[TripleCase]:
    rng = np.random.default_rng(RNG_SEED)
    cases: list[TripleCase] = []
    counter = 0
    for image in images:
        for psf in psfs:
            blur_clean = np.clip(blur_with_pyolimp(image.array, psf.shifted), 0.0, 1.0)
            for sigma in NOISE_LEVELS:
                counter += 1
                noise = rng.normal(0.0, sigma, size=blur_clean.shape).astype(np.float32)
                blur_noisy = np.clip(blur_clean + noise, 0.0, 1.0).astype(np.float32)
                case_id = (
                    f"triple_{counter:03d}_{image.name}_{psf.psf_id}_sigma_{sigma:.2f}"
                )
                triple_dir = TRIPLES_DIR / case_id
                triple_dir.mkdir(parents=True, exist_ok=True)
                save_npy_and_png(image.array, triple_dir / "original")
                save_npy_and_png(psf.centered, triple_dir / "psf")
                save_npy_and_png(blur_noisy, triple_dir / "blurred_noisy")
                metadata = {
                    "case_id": case_id,
                    "image_id": image.image_id,
                    "image_name": image.name,
                    "image_url": image.source_url,
                    "image_page": image.page_url,
                    "psf_id": psf.psf_id,
                    "psf_type": psf.psf_type,
                    "psf_description": psf.description,
                    "psf_params": psf.params,
                    "noise_sigma": sigma,
                }
                (triple_dir / "metadata.json").write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                cases.append(
                    TripleCase(
                        case_id=case_id,
                        image=image,
                        psf=psf,
                        noise_sigma=sigma,
                        blur_clean=blur_clean,
                        blur_noisy=blur_noisy,
                        fft_blur_noisy=np.fft.fft2(blur_noisy),
                        triple_dir=triple_dir,
                    )
                )
    return cases


def psnr_ssim_mae(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    estimate = np.clip(estimate.astype(np.float32), 0.0, 1.0)
    return {
        "psnr": float(peak_signal_noise_ratio(reference, estimate, data_range=1.0)),
        "ssim": float(structural_similarity(reference, estimate, data_range=1.0)),
        "mae": float(np.mean(np.abs(reference - estimate))),
    }


def wiener_restore(case: TripleCase, balance: float) -> np.ndarray:
    denom = case.psf.abs_otf_sq + balance
    restored_fft = case.psf.otf_conj * case.fft_blur_noisy / np.clip(denom, EPS, None)
    restored = np.real(np.fft.ifft2(restored_fft)).astype(np.float32)
    return np.clip(restored, 0.0, 1.0)


def richardson_lucy_restore(case: TripleCase, iterations: int) -> np.ndarray:
    estimate = np.clip(case.blur_noisy.copy(), EPS, 1.0)
    normalizer = np.real(
        np.fft.ifft2(case.psf.otf_conj * np.fft.fft2(np.ones_like(estimate)))
    )
    normalizer = np.clip(normalizer.astype(np.float32), EPS, None)
    for _ in range(iterations):
        conv_estimate = np.real(np.fft.ifft2(case.psf.otf * np.fft.fft2(estimate)))
        relative_blur = case.blur_noisy / np.clip(conv_estimate, EPS, None)
        correction = np.real(
            np.fft.ifft2(case.psf.otf_conj * np.fft.fft2(relative_blur))
        ).astype(np.float32)
        estimate *= correction / normalizer
        estimate = np.clip(estimate, 0.0, 1.0)
    return estimate.astype(np.float32)


def tv_montalto_restore(
    case: TripleCase, weight: float, outer_iters: int = TV_OUTER_ITERS
) -> np.ndarray:
    estimate = case.blur_noisy.copy()
    z = estimate.copy()
    t = 1.0
    lipschitz = float(np.max(case.psf.abs_otf_sq).real)
    step = 1.0 / max(lipschitz, EPS)
    blur_fft = case.fft_blur_noisy
    for _ in range(outer_iters):
        grad = np.real(
            np.fft.ifft2(
                case.psf.otf_conj * (case.psf.otf * np.fft.fft2(z) - blur_fft)
            )
        ).astype(np.float32)
        prox_input = np.clip(z - step * grad, 0.0, 1.0)
        next_estimate = denoise_tv_chambolle(
            prox_input,
            weight=weight * step,
            max_num_iter=20,
            channel_axis=None,
        ).astype(np.float32)
        next_estimate = np.clip(next_estimate, 0.0, 1.0)
        next_t = (1.0 + math.sqrt(1.0 + 4.0 * t * t)) / 2.0
        z = next_estimate + ((t - 1.0) / next_t) * (next_estimate - estimate)
        estimate = next_estimate
        t = next_t
    return estimate.astype(np.float32)


def tune_method(
    cases: list[TripleCase],
    noise_sigma: float,
    method: str,
    params: list[float | int],
) -> tuple[float | int, pd.DataFrame]:
    subset = [case for case in cases if math.isclose(case.noise_sigma, noise_sigma)]
    rows: list[dict[str, Any]] = []
    for param in params:
        ssim_scores: list[float] = []
        psnr_scores: list[float] = []
        for case in subset:
            if method == "wiener":
                restored = wiener_restore(case, float(param))
            elif method == "richardson_lucy":
                restored = richardson_lucy_restore(case, int(param))
            else:
                restored = tv_montalto_restore(case, float(param))
            metrics = psnr_ssim_mae(case.image.array, restored)
            ssim_scores.append(metrics["ssim"])
            psnr_scores.append(metrics["psnr"])
        rows.append(
            {
                "method": method,
                "noise_sigma": noise_sigma,
                "param": param,
                "mean_ssim": float(np.mean(ssim_scores)),
                "mean_psnr": float(np.mean(psnr_scores)),
            }
        )
    tuning_df = pd.DataFrame(rows).sort_values(
        by=["mean_ssim", "mean_psnr"], ascending=False
    )
    return tuning_df.iloc[0]["param"], tuning_df


def restore_all_cases(
    cases: list[TripleCase],
    best_params: dict[str, dict[float, float | int]],
) -> tuple[pd.DataFrame, dict[tuple[str, str], np.ndarray]]:
    records: list[dict[str, Any]] = []
    restored_cache: dict[tuple[str, str], np.ndarray] = {}
    for case in cases:
        baseline_metrics = psnr_ssim_mae(case.image.array, case.blur_noisy)
        records.append(
            {
                "case_id": case.case_id,
                "image_id": case.image.image_id,
                "image_name": case.image.name,
                "psf_id": case.psf.psf_id,
                "psf_type": case.psf.psf_type,
                "noise_sigma": case.noise_sigma,
                "method": "blurred",
                "param": np.nan,
                **baseline_metrics,
            }
        )
        for method in ["wiener", "richardson_lucy", "tv_montalto"]:
            param = best_params[method][case.noise_sigma]
            if method == "wiener":
                restored = wiener_restore(case, float(param))
            elif method == "richardson_lucy":
                restored = richardson_lucy_restore(case, int(param))
            else:
                restored = tv_montalto_restore(case, float(param))
            metrics = psnr_ssim_mae(case.image.array, restored)
            records.append(
                {
                    "case_id": case.case_id,
                    "image_id": case.image.image_id,
                    "image_name": case.image.name,
                    "psf_id": case.psf.psf_id,
                    "psf_type": case.psf.psf_type,
                    "noise_sigma": case.noise_sigma,
                    "method": method,
                    "param": param,
                    **metrics,
                }
            )
            restored_cache[(case.case_id, method)] = restored
            method_dir = RESTORED_DIR / method
            method_dir.mkdir(parents=True, exist_ok=True)
            save_npy_and_png(restored, method_dir / case.case_id)
    return pd.DataFrame(records), restored_cache


def compute_summary_tables(results_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    baseline = (
        results_df[results_df["method"] == "blurred"]
        .set_index("case_id")[["psnr", "ssim", "mae"]]
        .rename(
            columns={
                "psnr": "psnr_blurred",
                "ssim": "ssim_blurred",
                "mae": "mae_blurred",
            }
        )
    )
    merged = results_df.join(baseline, on="case_id")
    merged["delta_psnr_vs_blurred"] = merged["psnr"] - merged["psnr_blurred"]
    merged["delta_ssim_vs_blurred"] = merged["ssim"] - merged["ssim_blurred"]
    merged["delta_mae_vs_blurred"] = merged["mae"] - merged["mae_blurred"]
    tables: dict[str, pd.DataFrame] = {}
    tables["all_results"] = merged.sort_values(by=["case_id", "method"]).reset_index(
        drop=True
    )
    tables["overall_mean"] = (
        merged.groupby("method", as_index=False)[
            [
                "psnr",
                "ssim",
                "mae",
                "delta_psnr_vs_blurred",
                "delta_ssim_vs_blurred",
                "delta_mae_vs_blurred",
            ]
        ]
        .mean()
        .sort_values(by=["ssim", "psnr"], ascending=False)
        .reset_index(drop=True)
    )
    tables["mean_by_noise"] = (
        merged.groupby(["noise_sigma", "method"], as_index=False)[
            [
                "psnr",
                "ssim",
                "mae",
                "delta_psnr_vs_blurred",
                "delta_ssim_vs_blurred",
            ]
        ]
        .mean()
        .sort_values(by=["noise_sigma", "ssim"], ascending=[True, False])
        .reset_index(drop=True)
    )
    tables["mean_by_psf"] = (
        merged.groupby(["psf_id", "method"], as_index=False)[
            [
                "psnr",
                "ssim",
                "mae",
                "delta_psnr_vs_blurred",
                "delta_ssim_vs_blurred",
            ]
        ]
        .mean()
        .sort_values(by=["psf_id", "ssim"], ascending=[True, False])
        .reset_index(drop=True)
    )
    algo_only = merged[merged["method"].isin(["wiener", "richardson_lucy", "tv_montalto"])]
    ssim_winners = (
        algo_only.sort_values(by=["case_id", "ssim", "psnr"], ascending=[True, False, False])
        .groupby("case_id", as_index=False)
        .first()[["case_id", "method"]]
        .rename(columns={"method": "best_method_ssim"})
    )
    psnr_winners = (
        algo_only.sort_values(by=["case_id", "psnr", "ssim"], ascending=[True, False, False])
        .groupby("case_id", as_index=False)
        .first()[["case_id", "method"]]
        .rename(columns={"method": "best_method_psnr"})
    )
    wins = ssim_winners.join(psnr_winners.set_index("case_id"), on="case_id")
    tables["win_counts"] = pd.DataFrame(
        {
            "method": ["wiener", "richardson_lucy", "tv_montalto"],
            "ssim_wins": [
                int((wins["best_method_ssim"] == method).sum())
                for method in ["wiener", "richardson_lucy", "tv_montalto"]
            ],
            "psnr_wins": [
                int((wins["best_method_psnr"] == method).sum())
                for method in ["wiener", "richardson_lucy", "tv_montalto"]
            ],
        }
    ).sort_values(by=["ssim_wins", "psnr_wins"], ascending=False)
    return tables


def export_tables(tables: dict[str, pd.DataFrame], tuning_tables: list[pd.DataFrame]) -> None:
    for name, df in tables.items():
        df.to_csv(TABLES_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")
    tuning_df = pd.concat(tuning_tables, ignore_index=True)
    tuning_df.to_csv(
        TABLES_DIR / "hyperparameter_tuning.csv", index=False, encoding="utf-8-sig"
    )
    with pd.ExcelWriter(TABLES_DIR / "study_tables.xlsx", engine="openpyxl") as writer:
        for name, df in tables.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
        tuning_df.to_excel(writer, sheet_name="tuning", index=False)


def pretty_method(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def save_dataset_overview_figure(cases: list[TripleCase]) -> None:
    chosen_cases = [cases[0], cases[len(cases) // 2], cases[-1]]
    fig, axes = plt.subplots(len(chosen_cases), 3, figsize=(10, 9))
    for row, case in enumerate(chosen_cases):
        axes[row, 0].imshow(case.image.array, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(f"Original: {case.image.name}")
        axes[row, 1].imshow(
            case.psf.centered, cmap="gray", vmin=0, vmax=float(case.psf.centered.max())
        )
        axes[row, 1].set_title(f"PSF: {case.psf.psf_id}")
        axes[row, 2].imshow(case.blur_noisy, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].set_title(f"Blurred, sigma={case.noise_sigma:.2f}")
        for col in range(3):
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "dataset_overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_noise_plot(mean_by_noise: pd.DataFrame) -> None:
    ordered_methods = ["blurred", "wiener", "richardson_lucy", "tv_montalto"]
    display_names = {
        "blurred": "Искажённое",
        "wiener": "Винер",
        "richardson_lucy": "РЛ",
        "tv_montalto": "TV",
    }
    colors = {
        "blurred": "#b0b0b0",
        "wiener": "#4f86b5",
        "richardson_lucy": "#ff7f50",
        "tv_montalto": "#008a00",
    }
    noise_levels = sorted(mean_by_noise["noise_sigma"].unique())
    noise_labels = [f"σ={sigma:.2f}" for sigma in noise_levels]
    x = np.arange(len(noise_levels))
    width = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))
    fig.suptitle("Сравнение всех трёх методов: Винер vs РЛ vs TV", fontsize=15, fontweight="bold")

    for idx, method in enumerate(ordered_methods):
        subset = (
            mean_by_noise[mean_by_noise["method"] == method]
            .sort_values("noise_sigma")
            .reset_index(drop=True)
        )
        offset = (idx - 1.5) * width
        axes[0].bar(
            x + offset,
            subset["psnr"],
            width=width,
            color=colors[method],
            label=display_names[method],
        )
        axes[1].bar(
            x + offset,
            subset["ssim"],
            width=width,
            color=colors[method],
            label=display_names[method],
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
    output_path = FIGURES_DIR / "mean_metrics_by_noise.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "comparison_methods_bar_chart.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_internet_cases(psfs: list[PSFInfo]) -> list[TripleCase]:
    images = [prepare_internet_image(record) for record in INTERNET_IMAGES]
    image_map = {image.image_id: image for image in images}
    psf_map = {psf.psf_id: psf for psf in psfs}
    cases: list[TripleCase] = []
    for spec in INTERNET_EXAMPLE_CASES:
        image = image_map[spec["image_id"]]
        psf = psf_map[spec["psf_id"]]
        sigma = spec["noise_sigma"]
        blur_clean = np.clip(blur_with_pyolimp(image.array, psf.shifted), 0.0, 1.0)
        rng = np.random.default_rng(
            stable_seed(spec["target_method"], image.image_id, psf.psf_id, sigma)
        )
        blur_noisy = np.clip(
            blur_clean + rng.normal(0.0, sigma, size=blur_clean.shape).astype(np.float32),
            0.0,
            1.0,
        ).astype(np.float32)
        case_id = (
            f"internet_{spec['target_method']}_{image.image_id}_{psf.psf_id}_sigma_{sigma:.2f}"
        )
        triple_dir = RESULTS_DIR / "internet_cases" / case_id
        triple_dir.mkdir(parents=True, exist_ok=True)
        cases.append(
            TripleCase(
                case_id=case_id,
                image=image,
                psf=psf,
                noise_sigma=sigma,
                blur_clean=blur_clean,
                blur_noisy=blur_noisy,
                fft_blur_noisy=np.fft.fft2(blur_noisy),
                triple_dir=triple_dir,
            )
        )
    return cases


def select_and_render_internet_examples(
    internet_cases: list[TripleCase],
    best_params: dict[str, dict[float, float | int]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    predictions: dict[tuple[str, str], np.ndarray] = {}
    for case in internet_cases:
        for method in ["wiener", "richardson_lucy", "tv_montalto"]:
            param = best_params[method][case.noise_sigma]
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
                    "image_id": case.image.image_id,
                    "image_name": case.image.name,
                    "image_page": case.image.page_url,
                    "psf_id": case.psf.psf_id,
                    "noise_sigma": case.noise_sigma,
                    "method": method,
                    "param": param,
                    **metrics,
                }
            )
            predictions[(case.case_id, method)] = restored
    metrics_df = pd.DataFrame(rows)
    winners: list[dict[str, Any]] = []
    for spec in INTERNET_EXAMPLE_CASES:
        method = spec["target_method"]
        method_row = metrics_df[
            (metrics_df["method"] == method)
            & (metrics_df["image_id"] == spec["image_id"])
            & (metrics_df["psf_id"] == spec["psf_id"])
            & (metrics_df["noise_sigma"] == spec["noise_sigma"])
        ].iloc[0]
        competitor_rows = metrics_df[
            (metrics_df["case_id"] == method_row["case_id"])
            & (metrics_df["method"] != method)
        ]
        best_other_ssim = float(competitor_rows["ssim"].max())
        winners.append(
            {
                "target_method": method,
                "case_id": method_row["case_id"],
                "image_id": method_row["image_id"],
                "image_name": method_row["image_name"],
                "image_page": method_row["image_page"],
                "psf_id": method_row["psf_id"],
                "noise_sigma": method_row["noise_sigma"],
                "param": method_row["param"],
                "ssim": method_row["ssim"],
                "ssim_margin": method_row["ssim"] - best_other_ssim,
            }
        )
        case = next(item for item in internet_cases if item.case_id == method_row["case_id"])
        fig, axes = plt.subplots(1, 5, figsize=(18, 4))
        panels = [
            ("Original", case.image.array),
            ("Blurred", case.blur_noisy),
            ("Wiener", predictions[(case.case_id, "wiener")]),
            ("Richardson-Lucy", predictions[(case.case_id, "richardson_lucy")]),
            ("TV", predictions[(case.case_id, "tv_montalto")]),
        ]
        for ax, (title, panel) in zip(axes, panels):
            ax.imshow(panel, cmap="gray", vmin=0, vmax=1)
            ax.set_title(title)
            ax.axis("off")
        fig.suptitle(
            f"Internet example for {pretty_method(method)}: {case.image.name}, {case.psf.psf_id}, sigma={case.noise_sigma:.2f}"
        )
        fig.tight_layout()
        fig.savefig(
            FIGURES_DIR / f"internet_example_{method}.png", dpi=180, bbox_inches="tight"
        )
        plt.close(fig)
    metrics_df.to_csv(
        TABLES_DIR / "internet_example_metrics.csv", index=False, encoding="utf-8-sig"
    )
    winners_df = pd.DataFrame(winners)
    winners_df.to_csv(
        TABLES_DIR / "internet_example_winners.csv", index=False, encoding="utf-8-sig"
    )
    return winners_df


def main() -> None:
    ensure_dirs()
    sipi_images = prepare_sipi_images()
    psfs = generate_psfs(TARGET_SIZE)
    cases = build_triples(sipi_images, psfs)
    save_dataset_overview_figure(cases)
    best_params: dict[str, dict[float, float | int]] = {
        "wiener": {},
        "richardson_lucy": {},
        "tv_montalto": {},
    }
    tuning_tables: list[pd.DataFrame] = []
    for sigma in NOISE_LEVELS:
        best_param, tuning_df = tune_method(cases, sigma, "wiener", WIENER_BALANCES[sigma])
        best_params["wiener"][sigma] = float(best_param)
        tuning_tables.append(tuning_df)
        best_param, tuning_df = tune_method(
            cases, sigma, "richardson_lucy", RL_ITERATIONS[sigma]
        )
        best_params["richardson_lucy"][sigma] = int(best_param)
        tuning_tables.append(tuning_df)
        best_param, tuning_df = tune_method(cases, sigma, "tv_montalto", TV_WEIGHTS[sigma])
        best_params["tv_montalto"][sigma] = float(best_param)
        tuning_tables.append(tuning_df)
    results_df, _ = restore_all_cases(cases, best_params)
    tables = compute_summary_tables(results_df)
    export_tables(tables, tuning_tables)
    save_noise_plot(tables["mean_by_noise"])
    build_internet_cases(psfs)
    select_and_render_internet_examples(internet_cases, best_params)
    summary = {
        "num_cases": len(cases),
        "best_params": best_params,
        "report": str((OUTPUT_DIR / "report.md").resolve()),
        "excel": str((TABLES_DIR / "study_tables.xlsx").resolve()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
