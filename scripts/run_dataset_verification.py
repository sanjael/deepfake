"""
Phase 1: Dataset Verification & Integrity Execution Script
PhD Paper I: Frequency-Guided Spatial Attention Network
Author: Ramesh Prabhakaran R.
"""

import sys
import os
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.dataset_verifier import DatasetVerifier
from src.frequency.dwt_extractor import DWTExtractor2D
from src.frequency.dct_extractor import DCTExtractor2D


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_frequency_sample_visualizations(verifier: DatasetVerifier, output_dir: Path):
    """
    Selects representative Real and Fake samples from the test set,
    computes DWT subbands (LL, LH, HL, HH) and DCT log spectra,
    and produces a publication-grade scientific comparison figure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    test_df = verifier.dfs['test']
    
    # Pick 2 Real and 2 Fake representative samples
    real_samples = test_df[test_df['label'] == 1].head(2)['resolved_path'].tolist()
    fake_samples = test_df[test_df['label'] == 0].head(2)['resolved_path'].tolist()
    
    samples = [("Real Face (FFHQ)", p, "real") for p in real_samples] + \
              [("Fake Face (StyleGAN)", p, "fake") for p in fake_samples]

    dwt_ext = DWTExtractor2D(wavelet="haar")
    dct_ext = DCTExtractor2D(block_size=8)

    # 4 rows (2 real, 2 fake) x 6 columns: [RGB, DWT-LL, DWT-LH, DWT-HL, DWT-HH, DCT Spectrum]
    fig, axes = plt.subplots(len(samples), 6, figsize=(18, 12), dpi=200)
    col_titles = [
        "Spatial RGB (256x256)", 
        "DWT: LL (Approximation)", 
        "DWT: LH (Horizontal Detail)", 
        "DWT: HL (Vertical Detail)", 
        "DWT: HH (High-Freq Artifacts)", 
        "2D DCT Log-Spectrum"
    ]

    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=11, fontweight='bold', pad=10)

    for row_idx, (label_text, img_path, class_type) in enumerate(samples):
        # Load RGB
        img_pil = Image.open(img_path).convert('RGB')
        img_np = np.array(img_pil)
        
        # DWT decomposition
        dwt_bands = dwt_ext.extract_numpy(img_np)
        ll = np.clip(dwt_bands['LL'] / 255.0, 0, 1)
        lh = np.abs(dwt_bands['LH'])
        hl = np.abs(dwt_bands['HL'])
        hh = np.abs(dwt_bands['HH'])
        
        # Normalize high frequencies for visual contrast
        lh = lh / (lh.max() + 1e-6)
        hl = hl / (hl.max() + 1e-6)
        hh = hh / (hh.max() + 1e-6)

        # DCT log spectrum
        dct_spectrum = dct_ext.extract_full_dct_log_spectrum(img_np)

        # Plot RGB
        axes[row_idx, 0].imshow(img_np)
        axes[row_idx, 0].set_ylabel(f"{label_text}\n[{Path(img_path).name}]", fontsize=10, fontweight='bold')
        
        # Plot LL
        axes[row_idx, 1].imshow(ll)
        
        # Plot LH, HL, HH
        axes[row_idx, 2].imshow(lh, cmap='viridis' if lh.ndim==2 else None)
        axes[row_idx, 3].imshow(hl, cmap='viridis' if hl.ndim==2 else None)
        axes[row_idx, 4].imshow(hh, cmap='magma' if hh.ndim==2 else None)
        
        # Plot DCT Spectrum
        im_dct = axes[row_idx, 5].imshow(dct_spectrum, cmap='inferno')

        for c in range(6):
            axes[row_idx, c].set_xticks([])
            axes[row_idx, c].set_yticks([])

    plt.suptitle("Frequency-Domain Visual Forensic Analysis: Real vs. Synthetically Generated Faces\n(DWT Wavelet Sub-bands & 2D-DCT Spectra)", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    fig_path = output_dir / "frequency_domain_sample_analysis.png"
    plt.savefig(fig_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Sample frequency visualization saved to: {fig_path}")


def main():
    print("=" * 80)
    print("PhD Research Pipeline — Phase 1: Dataset Verification & Integrity Audit")
    print("Target: 140K Real and Fake Faces Dataset")
    print("=" * 80)

    config_file = PROJECT_ROOT / "configs" / "dataset_140k.yaml"
    cfg = load_config(str(config_file))

    dataset_cfg = cfg["dataset"]
    verifier = DatasetVerifier(
        root_dir=dataset_cfg["root_dir"],
        images_dir=dataset_cfg["images_dir"],
        csv_paths=dataset_cfg["csv_files"]
    )

    print("\n[Step 1/5] Loading metadata CSV files...")
    verifier.load_metadata()

    print("\n[Step 2/5] Auditing file existence & directory mappings...")
    audit_results = verifier.verify_file_existence()
    for split, info in audit_results.items():
        print(f"  - {split.upper()}: {info['csv_files_found_on_disk']:,} / {info['csv_total_rows']:,} files found on disk (Missing: {info['csv_files_missing']})")

    print("\n[Step 3/5] Computing class balance & distribution statistics...")
    balance_results = verifier.analyze_class_balance()
    for split in ['train', 'valid', 'test']:
        b = balance_results[split]
        print(f"  - {split.upper()}: Real={b['real_count']:,} ({b['real_percentage']}%), Fake={b['fake_count']:,} ({b['fake_percentage']}%) -> {b['status']}")
    ov = balance_results['overall']
    print(f"  - OVERALL: Real={ov['total_real']:,} ({ov['real_percentage']}%), Fake={ov['total_fake']:,} ({ov['fake_percentage']}%) -> {ov['status']}")

    print("\n[Step 4/5] Evaluating Data Leakage & Split Independence...")
    leakage_results = verifier.analyze_data_leakage(compute_hash_check=True)
    id_res = leakage_results['id_overlap']
    hash_res = leakage_results['hash_overlap']
    print(f"  - ID Overlap: Train-Val={id_res['train_valid_overlap_count']}, Train-Test={id_res['train_test_overlap_count']}, Val-Test={id_res['valid_test_overlap_count']}")
    print(f"  - Exact Content Hash Overlap: Train-Val={hash_res['train_valid_identical_images']}, Train-Test={hash_res['train_test_identical_images']}, Val-Test={hash_res['valid_test_identical_images']}")
    print(f"  - Leakage Detected: {leakage_results['leakage_detected']}")

    print("\n[Step 5/5] Checking image corruption, resolution & color channels...")
    integrity_results = verifier.verify_image_integrity_and_dimensions()
    for split, info in integrity_results.items():
        print(f"  - {split.upper()}: {info['checked_count']:,} checked, Corrupted: {info['corrupted_count']}, Dimensions: {info['dimension_distribution']}, Modes: {info['color_modes']}")

    print("\nGenerating frequency-domain forensic sample visualizations...")
    figures_dir = PROJECT_ROOT / "results" / "verification" / "figures"
    generate_frequency_sample_visualizations(verifier, figures_dir)

    print("\nCompiling Markdown and JSON reports...")
    report_md_path = PROJECT_ROOT / "results" / "verification" / "dataset_verification_report.md"
    verifier.generate_markdown_report(str(report_md_path))
    print(f"Verification report generated: {report_md_path}")
    print(f"JSON metrics written: {report_md_path.with_suffix('.json')}")
    print("\n" + "=" * 80)
    print("Phase 1 Verification Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
