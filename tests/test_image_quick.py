"""
Image detection accuracy test.

Usage:
    python tests/test_image_quick.py                         # test with sample images from web
    python tests/test_image_quick.py --mode quick            # fast ML-backed test
    python tests/test_image_quick.py --local path/to/img.jpg # test one local image
    python tests/test_image_quick.py --batch-ai folder/      # test folder of AI images
    python tests/test_image_quick.py --batch-real folder/     # test folder of real images
    python tests/test_image_quick.py --batch-ai ai/ --batch-real real/  # test both
"""
import sys
import os
import time
import argparse
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from PIL import Image
import io


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

# Sample test images from public sources
TEST_IMAGES = [
    {
        "name": "AI - SDXL astronaut",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/A_photograph_of_an_astronaut_riding_a_horse_2024-10-17.png/1024px-A_photograph_of_an_astronaut_riding_a_horse_2024-10-17.png",
        "expected": "AI Generated",
    },
    {
        "name": "AI - DALL-E cat",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2b/A_black_cat_with_orange_eyes_looking_at_the_camera_DALL-E_3.png/1024px-A_black_cat_with_orange_eyes_looking_at_the_camera_DALL-E_3.png",
        "expected": "AI Generated",
    },
    {
        "name": "Real - Nature",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/24701-nature-702702.jpg/1280px-24701-nature-702702.jpg",
        "expected": "Human Created",
    },
    {
        "name": "Real - Street photo",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Große_Freiheit_Hamburg_01.jpg/1280px-Große_Freiheit_Hamburg_01.jpg",
        "expected": "Human Created",
    },
]


def download_image(url: str, timeout: int = 30) -> Image.Image:
    response = requests.get(url, timeout=timeout, headers={"User-Agent": "TruthyTest/1.0"})
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def analyze_pil(pipeline, pil_image, mode):
    """Analyze a PIL image via the full preprocessing pipeline."""
    from core.image.preprocessor import load_image_from_path

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        pil_image.save(f, format="JPEG", quality=95)
        temp_path = f.name
    try:
        image_data = load_image_from_path(temp_path)
        return pipeline.analyze_image(image_data, mode=mode)
    finally:
        os.unlink(temp_path)


def analyze_path(pipeline, file_path, mode):
    """Analyze a local image file."""
    from core.image.preprocessor import load_image_from_path
    image_data = load_image_from_path(file_path)
    return pipeline.analyze_image(image_data, mode=mode)


def test_image(pipeline, name, expected, mode, pil_image=None, file_path=None):
    """Run detection on an image and return results dict."""
    start = time.perf_counter()
    if file_path:
        verdict = analyze_path(pipeline, file_path, mode)
    else:
        verdict = analyze_pil(pipeline, pil_image, mode)
    elapsed = time.perf_counter() - start

    correct = verdict.verdict == expected if expected != "Unknown" else None

    return {
        "name": name,
        "expected": expected,
        "verdict": verdict.verdict,
        "probability": verdict.overall_ai_probability,
        "confidence": verdict.confidence,
        "correct": correct,
        "time_s": round(elapsed, 1),
        "breakdown": verdict.signal_breakdown,
    }


def print_result(result):
    if result["correct"] is None:
        icon = "?"
    elif result["correct"]:
        icon = "+"
    else:
        icon = "X"

    print(f"  [{icon}] {result['name']}")
    print(f"      Expected: {result['expected']}  |  Got: {result['verdict']} "
          f"({result['probability']:.0%}, {result['confidence']})")
    print(f"      Time: {result['time_s']}s")

    signals = sorted(
        [s for s in result["breakdown"] if s.get("confidence", 0) > 0 and s.get("error") is None],
        key=lambda s: s["weight"] * s["confidence"],
        reverse=True,
    )[:5]
    for s in signals:
        bar = "#" * int(s["score"] * 20) + "." * (20 - int(s["score"] * 20))
        print(f"      {s['display_name']:25s} [{bar}] {s['score']:.2f} "
              f"(w={s['weight']:.1f}, c={s['confidence']:.2f})")
    print()


def get_images_from_folder(folder_path):
    """Get all image files from a folder."""
    folder = Path(folder_path)
    if not folder.is_dir():
        print(f"ERROR: {folder_path} is not a directory")
        return []
    return sorted([
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ])


def main():
    parser = argparse.ArgumentParser(description="Image detection accuracy test")
    parser.add_argument("--mode", default="standard", choices=["quick", "standard", "thorough"])
    parser.add_argument("--local", type=str, help="Test a single local image file")
    parser.add_argument("--batch-ai", type=str, help="Folder of AI-generated images to test")
    parser.add_argument("--batch-real", type=str, help="Folder of real photos to test")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Truthy Image Detection Test  |  mode={args.mode}")
    print(f"{'='*60}\n")

    print("Loading pipeline...")
    from core.image.pipeline import ImageDetectionPipeline
    pipeline = ImageDetectionPipeline()
    print(f"  {len(pipeline.analyzers)} analyzers loaded\n")

    results = []

    # --- Single local image ---
    if args.local:
        print(f"Testing: {args.local}\n")
        result = test_image(pipeline, Path(args.local).name, "Unknown", args.mode, file_path=args.local)
        print_result(result)
        return

    # --- Batch: AI folder ---
    if args.batch_ai:
        ai_files = get_images_from_folder(args.batch_ai)
        print(f"  AI images folder: {args.batch_ai} ({len(ai_files)} images)\n")
        for f in ai_files:
            print(f"  Analyzing: {f.name}...", end=" ", flush=True)
            result = test_image(pipeline, f"AI - {f.name}", "AI Generated", args.mode, file_path=str(f))
            results.append(result)
            icon = "+" if result["correct"] else "X"
            print(f"[{icon}] {result['verdict']} ({result['probability']:.0%}) - {result['time_s']}s")

    # --- Batch: Real folder ---
    if args.batch_real:
        real_files = get_images_from_folder(args.batch_real)
        print(f"  Real images folder: {args.batch_real} ({len(real_files)} images)\n")
        for f in real_files:
            print(f"  Analyzing: {f.name}...", end=" ", flush=True)
            result = test_image(pipeline, f"Real - {f.name}", "Human Created", args.mode, file_path=str(f))
            results.append(result)
            icon = "+" if result["correct"] else "X"
            print(f"[{icon}] {result['verdict']} ({result['probability']:.0%}) - {result['time_s']}s")

    # --- Web test images (default if no batch) ---
    if not args.batch_ai and not args.batch_real:
        for test_case in TEST_IMAGES:
            print(f"  Downloading: {test_case['name']}...", end=" ", flush=True)
            try:
                pil = download_image(test_case["url"])
                print(f"({pil.size[0]}x{pil.size[1]})")
            except Exception as e:
                print(f"FAILED: {e}")
                continue
            result = test_image(pipeline, test_case["name"], test_case["expected"], args.mode, pil_image=pil)
            results.append(result)

    if not results:
        print("No results to show.")
        return

    # --- Print detailed results ---
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}\n")

    for result in results:
        print_result(result)

    # --- Summary ---
    scored = [r for r in results if r["correct"] is not None]
    total = len(scored)
    correct = sum(1 for r in scored if r["correct"])
    accuracy = correct / total if total > 0 else 0

    # Per-category stats
    ai_results = [r for r in scored if r["expected"] == "AI Generated"]
    real_results = [r for r in scored if r["expected"] == "Human Created"]
    ai_correct = sum(1 for r in ai_results if r["correct"])
    real_correct = sum(1 for r in real_results if r["correct"])

    print(f"{'='*60}")
    print(f"  Overall: {correct}/{total} ({accuracy:.0%})")
    if ai_results:
        print(f"  AI detection:   {ai_correct}/{len(ai_results)} "
              f"({ai_correct/len(ai_results):.0%})")
    if real_results:
        print(f"  Real detection: {real_correct}/{len(real_results)} "
              f"({real_correct/len(real_results):.0%})")
    avg_time = sum(r["time_s"] for r in results) / len(results)
    print(f"  Avg time: {avg_time:.1f}s per image")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
