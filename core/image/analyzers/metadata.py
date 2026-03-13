"""
Metadata/EXIF Analyzer.
Checks for camera info, AI generator markers, and suspicious metadata patterns.
Low weight — metadata is easily faked/stripped, but free to check.
"""
from core.image.analyzers.base import BaseAnalyzer, AnalyzerResult
from core.image.preprocessor import ImageData


# Known AI generator software markers
AI_SOFTWARE_MARKERS = [
    "stable diffusion", "midjourney", "dall-e", "dalle",
    "comfyui", "automatic1111", "invoke ai", "novelai",
    "adobe firefly", "flux", "ideogram", "leonardo",
    "runway", "playground", "bing image creator",
    "copilot", "gemini", "chatgpt", "gpt-4",
]

# Legitimate camera/editing software
CAMERA_SOFTWARE = [
    "adobe photoshop", "adobe lightroom", "lightroom",
    "capture one", "dxo", "darktable", "rawtherapee",
    "snapseed", "vsco", "gimp",
]


class MetadataAnalyzer(BaseAnalyzer):

    name = "metadata"
    display_name = "Metadata Analysis"
    weight = 0.5  # Low weight — metadata is unreliable

    def _analyze(self, image_data: ImageData) -> AnalyzerResult:
        meta = image_data.metadata
        exif = meta.get("exif_data", {})

        signals = {}
        reasons = []

        # --- Signal 1: EXIF presence ---
        has_exif = meta.get("exif_present", False)
        exif_tag_count = meta.get("exif_tags", 0)

        if not has_exif:
            signals["no_exif"] = 0.6
            reasons.append("no EXIF metadata found (common in AI images)")
        elif exif_tag_count < 5:
            signals["sparse_exif"] = 0.4
            reasons.append("very sparse EXIF data")
        else:
            signals["has_exif"] = 0.0

        # --- Signal 2: Camera info ---
        camera_make = meta.get("camera_make")
        camera_model = meta.get("camera_model")

        if camera_make and camera_model:
            signals["has_camera"] = 0.0
            reasons.append(f"camera info: {camera_make} {camera_model}")
        elif has_exif and not camera_make:
            signals["no_camera"] = 0.4
            reasons.append("EXIF present but no camera info")

        # --- Signal 3: Software field ---
        software = meta.get("software", "") or exif.get("Software", "")
        software_lower = software.lower() if software else ""

        ai_detected = False
        for marker in AI_SOFTWARE_MARKERS:
            if marker in software_lower:
                signals["ai_software"] = 0.9
                reasons.append(f"AI generator marker in software field: '{software}'")
                ai_detected = True
                break

        if not ai_detected and software:
            is_legit = any(s in software_lower for s in CAMERA_SOFTWARE)
            if is_legit:
                signals["legit_software"] = 0.0
            else:
                signals["unknown_software"] = 0.3

        # --- Signal 4: Color profile ---
        has_profile = meta.get("has_color_profile", False)
        if not has_profile:
            signals["no_color_profile"] = 0.3
        else:
            signals["has_color_profile"] = 0.0

        # --- Signal 5: DateTime ---
        datetime_str = meta.get("datetime") or exif.get("DateTime", "")
        if not datetime_str and has_exif:
            signals["no_datetime"] = 0.2
        elif datetime_str:
            signals["has_datetime"] = 0.0

        # --- Signal 6: Image dimensions ---
        # Common AI generator resolutions
        w, h = image_data.final_size
        ai_resolutions = [
            (512, 512), (768, 768), (1024, 1024), (1536, 1024), (1024, 1536),
            (1024, 768), (768, 1024), (2048, 2048), (1792, 1024), (1024, 1792),
        ]
        is_ai_resolution = (w, h) in ai_resolutions or (h, w) in ai_resolutions
        if is_ai_resolution:
            signals["ai_resolution"] = 0.3
            reasons.append(f"resolution {w}x{h} matches common AI generator output")
        # Perfect squares are slightly suspicious
        elif w == h and w in [256, 384, 512, 640, 768, 896, 1024, 1280, 1536, 2048]:
            signals["square_power"] = 0.2

        # --- Compute final score ---
        if signals:
            signal_score = max(signals.values())
            # Average in the second-highest signal for robustness
            sorted_scores = sorted(signals.values(), reverse=True)
            if len(sorted_scores) > 1:
                signal_score = 0.6 * sorted_scores[0] + 0.4 * sorted_scores[1]
        else:
            signal_score = 0.3  # No info = slightly suspicious

        signal_score = min(max(signal_score, 0.0), 1.0)

        # Confidence is always low for metadata (easily manipulated)
        confidence = 0.30 if not ai_detected else 0.60

        if signal_score > 0.5:
            reasoning = "Metadata suggests AI generation: " + "; ".join(reasons[:3]) if reasons else \
                "Metadata patterns are suspicious"
        else:
            reasoning = "Metadata is consistent with authentic image capture"
            if reasons:
                reasoning += " (" + "; ".join(reasons[:2]) + ")"

        return AnalyzerResult(
            analyzer_name=self.name,
            display_name=self.display_name,
            signal_score=float(signal_score),
            confidence=float(confidence),
            reasoning=reasoning,
            details={
                "exif_present": has_exif,
                "exif_tag_count": exif_tag_count,
                "camera_make": camera_make,
                "camera_model": camera_model,
                "software": software or None,
                "has_color_profile": has_profile,
                "datetime": datetime_str or None,
                "resolution": f"{w}x{h}",
                "is_ai_resolution": is_ai_resolution,
                "signals": signals,
            },
        )
