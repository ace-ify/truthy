"""
Self-contained printable evidence report (HTML -> browser "Save as PDF").

No PDF library needed: this returns a single inline-styled HTML document. It is the
exportable, shareable case artifact required by Track 3 (chain of custody: case id,
SHA-256, timestamp, verdict, recommendation, evidence signals, provenance, reviewer).
"""
import html
import json
from typing import Any, Dict

_RISK_COLOR = {"high": "#ef4444", "medium": "#f59e0b", "low": "#22c55e"}
_VERDICT_LABEL = {
    "AI_GENERATED": "Likely fake / AI-generated",
    "HUMAN": "Likely authentic",
    "INCONCLUSIVE": "Needs review — not confirmed",
}


def _esc(x: Any) -> str:
    return html.escape(str(x if x is not None else ""))


def _signal_rows(signals) -> str:
    rows = []
    for s in sorted(signals, key=lambda r: (r.get("tier") != "primary", -abs(r.get("score", 0.5) - 0.5))):
        pct = round(float(s.get("score", 0.5)) * 100)
        tier = s.get("tier", "supporting")
        badge = "PRIMARY" if tier == "primary" else "supporting"
        badge_c = "#3b82f6" if tier == "primary" else "#6b7280"
        rows.append(
            f"<tr><td><span class='tier' style='background:{badge_c}'>{badge}</span> "
            f"{_esc(s.get('display_name'))}</td><td class='num'>{pct}%</td>"
            f"<td>{_esc(s.get('reasoning'))}</td></tr>"
        )
    return "\n".join(rows) or "<tr><td colspan='3'>No signals recorded.</td></tr>"


def _provenance_block(prov: Dict[str, Any]) -> str:
    if not prov or not prov.get("flag"):
        return "<p class='muted'>No provenance data (no Content Credential / usable metadata found).</p>"
    c = prov.get("c2pa", {}) or {}
    e = prov.get("exif", {}) or {}
    q = prov.get("quant", {}) or {}
    rows = [f"<li><b>Assessment:</b> {_esc(prov.get('summary'))} "
            f"(flag: {_esc(prov.get('flag'))}, strength: {_esc(prov.get('strength'))})</li>"]
    c2pa_txt = ("verified" if c.get("verified") else "present, unverified") if c.get("present") else "none found"
    if c.get("claim_generator"):
        c2pa_txt += f" — {_esc(c.get('claim_generator'))}"
    rows.append(f"<li><b>Content Credentials (C2PA):</b> {c2pa_txt}</li>")
    if e.get("present"):
        cam = " ".join(x for x in [e.get("make"), e.get("model")] if x) or "—"
        extra = (f", {_esc(e.get('datetime'))}" if e.get("datetime") else "")
        sw = f" · software: {_esc(e.get('software'))}" if e.get("software") else ""
        rows.append(f"<li><b>Camera EXIF:</b> {_esc(cam)}{extra}{sw}</li>")
    else:
        rows.append("<li><b>Camera EXIF:</b> none (stripped by messaging apps, screenshots, or AI tools)</li>")
    if q.get("note"):
        rows.append(f"<li><b>Compression:</b> {_esc(q.get('note'))}</li>")
    return "<ul class='prov'>" + "".join(rows) + "</ul>"


def render_report_html(case: Dict[str, Any]) -> str:
    """case: a row dict from governance.store.get_case (includes report_json)."""
    env = json.loads(case.get("report_json") or "{}")
    verdict = case.get("verdict", "INCONCLUSIVE")
    risk = case.get("risk_level", "medium")
    risk_c = _RISK_COLOR.get(risk, "#f59e0b")
    review_status = case.get("review_status", "none")
    reviewer_decision = case.get("reviewer_decision")
    reviewer_note = case.get("reviewer_note")

    cal = "calibrated" if env.get("calibrated") else "not yet calibrated (small validation set)"
    model_backed = "model-backed primary detector" if env.get("modelBacked") else "no confident model signal"

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Truthy Evidence Report {_esc(case.get('case_id'))}</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Oswald:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  * {{ box-sizing: border-box; margin:0; padding:0; }}
  body {{ font-family: 'Inter', -apple-system, Segoe UI, sans-serif; background:#000; color:#e5e7eb; margin:0; padding:32px; }}
  .sheet {{ max-width: 820px; margin:0 auto; background:#0b0b0d; border:1px solid #1f2229; border-radius:16px; padding:32px; }}
  h1 {{ font-family:'Oswald',sans-serif; font-weight:600; font-size:24px; margin:0 0 4px; letter-spacing:.01em; }}
  .muted {{ color:#8a8e96; font-size:13px; }}
  .verdict {{ display:flex; align-items:center; gap:16px; margin:20px 0; padding:18px; border-radius:12px; background:#0b0e14; border:1px solid {risk_c}55; }}
  .riskdot {{ width:16px; height:16px; border-radius:50%; background:{risk_c}; box-shadow:0 0 14px {risk_c}; }}
  .vlabel {{ font-size:22px; font-weight:700; }}
  .badge {{ display:inline-block; padding:2px 10px; border-radius:999px; font-size:11px; font-weight:700; color:#fff; text-transform:uppercase; }}
  .rec {{ border-left:3px solid {risk_c}; padding:12px 16px; background:#0b0e14; border-radius:8px; margin:12px 0; font-size:15px; }}
  table {{ width:100%; border-collapse:collapse; margin-top:8px; font-size:13px; }}
  th, td {{ text-align:left; padding:8px 10px; border-bottom:1px solid #232a36; vertical-align:top; }}
  th {{ color:#9ca3af; font-weight:600; }}
  td.num {{ font-variant-numeric: tabular-nums; font-weight:700; white-space:nowrap; }}
  .tier {{ display:inline-block; padding:1px 6px; border-radius:6px; font-size:9px; font-weight:700; color:#fff; margin-right:6px; }}
  .kv {{ display:grid; grid-template-columns: 180px 1fr; gap:6px 12px; font-size:13px; margin:12px 0; }}
  .kv div:nth-child(odd) {{ color:#9ca3af; }}
  .mono {{ font-family: ui-monospace, Menlo, Consolas, monospace; word-break:break-all; }}
  h2 {{ font-size:14px; text-transform:uppercase; letter-spacing:.05em; color:#9ca3af; margin:28px 0 6px; }}
  .prov {{ font-size:13px; }}
  .foot {{ margin-top:28px; font-size:11px; color:#6b7280; border-top:1px solid #232a36; padding-top:12px; }}
  @media print {{ body {{ background:#fff; color:#000; padding:0; }} .sheet {{ border:none; background:#fff; }} th,td{{border-color:#ddd;}} .verdict,.rec{{background:#f7f7f7;}} }}
</style></head>
<body><div class="sheet">
  <h1>Truthy — Financial Communication Verification Report</h1>
  <div class="muted">Case {_esc(case.get('case_id'))} · generated {_esc(case.get('created_at'))}</div>

  <div class="verdict">
    <div class="riskdot"></div>
    <div>
      <div class="vlabel">{_esc(_VERDICT_LABEL.get(verdict, verdict))}</div>
      <div class="muted">Risk: <b style="color:{risk_c}">{_esc(risk.upper())}</b> ·
        Confidence: {_esc(case.get('confidence'))} · AI probability: {round(float(case.get('ai_probability') or 0.5)*100)}%</div>
    </div>
  </div>

  <div class="rec"><b>Recommended action:</b> {_esc(env.get('recommendation'))}</div>
  <p>{_esc(env.get('explanation'))}</p>

  <h2>Chain of custody</h2>
  <div class="kv">
    <div>Media type</div><div>{_esc(case.get('media_type'))}</div>
    <div>File name</div><div>{_esc(case.get('filename'))}</div>
    <div>Size</div><div>{_esc(case.get('size_bytes'))} bytes</div>
    <div>SHA-256</div><div class="mono">{_esc(case.get('sha256'))}</div>
    <div>Basis</div><div>{_esc(model_backed)}; confidence {_esc(cal)}</div>
    <div>Review status</div><div>{_esc(review_status)}{(' — ' + _esc(reviewer_decision)) if reviewer_decision else ''}</div>
  </div>

  <h2>Evidence signals</h2>
  <table><thead><tr><th>Signal</th><th>AI score</th><th>Why</th></tr></thead>
  <tbody>{_signal_rows(env.get('signals', []))}</tbody></table>

  <h2>Provenance</h2>
  {_provenance_block(env.get('provenance', {}))}

  {('<h2>Reviewer note</h2><p>' + _esc(reviewer_note) + '</p>') if reviewer_note else ''}

  <div class="foot">
    Truthy is a decision-support tool, not legal proof. A detector score is not a verdict on a person.
    Confidence is calibrated on a small validation set; treat borderline and inconclusive results as
    "verify through a trusted channel." Media is processed transiently and not stored — only this
    metadata and its SHA-256 hash are retained.
  </div>
</div></body></html>"""


if __name__ == "__main__":
    # ponytail: self-check — renders without KeyError on a minimal case row.
    demo = {
        "case_id": "TRV-DEADBEEF", "created_at": "2026-09-29T00:00:00Z", "media_type": "image",
        "filename": "wire.jpg", "size_bytes": 1234, "sha256": "abc123", "verdict": "AI_GENERATED",
        "risk_level": "high", "confidence": "High", "ai_probability": 0.9, "review_status": "queued",
        "report_json": json.dumps({
            "recommendation": "Do NOT pay.", "explanation": "Likely fake.",
            "modelBacked": True, "calibrated": True,
            "signals": [{"display_name": "AI image classifier", "tier": "primary", "score": 0.9, "reasoning": "both CNNs agree AI"}],
            "provenance": {"c2pa": "none", "exif_software": "Adobe Firefly"},
        }),
    }
    out = render_report_html(demo)
    assert "TRV-DEADBEEF" in out and "Do NOT pay." in out and "PRIMARY" in out
    print("report_html self-check OK (%d chars)" % len(out))
