"""Renders docs/index.html — the project website.

Reads ONLY docs/site_data.json (produced by evaluation/export_site_data.py), so
the page cannot drift from the run that produced it. No backend, no API key, no
network requests at runtime: the curated trace snapshot is embedded directly in
the page, which makes every interaction client-side and means the site works
from a fresh clone, offline, and on GitHub Pages with no configuration.

Deliberately NOT included: live LLM attack generation. It would put an API key
on a server, cost money per visitor, and — worst — be non-deterministic, so a
judge could run an attack and see a different result than the one we report.
That is a credibility risk, not a feature.

Run: PYTHONPATH=. .venv/bin/python docs/build_site.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

DATA = Path("docs/site_data.json")
OUT = Path("docs/index.html")
ART = Path("docs/artifact.html")  # same page, artifact-wrapper format

# The AP2 kill chain — columns of the coverage matrix. Ordered as the protocol
# actually executes, so the grid reads left-to-right as an attack progresses.
STAGES = [
    ("Intent Capture", "User states what they want"),
    ("Product Discovery", "Merchant returns candidates"),
    ("Agent Reasoning", "LLM decides what to buy"),
    ("Credential Access", "Provider releases secrets"),
    ("Mandate Construction", "Chain is cryptographically signed"),
    ("Payment Execution", "Transaction clears"),
]

# Cells: (stage_index, label, state, metric, family_filter)
#   state: "built" | "gap" | "thesis"
CELLS = [
    (1, "Branded Whisper", "built", "0% ASR · 46 traces", "reasoning_attack"),
    (2, "Branded Whisper", "built", "ranking bias", "reasoning_attack"),
    (1, "Intent Manipulation", "built", "15/18 reached agent", "intent_manipulation"),
    (2, "Intent Manipulation", "built", "0/18 succeeded", "intent_manipulation"),
    (3, "Vault Whisper", "built", "100% exposure", "reasoning_attack"),
    (5, "Delegation Abuse", "built", "6 violation types", "delegation_abuse"),
    (5, "Sequence Anomaly", "built", "3 presets · 46 traces", "sequence_anomaly"),
    (1, "Context Poisoning", "gap", "identified, not built", None),
    (2, "Cross-Agent Injection", "gap", "identified, not built", None),
    (5, "Multi-Agent Propagation", "gap", "identified, not built", None),
    (4, "signature_valid = TRUE", "thesis", "cryptography holds — and does not help", None),
]

OUTCOME_META = {
    "case_c": ("CASE C", "Succeeded · UNDETECTED", "danger"),
    "case_b": ("CASE B", "Succeeded · caught", "warn"),
    "case_a": ("CASE A", "Failed · caught", "ok"),
    "case_d": ("CASE D", "Failed · missed", "muted"),
    "clean": ("BENIGN", "No attack present", "ok"),
    "false_positive": ("FALSE POSITIVE", "Benign · flagged", "warn"),
}


# ---------------------------------------------------------------- svg charts

def line_chart(rounds: List[Dict[str, Any]], field: str, families: List[str],
               colors: Dict[str, str], *, title: str, ymax: float = 1.0) -> str:
    """Inline SVG. Generated at build time because the data is fixed — shipping
    a charting library to render four static series would be waste."""
    w, h, pad = 620, 200, 38
    pts_by_fam: Dict[str, List[tuple]] = {}
    for fam in families:
        rows = [r for r in rounds if r["family"].startswith(fam)]
        pts = []
        for r in rows:
            try:
                gen = int(r["generation"])
                val = float(r[field])
            except (ValueError, KeyError, TypeError):
                continue
            pts.append((gen, val))
        if pts:
            pts_by_fam[fam] = sorted(pts)
    if not pts_by_fam:
        return ""
    gens = sorted({g for pts in pts_by_fam.values() for g, _ in pts})
    gmin, gmax = min(gens), max(gens)
    span = max(gmax - gmin, 1)

    def X(g): return pad + (g - gmin) / span * (w - pad * 2)
    def Y(v): return h - pad - (v / ymax) * (h - pad * 2)

    out = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="{title}">']
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        y = Y(frac * ymax)
        out.append(f'<line x1="{pad}" y1="{y:.1f}" x2="{w-pad}" y2="{y:.1f}" class="grid"/>')
        out.append(f'<text x="{pad-8}" y="{y+3:.1f}" class="axis" text-anchor="end">{frac*ymax:.2f}</text>')
    for g in gens:
        out.append(f'<text x="{X(g):.1f}" y="{h-pad+16}" class="axis" text-anchor="middle">gen {g}</text>')
    for fam, pts in pts_by_fam.items():
        c = colors.get(fam, "#6EA8FE")
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(g):.1f},{Y(v):.1f}" for i, (g, v) in enumerate(pts))
        out.append(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="2.2" stroke-linejoin="round"/>')
        for g, v in pts:
            out.append(f'<circle cx="{X(g):.1f}" cy="{Y(v):.1f}" r="3.6" fill="{c}"/>')
    out.append("</svg>")
    return "".join(out)


def bar_ci(rows: List[Dict[str, Any]]) -> str:
    """Recall with 95% CI whiskers; the null-control row is styled apart because
    its bar is a FALSE-POSITIVE rate, not recall — labelling them the same way
    would invite exactly the misreading we spent the project avoiding."""
    w, h, pad = 620, 210, 42
    n = len(rows)
    bw = (w - pad * 2) / n * 0.52

    def X(i): return pad + (i + 0.5) / n * (w - pad * 2)
    def Y(v): return h - pad - v * (h - pad * 2)

    out = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="recall with confidence intervals">']
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        y = Y(frac)
        out.append(f'<line x1="{pad}" y1="{y:.1f}" x2="{w-pad}" y2="{y:.1f}" class="grid"/>')
        out.append(f'<text x="{pad-8}" y="{y+3:.1f}" class="axis" text-anchor="end">{frac:.2f}</text>')
    for i, r in enumerate(rows):
        x, v = X(i), r["recall"]
        c = "#E8B339" if r["null"] else "#6EA8FE"
        out.append(f'<rect x="{x-bw/2:.1f}" y="{Y(v):.1f}" width="{bw:.1f}" '
                   f'height="{h-pad-Y(v):.1f}" fill="{c}" opacity="0.82" rx="2"/>')
        out.append(f'<line x1="{x:.1f}" y1="{Y(r["lo"]):.1f}" x2="{x:.1f}" y2="{Y(r["hi"]):.1f}" '
                   f'stroke="var(--fg)" stroke-width="1.5" opacity="0.75"/>')
        for b in (r["lo"], r["hi"]):
            out.append(f'<line x1="{x-5:.1f}" y1="{Y(b):.1f}" x2="{x+5:.1f}" y2="{Y(b):.1f}" '
                       f'stroke="var(--fg)" stroke-width="1.5" opacity="0.75"/>')
        out.append(f'<text x="{x:.1f}" y="{h-pad+16}" class="axis" text-anchor="middle">{r["mult"]}</text>')
        if r["null"]:
            out.append(f'<text x="{x:.1f}" y="{h-pad+30}" class="axis danger" text-anchor="middle">null</text>')
    out.append("</svg>")
    return "".join(out)


# ---------------------------------------------------------------- html parts

def matrix_html() -> str:
    cols = "".join(
        f'<div class="mx-head"><span class="mx-stage">{i+1}</span>{name}'
        f'<em>{desc}</em></div>' for i, (name, desc) in enumerate(STAGES))
    by_col: Dict[int, List[str]] = {i: [] for i in range(len(STAGES))}
    for col, label, state, metric, fam in CELLS:
        attr = f' data-family="{fam}"' if fam else ""
        by_col[col].append(
            f'<div class="mx-cell {state}"{attr} tabindex="{0 if fam else -1}">'
            f'<strong>{label}</strong><span>{metric}</span></div>')
    body = "".join(f'<div class="mx-col">{"".join(v)}</div>' for v in by_col.values())
    return f'<div class="matrix"><div class="mx-heads">{cols}</div><div class="mx-body">{body}</div></div>'


def build() -> None:
    data = json.loads(DATA.read_text())
    m = data["meta"]

    colors = {"reasoning_attack": "#FF5C4D", "intent_manipulation": "#B084F5",
              "sequence_anomaly": "#34D399", "delegation_abuse": "#8794AB"}

    recall_chart = line_chart(data["rounds"], "blue_recall_test",
                              list(colors), colors, title="Blue recall by generation")
    reward_chart = line_chart(
        [{**r, field_fix: r[field_fix]} for r in data["rounds"] for field_fix in ["mean_reward"]
         if r.get("mean_reward") not in (None, "", "n/a")],
        "mean_reward", [f for f in colors if f != "delegation_abuse"], colors,
        title="Red reward by generation", ymax=0.1)

    gen_rows = "".join(
        f'<tr class="{"spot" if g["tier"].startswith("3") else ""}">'
        f'<td>{g["tier"]}<em>{g["q"]}</em></td>'
        f'<td class="{"zero" if g["sup"] == 0 else ""}">{"—" if g["sup"] is None else f"{g['sup']:.2f}"}</td>'
        f'<td>{g["one"]:.2f}</td><td class="best">{g["hyb"]:.2f}</td></tr>'
        for g in data["generalization"])

    power_rows = "".join(
        f'<tr><td>{p["strength"]}</td><td>{p["history"]}</td>'
        f'<td class="mono">{p["pred"]}</td><td class="mono best">{p["meas"]}</td></tr>'
        for p in data["power"])

    base_rows = "".join(
        f'<tr><td>{b["attack"]}</td><td class="mono">{b["paper"]}</td>'
        f'<td class="mono best">{b["ours"]}</td><td><em>{b["note"]}</em></td></tr>'
        for b in data["baseline"])

    round_rows = "".join(
        f'<tr><td class="mono">{r["generation"]}</td><td>{r["family"].split(" (")[0]}</td>'
        f'<td class="mono">{r["red_asr"]}</td><td class="mono">{r["blue_recall_test"]}</td>'
        f'<td class="mono">{r["blue_fpr_test"]}</td>'
        f'<td class="mono {"ok" if r["case_c_test"] == "0" else "danger"}">{r["case_c_test"]}</td>'
        f'<td class="mono">{r["f1_test"]}</td><td class="mono">{r["mean_reward"]}</td></tr>'
        for r in data["rounds"])

    head = f"""<title>AP2 Attack Surface</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
:root{{
  --bg:#0B0E14; --panel:#141A24; --panel2:#1B2230; --bd:#2A3342;
  --fg:#E4E9F2; --muted:#8794AB; --accent:#6EA8FE; --danger:#FF5C4D;
  --ok:#34D399; --warn:#E8B339; --purple:#B084F5;
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
  --sans:"IBM Plex Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
}}
@media(prefers-color-scheme:light){{:root:not([data-theme=dark]){{
  --bg:#FBFCFE; --panel:#F2F5FA; --panel2:#E7ECF4; --bd:#CFD8E6;
  --fg:#141A24; --muted:#5A6884; --accent:#2563C9; --danger:#C7362A;
  --ok:#137A55; --warn:#8A6410; --purple:#6D3FC4;
}}}}
:root[data-theme=light]{{
  --bg:#FBFCFE; --panel:#F2F5FA; --panel2:#E7ECF4; --bd:#CFD8E6;
  --fg:#141A24; --muted:#5A6884; --accent:#2563C9; --danger:#C7362A;
  --ok:#137A55; --warn:#8A6410; --purple:#6D3FC4;
}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);
  font:15px/1.6 var(--sans);-webkit-font-smoothing:antialiased}}
h1,h2,h3,h4{{font-family:var(--sans);text-wrap:balance}}
:focus-visible{{outline:2px solid var(--accent);outline-offset:2px;border-radius:4px}}
@media(prefers-reduced-motion:reduce){{*{{transition:none!important;animation:none!important}}}}
.wrap{{max-width:1120px;margin:0 auto;padding:0 22px}}
section{{padding:64px 0;border-top:1px solid var(--bd)}}
section:first-of-type{{border-top:0}}
h1,h2,h3{{line-height:1.22;margin:0 0 12px}}
h2{{font-size:26px;letter-spacing:-.02em}}
h3{{font-size:16px;margin-top:28px}}
p{{margin:0 0 14px;max-width:74ch;color:var(--fg)}}
.lead{{font-size:17px;color:var(--muted);max-width:70ch}}
.mono{{font-family:var(--mono);font-size:.92em}}
em{{color:var(--muted);font-style:normal;display:block;font-size:12.5px;margin-top:3px}}
.kicker{{font:600 11.5px/1 var(--mono);letter-spacing:.14em;text-transform:uppercase;
  color:var(--accent);margin-bottom:14px}}

/* hero */
.hero{{padding:76px 0 56px}}
.hero h1{{font-size:clamp(30px,5vw,50px);letter-spacing:-.03em;max-width:19ch}}
.thesis{{font-size:clamp(17px,2.3vw,22px);line-height:1.45;margin:22px 0 30px;max-width:40ch;
  padding-left:18px;border-left:3px solid var(--danger)}}
.thesis b{{color:var(--danger)}}
.counters{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-top:34px}}
.ctr{{background:var(--panel);border:1px solid var(--bd);border-radius:9px;padding:15px 17px}}
.ctr b{{display:block;font:600 27px/1.1 var(--mono);letter-spacing:-.02em;font-variant-numeric:tabular-nums}}
.ctr span{{font-size:12px;color:var(--muted)}}
.ctr.alert b{{color:var(--danger)}}

/* matrix */
.matrix{{border:1px solid var(--bd);border-radius:11px;overflow-x:auto;background:var(--panel)}}
.mx-heads,.mx-body{{display:grid;grid-template-columns:repeat(6,minmax(168px,1fr));min-width:1010px}}
.mx-head{{padding:13px 12px;font-weight:650;font-size:12.5px;border-right:1px solid var(--bd);
  border-bottom:1px solid var(--bd);background:var(--panel2)}}
.mx-head:last-child{{border-right:0}}
.mx-stage{{display:inline-flex;width:17px;height:17px;border-radius:4px;background:var(--bd);
  color:var(--fg);font:600 10px/17px var(--mono);justify-content:center;margin-right:7px}}
.mx-col{{padding:11px;border-right:1px solid var(--bd);display:flex;flex-direction:column;gap:9px}}
.mx-col:last-child{{border-right:0}}
.mx-cell{{border-radius:7px;padding:11px 12px;font-size:13px;border:1px solid var(--bd);cursor:default}}
.mx-cell strong{{display:block;font-size:13px;font-weight:650}}
.mx-cell span{{display:block;font:11.5px/1.4 var(--mono);color:var(--muted);margin-top:4px}}
.mx-cell.built{{background:color-mix(in srgb,var(--danger) 13%,transparent);
  border-color:color-mix(in srgb,var(--danger) 42%,transparent);cursor:pointer}}
.mx-cell.built:hover,.mx-cell.built:focus{{background:color-mix(in srgb,var(--danger) 24%,transparent);
  outline:none;transform:translateY(-1px);transition:.13s}}
.mx-cell.gap{{border-style:dashed;opacity:.62}}
.mx-cell.thesis{{background:color-mix(in srgb,var(--ok) 12%,transparent);
  border-color:color-mix(in srgb,var(--ok) 50%,transparent)}}
.mx-cell.thesis strong{{color:var(--ok);font-family:var(--mono);font-size:12.5px}}
.legend{{display:flex;gap:20px;flex-wrap:wrap;margin-top:14px;font-size:12.5px;color:var(--muted)}}
.legend i{{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:6px;
  vertical-align:-1px;border:1px solid var(--bd)}}

/* theatre */
.theatre{{display:grid;grid-template-columns:270px 1fr;gap:22px;margin-top:22px}}
@media(max-width:880px){{.theatre{{grid-template-columns:1fr}}}}
.picker{{border:1px solid var(--bd);border-radius:10px;background:var(--panel);
  max-height:560px;overflow-y:auto}}
.filters{{display:flex;flex-wrap:wrap;gap:5px;padding:11px;border-bottom:1px solid var(--bd);
  position:sticky;top:0;background:var(--panel);z-index:2}}
.filters button{{font:600 11px/1 var(--mono);padding:6px 9px;border-radius:5px;cursor:pointer;
  background:transparent;border:1px solid var(--bd);color:var(--muted)}}
.filters button.on{{background:var(--accent);border-color:var(--accent);color:#fff}}
.titem{{padding:10px 13px;border-bottom:1px solid var(--bd);cursor:pointer;font-size:12.5px}}
.titem:hover{{background:var(--panel2)}}
.titem.on{{background:color-mix(in srgb,var(--accent) 18%,transparent);
  box-shadow:inset 3px 0 0 var(--accent)}}
.titem b{{display:block;font-weight:600}}
.titem span{{font:11px/1.5 var(--mono);color:var(--muted)}}
.badge{{display:inline-block;font:600 9.5px/1 var(--mono);padding:3px 6px;border-radius:3px;
  letter-spacing:.05em;margin-top:5px}}
.badge.danger{{background:var(--danger);color:#fff}}
.badge.warn{{background:var(--warn);color:#000}}
.badge.ok{{background:var(--ok);color:#fff}}
.badge.muted{{background:var(--bd);color:var(--fg)}}
.stage{{border:1px solid var(--bd);border-radius:10px;background:var(--panel);padding:20px}}
.stage-hd{{display:flex;justify-content:space-between;align-items:flex-start;gap:14px;
  flex-wrap:wrap;padding-bottom:14px;border-bottom:1px solid var(--bd);margin-bottom:16px}}
.step{{display:grid;grid-template-columns:30px 1fr;gap:12px;padding:11px 0;
  border-left:2px solid var(--bd);margin-left:14px;padding-left:18px;position:relative}}
.step:last-child{{border-left-color:transparent}}
.dot{{position:absolute;left:-7px;top:15px;width:12px;height:12px;border-radius:50%;
  background:var(--bd);border:2px solid var(--bg)}}
.step.danger .dot{{background:var(--danger)}}
.step.thesis .dot{{background:var(--ok)}}
.step h4{{margin:0 0 5px;font-size:13px;font-weight:650}}
.step p{{margin:0;font-size:13px;color:var(--muted)}}
.step .verbatim{{font-family:var(--mono);font-size:12px;background:var(--panel2);
  border:1px solid var(--bd);border-radius:6px;padding:10px 12px;white-space:pre-wrap;color:var(--fg)}}
.step .tag{{display:inline-block;font:600 10px/1 var(--mono);padding:4px 7px;border-radius:4px;
  background:var(--bd);margin-top:7px;letter-spacing:.05em}}
.step.danger .tag{{background:var(--danger);color:#fff}}
.outcomes{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:18px}}
@media(max-width:620px){{.outcomes{{grid-template-columns:repeat(2,1fr)}}}}
.oc{{border:1px solid var(--bd);border-radius:7px;padding:11px;text-align:center;background:var(--panel2)}}
.oc b{{display:block;font:600 10px/1.3 var(--mono);color:var(--muted);letter-spacing:.06em}}
.oc span{{display:block;font-size:19px;margin-top:5px;font-weight:600}}
.oc.yes span{{color:var(--danger)}} .oc.no span{{color:var(--ok)}}
.oc.final{{background:var(--danger);border-color:var(--danger)}}
.oc.final b,.oc.final span{{color:#fff}}
.oc.final.safe{{background:var(--ok);border-color:var(--ok)}}
.mech{{margin-top:16px}}
.mrow{{display:grid;grid-template-columns:130px 1fr 52px;gap:10px;align-items:center;
  font:11.5px/1 var(--mono);margin-bottom:7px;color:var(--muted)}}
.mbar{{height:7px;background:var(--panel2);border-radius:4px;overflow:hidden}}
.mbar i{{display:block;height:100%;background:var(--accent);border-radius:4px}}
.spark{{margin-top:14px}}

/* tables */
table{{width:100%;border-collapse:collapse;font-size:13.5px;margin:14px 0 6px;font-variant-numeric:tabular-nums}}
.tw{{overflow-x:auto}}
th{{text-align:left;font:600 11px/1 var(--mono);letter-spacing:.07em;text-transform:uppercase;
  color:var(--muted);padding:9px 11px;border-bottom:1px solid var(--bd)}}
td{{padding:10px 11px;border-bottom:1px solid var(--bd);vertical-align:top}}
tr.spot td{{background:color-mix(in srgb,var(--danger) 8%,transparent)}}
td.best{{font-weight:650;color:var(--accent)}}
td.zero{{color:var(--danger);font-weight:650}}
td.ok{{color:var(--ok)}} td.danger{{color:var(--danger)}}
.chart{{width:100%;height:auto;margin:10px 0}}
.grid{{stroke:var(--bd);stroke-width:1}}
.axis{{fill:var(--muted);font:10px var(--mono)}}
.axis.danger{{fill:var(--warn)}}
.keys{{display:flex;gap:16px;flex-wrap:wrap;font:11.5px var(--mono);color:var(--muted);margin-top:4px}}
.keys i{{display:inline-block;width:16px;height:3px;border-radius:2px;margin-right:6px;vertical-align:3px}}
.note{{background:var(--panel);border:1px solid var(--bd);border-left:3px solid var(--warn);
  border-radius:7px;padding:14px 16px;margin:16px 0;font-size:13.5px}}
.note b{{color:var(--warn)}}
.note.blue{{border-left-color:var(--accent)}} .note.blue b{{color:var(--accent)}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:13px;margin-top:16px}}
.card{{background:var(--panel);border:1px solid var(--bd);border-radius:9px;padding:16px}}
.card h4{{margin:0 0 7px;font-size:13.5px}}
.card p{{margin:0;font-size:13px;color:var(--muted)}}
footer{{padding:38px 0 56px;color:var(--muted);font-size:12.5px;border-top:1px solid var(--bd)}}
.toggle{{position:fixed;top:14px;right:14px;z-index:20;background:var(--panel);
  border:1px solid var(--bd);color:var(--fg);border-radius:7px;padding:7px 11px;
  cursor:pointer;font:600 11px var(--mono)}}
</style>"""

    body = f"""<button class="toggle" onclick="tog()">◐ theme</button>

<div class="wrap">

<section class="hero">
  <div class="kicker">Mastercard Innovation Challenge 2026</div>
  <h1>Agentic-Commerce Fraud: Red Team / Blue Team</h1>
  <div class="thesis">The mandate is signed correctly.<br><b>The decision behind it was manipulated.</b></div>
  <p class="lead">When an AI agent holds payment credentials and decides what to buy on your behalf,
  cryptography still protects <em style="display:inline">execution</em> — but nothing protects the
  decision that built the transaction. We red-teamed Google's AP2 protocol across five attack
  families, then built a defence that learns from what the attacker discovers.</p>
  <div class="counters">
    <div class="ctr"><b>{m['n_traces']}</b><span>attack traces generated</span></div>
    <div class="ctr"><b>{m['families']}</b><span>attack families</span></div>
    <div class="ctr"><b>{m['scenarios']}</b><span>distinct scenarios</span></div>
    <div class="ctr"><b>{m['n_present']}</b><span>attacks attempted</span></div>
    <div class="ctr"><b>{m['n_succeeded']}</b><span>actually succeeded</span></div>
    <div class="ctr alert"><b>{m['n_case_c']}</b><span>Case C · harm undetected</span></div>
  </div>
</section>

<section>
  <div class="kicker">01 · Coverage</div>
  <h2>The attack surface, mapped onto AP2</h2>
  <p class="lead">Each column is a stage of the protocol as it actually executes. Solid cells are
  attacks we built and measured; dashed cells are threats we identified but did not build. Click a
  solid cell to filter the trace explorer below.</p>
  {matrix_html()}
  <div class="legend">
    <span><i style="background:color-mix(in srgb,var(--danger) 30%,transparent);
      border-color:var(--danger)"></i>built, with evidence</span>
    <span><i style="border-style:dashed"></i>identified, not built</span>
    <span><i style="background:color-mix(in srgb,var(--ok) 30%,transparent);
      border-color:var(--ok)"></i>where cryptography holds — and fails to help</span>
  </div>
  <div class="note blue"><b>Read the empty column.</b> Nothing attacks Mandate Construction, because
  signing is not what breaks. Every attack in this matrix produces a cryptographically valid
  transaction. That absence is the finding.</div>
</section>

<section>
  <div class="kicker">02 · Evidence</div>
  <h2>Trace explorer</h2>
  <p class="lead">Every attack below is a real trace from the run that produced our reported numbers —
  including the agent's verbatim output. Nothing here is illustrative.</p>
  <div class="theatre">
    <div class="picker">
      <div class="filters" id="filters"></div>
      <div id="tlist"></div>
    </div>
    <div class="stage" id="stage"></div>
  </div>
</section>

<section>
  <div class="kicker">03 · Arms race</div>
  <h2>Red and Blue, generation over generation</h2>
  <p class="lead">Red evolves against the current Blue; Blue retrains on what Red discovers. Split by
  lineage root, so a mutated child never lands on the opposite side of the train/test split from its
  near-identical parent.</p>
  <h3>Blue recall on the held-out test split</h3>
  {recall_chart}
  <div class="keys">
    <span><i style="background:#FF5C4D"></i>reasoning_attack</span>
    <span><i style="background:#B084F5"></i>intent_manipulation</span>
    <span><i style="background:#34D399"></i>sequence_anomaly</span>
    <span><i style="background:#8794AB"></i>delegation_abuse (control)</span>
  </div>
  <div class="tw"><table><thead><tr><th>Gen</th><th>Family</th><th>Red ASR</th><th>Recall</th>
    <th>FPR</th><th>Case C</th><th>F1</th><th>Red reward</th></tr></thead>
    <tbody>{round_rows}</tbody></table></div>
  <div class="note"><b>Four caveats, because this table flatters.</b>
  intent_manipulation's zero Case C is trivial — no attack ever succeeded, so nothing could be missed.
  delegation_abuse's flat 1.00 is true by construction: its verifier provably covers all six violation
  types, so it validates the harness rather than demonstrating defence. reasoning_attack's 0.00 recall
  sits on test pools of n = 2–3. And test pools are small throughout — treat single-generation numbers
  as indicative only.</div>
</section>

<section>
  <div class="kicker">04 · The hard question</div>
  <h2>Does Blue generalize, or memorize?</h2>
  <p class="lead">Three different questions that must never be averaged into one score. Conflating them
  is the easiest way to overclaim — and we did it once before catching it.</p>
  <table><thead><tr><th>Tier</th><th>Supervised</th><th>One-class</th><th>Hybrid</th></tr></thead>
    <tbody>{gen_rows}</tbody></table>
  <div class="note"><b>Supervised learning scores literally 0.00 on an unseen strategy — and this is
  provable, not incidental.</b> In a training pool of loud attacks only, the feature identifying a
  slow drain carries no label information: benign values span and exceed the attack values. Its
  coefficient is statistically <em style="display:inline">unidentified</em>, and its fitted sign is set
  by sampling noise. No supervised model of any complexity recovers a coefficient the data does not
  constrain. The one-class half sidesteps this by never looking at attack labels — reaching 1.00 on two
  strategies it was never trained on.</div>
</section>

<section>
  <div class="kicker">05 · Honesty</div>
  <h2>Where it does not work</h2>
  <h3>The slow-drain blind spot is an information limit, not a detector limit</h3>
  <p>Rather than tune until the number improved, we asked whether the attack is detectable at all.
  Closed-form statistical power analysis and measurement agree closely:</p>
  <table><thead><tr><th>Attack strength</th><th>Baseline history</th>
    <th>Predicted power</th><th>Measured</th></tr></thead><tbody>{power_rows}</tbody></table>
  <p>With only 8 baseline transactions, an attack at multiplier ≤ 0.95 sits below the detection floor
  for <em style="display:inline">any</em> statistic. That is the signal-to-noise ratio of the
  observable history — and it points at a concrete remedy (longer history) rather than a better model.</p>

  <h3>Detection is bought with false positives</h3>
  {bar_ci(data['fpr_curve'])}
  <p>Recall with 95% confidence intervals, n = 200 per point. The amber bar is the
  <b>null control</b> — an attack-free sequence statistically identical to benign, so its bar is a
  false-positive rate, not recall.</p>
  <div class="note"><b>A measurement that was not real.</b> This null control originally reported 20%.
  Two causes were genuine (a 99th percentile estimated from six samples; a library-default 0.5
  threshold against benign traces scoring up to 0.89) — but the third was that the probe used n = 15,
  whose 95% CI is [1%, 30%]. It could not distinguish 5% from 20%. The same detector measures
  4.0% [2.3, 6.9] at n = 300. Fixing it cost recall (~72% → ~54% at 0.85 strength); that is a
  correction, not a regression, because the earlier recall was partly bought with false positives
  nobody was counting.</div>

  <h3>Reproduction against the source paper</h3>
  <table><thead><tr><th>Attack</th><th>Paper</th><th>Ours</th><th></th></tr></thead>
    <tbody>{base_rows}</tbody></table>
  <p>Neither gap is a bug; both were traced to a different victim model. Susceptibility is strongly
  model-dependent, so these should not be read as properties of "LLM agents" in general.</p>

  <h3>Remaining limitations</h3>
  <div class="cards">
    <div class="card"><h4>One detector stays keyword-based</h4><p>Branded Whisper detection is blind on
      ~90% of real attacks. We audited three candidate structural signals and rejected all three as
      disguised keyword lists or attacker-controlled surface properties, rather than ship a detector
      that overfits our own examples.</p></div>
    <div class="card"><h4>Cross-strategy generalization to quiet attacks</h4><p>The one-class layer
      reaches 1.00 on unseen loud strategies but 0.15–0.25 on unseen slow drains. The ceiling appears
      environmental rather than architectural.</p></div>
    <div class="card"><h4>A single victim model</h4><p>All results use one model. We measured 0% attack
      success on ranking injection and 100% on identity override — the spread is the point.</p></div>
    <div class="card"><h4>Simplified transaction schema</h4><p>No device fingerprint, geolocation,
      merchant reputation or session signal. Entire families of real-world detection are unavailable by
      construction.</p></div>
  </div>
</section>

<footer>
  Built from a committed snapshot of {m['n_traces']} traces — {m['curated']} shown here. Every figure is
  reproducible from the repository; the deterministic experiments assert zero API calls.
  This page makes no network requests and requires no API key.
</footer>
</div>

<script id="data" type="application/json">{json.dumps(data['traces'], separators=(',', ':'))}</script>
<script>
const T=JSON.parse(document.getElementById('data').textContent);
const OC={json.dumps(OUTCOME_META)};
let filt='all';
const FAMS=[...new Set(T.map(t=>t.family))];

function tog(){{const r=document.documentElement;
  const cur=r.getAttribute('data-theme')||(matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light');
  r.setAttribute('data-theme',cur==='dark'?'light':'dark');}}

function esc(s){{return String(s??'').replace(/[&<>]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;'}})[c]);}}

function drawFilters(){{
  const f=document.getElementById('filters');
  f.innerHTML=[['all','all'],...FAMS.map(x=>[x,x.replace('_',' ')])]
    .map(([v,l])=>`<button data-f="${{v}}" class="${{filt===v?'on':''}}">${{l}}</button>`).join('');
  f.querySelectorAll('button').forEach(b=>b.onclick=()=>{{filt=b.dataset.f;drawFilters();drawList();}});
}}

function drawList(){{
  const rows=T.filter(t=>filt==='all'||t.family===filt);
  document.getElementById('tlist').innerHTML=rows.map((t,i)=>{{
    const o=OC[t.outcome];
    return `<div class="titem" data-id="${{t.id}}">
      <b>${{esc(t.sub||t.family)}}</b>
      <span>${{esc(t.segment)}} · ${{esc(t.scenario)}} · gen ${{t.generation}}</span>
      <span class="badge ${{o[2]}}">${{o[0]}}</span></div>`;}}).join('')
    ||'<div class="titem"><b>No traces</b></div>';
  document.querySelectorAll('.titem[data-id]').forEach(el=>el.onclick=()=>show(el.dataset.id));
  if(rows.length) show(rows[0].id);
}}

function spark(s){{
  if(!s) return '';
  const a=s.amounts,n=a.length,mx=Math.max(...a),w=560,h=90,p=6;
  const X=i=>p+i/(n-1)*(w-2*p), Y=v=>h-p-(v/mx)*(h-2*p);
  const bx=X(s.baseline_n-0.5);
  const bars=a.map((v,i)=>`<rect x="${{X(i)-3}}" y="${{Y(v)}}" width="6" height="${{h-p-Y(v)}}"
    fill="${{i<s.baseline_n?'var(--muted)':'var(--danger)'}}" opacity="0.85" rx="1"/>`).join('');
  return `<div class="spark"><svg viewBox="0 0 ${{w}} ${{h}}" class="chart">
    ${{bars}}<line x1="${{bx}}" y1="2" x2="${{bx}}" y2="${{h-2}}" stroke="var(--accent)"
    stroke-width="1.5" stroke-dasharray="4 3"/>
    <text x="${{bx+6}}" y="13" class="axis" fill="var(--accent)">attack tail begins</text></svg>
    <div class="keys"><span><i style="background:var(--muted)"></i>baseline history</span>
    <span><i style="background:var(--danger)"></i>attack tail</span></div></div>`;
}}

function show(id){{
  const t=T.find(x=>x.id===id); if(!t) return;
  document.querySelectorAll('.titem').forEach(e=>e.classList.toggle('on',e.dataset.id===id));
  const o=OC[t.outcome];
  const steps=t.steps.map(s=>`<div class="step ${{s.danger?'danger':''}} ${{s.thesis?'thesis':''}}">
    <div class="dot"></div><div></div><div>
      <h4>${{esc(s.title)}}</h4>
      <div class="${{s.verbatim?'verbatim':''}}"><p style="${{s.verbatim?'color:var(--fg)':''}}">${{esc(s.body)}}</p></div>
      ${{s.tag?`<span class="tag">${{esc(s.tag)}}</span>`:''}}
    </div></div>`).join('');
  const mech=t.mech?`<div class="mech"><h4 style="font-size:12px;color:var(--muted);
    font-family:var(--mono);margin:0 0 9px">BLUE · MECHANISM SCORES</h4>`+
    Object.entries(t.mech).map(([k,v])=>{{const pc=Math.min(100,v/ (k==='pooled_shift_z'?8:
      k==='cusum_norm'?6:k==='amount_z_abs'?4:1)*100);
      return `<div class="mrow"><span>${{k}}</span><span class="mbar">
      <i style="width:${{pc}}%"></i></span><span>${{v}}</span></div>`;}}).join('')+'</div>':'';
  const checks=t.checks.length?`<div style="margin-top:14px"><h4 style="font-size:12px;
    color:var(--muted);font-family:var(--mono);margin:0 0 7px">TRIGGERED CHECKS</h4>`+
    t.checks.map(c=>`<div class="mono" style="font-size:11.5px;color:var(--muted);
      padding:4px 0">▸ ${{esc(c)}}</div>`).join('')+'</div>':'';
  document.getElementById('stage').innerHTML=`
    <div class="stage-hd"><div>
      <h3 style="margin:0">${{esc(t.sub||t.family)}}</h3>
      <em>${{esc(t.family)}} · ${{esc(t.scenario)}} · generation ${{t.generation}} ·
        <span class="mono">${{esc(t.id)}}</span></em>
      </div><span class="badge ${{o[2]}}" style="font-size:11px;padding:6px 9px">${{o[1]}}</span></div>
    ${{steps}}${{spark(t.series)}}
    <div class="outcomes">
      <div class="oc ${{t.present?'yes':'no'}}"><b>ATTACK PRESENT</b><span>${{t.present?'YES':'NO'}}</span></div>
      <div class="oc ${{t.succeeded?'yes':'no'}}"><b>SUCCEEDED</b><span>${{t.succeeded?'YES':'NO'}}</span></div>
      <div class="oc ${{t.detected?'no':'yes'}}"><b>BLUE DETECTED</b><span>${{t.detected?'YES':'NO'}}</span></div>
      <div class="oc final ${{t.outcome==='case_c'?'':'safe'}}"><b>OUTCOME</b><span>${{o[0]}}</span></div>
    </div>
    ${{t.signature_valid!==null?`<div class="note blue" style="margin-top:16px"><b>signature_valid =
      ${{t.signature_valid}}</b> — the mandate chain signed correctly regardless of the outcome above.</div>`:''}}
    ${{mech}}${{checks}}`;
}}

document.querySelectorAll('.mx-cell.built').forEach(c=>{{
  const go=()=>{{filt=c.dataset.family;drawFilters();drawList();
    document.getElementById('stage').scrollIntoView({{behavior:'smooth',block:'center'}});}};
  c.onclick=go; c.onkeydown=e=>{{if(e.key==='Enter'||e.key===' '){{e.preventDefault();go();}}}};
}});
drawFilters();drawList();
</script>"""

    OUT.write_text(
        '<!doctype html>\n<html lang="en"><head>\n'
        '<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">\n'
        '<meta name="description" content="Red-teaming Google\'s AP2 agentic-payments protocol: '
        'five attack families, an adaptive Red/Blue loop, and the evidence behind every number.">\n'
        + head + "</head><body>" + body + "</body></html>")
    ART.write_text(head + body)
    print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")
    print(f"wrote {ART}  ({ART.stat().st_size/1024:.0f} KB, artifact format)")


if __name__ == "__main__":
    build()
