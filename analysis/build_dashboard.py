"""Build the interactive results page from results/*.json.

    python -m analysis.build_dashboard            # -> docs/index.html (GitHub Pages ready)
    python -m analysis.build_dashboard --fragment out.html   # body-only variant for embedding
"""

import argparse
import json
import os

from analysis.summarize import ABLATIONS, ablation_curve

R = "results"
ARCHS = ["transformer", "mamba", "retnet", "hybrid"]


def j(path):
    return json.load(open(path)) if os.path.exists(path) else None


def jl(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def collect():
    data = {"models": {}, "env": None}
    ev = j(f"{R}/enwik8/eval.json") or {}
    for a in ARCHS:
        s = j(f"{R}/enwik8/{a}/summary.json")
        if not s:
            continue
        log = jl(f"{R}/enwik8/{a}/log.jsonl")
        e = ev.get(a, {})
        data["models"][a] = {
            "layers": s["layers"], "params": s["non_embedding_params"], "val_bpc": s["final_val_bpc"],
            "test_bpc": e.get("test_bpc"),
            "bpc_by_length": [[int(k), v] for k, v in e.get("bpc_by_length", {}).items() if v is not None],
            "position_loss": [[(b["start"] + b["end"]) / 2, b["bpc"]] for b in e.get("position_loss", [])],
            "curve": [[r["tokens"] / 1e6, r["val_bpc"]] for r in log if "val_bpc" in r and r["step"] > 0],
        }
    for name in ("models", "decode", "scan"):
        d = j(f"{R}/bench/{name}.json")
        if d:
            data["bench_" + name] = d["runs"]
            data["env"] = d["env"]
            if name == "models":
                data["train_tp"] = d.get("train_b16_l512", {})
    data["ablation"] = [{"name": n, "desc": desc, "curve": [[s, v] for s, v in ablation_curve(dd) if s > 0]}
                        for n, dd, desc in ABLATIONS]
    data["mqar"] = jl(f"{R}/synthetic/mqar.jsonl")
    data["selective"] = jl(f"{R}/synthetic/selective.jsonl")
    dl = j(f"{R}/delta.json")
    if dl:
        data["delta"] = {"text": dl["text"], "rel": [[round(v, 3) for v in row] for row in dl["delta_rel"]],
                         "classes": {k: v["rel_delta_by_layer"] for k, v in dl.get("class_stats", {}).items()},
                         "selective": dl.get("selective", {})}
    return data


HEAD = """<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
"""

PAGE = r"""<title>Mamba vs Transformer, From Scratch</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&display=swap">
<style>
/* Layout: one reading column like a paper; each finding = caption + one interactive figure. */
:root {
  --bg: #f6f6f3; --surface: #ffffff; --fg: #17181b; --fg-2: #50535a; --fg-3: #85878d;
  --rule: #e2e2dd; --grid: #ebebe7; --chip: #ecece8; --heat: 42, 120, 214;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --s5: #e87ba4; --s6: #008300;
  --display: "Source Serif 4", Georgia, "Times New Roman", serif;
  --body: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, Consolas, monospace;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #121313; --surface: #1a1a19; --fg: #f2f2ef; --fg-2: #c3c2b7; --fg-3: #8e8d87;
  --rule: #2d2d2a; --grid: #262624; --chip: #262624; --heat: 92, 156, 236;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --s6: #2f9a2f;
  color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #121313; --surface: #1a1a19; --fg: #f2f2ef; --fg-2: #c3c2b7; --fg-3: #8e8d87;
  --rule: #2d2d2a; --grid: #262624; --chip: #262624; --heat: 92, 156, 236;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --s6: #2f9a2f;
  color-scheme: dark; }
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--fg); font: 15px/1.6 var(--body); margin: 0; }
.page { max-width: 1040px; margin: 0 auto; padding-inline: 20px; padding-block: 40px 64px; }
h1, h2 { font-family: var(--display); font-weight: 600; text-wrap: balance; letter-spacing: -0.01em; margin: 0; }
h1 { font-size: clamp(30px, 5vw, 44px); line-height: 1.1; }
h2 { font-size: 24px; line-height: 1.25; }
p { margin: 0; max-width: 68ch; color: var(--fg-2); }
a { color: inherit; text-underline-offset: 3px; }
.eyebrow { font: 500 11.5px/1 var(--mono); text-transform: uppercase; letter-spacing: 0.08em; color: var(--fg-3); }
header { display: grid; gap: 14px; padding-bottom: 28px; border-bottom: 1px solid var(--rule); }
.env { font: 12.5px/1.5 var(--mono); color: var(--fg-3); }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-top: 28px; }
.tile { background: var(--surface); border: 1px solid var(--rule); border-radius: 10px; padding: 16px 18px; display: grid; gap: 6px; align-content: start; }
.tile .v { font: 600 28px/1.1 var(--display); font-variant-numeric: tabular-nums; }
.tile .k { font-size: 13px; color: var(--fg-2); line-height: 1.4; }
section { display: grid; gap: 14px; padding-block: 40px 8px; }
.fig { background: var(--surface); border: 1px solid var(--rule); border-radius: 10px; padding: 14px 14px 8px; min-width: 0; }
.fig-head { display: flex; flex-wrap: wrap; gap: 10px 18px; align-items: center; justify-content: space-between; padding: 2px 4px 8px; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 13px; color: var(--fg); }
.legend span { display: inline-flex; align-items: center; gap: 7px; white-space: nowrap; }
.legend i { width: 18px; height: 0; border-top: 2px solid; display: inline-block; }
.legend i.dash { border-top-style: dashed; }
.seg { display: inline-flex; background: var(--chip); border-radius: 8px; padding: 3px; gap: 2px; }
.seg button { font: 500 12.5px var(--body); color: var(--fg-2); background: transparent; border: 0; padding: 6px 11px; border-radius: 6px; cursor: pointer; }
.seg button[aria-pressed="true"] { background: var(--surface); color: var(--fg); box-shadow: 0 1px 2px rgba(0,0,0,.12); }
.seg button:focus-visible, input:focus-visible { outline: 2px solid var(--s1); outline-offset: 2px; }
.chart { position: relative; width: 100%; }
.chart svg { display: block; width: 100%; height: auto; overflow: visible; }
.chart text { font: 11.5px var(--mono); fill: var(--fg-3); }
.chart .lab { font: 12px var(--body); fill: var(--fg); }
.tip { position: absolute; pointer-events: none; background: var(--surface); border: 1px solid var(--rule); border-radius: 8px; padding: 8px 10px; font: 12px/1.5 var(--mono); color: var(--fg); box-shadow: 0 4px 16px rgba(0,0,0,.12); white-space: nowrap; z-index: 2; }
.tip b { font-weight: 500; }
.tip .sw { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
.two { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 14px; }
.note { font-size: 13px; color: var(--fg-3); padding: 4px 4px 6px; max-width: none; }
.tbl { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 8px 10px; border-bottom: 1px solid var(--rule); white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
th { font: 500 11.5px var(--mono); text-transform: uppercase; letter-spacing: .05em; color: var(--fg-3); }
td.best { font-weight: 600; }
.swatch { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 8px; vertical-align: 0; }
.bars { display: grid; gap: 8px; padding: 6px 4px 10px; }
.bar { display: grid; grid-template-columns: minmax(120px, 210px) 1fr 58px; gap: 12px; align-items: center; font-size: 13px; }
.bar .track { background: var(--chip); border-radius: 4px; height: 14px; position: relative; overflow: hidden; }
.bar .fill { position: absolute; inset: 0 auto 0 0; border-radius: 4px; }
.bar .fill.hatch { background-image: repeating-linear-gradient(45deg, transparent 0 4px, rgba(255,255,255,.45) 4px 7px); }
.bar .num { font: 12.5px var(--mono); text-align: right; font-variant-numeric: tabular-nums; }
.delta-text { font: 15px/2.1 var(--mono); padding: 10px 6px; word-break: break-all; }
.delta-text span { padding: 2px 0; border-radius: 2px; cursor: default; }
.ctrl { display: flex; flex-wrap: wrap; align-items: center; gap: 10px 14px; font-size: 13px; color: var(--fg-2); }
.ctrl input[type=range] { width: 180px; accent-color: var(--s2); }
.scale { display: flex; align-items: center; gap: 8px; font: 11.5px var(--mono); color: var(--fg-3); }
.scale i { width: 120px; height: 10px; border-radius: 3px; background: linear-gradient(90deg, rgba(var(--heat), .04), rgba(var(--heat), .85)); display: inline-block; }
footer { margin-top: 48px; padding-top: 20px; border-top: 1px solid var(--rule); display: grid; gap: 10px; }
code, pre { font-family: var(--mono); font-size: 13px; }
pre { background: var(--surface); border: 1px solid var(--rule); border-radius: 8px; padding: 12px 14px; overflow-x: auto; margin: 0; color: var(--fg); }
@media (prefers-reduced-motion: no-preference) { .tip { transition: opacity .1s; } }
</style>

<div class="page">
<header>
  <div class="eyebrow">Byte-level language modeling · RTX 3050 Laptop (4 GB, 2.97 GiB usable)</div>
  <h1>Mamba vs Transformer, from scratch</h1>
  <p>A selective state-space model (Mamba/S6), a GPT-style Transformer, RetNet-style linear attention and a Mamba–attention
  hybrid, all written in plain PyTorch with a hand-written Triton kernel for the selective scan. The four models are
  parameter-matched at about 4.7M parameters and measured for speed, memory and quality as sequences get longer.
  Every chart below is generated from the project's result files. <b>Status: preliminary.</b> Quality numbers come
  from one training seed with untuned learning rates; see docs/AUDIT.md for the open items.</p>
  <div class="env" id="env"></div>
  <div class="tiles" id="tiles"></div>
</header>

<section id="speed">
  <div class="eyebrow">Speed and memory</div>
  <h2>Efficiency depends on the kernel as much as the architecture</h2>
  <p>One forward pass or training step at batch size 1, until PyTorch's 2.97 GiB memory cap is hit (marked ×). Compare
  like with like: fused attention (PyTorch SDPA) with the fused Triton scan, or naive attention with the naive PyTorch scan.
  With fused kernels on both sides, attention is faster up to about 32K tokens and Mamba is faster beyond.</p>
  <div class="fig">
    <div class="fig-head"><div class="legend" id="lg-speed"></div>
      <div class="seg" id="seg-speed"><button data-k="fwd:ms">Forward time</button><button data-k="fwd:peak_mb">Forward memory</button><button data-k="train:ms">Training step</button></div></div>
    <div class="chart" id="c-speed"></div>
  </div>
  <div class="tbl" id="t-tp"></div>
</section>

<section id="decode">
  <div class="eyebrow">Generation</div>
  <h2>A KV cache grows with the context; a recurrent state does not</h2>
  <p>Cost of producing one token when the model already holds P tokens of context. Eager mode launches every kernel from
  Python. CUDA-graph mode replays a captured step, which removes launch overhead and shows the actual GPU work.</p>
  <div class="two">
    <div class="fig"><div class="fig-head"><div class="legend" id="lg-dec"></div>
      <div class="seg" id="seg-dec"><button data-k="ms_graph">CUDA graph</button><button data-k="ms">Eager</button></div></div>
      <div class="chart" id="c-dec"></div></div>
    <div class="fig"><div class="fig-head"><div class="legend"><span>Cache or state size, MB</span></div></div>
      <div class="chart" id="c-state"></div></div>
  </div>
</section>

<section id="quality">
  <div class="eyebrow">Quality on enwik8</div>
  <h2>Same parameters, same data: who compresses Wikipedia best?</h2>
  <p>Bits per byte on the held-out test split (lower is better), after 32.8M training bytes each, one seed per model.
  The right chart evaluates the same models on contexts longer than the 512 bytes they were trained on; the Transformer
  uses rotary embeddings without any length-extension method.</p>
  <div class="tbl" id="t-q"></div>
  <div class="two">
    <div class="fig"><div class="fig-head"><div class="legend" id="lg-curve"></div></div><div class="chart" id="c-curve"></div></div>
    <div class="fig"><div class="fig-head"><div class="legend"><span>Test bpc vs evaluation context</span></div>
      <div class="seg" id="seg-ext"><button data-k="len">By context length</button><button data-k="pos">By position</button></div></div>
      <div class="chart" id="c-ext"></div></div>
  </div>
</section>

<section id="ablation">
  <div class="eyebrow">Ablations</div>
  <h2>What happens without selection, and with other discretizations</h2>
  <p>Mamba variants trained on the same data with the same learning-rate schedule, compared over their first 1,000 steps.
  Preliminary: one seed, and the no-selection variant has 7.5% fewer parameters.</p>
  <div class="fig"><div class="fig-head"><div class="legend" id="lg-abl"></div></div><div class="chart" id="c-abl"></div></div>
</section>

<section id="scan">
  <div class="eyebrow">Systems</div>
  <h2>Six ways to compute the same recurrence</h2>
  <p>One Mamba layer's selective scan (1 sequence, 512 channels, 16 state dimensions). The PyTorch scans write every
  intermediate state to GPU memory; the fused Triton kernel keeps the state in registers.</p>
  <div class="fig"><div class="fig-head"><div class="legend" id="lg-scan"></div>
    <div class="seg" id="seg-scan"><button data-k="fwd:ms">Forward</button><button data-k="fwd_bwd:ms">Forward + backward</button><button data-k="fwd_bwd:peak_mb">Memory</button></div></div>
    <div class="chart" id="c-scan"></div></div>
</section>

<section id="recall">
  <div class="eyebrow">Synthetic tasks</div>
  <h2>Exact recall and selective copying</h2>
  <p>Small two-layer-sized models, 3,000 training steps each. Associative recall asks the model to retrieve the values of
  N stored keys. Selective copying asks it to reproduce 8 tokens scattered among noise. These charts measure learning
  speed under a fixed budget, not capability: pure Mamba reached 52.9% on 4-pair recall when trained for 9,000 steps.</p>
  <div class="two">
    <div class="fig"><div class="fig-head"><div class="legend" id="lg-mqar"></div></div><div class="chart" id="c-mqar"></div></div>
    <div class="fig"><div class="fig-head"><div class="legend"><span>Selective copying accuracy</span></div></div><div class="bars" id="b-sel"></div></div>
  </div>
</section>

<section id="delta">
  <div class="eyebrow">Inside the model</div>
  <h2>What the trained Mamba chooses to remember</h2>
  <p>Mamba's step size Δ decides how much of each byte enters the state. Darker means a larger Δ: the state is
  overwritten by this byte. Lighter means the byte is mostly skipped. Values are relative to the layer's mean.
  Hover a character to read its value.</p>
  <div class="fig">
    <div class="fig-head"><div class="ctrl"><label for="layer">Layer <b id="layer-v">1</b></label><input id="layer" type="range" min="1" max="1" value="1"></div>
      <div class="scale">skipped <i></i> overwrites</div></div>
    <div class="delta-text" id="dtext"></div>
  </div>
  <div class="fig"><div class="fig-head"><div class="legend"><span>Mean relative Δ by character type, selected layer</span></div></div><div class="bars" id="b-class"></div></div>
</section>

<footer>
  <div class="eyebrow">Reproduce</div>
  <pre>pip install -r requirements.txt
python reproduce.py        # tests, training, ablations, evaluation, benchmarks, synthetic tasks, figures</pre>
  <p class="note">Source: <a href="https://github.com/aman078-ai/mamba-ssm-benchmark">github.com/aman078-ai/mamba-ssm-benchmark</a></p>
</footer>
</div>

<script id="data" type="application/json">__DATA__</script>
<script>
(() => {
const D = JSON.parse(document.getElementById("data").textContent);
const NAMES = { transformer: "Transformer", mamba: "Mamba", retnet: "RetNet", hybrid: "Hybrid",
  transformer_sdpa: "Transformer · fused attn.", mamba_torchscan: "Mamba · PyTorch scan", mamba_lti: "Mamba w/o selection" };
const COLOR = { transformer: "--s1", mamba: "--s2", retnet: "--s3", hybrid: "--s4",
  transformer_sdpa: "--s1", mamba_torchscan: "--s2", mamba_lti: "--s2" };
const DASH = { transformer_sdpa: 1, mamba_torchscan: 1, mamba_lti: 1 };
const SCAN = { reference: ["autograd loop", "--s5"], sequential: ["sequential", "--s6"], hillis_steele: ["Hillis–Steele", "--s3"],
  blelloch: ["Blelloch", "--s4"], chunked: ["chunked", "--s1"], triton: ["Triton kernel", "--s2"] };
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const fmtLen = v => v >= 1024 ? (v / 1024 >= 1024 ? (v / 1048576) + "M" : (v / 1024) + "K") : String(v);
const fmtNum = v => v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v >= 1 ? v.toFixed(2) : v.toPrecision(2);
const pct = v => (v * 100).toFixed(1) + "%";
const $ = id => document.getElementById(id);
const NS = "http://www.w3.org/2000/svg";
const el = (tag, attrs = {}, parent) => { const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };

function legend(id, items) {
  $(id).innerHTML = items.map(s => `<span><i class="${s.dash ? "dash" : ""}" style="border-color:var(${s.color})"></i>${s.label}</span>`).join("");
}

// ---------- generic line chart: log or linear axes, OOM markers, crosshair tooltip ----------
function lineChart(host, o) {
  const draw = () => {
    host.innerHTML = "";
    const W = Math.max(300, host.clientWidth), H = o.height || Math.round(Math.min(380, Math.max(240, W * 0.42)));
    const m = { l: 58, r: o.labels === false ? 16 : 108, t: 12, b: 42 };
    if (W < 520) m.r = 16;
    const series = o.series.filter(s => s.points.length);
    const xs = series.flatMap(s => s.points.map(p => p[0]).concat(s.oom ? [s.oom[0]] : []));
    const ys = series.flatMap(s => s.points.map(p => p[1])).concat(o.yExtra || []);
    if (!xs.length) { host.innerHTML = '<p class="note">No data yet.</p>'; return; }
    const sx = scale(o.xlog, Math.min(...xs), Math.max(...xs), m.l, W - m.r, o.xlog ? 1.15 : 0.02);
    const sy = scale(o.ylog, o.ymin ?? Math.min(...ys), o.ymax ?? Math.max(...ys), H - m.b, m.t, o.ylog ? 1.25 : 0.06);
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.aria || "" }, host);
    // grid + ticks
    for (const t of sy.ticks(o.yfmt)) {
      el("line", { x1: m.l, x2: W - m.r, y1: sy(t.v), y2: sy(t.v), stroke: css("--grid") }, svg);
      el("text", { x: m.l - 8, y: sy(t.v) + 4, "text-anchor": "end" }, svg).textContent = t.s;
    }
    for (const t of sx.ticks(o.xfmt)) {
      el("line", { x1: sx(t.v), x2: sx(t.v), y1: m.t, y2: H - m.b, stroke: css("--grid") }, svg);
      el("text", { x: sx(t.v), y: H - m.b + 17, "text-anchor": "middle" }, svg).textContent = t.s;
    }
    el("text", { x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle" }, svg).textContent = o.xlabel;
    const yl = el("text", { x: 12, y: (m.t + H - m.b) / 2, "text-anchor": "middle", transform: `rotate(-90 12 ${(m.t + H - m.b) / 2})` }, svg);
    yl.textContent = o.ylabel;
    for (const r of (o.refs || [])) {
      const x = sx(r.x);
      el("line", { x1: x, x2: x, y1: m.t, y2: H - m.b, stroke: css("--fg-3"), "stroke-dasharray": "3 4" }, svg);
      el("text", { x: x + 5, y: m.t + 12 }, svg).textContent = r.label;
    }
    if (o.hline) {
      const y = sy(o.hline.y);
      el("line", { x1: m.l, x2: W - m.r, y1: y, y2: y, stroke: css("--fg-3"), "stroke-dasharray": "3 4" }, svg);
      el("text", { x: m.l + 6, y: y - 6 }, svg).textContent = o.hline.label;
    }
    // series
    const ends = [];
    for (const s of series) {
      const c = css(s.color);
      const d = s.points.map((p, i) => (i ? "L" : "M") + sx(p[0]).toFixed(1) + "," + sy(p[1]).toFixed(1)).join("");
      el("path", { d, fill: "none", stroke: c, "stroke-width": 2, "stroke-dasharray": s.dash ? "6 4" : "", "stroke-linejoin": "round" }, svg);
      if (o.markers !== false) for (const p of s.points)
        el("circle", { cx: sx(p[0]), cy: sy(p[1]), r: 3.6, fill: s.dash ? css("--surface") : c, stroke: s.dash ? c : css("--surface"), "stroke-width": s.dash ? 1.6 : 1.5 }, svg);
      if (s.oom) {
        const x = sx(s.oom[0]), y = sy(s.points[s.points.length - 1][1]);
        el("path", { d: `M${x - 5},${y - 5}L${x + 5},${y + 5}M${x - 5},${y + 5}L${x + 5},${y - 5}`, stroke: c, "stroke-width": 2.4, fill: "none" }, svg);
        if (o.labels === false || W < 520) el("text", { x: x + 8, y: y + 4 }, svg).textContent = "OOM";
      }
      const last = s.points[s.points.length - 1];
      const xMax = Math.max(...xs);
      if (last[0] === xMax || s.oom) ends.push({ y: sy(last[1]), x: sx(last[0]), label: s.label + (s.oom ? " · OOM" : ""), dash: s.dash });
    }
    if (m.r > 40 && o.labels !== false) {   // direct labels at line ends, nudged apart
      ends.filter(e => !e.dash).sort((a, b) => a.y - b.y).reduce((prev, e) => { e.y = Math.max(e.y, prev + 14); return e.y; }, -1e9);
      for (const e of ends.filter(e => !e.dash)) el("text", { x: W - m.r + 8, y: e.y + 4, class: "lab" }, svg).textContent = e.label;
    }
    // hover
    const tip = document.createElement("div"); tip.className = "tip"; tip.hidden = true; host.appendChild(tip);
    const cross = el("line", { y1: m.t, y2: H - m.b, stroke: css("--fg-3"), "stroke-width": 1, visibility: "hidden" }, svg);
    const xsU = [...new Set(series.flatMap(s => s.points.map(p => p[0])))].sort((a, b) => a - b);
    const hit = el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" }, svg);
    const move = ev => {
      const r = svg.getBoundingClientRect(), px = (ev.clientX - r.left) * W / r.width;
      let best = xsU[0]; for (const x of xsU) if (Math.abs(sx(x) - px) < Math.abs(sx(best) - px)) best = x;
      cross.setAttribute("x1", sx(best)); cross.setAttribute("x2", sx(best)); cross.setAttribute("visibility", "visible");
      const rows = series.map(s => { const p = s.points.find(q => q[0] === best);
        const oom = !p && s.oom && s.oom[0] <= best;
        return p || oom ? `<div><span class="sw" style="background:${css(s.color)}"></span>${s.label}: <b>${p ? (o.vfmt || fmtNum)(p[1]) : "out of memory"}</b></div>` : ""; }).join("");
      tip.innerHTML = `<div style="color:var(--fg-3)">${(o.tipx || (v => v))(best)}</div>` + rows; tip.hidden = false;
      const left = sx(best) * r.width / W; const tw = tip.offsetWidth;
      tip.style.left = (left + 14 + tw > r.width ? left - tw - 14 : left + 14) + "px"; tip.style.top = "8px";
    };
    hit.addEventListener("pointermove", move);
    hit.addEventListener("pointerleave", () => { tip.hidden = true; cross.setAttribute("visibility", "hidden"); });
  };
  draw(); host._redraw = draw;
}
new ResizeObserver(es => es.forEach(e => e.target._redraw && e.target._redraw())).observe(document.body);
window.addEventListener("resize", () => document.querySelectorAll(".chart").forEach(c => c._redraw && c._redraw()));
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => document.querySelectorAll(".chart").forEach(c => c._redraw && c._redraw()));
new MutationObserver(() => document.querySelectorAll(".chart").forEach(c => c._redraw && c._redraw())).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

function scale(log, lo, hi, a, b, pad) {
  if (log) { lo = lo / pad; hi = hi * pad; } else { const d = (hi - lo) || 1; lo -= d * pad; hi += d * pad; }
  const f = log ? (v => a + (Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo)) * (b - a)) : (v => a + (v - lo) / (hi - lo) * (b - a));
  f.ticks = fmt => {
    const out = [];
    if (log) {
      const base = (hi / lo > 40 && fmt === "len") ? 4 : 10;
      if (fmt === "len") { for (let v = 2 ** Math.ceil(Math.log2(lo)); v <= hi; v *= base === 4 ? 4 : 2) out.push({ v, s: fmtLen(v) }); }
      else { for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) for (const k of [1, 2, 5]) { const v = k * 10 ** e; if (v >= lo && v <= hi && (k === 1 || hi / lo < 300)) out.push({ v, s: fmtNum(v) }); } }
    } else {
      const step = nice((hi - lo) / 5); for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) out.push({ v, s: fmt === "pct" ? Math.round(v * 100) + "%" : +v.toFixed(3) + "" });
    }
    return out;
  };
  return f;
}
function nice(x) { const e = 10 ** Math.floor(Math.log10(x)); const f = x / e; return (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * e; }

function segmented(id, onPick) {
  const btns = [...$(id).querySelectorAll("button")];
  const pick = b => { btns.forEach(x => x.setAttribute("aria-pressed", x === b)); onPick(b.dataset.k); };
  btns.forEach(b => b.addEventListener("click", () => pick(b))); pick(btns[0]);
}

// ---------- header ----------
if (D.env) $("env").textContent = `${D.env.gpu} · ${D.env.vram_gib.toFixed(1)} GB · PyTorch ${D.env.torch} · CUDA ${D.env.cuda}`;
const tiles = [];
const ms = D.models;
const tested = Object.entries(ms).filter(([, m]) => m.test_bpc);
if (tested.length) { const [n, m] = tested.reduce((a, b) => a[1].test_bpc < b[1].test_bpc ? a : b);
  tiles.push([m.test_bpc.toFixed(3) + " bpc", `lowest enwik8 test bits per byte (${NAMES[n]}); one seed, preliminary`]); }
if (D.bench_models) {
  const at = (v, L) => D.bench_models.find(r => r.variant === v && r.L === L && !r.fwd.oom);
  const L = Math.max(...D.bench_models.filter(r => r.variant === "mamba" && !r.fwd.oom).map(r => r.L));
  const a = at("transformer_sdpa", L), b = at("mamba", L);
  if (a && b) tiles.push([(a.fwd.ms / b.fwd.ms).toFixed(1) + "× faster", `Mamba vs. fused attention, one forward pass at ${fmtLen(L)} tokens (fused attention wins below ~32K)`]);
}
if (D.bench_decode) {
  const big = Math.max(...D.bench_decode.map(r => r.context));
  const g = n => D.bench_decode.find(r => r.model === n && r.context === big);
  const t = g("transformer"), mb = g("mamba");
  if (t && mb && !mb.oom) tiles.push([t.oom ? "OOM vs " + fmtNum(mb.state_mb) + " MB" : fmtNum(t.state_mb) + " vs " + fmtNum(mb.state_mb) + " MB", `cache vs state at ${fmtLen(big)} tokens of context (Transformer vs Mamba)`]);
}
if (D.bench_scan) {
  const tri = D.bench_scan.filter(r => r.scan === "triton" && !r.fwd.oom);
  const L = Math.max(...tri.map(r => r.L));
  const b = tri.find(r => r.L === L);
  const torch = D.bench_scan.filter(r => r.L === L && r.scan !== "triton" && !r.fwd.oom);
  if (b && torch.length) { const best = torch.reduce((x, y) => x.fwd.ms < y.fwd.ms ? x : y);
    tiles.push([(best.fwd.ms / b.fwd.ms).toFixed(1) + "× faster", `fused Triton scan vs the best PyTorch scan (${best.scan}), forward pass at ${fmtLen(L)} tokens`]); }
}
$("tiles").innerHTML = tiles.map(([v, k]) => `<div class="tile"><div class="v">${v}</div><div class="k">${k}</div></div>`).join("");

// ---------- speed ----------
if (D.bench_models) {
  const variants = [...new Set(D.bench_models.map(r => r.variant))];
  legend("lg-speed", variants.map(v => ({ label: NAMES[v], color: COLOR[v], dash: DASH[v] })));
  segmented("seg-speed", k => {
    const [metric, field] = k.split(":");
    const series = variants.map(v => { const rr = D.bench_models.filter(r => r.variant === v);
      const ok = rr.filter(r => !r[metric].oom); const bad = rr.find(r => r[metric].oom && !r[metric].skipped);
      return { label: NAMES[v], color: COLOR[v], dash: DASH[v], points: ok.map(r => [r.L, r[metric][field]]), oom: bad ? [bad.L] : null }; });
    lineChart($("c-speed"), { series, xlog: true, ylog: true, xfmt: "len", xlabel: "sequence length (bytes)", tipx: v => fmtLen(v) + " bytes",
      ylabel: field === "ms" ? "milliseconds" : "MB", vfmt: field === "ms" ? (v => fmtNum(v) + " ms") : (v => fmtNum(v) + " MB"),
      hline: field === "peak_mb" && D.env ? { y: D.env.vram_gib * 1024, label: "GPU memory" } : null,
      yExtra: field === "peak_mb" && D.env ? [D.env.vram_gib * 1024] : [] });
  });
  const tp = D.train_tp || {};
  const rows = Object.entries(tp).filter(([, r]) => r.tokens_per_s);
  if (rows.length) $("t-tp").innerHTML = `<table><thead><tr><th>Training at batch 16 × 512</th><th>tokens / s</th><th>ms / step</th><th>peak memory</th></tr></thead><tbody>` +
    rows.map(([v, r]) => `<tr><td><span class="swatch" style="background:var(${COLOR[v]})"></span>${NAMES[v]}</td><td>${(r.tokens_per_s / 1000).toFixed(1)}k</td><td>${r.ms.toFixed(0)}</td><td>${(r.peak_mb / 1024).toFixed(2)} GB</td></tr>`).join("") + `</tbody></table>`;
}

// ---------- decode ----------
if (D.bench_decode) {
  const models = ["transformer", "mamba", "retnet", "hybrid"].filter(n => D.bench_decode.some(r => r.model === n));
  legend("lg-dec", models.map(n => ({ label: NAMES[n], color: COLOR[n] })));
  const ser = f => models.map(n => { const rr = D.bench_decode.filter(r => r.model === n);
    const ok = rr.filter(r => !r.oom && r[f] != null); const bad = rr.find(r => r.oom);
    return { label: NAMES[n], color: COLOR[n], points: ok.map(r => [r.context, r[f]]), oom: bad ? [bad.context] : null }; });
  segmented("seg-dec", f => lineChart($("c-dec"), { series: ser(f), xlog: true, ylog: true, xfmt: "len", xlabel: "tokens already in context",
    ylabel: "ms per token", vfmt: v => v.toFixed(2) + " ms", tipx: v => fmtLen(v) + " tokens of context" }));
  lineChart($("c-state"), { series: ser("state_mb"), xlog: true, ylog: true, xfmt: "len", xlabel: "tokens already in context",
    ylabel: "MB", vfmt: v => fmtNum(v) + " MB", tipx: v => fmtLen(v) + " tokens of context" });
}

// ---------- quality ----------
const qa = ["transformer", "mamba", "retnet", "hybrid"].filter(a => ms[a]);
if (qa.length) {
  const lens = [...new Set(qa.flatMap(a => ms[a].bpc_by_length.map(p => p[0])))].sort((a, b) => a - b).filter(L => L > 512);
  const best = Math.min(...qa.map(a => ms[a].test_bpc || 99));
  $("t-q").innerHTML = `<table><thead><tr><th>Model</th><th>Layers</th><th>Params</th><th>Val bpc</th><th>Test bpc</th>${lens.map(L => `<th>Test @ ${fmtLen(L)}</th>`).join("")}</tr></thead><tbody>` +
    qa.map(a => { const m = ms[a]; const by = Object.fromEntries(m.bpc_by_length);
      return `<tr><td><span class="swatch" style="background:var(${COLOR[a]})"></span>${NAMES[a]}</td><td><code>${m.layers}</code></td><td>${(m.params / 1e6).toFixed(2)}M</td><td>${m.val_bpc.toFixed(3)}</td>` +
        `<td class="${m.test_bpc === best ? "best" : ""}">${m.test_bpc ? m.test_bpc.toFixed(3) : "pending"}</td>${lens.map(L => `<td>${by[L] != null ? by[L].toFixed(3) : "OOM"}</td>`).join("")}</tr>`; }).join("") + `</tbody></table>`;
  legend("lg-curve", qa.map(a => ({ label: NAMES[a], color: COLOR[a] })));
  lineChart($("c-curve"), { series: qa.map(a => ({ label: NAMES[a], color: COLOR[a], points: ms[a].curve })), markers: false,
    xlabel: "training bytes (millions)", ylabel: "validation bpc", ymax: 3.0, vfmt: v => v.toFixed(3) + " bpc", tipx: v => v.toFixed(1) + "M bytes" });
  segmented("seg-ext", k => {
    if (k === "len") lineChart($("c-ext"), { series: qa.map(a => ({ label: NAMES[a], color: COLOR[a], points: ms[a].bpc_by_length })), xlog: true, xfmt: "len",
      xlabel: "evaluation context (trained at 512)", ylabel: "test bpc", refs: [{ x: 512, label: "train length" }], vfmt: v => v.toFixed(3) + " bpc", tipx: v => fmtLen(v) + " context" });
    else lineChart($("c-ext"), { series: qa.map(a => ({ label: NAMES[a], color: COLOR[a], points: ms[a].position_loss })), xlog: true, xfmt: "len", markers: false,
      ymin: 1.0, ymax: 4.0, xlabel: "position inside a 4K context", ylabel: "bpc at this position", refs: [{ x: 512, label: "train length" }], vfmt: v => v.toFixed(2) + " bpc", tipx: v => "position ≈ " + Math.round(v) });
  });
}

// ---------- ablation ----------
const ab = (D.ablation || []).filter(a => a.curve.length);
if (ab.length > 1) {
  const spec = { mamba: ["selective, exact ZOH (default)", "--s2", 0], mamba_lti: ["no selection (LTI)", "--s2", 1], mamba_euler: ["selective, Euler", "--s1", 0], mamba_bilinear: ["selective, bilinear", "--s3", 0] };
  legend("lg-abl", ab.map(a => ({ label: spec[a.name][0], color: spec[a.name][1], dash: spec[a.name][2] })));
  lineChart($("c-abl"), { series: ab.map(a => ({ label: spec[a.name][0], color: spec[a.name][1], dash: spec[a.name][2], points: a.curve })),
    xlabel: "training step (16 × 512 bytes each)", ylabel: "validation bpc", vfmt: v => v.toFixed(3) + " bpc", tipx: v => "step " + v, labels: false });
} else $("ablation").hidden = true;

// ---------- scans ----------
if (D.bench_scan) {
  const names = [...new Set(D.bench_scan.map(r => r.scan))];
  legend("lg-scan", names.map(n => ({ label: SCAN[n][0], color: SCAN[n][1] })));
  segmented("seg-scan", k => { const [metric, field] = k.split(":");
    lineChart($("c-scan"), { series: names.map(n => ({ label: SCAN[n][0], color: SCAN[n][1],
      points: D.bench_scan.filter(r => r.scan === n && !r[metric].oom).map(r => [r.L, r[metric][field]]) })),
      xlog: true, ylog: true, xfmt: "len", xlabel: "sequence length", ylabel: field === "ms" ? "milliseconds" : "MB",
      vfmt: field === "ms" ? (v => fmtNum(v) + " ms") : (v => fmtNum(v) + " MB"), tipx: v => fmtLen(v) + " tokens" }); });
}

// ---------- synthetic ----------
if ((D.mqar || []).length) {
  const archs = ["transformer", "mamba", "retnet", "hybrid"].filter(a => D.mqar.some(r => r.arch === a));
  legend("lg-mqar", archs.map(a => ({ label: NAMES[a], color: COLOR[a] })));
  lineChart($("c-mqar"), { series: archs.map(a => ({ label: NAMES[a], color: COLOR[a], points: D.mqar.filter(r => r.arch === a).map(r => [r.n_kv, r.acc]).sort((x, y) => x[0] - y[0]) })),
    xlog: true, xfmt: "len", ymin: 0, ymax: 1, yfmt: "pct", xlabel: "key-value pairs to recall (sequence of 128)", ylabel: "accuracy", vfmt: pct, tipx: v => v + " pairs" });
} else $("c-mqar").innerHTML = '<p class="note">Associative recall results pending.</p>';
function bars(id, rows, max) {
  $(id).innerHTML = rows.map(r => `<div class="bar"><span>${r.label}</span><span class="track"><span class="fill ${r.dash ? "hatch" : ""}" style="width:${Math.max(0, r.v / max * 100)}%;background-color:var(${r.color})"></span></span><span class="num">${r.text}</span></div>`).join("");
}
if ((D.selective || []).length) bars("b-sel", D.selective.map(r => ({ label: NAMES[r.arch], color: COLOR[r.arch], dash: DASH[r.arch], v: r.acc, text: pct(r.acc) })), 1);

// ---------- delta ----------
if (D.delta) {
  const L = D.delta.rel.length, s = $("layer"); s.max = L;
  const render = () => {
    const li = +s.value - 1; $("layer-v").textContent = s.value;
    const row = D.delta.rel[li];
    $("dtext").innerHTML = [...D.delta.text].map((ch, i) => { const v = row[i] ?? 1;
      const a = Math.min(0.9, Math.max(0.03, (Math.log2(v) + 1) / 3 * 0.9));
      const c = ch === " " ? "␣" : ch === "\n" ? "↵" : ch.replace(/&/g, "&amp;").replace(/</g, "&lt;");
      return `<span title="Δ = ${v.toFixed(2)}× layer mean" style="background:rgba(var(--heat),${a.toFixed(3)})">${c}</span>`; }).join("");
    const cls = Object.entries(D.delta.classes);
    if (cls.length) bars("b-class", cls.map(([k, v]) => ({ label: k, color: "--s2", v: v[li], text: v[li].toFixed(2) + "×" })).sort((a, b) => b.v - a.v),
      Math.max(...cls.map(([, v]) => v[li])) * 1.05);
  };
  s.addEventListener("input", render); render();
} else $("delta").hidden = true;
})();
</script>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/index.html")
    ap.add_argument("--fragment", default="", help="also write a body-only version (no <html>/<head>) here")
    args = ap.parse_args()
    data = json.dumps(collect(), separators=(",", ":")).replace("</", "<\\/")
    page = PAGE.replace("__DATA__", data)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("<!doctype html>\n<html lang=\"en\">\n<head>\n" + HEAD + "</head>\n<body>\n" + page + "\n</body>\n</html>\n")
    print(f"wrote {args.out}")
    if args.fragment:
        with open(args.fragment, "w", encoding="utf-8") as f:
            f.write(page)
        print(f"wrote {args.fragment}")


if __name__ == "__main__":
    main()
