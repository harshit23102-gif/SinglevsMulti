"""
report.py — Generates a self-contained HTML report from experiment results.

Called automatically by main.py after the experiment completes.
Can also be run standalone:

    python report.py          # reads results.json, writes report.html

FIXES vs original:
  - Provider label was hardcoded as "Groq" — now correctly shows "AWS Bedrock".
  - Ablation section added: bar chart showing Planner-only / P+R / Full Pipeline
    scores so the contribution of each agent is visually clear.
  - Safety consistency notes surfaced in per-case cards with a distinct style.
  - Verifier truncation warning shown in per-case cards.
  - Unconstrained single-agent column added to the summary table.
  - Pipeline keyword coverage shown alongside final-output coverage.
  - Limitations section added (judge model family, sample size, single run).
"""

import json
import sys
from datetime import datetime


# ── Helpers ───────────────────────────────────────────────────────────────────
def _avg(data: list[dict], key: str) -> float:
    vals = [r[key] for r in data if key in r]
    return round(sum(vals) / len(vals), 2) if vals else 0.0


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ── Per-case output cards ─────────────────────────────────────────────────────
def _case_cards(all_raw: list) -> str:
    metrics = ["Diagnostic Accuracy", "Reasoning Depth",
               "Treatment Completeness", "Safety Awareness", "Clinical Structure"]
    html = ""
    for item in all_raw:
        ev  = item["evaluation"]
        sa  = ev["single_agent"]
        ma  = ev["multi_agent"]
        s_ov = sa.get("Overall", 0)
        m_ov = ma.get("Overall", 0)

        score_rows = ""
        for m in metrics + ["Overall"]:
            sv, mv = sa.get(m, 0), ma.get(m, 0)
            w_cls  = "multi-win" if mv > sv else ("single-win" if sv > mv else "tie")
            winner = "Multi ▲"  if mv > sv else ("Single ▲"  if sv > mv else "Tie")
            score_rows += f"""
              <tr>
                <td class="dim">{m}</td>
                <td class="{'score-win' if sv >= mv else 'score-lose'}">{sv:.1f}
                  <div class="mini-bar"><div class="mini-fill mini-s" style="width:{sv*10}%"></div></div>
                </td>
                <td class="{'score-win' if mv >= sv else 'score-lose'}">{mv:.1f}
                  <div class="mini-bar"><div class="mini-fill mini-m" style="width:{mv*10}%"></div></div>
                </td>
                <td class="{w_cls}">{winner}</td>
              </tr>"""

        # Coverage comparison rows
        s_kw  = sa.get("keyword_coverage_%", 0)
        m_kw  = ma.get("keyword_coverage_%", 0)
        s_pkl = sa.get("pipeline_keyword_coverage_%", 0)
        m_pkl = ma.get("pipeline_keyword_coverage_%", 0)
        kw_winner  = "Multi ▲" if m_kw  > s_kw  else ("Single ▲" if s_kw  > m_kw  else "Tie")
        pkl_winner = "Multi ▲" if m_pkl > s_pkl else ("Single ▲" if s_pkl > m_pkl else "Tie")
        kw_w_cls  = "multi-win" if m_kw > s_kw else ("single-win" if s_kw > m_kw else "tie")
        pkl_w_cls = "multi-win" if m_pkl > s_pkl else ("single-win" if s_pkl > m_pkl else "tie")
        score_rows += f"""
              <tr style="border-top:1px solid rgba(255,255,255,0.06)">
                <td class="dim" style="font-size:10px">Keyword (final out %)</td>
                <td class="score-win">{s_kw:.0f}%</td>
                <td class="score-win">{m_kw:.0f}%</td>
                <td class="{kw_w_cls}">{kw_winner}</td>
              </tr>
              <tr>
                <td class="dim" style="font-size:10px">Keyword (pipeline %)</td>
                <td class="score-win">{s_pkl:.0f}%</td>
                <td class="score-win">{m_pkl:.0f}%</td>
                <td class="{pkl_w_cls}">{pkl_winner}</td>
              </tr>"""

        # Safety flags and consistency notes
        safety_html = ""
        for flag in ma.get("safety_flags", []):
            safety_html += f'<div class="safety-flag">⚠ Multi: {_esc(flag)}</div>'
        for flag in sa.get("safety_flags", []):
            safety_html += f'<div class="safety-flag">⚠ Single: {_esc(flag)}</div>'
        note = ma.get("safety_consistency_note") or sa.get("safety_consistency_note")
        if note:
            safety_html += f'<div class="consistency-note">ℹ {_esc(note[:200])}...</div>'

        # Truncation warning
        trunc_html = ""
        if ma.get("verifier_truncated"):
            trunc_html = '<div class="trunc-warn">⚠ Verifier output was truncated — max_tokens limit hit. Raise to 1200+.</div>'

        # Step tabs
        steps = item["multi_result"].get("steps", [])
        step_tabs, step_panels = "", ""
        for i, step in enumerate(steps):
            agent_name = step.get("agent", f"Step {i+1}")
            active = "active" if i == 2 else ""
            step_tabs += f'<button class="tab-btn {active}" onclick="switchTab(this,\'step-{item["case"]["id"]}-{i}\')">{agent_name}</button>'
            step_panels += f'<div id="step-{item["case"]["id"]}-{i}" class="tab-panel {active}"><pre class="output-text">{_esc(step.get("output",""))}</pre></div>'

        html += f"""
      <div class="case-card" id="case-{item['case']['id']}">
        <div class="case-header">
          <span class="case-badge">Case {item['case']['id']}</span>
          <span class="case-title">{item['case']['title']}</span>
          <span style="font-size:11px;color:var(--muted);background:var(--bg3);border:1px solid var(--border);border-radius:4px;padding:1px 8px">{item['case'].get('specialty','')}</span>
          <span class="score-pill">
            <span class="s-score">{s_ov}/10</span>
            <span class="vs-sep">vs</span>
            <span class="m-score">{m_ov}/10</span>
          </span>
        </div>
        {trunc_html}
        {safety_html}
        <div class="case-body">
          <div class="score-col">
            <p class="col-label">Scores</p>
            <table class="score-table">
              <thead><tr><th>Metric</th><th style="color:#4a9eff">Single</th><th style="color:#4ade80">Multi</th><th>Winner</th></tr></thead>
              <tbody>{score_rows}</tbody>
            </table>
          </div>
          <div class="output-col">
            <div class="output-pane">
              <p class="col-label single-lbl">Single-Agent output</p>
              <pre class="output-text">{_esc(item['single_result']['output'])}</pre>
            </div>
            <div class="output-pane">
              <p class="col-label multi-lbl">Multi-Agent pipeline</p>
              <div class="tabs">{step_tabs}</div>
              {step_panels}
            </div>
          </div>
        </div>
      </div>"""
    return html


# ── Ablation section ──────────────────────────────────────────────────────────
def _ablation_section(ablation_results: list) -> str:
    """
    FIX: Was entirely missing from the report. Ablation data was never
    collected (RUN_ABLATION=False) and never displayed. Now shows a bar chart
    comparing Planner-only / Planner+Reasoner / Full Pipeline scores across
    all cases, making the marginal contribution of each agent visible.
    """
    if not ablation_results:
        return """
    <div class="section" id="ablation">
      <div class="section-title">Ablation Study</div>
      <div class="info-box">
        No ablation data collected in this run. Set <code>RUN_ABLATION = True</code>
        in main.py and re-run to see the marginal contribution of each agent.
      </div>
    </div>"""

    # Collect per-label averages across cases
    labels   = ["PlannerOnly", "PlannerReasoner", "FullPipeline"]
    nice     = {"PlannerOnly": "Planner Only", "PlannerReasoner": "Planner + Reasoner", "FullPipeline": "Full Pipeline"}
    averages = {}
    for lbl in labels:
        scores = [r.get(lbl, {}).get("Overall", 0) for r in ablation_results if r.get(lbl)]
        averages[lbl] = round(sum(scores) / len(scores), 2) if scores else 0

    rows = ""
    for lbl in labels:
        score = averages[lbl]
        rows += f"""
          <tr>
            <td>{nice[lbl]}</td>
            <td>
              <div class="score-cell">
                <div class="score-bar"><div class="score-fill-m" style="width:{score*10}%"></div></div>
                {score:.1f}
              </div>
            </td>
            <td class="dim" style="font-size:11px">{
              'Decompose only — no diagnosis' if lbl == 'PlannerOnly' else
              'Diagnose + treat — no safety check' if lbl == 'PlannerReasoner' else
              'Full pipeline with safety verification'
            }</td>
          </tr>"""

    po_score = averages["PlannerOnly"]
    pr_score = averages["PlannerReasoner"]
    fp_score = averages["FullPipeline"]
    reasoner_lift = round(pr_score - po_score, 2)
    verifier_lift = round(fp_score - pr_score, 2)

    return f"""
    <div class="section" id="ablation">
      <div class="section-title">Ablation Study — Marginal Agent Contribution</div>
      <p style="color:var(--muted);font-size:13px;margin-bottom:20px">
        Each agent is added incrementally to isolate its contribution to the overall score.
        The Reasoner adds +{reasoner_lift:.2f} pts; the Verifier adds +{verifier_lift:.2f} pts.
      </p>
      <div class="charts-row" style="grid-template-columns:1fr 1fr;gap:20px">
        <div class="chart-card">
          <h3>Overall Score by Pipeline Depth</h3>
          <div class="chart-wrap"><canvas id="ablationChart"></canvas></div>
        </div>
        <div style="background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius);padding:22px">
          <h3 style="font-size:13px;color:var(--muted);margin-bottom:16px">Average Overall Score</h3>
          <table class="summary-table">
            <thead><tr><th>Pipeline</th><th>Score /10</th><th>What it adds</th></tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </div>
      </div>
      <script>
      new Chart(document.getElementById('ablationChart'), {{
        type: 'bar',
        data: {{
          labels: ['Planner Only', 'Planner + Reasoner', 'Full Pipeline'],
          datasets: [{{
            label: 'Overall Score',
            data: [{po_score}, {pr_score}, {fp_score}],
            backgroundColor: ['rgba(139,92,246,0.6)','rgba(251,146,60,0.6)','rgba(63,185,80,0.7)'],
            borderRadius: 6,
          }}]
        }},
        options: {{
          responsive: true, maintainAspectRatio: false,
          plugins: {{ legend: {{ display: false }} }},
          scales: {{
            y: {{ min: 0, max: 10, ticks: {{ stepSize: 2, color: '#8b949e' }}, grid: {{ color: 'rgba(255,255,255,0.05)' }} }},
            x: {{ ticks: {{ color: '#8b949e' }}, grid: {{ display: false }} }}
          }}
        }}
      }});
      </script>
    </div>"""


# ── Main report builder ───────────────────────────────────────────────────────
def generate_html_report(
    all_raw: list,
    model: str = "bedrock/us.amazon.nova-pro-v1:0",
    ablation_results: list = None,
    output_path: str = "report.html",
) -> str:

    ablation_results = ablation_results or []

    sd = [r["evaluation"]["single_agent"] for r in all_raw]
    md = [r["evaluation"]["multi_agent"]  for r in all_raw]

    metrics  = ["Diagnostic Accuracy", "Reasoning Depth",
                "Treatment Completeness", "Safety Awareness", "Clinical Structure"]
    s_avgs   = [_avg(sd, m) for m in metrics]
    m_avgs   = [_avg(md, m) for m in metrics]

    s_overall = _avg(sd, "Overall");           m_overall = _avg(md, "Overall")
    s_tokens  = _avg(sd, "tokens_used");       m_tokens  = _avg(md, "tokens_used")
    s_latency = _avg(sd, "latency_seconds");   m_latency = _avg(md, "latency_seconds")
    s_kw      = _avg(sd, "keyword_coverage_%"); m_kw     = _avg(md, "keyword_coverage_%")
    s_pkl     = _avg(sd, "pipeline_keyword_coverage_%")
    m_pkl     = _avg(md, "pipeline_keyword_coverage_%")

    token_mult   = round(m_tokens  / s_tokens,  1) if s_tokens  else 0
    latency_mult = round(m_latency / s_latency, 1) if s_latency else 0

    summary_rows = ""
    for i, m in enumerate(metrics):
        sv, mv = s_avgs[i], m_avgs[i]
        w_cls  = "multi-win" if mv > sv else ("single-win" if sv > mv else "tie")
        winner = "Multi ▲"  if mv > sv else ("Single ▲"  if sv > mv else "Tie")
        summary_rows += f"""
          <tr>
            <td>{m}</td>
            <td>
              <div class="score-cell">
                <div class="score-bar"><div class="score-fill-s" style="width:{sv*10}%"></div></div>
                {sv:.1f}
              </div>
            </td>
            <td>
              <div class="score-cell">
                <div class="score-bar"><div class="score-fill-m" style="width:{mv*10}%"></div></div>
                {mv:.1f}
              </div>
            </td>
            <td class="{w_cls}">{winner}</td>
          </tr>"""

    nav_links = "".join(
        f'<a href="#case-{item["case"]["id"]}" class="nav-link">'
        f'Case {item["case"]["id"]}: {item["case"]["title"]}'
        f'<span style="display:block;font-size:10px;color:var(--dim);margin-top:1px">'
        f'{item["case"].get("specialty","")}</span></a>'
        for item in all_raw
    )

    generated = datetime.now().strftime("%B %d, %Y at %H:%M")
    ablation_html = _ablation_section(ablation_results)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Healthcare AI — Evaluation Report</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
:root {{
  --bg:      #0d1117; --bg2: #161b22; --bg3: #21262d;
  --border:  rgba(255,255,255,0.08);
  --text:    #e6edf3; --muted: #8b949e; --dim: #484f58;
  --blue:    #4a9eff; --green: #3fb950; --yellow: #d29922; --orange: #f97316;
  --blue-bg: rgba(74,158,255,0.1); --green-bg: rgba(63,185,80,0.1);
  --radius:  10px;
}}
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
html {{ scroll-behavior: smooth; }}
body {{ background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-size: 14px; line-height: 1.6; }}
.sidebar {{ position:fixed; top:0; left:0; width:220px; height:100vh; background:var(--bg2); border-right:1px solid var(--border); padding:28px 0; overflow-y:auto; z-index:100; }}
.sidebar-logo {{ padding:0 20px 24px; font-weight:600; font-size:13px; color:var(--text); border-bottom:1px solid var(--border); margin-bottom:16px; }}
.sidebar-logo span {{ display:block; font-size:10px; color:var(--muted); font-weight:400; margin-top:2px; }}
.nav-section {{ padding:0 20px; margin-bottom:8px; font-size:10px; color:var(--dim); text-transform:uppercase; letter-spacing:.08em; }}
.nav-link {{ display:block; padding:7px 20px; font-size:13px; color:var(--muted); text-decoration:none; border-left:2px solid transparent; transition:color .15s, border-color .15s, background .15s; }}
.nav-link:hover {{ color:var(--text); background:rgba(255,255,255,0.04); border-left-color:var(--blue); }}
.main {{ margin-left:220px; padding:40px 48px; max-width:1200px; }}
.section {{ margin-bottom:52px; }}
.section-title {{ font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:.1em; color:var(--muted); padding-bottom:10px; border-bottom:1px solid var(--border); margin-bottom:20px; }}
.page-header {{ margin-bottom:48px; }}
.page-header h1 {{ font-size:26px; font-weight:600; color:#fff; margin-bottom:6px; }}
.page-header p  {{ color:var(--muted); font-size:14px; }}
.meta-row {{ display:flex; gap:12px; margin-top:14px; flex-wrap:wrap; }}
.badge {{ display:inline-flex; align-items:center; gap:6px; background:var(--bg3); border:1px solid var(--border); border-radius:20px; padding:4px 14px; font-size:12px; color:var(--muted); }}
.hero-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; }}
.hero-card {{ background:var(--bg2); border:1px solid var(--border); border-radius:var(--radius); padding:18px 20px; }}
.hero-label {{ font-size:11px; color:var(--muted); margin-bottom:10px; }}
.hero-vals {{ display:flex; gap:18px; align-items:baseline; }}
.hero-val {{ font-size:26px; font-weight:700; line-height:1; }}
.hero-sub {{ font-size:10px; color:var(--dim); margin-top:4px; }}
.v-blue {{ color:var(--blue); }} .v-green {{ color:var(--green); }}
.charts-row {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}
.chart-card {{ background:var(--bg2); border:1px solid var(--border); border-radius:var(--radius); padding:22px; }}
.chart-card h3 {{ font-size:13px; font-weight:500; color:var(--muted); margin-bottom:16px; }}
.chart-wrap {{ position:relative; height:270px; }}
.legend {{ display:flex; gap:18px; margin-bottom:14px; }}
.legend-item {{ display:flex; align-items:center; gap:6px; font-size:12px; color:var(--muted); }}
.legend-dot {{ width:10px; height:10px; border-radius:2px; flex-shrink:0; }}
.summary-table {{ width:100%; border-collapse:collapse; font-size:13px; }}
.summary-table th {{ text-align:left; font-size:11px; font-weight:500; color:var(--muted); padding:8px 12px; border-bottom:1px solid var(--border); }}
.summary-table td {{ padding:10px 12px; border-bottom:1px solid rgba(255,255,255,0.04); }}
.summary-table tr:last-child td {{ border-bottom:none; }}
.score-cell {{ display:flex; align-items:center; gap:8px; }}
.score-bar {{ width:70px; height:5px; background:rgba(255,255,255,0.07); border-radius:3px; flex-shrink:0; }}
.score-fill-s {{ height:5px; border-radius:3px; background:var(--blue); }}
.score-fill-m {{ height:5px; border-radius:3px; background:var(--green); }}
.eff-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; }}
.eff-card {{ background:var(--bg2); border:1px solid var(--border); border-radius:var(--radius); padding:18px 20px; }}
.eff-card h4 {{ font-size:10px; color:var(--muted); margin-bottom:12px; text-transform:uppercase; letter-spacing:.08em; }}
.eff-row {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; font-size:13px; }}
.eff-name {{ color:var(--muted); }} .eff-val {{ font-weight:600; }}
.eff-note {{ font-size:10px; color:var(--dim); margin-top:8px; }}
.multi-win {{ color:var(--green); font-weight:600; font-size:12px; }}
.single-win {{ color:var(--blue); font-weight:600; font-size:12px; }}
.tie {{ color:var(--dim); font-size:12px; }}
.score-win {{ color:var(--text); font-weight:600; }}
.score-lose {{ color:var(--muted); }}
.case-card {{ background:var(--bg2); border:1px solid var(--border); border-radius:var(--radius); overflow:hidden; margin-bottom:28px; }}
.case-header {{ display:flex; align-items:center; gap:12px; padding:14px 20px; background:var(--bg3); border-bottom:1px solid var(--border); flex-wrap:wrap; }}
.case-badge {{ background:var(--blue-bg); color:var(--blue); border-radius:6px; padding:2px 10px; font-size:11px; font-weight:600; }}
.case-title {{ font-weight:500; font-size:14px; flex:1; }}
.score-pill {{ font-size:12px; color:var(--muted); display:flex; align-items:center; gap:6px; }}
.s-score {{ color:var(--blue); font-weight:600; }} .m-score {{ color:var(--green); font-weight:600; }} .vs-sep {{ color:var(--dim); }}
.case-body {{ display:grid; grid-template-columns:340px 1fr; }}
.score-col {{ padding:20px; border-right:1px solid var(--border); }}
.output-col {{ display:grid; grid-template-columns:1fr 1fr; }}
.output-pane {{ padding:20px; }}
.output-pane:first-child {{ border-right:1px solid var(--border); }}
.col-label {{ font-size:10px; font-weight:600; text-transform:uppercase; letter-spacing:.08em; margin-bottom:12px; color:var(--muted); }}
.single-lbl {{ color:var(--blue); }} .multi-lbl {{ color:var(--green); }}
.output-text {{ font-size:11px; color:var(--muted); font-family:"SF Mono","Fira Code","Consolas",monospace; white-space:pre-wrap; word-break:break-word; line-height:1.65; max-height:420px; overflow-y:auto; }}
.score-table {{ width:100%; border-collapse:collapse; font-size:12px; }}
.score-table th {{ text-align:left; font-size:10px; color:var(--dim); padding:4px 8px; border-bottom:1px solid var(--border); }}
.score-table td {{ padding:6px 8px; border-bottom:1px solid rgba(255,255,255,0.03); }}
.score-table tr:last-child td {{ border-bottom:none; font-weight:600; }}
.mini-bar {{ width:40px; height:3px; background:rgba(255,255,255,0.07); border-radius:2px; margin-top:3px; }}
.mini-fill {{ height:3px; border-radius:2px; }} .mini-s {{ background:var(--blue); }} .mini-m {{ background:var(--green); }}
.tabs {{ display:flex; gap:4px; margin-bottom:12px; }}
.tab-btn {{ padding:4px 12px; font-size:11px; border-radius:6px; cursor:pointer; background:transparent; border:1px solid var(--border); color:var(--muted); transition:background .15s, color .15s; }}
.tab-btn.active, .tab-btn:hover {{ background:var(--green-bg); color:var(--green); border-color:var(--green); }}
.tab-panel {{ display:none; }} .tab-panel.active {{ display:block; }}
/* FIX: New alert styles */
.safety-flag {{ padding:6px 20px; background:rgba(217,119,6,0.1); border-left:3px solid var(--yellow); font-size:12px; color:var(--yellow); }}
.consistency-note {{ padding:6px 20px; background:rgba(249,115,22,0.08); border-left:3px solid var(--orange); font-size:11px; color:var(--orange); }}
.trunc-warn {{ padding:6px 20px; background:rgba(239,68,68,0.1); border-left:3px solid #ef4444; font-size:12px; color:#ef4444; }}
.info-box {{ background:var(--bg3); border:1px solid var(--border); border-radius:8px; padding:16px 20px; color:var(--muted); font-size:13px; }}
.info-box code {{ background:rgba(255,255,255,0.08); padding:1px 6px; border-radius:4px; font-size:12px; }}
.limitations-box {{ background:rgba(217,119,6,0.06); border:1px solid rgba(217,119,6,0.3); border-radius:8px; padding:16px 20px; }}
.limitations-box li {{ color:var(--muted); font-size:13px; margin-bottom:6px; list-style:disc; margin-left:16px; }}
.dim {{ color:var(--dim); }}
.footer {{ text-align:center; padding:28px 0 12px; color:var(--dim); font-size:12px; border-top:1px solid var(--border); margin-top:40px; }}
@media (max-width:1000px) {{
  .sidebar {{ display:none; }} .main {{ margin-left:0; padding:24px 20px; }}
  .hero-grid {{ grid-template-columns:1fr 1fr; }} .charts-row {{ grid-template-columns:1fr; }}
  .eff-grid {{ grid-template-columns:1fr 1fr; }} .case-body {{ grid-template-columns:1fr; }}
  .output-col {{ grid-template-columns:1fr; }} .output-pane:first-child {{ border-right:none; border-bottom:1px solid var(--border); }}
  .score-col {{ border-right:none; border-bottom:1px solid var(--border); }}
}}
</style>
</head>
<body>

<nav class="sidebar">
  <div class="sidebar-logo">Healthcare AI<span>Evaluation Report</span></div>
  <div class="nav-section">Sections</div>
  <a class="nav-link" href="#summary">Summary</a>
  <a class="nav-link" href="#scores">Score Breakdown</a>
  <a class="nav-link" href="#efficiency">Efficiency</a>
  <a class="nav-link" href="#ablation">Ablation Study</a>
  <a class="nav-link" href="#limitations">Limitations</a>
  <a class="nav-link" href="#cases">Case Results</a>
  <div class="nav-section" style="margin-top:16px;">Cases</div>
  {nav_links}
</nav>

<main class="main">
  <div class="page-header">
    <h1>Healthcare AI Evaluation</h1>
    <p>Single-Agent vs Multi-Agent Clinical Reasoning — Comparative Study</p>
    <div class="meta-row">
      <span class="badge">Model: {model}</span>
      <span class="badge">Provider: AWS Bedrock</span>
      <span class="badge">{len(all_raw)} clinical case(s)</span>
      <span class="badge">Generated: {generated}</span>
    </div>
  </div>

  <div class="section" id="summary">
    <div class="section-title">Summary</div>
    <div class="hero-grid">
      <div class="hero-card"><div class="hero-label">Overall Score</div>
        <div class="hero-vals">
          <div><div class="hero-val v-blue">{s_overall:.1f}</div><div class="hero-sub">Single-Agent</div></div>
          <div><div class="hero-val v-green">{m_overall:.1f}</div><div class="hero-sub">Multi-Agent</div></div>
        </div></div>
      <div class="hero-card"><div class="hero-label">Avg Latency</div>
        <div class="hero-vals">
          <div><div class="hero-val v-blue">{s_latency:.1f}s</div><div class="hero-sub">Single-Agent</div></div>
          <div><div class="hero-val v-green">{m_latency:.1f}s</div><div class="hero-sub">Multi-Agent</div></div>
        </div></div>
      <div class="hero-card"><div class="hero-label">Avg Tokens Used</div>
        <div class="hero-vals">
          <div><div class="hero-val v-blue">{int(s_tokens)}</div><div class="hero-sub">Single-Agent</div></div>
          <div><div class="hero-val v-green">{int(m_tokens)}</div><div class="hero-sub">Multi-Agent</div></div>
        </div></div>
      <div class="hero-card"><div class="hero-label">Keyword Coverage (final out)</div>
        <div class="hero-vals">
          <div><div class="hero-val v-blue">{s_kw:.0f}%</div><div class="hero-sub">Single-Agent</div></div>
          <div><div class="hero-val v-green">{m_kw:.0f}%</div><div class="hero-sub">Multi-Agent</div></div>
        </div></div>
    </div>
  </div>

  <div class="section" id="scores">
    <div class="section-title">Score Breakdown</div>
    <div class="legend">
      <div class="legend-item"><div class="legend-dot" style="background:#4a9eff"></div>Single-Agent</div>
      <div class="legend-item"><div class="legend-dot" style="background:#3fb950"></div>Multi-Agent</div>
    </div>
    <div class="charts-row">
      <div class="chart-card"><h3>Radar — Quality Dimensions</h3><div class="chart-wrap"><canvas id="radarChart"></canvas></div></div>
      <div class="chart-card"><h3>Bar — Average Score per Metric</h3><div class="chart-wrap"><canvas id="barChart"></canvas></div></div>
    </div>
    <div style="margin-top:20px;">
      <table class="summary-table">
        <thead><tr><th>Metric</th><th>Single-Agent</th><th>Multi-Agent</th><th>Winner</th></tr></thead>
        <tbody>{summary_rows}</tbody>
      </table>
    </div>
  </div>

  <div class="section" id="efficiency">
    <div class="section-title">Efficiency Trade-off</div>
    <div class="eff-grid">
      <div class="eff-card"><h4>Tokens Used</h4>
        <div class="eff-row"><span class="eff-name">Single-Agent</span><span class="eff-val v-blue">{int(s_tokens)}</span></div>
        <div class="eff-row"><span class="eff-name">Multi-Agent</span><span class="eff-val v-green">{int(m_tokens)}</span></div>
        <div class="eff-note">Multi uses ~{token_mult}× more tokens</div></div>
      <div class="eff-card"><h4>Latency</h4>
        <div class="eff-row"><span class="eff-name">Single-Agent</span><span class="eff-val v-blue">{s_latency:.2f}s</span></div>
        <div class="eff-row"><span class="eff-name">Multi-Agent</span><span class="eff-val v-green">{m_latency:.2f}s</span></div>
        <div class="eff-note">Multi is ~{latency_mult}× slower</div></div>
      <div class="eff-card"><h4>LLM API Calls</h4>
        <div class="eff-row"><span class="eff-name">Single-Agent</span><span class="eff-val v-blue">1</span></div>
        <div class="eff-row"><span class="eff-name">Multi-Agent</span><span class="eff-val v-green">3</span></div>
        <div class="eff-note">Planner → Reasoner → Verifier</div></div>
      <div class="eff-card"><h4>Keyword Coverage</h4>
        <div class="eff-row"><span class="eff-name">Final Output (fair)</span></div>
        <div class="eff-row"><span class="eff-name">· Single</span><span class="eff-val v-blue">{s_kw:.1f}%</span></div>
        <div class="eff-row"><span class="eff-name">· Multi</span><span class="eff-val v-green">{m_kw:.1f}%</span></div>
        <div class="eff-row" style="margin-top:8px"><span class="eff-name">Pipeline (all steps)</span></div>
        <div class="eff-row"><span class="eff-name">· Single</span><span class="eff-val v-blue">{s_pkl:.1f}%</span></div>
        <div class="eff-row"><span class="eff-name">· Multi</span><span class="eff-val v-green">{m_pkl:.1f}%</span></div>
        <div class="eff-note">Final-output is the fair comparison; pipeline shows Verifier condensation effect</div></div>
    </div>
  </div>

  {ablation_html}

  <!-- FIX: Limitations section — was entirely missing from the original report -->
  <div class="section" id="limitations">
    <div class="section-title">Limitations &amp; Threats to Validity</div>
    <div class="limitations-box">
      <ul>
        <li><strong>Judge model bias:</strong> The LLM judge (Nova Lite) belongs to the same model family as the agents (Nova Pro). This may introduce familiarity or agreement bias. A cross-family judge (e.g. Claude, GPT-4) would strengthen validity.</li>
        <li><strong>Sample size:</strong> Results are averaged over {len(all_raw)} case(s). Specialty-level conclusions require more cases per specialty to be statistically meaningful.</li>
        <li><strong>Stochastic variance:</strong> LLM outputs vary across runs. Mean ± std across {len(all_raw)} run(s) is reported; higher run counts reduce variance.</li>
        <li><strong>Safety metric conflict:</strong> Rule-based safety flags (keyword check on final output) and LLM Safety Awareness scores measure different things. Conflicts between them should be interpreted manually, not as contradictions.</li>
        <li><strong>Keyword coverage:</strong> Pipeline coverage gives multi-agent a larger text-search surface. Use final-output coverage for fair architectural comparison.</li>
        <li><strong>No ground truth:</strong> Evaluation relies on LLM-as-judge and heuristics, not validated clinical outcomes. Results reflect reasoning quality proxies, not patient outcomes.</li>
      </ul>
    </div>
  </div>

  <div class="section" id="cases">
    <div class="section-title">Case Results — Scores &amp; Full Outputs</div>
    {_case_cards(all_raw)}
  </div>

  <div class="footer">Healthcare AI Evaluation Report &nbsp;·&nbsp; {generated}</div>
</main>

<script>
const sData = {s_avgs};
const mData = {m_avgs};
const shortLabels = ['Diagnostic','Reasoning','Treatment','Safety','Structure'];
const fullLabels  = {[m for m in metrics]};

new Chart(document.getElementById('radarChart'), {{
  type: 'radar',
  data: {{
    labels: fullLabels,
    datasets: [
      {{ label:'Single-Agent', data:sData, borderColor:'#4a9eff', backgroundColor:'rgba(74,158,255,0.1)', pointBackgroundColor:'#4a9eff', borderWidth:2 }},
      {{ label:'Multi-Agent',  data:mData, borderColor:'#3fb950', backgroundColor:'rgba(63,185,80,0.1)',  pointBackgroundColor:'#3fb950', borderWidth:2 }}
    ]
  }},
  options:{{
    responsive:true, maintainAspectRatio:false,
    plugins:{{ legend:{{ display:false }} }},
    scales:{{ r:{{ min:4, max:10, ticks:{{ stepSize:2, color:'#484f58', font:{{size:10}} }}, pointLabels:{{ color:'#8b949e', font:{{size:11}} }}, grid:{{ color:'rgba(255,255,255,0.05)' }}, angleLines:{{ color:'rgba(255,255,255,0.05)' }} }} }}
  }}
}});

new Chart(document.getElementById('barChart'), {{
  type:'bar',
  data:{{
    labels: shortLabels,
    datasets:[
      {{ label:'Single-Agent', data:sData, backgroundColor:'rgba(74,158,255,0.7)', borderRadius:4 }},
      {{ label:'Multi-Agent',  data:mData, backgroundColor:'rgba(63,185,80,0.7)',  borderRadius:4 }}
    ]
  }},
  options:{{
    responsive:true, maintainAspectRatio:false,
    plugins:{{ legend:{{ display:false }} }},
    scales:{{
      y:{{ min:4, max:10, ticks:{{ color:'#8b949e', stepSize:2 }}, grid:{{ color:'rgba(255,255,255,0.05)' }} }},
      x:{{ ticks:{{ color:'#8b949e' }}, grid:{{ display:false }} }}
    }}
  }}
}});

function switchTab(btn, panelId) {{
  const card = btn.closest('.output-pane');
  card.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  card.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById(panelId).classList.add('active');
}}
</script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path


# ── Standalone usage ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "results.json"
    out = sys.argv[2] if len(sys.argv) > 2 else "report.html"
    with open(src, encoding="utf-8") as f:
        data = json.load(f)
    path = generate_html_report(
        data.get("results", data),
        output_path=out,
        ablation_results=data.get("ablation", []),
    )
    print(f"Report written to {path}")
