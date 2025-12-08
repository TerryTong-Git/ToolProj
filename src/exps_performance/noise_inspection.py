from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from simple_parsing import parse

from src.exps_performance.arms import Arm1, Arm2, Arm3, Arm4
from src.exps_performance.dataset import make_dataset
from src.exps_performance.llm import llm
from src.exps_performance.noise import NOISE_FUNCS, clamp_sigma, perturb
from src.exps_performance.utils import seed_all_and_setup


@dataclass
class InspectionArgs:
    model: str = "dummy"
    backend: str = "dummy"
    batch_size: int = 2
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    exec_workers: int = 2
    checkpoint_every: int = 8
    seed: int = 1
    kinds: List[str] = field(
        default_factory=lambda: [
            "add",
            "sub",
            "mul",
            "lcs",
            "knap",
            "rod",
            "ilp_assign",
            "ilp_prod",
            "ilp_partition",
        ]
    )
    digits_list: List[int] = field(default_factory=lambda: [2])
    n: int = 2
    sigma: List[float] = field(default_factory=lambda: [0.0, 0.3, 0.6, 0.9])
    noise_types: List[str] = field(default_factory=lambda: list(NOISE_FUNCS.keys()))
    outdir: str = "results_noise/inspection"


def _run_all_arms(data: List[Any], args: InspectionArgs, client: Any) -> List[Any]:
    pipeline = (Arm2, Arm3, Arm4, Arm1)
    updated = data
    for arm_cls in pipeline:
        arm = arm_cls(updated, args, client)
        _, updated = arm.run()
    return updated


def _record_snapshot(q: Any) -> Dict[str, Any]:
    rec = q.record
    return {
        "nl": {"answer": rec.nl_answer, "correct": bool(rec.nl_correct), "err": rec.nl_err_msg},
        "sim": {"answer": rec.sim_answer, "correct": bool(rec.sim_correct), "err": rec.sim_err_msg},
        "controlsim": {"answer": rec.controlsim_answer, "correct": bool(rec.controlsim_correct), "err": rec.controlsim_err_msg},
        "code": {"answer": rec.code_answer, "correct": bool(rec.code_correct), "err": rec.code_err_msg},
    }


def _collect_cases(clean: List[Any], noisy: List[Any], noise_type: str, sigma: float) -> List[Dict[str, Any]]:
    cases = []
    for base_q, noisy_q in zip(clean, noisy):
        base_view = _record_snapshot(base_q)
        noisy_view = _record_snapshot(noisy_q)
        regressions = []
        improvements = []
        for arm in base_view:
            b_ok = base_view[arm]["correct"]
            n_ok = noisy_view[arm]["correct"]
            if b_ok and not n_ok:
                regressions.append(arm)
            if (not b_ok) and n_ok:
                improvements.append(arm)
        cases.append(
            {
                "kind": base_q.kind,
                "digit": base_q.digits,
                "noise_type": noise_type,
                "sigma": sigma,
                "question_clean": base_q.question,
                "question_noisy": noisy_q.question,
                "answer": base_q.answer,
                "baseline": base_view,
                "noisy": noisy_view,
                "regressions": regressions,
                "improvements": improvements,
            }
        )
    return cases


def _write_json(cases: List[Dict[str, Any]], outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / "noise_inspection.json"
    json_path.write_text(json.dumps(cases, indent=2))
    return json_path


def _write_html(cases: List[Dict[str, Any]], outdir: Path) -> Path:
    html_path = outdir / "noise_inspection.html"
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Noise Inspection</title>
  <style>
    body {{ font-family: Arial, sans-serif; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; }}
    th {{ background: #f4f4f4; }}
    .regression {{ background: #ffe0e0; }}
    .improvement {{ background: #e0ffe0; }}
  </style>
</head>
<body>
  <h2>Noise Inspection Report</h2>
  <label>Filter by noise type: <input id="noiseFilter" placeholder="e.g., numerical"></label>
  <table id="cases">
    <thead>
      <tr>
        <th>noise_type</th><th>sigma</th><th>kind</th><th>question (clean → noisy)</th><th>answer</th>
        <th>baseline</th><th>noisy</th><th>regressions</th><th>improvements</th>
      </tr>
    </thead>
    <tbody></tbody>
  </table>
  <script>
    const data = {json.dumps(cases)};
    const tbody = document.querySelector('#cases tbody');
    function render(filter="") {{
      tbody.innerHTML = "";
      data.filter(d => !filter || d.noise_type.includes(filter)).forEach(item => {{
        const tr = document.createElement('tr');
        const regressionClass = item.regressions.length ? "regression" : "";
        const improvementClass = item.improvements.length ? "improvement" : "";
        tr.innerHTML = `
          <td>${{item.noise_type}}</td>
          <td>${{item.sigma}}</td>
          <td>${{item.kind}}</td>
          <td>${{item.question_clean}}<hr>${{item.question_noisy}}</td>
          <td>${{item.answer}}</td>
          <td><pre>${{JSON.stringify(item.baseline, null, 2)}}</pre></td>
          <td><pre>${{JSON.stringify(item.noisy, null, 2)}}</pre></td>
          <td class="${{regressionClass}}">${{item.regressions.join(", ")}}</td>
          <td class="${{improvementClass}}">${{item.improvements.join(", ")}}</td>
        `;
        tbody.appendChild(tr);
      }});
    }}
    document.getElementById('noiseFilter').addEventListener('input', (e) => render(e.target.value));
    render();
  </script>
</body>
</html>
"""
    html_path.write_text(html)
    return html_path


def run(args: InspectionArgs) -> None:
    seed_all_and_setup(args)
    client = llm(args)
    base = list(make_dataset(args.kinds, args.n, args.digits_list))

    # baseline (clean)
    clean_data = copy.deepcopy(base)
    clean_completed = _run_all_arms(clean_data, args, client)

    cases: List[Dict[str, Any]] = []
    for noise_type in args.noise_types:
        for sigma in args.sigma:
            sigma_val = clamp_sigma(sigma)
            noisy_data = copy.deepcopy(base)
            for q in noisy_data:
                q.question = perturb(q.question, noise_type, sigma_val, args.seed)
            noisy_completed = _run_all_arms(noisy_data, args, client)
            cases.extend(_collect_cases(clean_completed, noisy_completed, noise_type, sigma_val))

    outdir = Path(args.outdir)
    json_path = _write_json(cases, outdir)
    html_path = _write_html(cases, outdir)
    print(f"Wrote inspection JSON to {json_path}")
    print(f"Wrote inspection HTML to {html_path}")


def main(args: Optional[list[str]] = None) -> None:
    parsed = parse(InspectionArgs, args=args)
    run(parsed)


if __name__ == "__main__":
    main()
