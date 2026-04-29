from __future__ import annotations

import html
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from inference import run_us_daily as us_daily
from reporter.send_report import save_report_locally, send_email

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("us_watchlist_report")

DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path.home() / "quantpilot_data")))
WATCHLIST_FILE = Path(
    os.environ.get("US_WATCHLIST_FILE", str(DATA_DIR / "config" / "us_watchlist.json"))
).expanduser()
REPORT_DIR = Path(
    os.environ.get("US_WATCHLIST_REPORT_DIR", str(DATA_DIR / "reports" / "us_watchlist"))
).expanduser()
SEND_EMAIL = os.environ.get("US_WATCHLIST_SEND_EMAIL", "true").lower() == "true"
WATCHLIST_CONCURRENCY = int(os.environ.get("US_WATCHLIST_CONCURRENCY", "2"))
WATCHLIST_TIMEOUT_SECONDS = int(
    os.environ.get("US_WATCHLIST_TIMEOUT_SECONDS", os.environ.get("US_ANALYSIS_TIMEOUT_SECONDS", "3600"))
)
WATCHLIST_RETRY_COUNT = int(os.environ.get("US_WATCHLIST_RETRY_COUNT", "0"))
WATCHLIST_RETRY_DELAY_SECONDS = int(os.environ.get("US_WATCHLIST_RETRY_DELAY_SECONDS", "5"))

DEFAULT_WATCHLIST = {
    "enabled": True,
    "updated_at": "",
    "symbols": [
        {"symbol": "LI", "name": "Li Auto", "enabled": True, "notes": ""},
        {"symbol": "SPY", "name": "SPDR S&P 500 ETF", "enabled": True, "notes": ""},
        {"symbol": "YINN", "name": "Direxion Daily FTSE China Bull 3X", "enabled": True, "notes": ""},
        {"symbol": "CQQQ", "name": "Invesco China Technology ETF", "enabled": True, "notes": ""},
    ],
    "analysis": {
        "concurrency": WATCHLIST_CONCURRENCY,
        "timeout_seconds": WATCHLIST_TIMEOUT_SECONDS,
        "retry_count": WATCHLIST_RETRY_COUNT,
    },
}


def normalize_symbol(symbol: str) -> str:
    raw = str(symbol or "").strip().upper()
    if raw.startswith("US."):
        raw = raw[3:]
    raw = raw.replace("-", ".")
    return raw


def normalize_code(symbol: str) -> str:
    raw = normalize_symbol(symbol)
    if not raw:
        raise ValueError("empty symbol")
    return f"US.{raw}"


def default_watchlist() -> dict[str, Any]:
    payload = dict(DEFAULT_WATCHLIST)
    payload["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    return payload


def load_watchlist(path: Path = WATCHLIST_FILE) -> dict[str, Any]:
    if not path.exists():
        return default_watchlist()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"watchlist must be a JSON object: {path}")
    payload.setdefault("enabled", True)
    payload.setdefault("symbols", [])
    payload.setdefault("analysis", {})
    return payload


def save_watchlist(payload: dict[str, Any], path: Path = WATCHLIST_FILE) -> Path:
    normalized_items = []
    seen: set[str] = set()
    for item in payload.get("symbols", []):
        if isinstance(item, str):
            item = {"symbol": item, "enabled": True, "name": "", "notes": ""}
        if not isinstance(item, dict):
            continue
        symbol = normalize_symbol(str(item.get("symbol", "")))
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        normalized_items.append(
            {
                "symbol": symbol,
                "name": str(item.get("name", "")).strip(),
                "enabled": bool(item.get("enabled", True)),
                "notes": str(item.get("notes", "")).strip(),
            }
        )

    payload = {
        "enabled": bool(payload.get("enabled", True)),
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "symbols": normalized_items,
        "analysis": dict(payload.get("analysis", {})),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)
    return path


def enabled_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    items = []
    for item in payload.get("symbols", []):
        if isinstance(item, str):
            item = {"symbol": item, "enabled": True, "name": "", "notes": ""}
        if not isinstance(item, dict) or not item.get("enabled", True):
            continue
        symbol = normalize_symbol(str(item.get("symbol", "")))
        if not symbol:
            continue
        items.append(
            {
                "symbol": symbol,
                "code": normalize_code(symbol),
                "name": str(item.get("name", "")).strip(),
                "notes": str(item.get("notes", "")).strip(),
            }
        )
    seen: set[str] = set()
    unique = []
    for item in items:
        if item["code"] in seen:
            continue
        seen.add(item["code"])
        unique.append(item)
    return unique


def apply_analysis_settings(payload: dict[str, Any]) -> None:
    settings = payload.get("analysis") if isinstance(payload.get("analysis"), dict) else {}
    concurrency = int(settings.get("concurrency") or WATCHLIST_CONCURRENCY)
    timeout = int(settings.get("timeout_seconds") or WATCHLIST_TIMEOUT_SECONDS)
    retry_count = int(settings.get("retry_count") if settings.get("retry_count") is not None else WATCHLIST_RETRY_COUNT)

    us_daily.US_ANALYSIS_CONCURRENCY = max(1, concurrency)
    us_daily.US_ANALYSIS_TIMEOUT_SECONDS = max(60, timeout)
    us_daily.US_ANALYSIS_RETRY_COUNT = max(0, retry_count)
    us_daily.US_ANALYSIS_RETRY_DELAY_SECONDS = max(0, WATCHLIST_RETRY_DELAY_SECONDS)


def is_budget_error(message: str) -> bool:
    lowered = str(message or "").lower()
    return "daily budget exceeded" in lowered or "request rejected (429)" in lowered


def _analyze_watch_code(
    code: str,
    candidate_scores: dict[str, float],
    expected_date: str,
) -> dict[str, Any]:
    log.info(f"Analyzing watchlist symbol {code} via deep-analysis ...")
    state = us_daily.run_deep_analysis(code)
    action, rating = us_daily.validate_trading_agents_state(state, code, expected_date)
    return {
        "code": code,
        "action": action,
        "rating": rating,
        "decision_score": us_daily.RATING_SCORES[rating],
        "candidate_score": float(candidate_scores.get(code, 0.0)),
        "run_id": state.get("run_id"),
        "state_path": state.get("state_path"),
        "reports": state.get("reports", {}),
        "analyst_summary": state.get("analyst_summary", ""),
        "debate_history": state.get("debate_history", []),
        "investment_plan": state.get("investment_plan", ""),
        "trade_proposal": state.get("trade_proposal", ""),
        "risk_history": state.get("risk_history", []),
        "final_decision": state.get("final_decision", ""),
        "contract_version": state.get("contract_version", ""),
        "status": "ok",
    }


def analyze_watchlist(items: list[dict[str, Any]], expected_date: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    analyses: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    if not items:
        return analyses, failures, skipped

    max_workers = max(1, us_daily.US_ANALYSIS_CONCURRENCY)
    candidate_scores = {item["code"]: float(len(items) - index) for index, item in enumerate(items)}
    code_to_item = {item["code"]: item for item in items}
    codes = [item["code"] for item in items]

    for offset in range(0, len(codes), max_workers):
        batch = codes[offset : offset + max_workers]
        results_by_code: dict[str, dict[str, Any]] = {}
        failures_by_code: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(batch)))) as executor:
            future_map = {
                executor.submit(_analyze_watch_code, code, candidate_scores, expected_date): code
                for code in batch
            }
            for future in as_completed(future_map):
                code = future_map[future]
                try:
                    results_by_code[code] = future.result()
                except Exception as exc:
                    message = str(exc)
                    log.error(f"watchlist deep-analysis failed for {code}: {message}")
                    failures_by_code[code] = {
                        "code": code,
                        "candidate_score": float(candidate_scores.get(code, 0.0)),
                        "status": "failed",
                        "error": message,
                    }
        batch_results = [results_by_code[code] for code in batch if code in results_by_code]
        batch_failures = [failures_by_code[code] for code in batch if code in failures_by_code]
        analyses.extend(batch_results)
        failures.extend(batch_failures)
        if any(is_budget_error(item.get("error", "")) for item in batch_failures):
            for code in codes[offset + max_workers :]:
                original = code_to_item.get(code, {})
                skipped.append(
                    {
                        "code": code,
                        "symbol": original.get("symbol", code.replace("US.", "")),
                        "name": original.get("name", ""),
                        "status": "skipped",
                        "error": "Claude daily budget exceeded; remaining watchlist skipped",
                    }
                )
            break

    for item in analyses:
        original = code_to_item.get(item["code"], {})
        item["symbol"] = original.get("symbol", item["code"].replace("US.", ""))
        item["name"] = original.get("name", "")
        item["notes"] = original.get("notes", "")
    for item in failures:
        original = code_to_item.get(item["code"], {})
        item["symbol"] = original.get("symbol", item["code"].replace("US.", ""))
        item["name"] = original.get("name", "")
        item["notes"] = original.get("notes", "")
    return analyses, failures, skipped


def _section(title: str, value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, (list, tuple)):
        text = "\n\n".join(str(item) for item in value if str(item).strip())
    elif isinstance(value, dict):
        text = json.dumps(value, ensure_ascii=False, indent=2)
    else:
        text = str(value)
    if not text.strip():
        return ""
    return (
        f"<section class=\"detail-section\">"
        f"<h4>{html.escape(title)}</h4>"
        f"<div class=\"prewrap\">{html.escape(text)}</div>"
        f"</section>"
    )


def build_report_html(payload: dict[str, Any]) -> str:
    analyses = payload["analyses"]
    failures = payload["failures"]
    skipped = payload["skipped"]
    generated_at = payload["generated_at"]
    date = payload["date"]
    source_path = payload["watchlist_file"]

    cards = []
    for item in analyses:
        reports = item.get("reports") if isinstance(item.get("reports"), dict) else {}
        parts = [
            _section("投组经理最终决策", item.get("final_decision", "")),
            _section("投资计划", item.get("investment_plan", "")),
            _section("交易员提案（仅作分析记录，不自动执行）", item.get("trade_proposal", "")),
            _section("分析师摘要", item.get("analyst_summary", "")),
            _section("市场/技术分析", reports.get("market", "")),
            _section("舆情/情绪分析", reports.get("sentiment", "")),
            _section("新闻/宏观分析", reports.get("news", "")),
            _section("基本面分析", reports.get("fundamentals", "")),
            _section("多空辩论", item.get("debate_history", [])),
            _section("风控讨论", item.get("risk_history", [])),
        ]
        title = item.get("symbol") or item.get("code", "").replace("US.", "")
        subtitle = " / ".join(part for part in [item.get("name", ""), item.get("rating", ""), item.get("action", "")] if part)
        cards.append(
            "<article class=\"stock-card\">"
            f"<h2>{html.escape(title)}</h2>"
            f"<div class=\"subtitle\">{html.escape(subtitle)}</div>"
            f"<div class=\"meta\">state: {html.escape(str(item.get('state_path', '')))}</div>"
            + "".join(parts)
            + "</article>"
        )

    issue_rows = []
    for item in failures + skipped:
        issue_rows.append(
            "<tr>"
            f"<td>{html.escape(str(item.get('symbol') or item.get('code', '')))}</td>"
            f"<td>{html.escape(str(item.get('status', 'failed')))}</td>"
            f"<td>{html.escape(str(item.get('error', ''))[:1000])}</td>"
            "</tr>"
        )
    issue_table = ""
    if issue_rows:
        issue_table = (
            "<h2>未完成股票</h2>"
            "<table><thead><tr><th>Symbol</th><th>Status</th><th>Error</th></tr></thead>"
            f"<tbody>{''.join(issue_rows)}</tbody></table>"
        )

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #1f2933; margin: 0; background: #f7f8fa; }}
.wrap {{ max-width: 980px; margin: 0 auto; padding: 28px 20px 48px; }}
h1 {{ margin: 0 0 6px; font-size: 28px; }}
h2 {{ margin: 0 0 8px; font-size: 22px; }}
h3 {{ margin: 18px 0 8px; }}
h4 {{ margin: 0 0 8px; font-size: 15px; color: #23395d; }}
.summary {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin: 20px 0; }}
.metric {{ background: #fff; border: 1px solid #d9dee7; border-radius: 8px; padding: 12px; }}
.metric-value {{ font-size: 24px; font-weight: 700; }}
.metric-label {{ font-size: 12px; color: #667085; margin-top: 4px; }}
.stock-card {{ background: #fff; border: 1px solid #d9dee7; border-radius: 8px; padding: 20px; margin: 18px 0; }}
.subtitle, .meta {{ color: #667085; font-size: 13px; margin-bottom: 8px; }}
.detail-section {{ border-top: 1px solid #edf0f5; padding-top: 14px; margin-top: 14px; }}
.prewrap {{ white-space: pre-wrap; line-height: 1.55; font-size: 14px; }}
table {{ border-collapse: collapse; width: 100%; background: #fff; margin-top: 10px; }}
th, td {{ border: 1px solid #d9dee7; padding: 8px; text-align: left; vertical-align: top; }}
th {{ background: #23395d; color: #fff; }}
.note {{ color: #667085; font-size: 13px; }}
@media (max-width: 700px) {{ .summary {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
</style>
</head>
<body>
<div class="wrap">
<h1>QuantPilot Watchlist Deep Analysis</h1>
<div class="note">Date: {html.escape(date)} | Generated: {html.escape(generated_at)}</div>
<div class="note">Watchlist: {html.escape(source_path)}</div>
<div class="summary">
  <div class="metric"><div class="metric-value">{payload["watchlist_count"]}</div><div class="metric-label">Enabled</div></div>
  <div class="metric"><div class="metric-value">{len(analyses)}</div><div class="metric-label">Succeeded</div></div>
  <div class="metric"><div class="metric-value">{len(failures)}</div><div class="metric-label">Failed</div></div>
  <div class="metric"><div class="metric-value">{len(skipped)}</div><div class="metric-label">Skipped</div></div>
</div>
<p class="note">This report is analysis-only. No automatic US trade plan or order execution is produced by this job.</p>
{''.join(cards) if cards else '<div class="stock-card"><h2>No completed analysis</h2></div>'}
{issue_table}
</div>
</body>
</html>
"""


def write_json_summary(payload: dict[str, Any], report_dir: Path = REPORT_DIR) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"us_watchlist_report_{payload['tag']}.json"
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)
    latest = report_dir / "us_watchlist_report_latest.json"
    latest_tmp = latest.with_suffix(f"{latest.suffix}.tmp")
    latest_tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    latest_tmp.replace(latest)
    return path


def run_watchlist_report() -> dict[str, Any]:
    watchlist = load_watchlist(WATCHLIST_FILE)
    save_watchlist(watchlist, WATCHLIST_FILE)
    if not watchlist.get("enabled", True):
        log.info("US watchlist report is disabled in config")

    apply_analysis_settings(watchlist)
    items = enabled_items(watchlist) if watchlist.get("enabled", True) else []
    expected_date = datetime.now().strftime("%Y-%m-%d")
    tag = datetime.now().strftime("%Y%m%d")
    log.info(f"US watchlist loaded: path={WATCHLIST_FILE} enabled_count={len(items)}")

    analyses, failures, skipped = analyze_watchlist(items, expected_date)
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    payload = {
        "tag": tag,
        "date": expected_date,
        "generated_at": generated_at,
        "watchlist_file": str(WATCHLIST_FILE),
        "watchlist_count": len(items),
        "analyses": analyses,
        "failures": failures,
        "skipped": skipped,
    }
    json_path = write_json_summary(payload, REPORT_DIR)
    html_report = build_report_html(payload)
    report_name = f"us_watchlist_report_{tag}.html"
    subject = f"QuantPilot Watchlist Deep Analysis - {expected_date}"
    if SEND_EMAIL:
        sent = send_email(
            html_report,
            subject,
            report_filename=report_name,
            report_dir=REPORT_DIR,
            attachment_paths=[json_path],
        )
        if not sent:
            raise RuntimeError("failed to send watchlist report email")
    else:
        save_report_locally(html_report, filename=report_name, report_dir=REPORT_DIR)

    return {
        "watchlist_count": len(items),
        "analysis_count": len(analyses),
        "failure_count": len(failures),
        "skipped_count": len(skipped),
        "report": str(REPORT_DIR / report_name),
        "summary": str(json_path),
    }


def main() -> None:
    try:
        result = run_watchlist_report()
        log.info(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as exc:
        log.error(f"US watchlist report failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
