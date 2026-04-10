from scripts import pretrade_watchdog


def test_inference_output_tag_strips_dashes():
    assert pretrade_watchdog.inference_output_tag("2026-04-09") == "20260409"


def test_ensure_signal_ready_syncs_before_inference(monkeypatch):
    states = iter(
        [
            pretrade_watchdog.WatchdogState(
                local_completed="2026-04-08",
                local_latest="2026-04-08",
                signal_date="2026-04-08",
                nas_completed="2026-04-09",
            ),
            pretrade_watchdog.WatchdogState(
                local_completed="2026-04-09",
                local_latest="2026-04-09",
                signal_date="2026-04-08",
                nas_completed="2026-04-09",
            ),
        ]
    )
    sync_targets: list[str] = []
    inference_targets: list[str] = []

    monkeypatch.setattr(pretrade_watchdog, "collect_state", lambda: next(states))
    monkeypatch.setattr(pretrade_watchdog, "sync_if_needed", lambda target: sync_targets.append(target))
    monkeypatch.setattr(pretrade_watchdog, "process_running", lambda patterns: False)
    monkeypatch.setattr(pretrade_watchdog, "run_inference", lambda target: inference_targets.append(target))
    monkeypatch.setattr(pretrade_watchdog, "latest_signal_date", lambda: "2026-04-09")

    assert pretrade_watchdog.ensure_signal_ready() == 0
    assert sync_targets == ["2026-04-09"]
    assert inference_targets == ["2026-04-09"]


def test_ensure_signal_ready_noops_when_aligned(monkeypatch):
    monkeypatch.setattr(
        pretrade_watchdog,
        "collect_state",
        lambda: pretrade_watchdog.WatchdogState(
            local_completed="2026-04-09",
            local_latest="2026-04-09",
            signal_date="2026-04-09",
            nas_completed="2026-04-09",
        ),
    )
    called = {"inference": False}
    monkeypatch.setattr(pretrade_watchdog, "run_inference", lambda target: called.__setitem__("inference", True))

    assert pretrade_watchdog.ensure_signal_ready() == 0
    assert called["inference"] is False
