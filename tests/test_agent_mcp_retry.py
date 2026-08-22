# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Restarting an agent whose MCP server did not connect, and capping how many start at once.

A failed MCP server is not a crash. The agent keeps its built-in tools, loses score/submit/task
entirely, burns its whole budget and exits reporting success -- measured on qwen 604475, where one
such agent ran 36 minutes over 54 turns and called a `Submit` tool that does not exist. The harness
records rc=0 and the data point is simply gone, so the driver has to notice and relaunch.
"""
import importlib
import json
import pathlib
import sys
import threading
import time

EXAMPLE = pathlib.Path(__file__).resolve().parents[1] / "containers/cluster/example-script"


def load_driver(monkeypatch, **env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    sys.path.insert(0, str(EXAMPLE))
    try:
        return importlib.reload(importlib.import_module("agent_driver"))
    finally:
        sys.path.remove(str(EXAMPLE))


def init_line(status):
    return json.dumps({"subtype": "init", "mcp_servers": [{"name": "optarena", "status": status}]}) + "\n"


def fake_popen_class(statuses, seen=None):
    """Popen stand-in that writes one init event per spawn, taking each status in turn."""

    class FakePopen:
        spawned = 0
        live = 0

        def __init__(self, command, cwd=None, env=None, stdout=None, stderr=None):
            cls = type(self)
            cls.spawned += 1
            cls.live += 1
            if seen is not None:
                seen.append(cls.live)
            self.returncode = None
            status = statuses[min(cls.spawned - 1, len(statuses) - 1)]
            stdout.write(init_line(status))
            stdout.flush()

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = self.returncode or 0
            return self.returncode

        def terminate(self):
            type(self).live -= 1
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    return FakePopen


def start(driver, monkeypatch, tmp_path, statuses, seen=None):
    monkeypatch.setattr(driver.subprocess, "Popen", fake_popen_class(statuses, seen))
    log_path = tmp_path / "claude.log"
    with log_path.open("w", encoding="utf-8") as log:
        process, attempts = driver.start_agent(["claude"], tmp_path, {}, log, log_path)
    return process, attempts, log_path


def test_a_connected_server_is_not_retried(monkeypatch, tmp_path):
    driver = load_driver(monkeypatch)
    _, attempts, _ = start(driver, monkeypatch, tmp_path, ["connected"])
    assert attempts == 1


def test_a_failed_server_is_relaunched_until_it_connects(monkeypatch, tmp_path):
    driver = load_driver(monkeypatch)
    _, attempts, log_path = start(driver, monkeypatch, tmp_path, ["failed", "failed", "connected"])
    assert attempts == 3
    assert driver.mcp_failed(log_path) is False, "the surviving transcript is the connected attempt"


def test_the_retries_are_bounded_and_the_agent_still_runs(monkeypatch, tmp_path):
    """Exhausting the attempts must not lose the agent -- a crippled run still beats no run, and the
    log has to say which one this was."""
    driver = load_driver(monkeypatch, AGENT_MCP_ATTEMPTS="2")
    process, attempts, log_path = start(driver, monkeypatch, tmp_path, ["failed"])
    assert attempts == 2
    assert process is not None
    assert "MCP still not connected" in log_path.read_text(encoding="utf-8")


def test_a_retry_leaves_no_half_transcript(monkeypatch, tmp_path):
    """Downstream readers take the LAST result event; a dead attempt's output must not linger."""
    driver = load_driver(monkeypatch)
    _, _, log_path = start(driver, monkeypatch, tmp_path, ["failed", "connected"])
    assert log_path.read_text(encoding="utf-8").count('"subtype": "init"') == 1


def test_mcp_failed_distinguishes_not_yet_from_connected(monkeypatch, tmp_path):
    driver = load_driver(monkeypatch)
    log_path = tmp_path / "claude.log"
    log_path.write_text("", encoding="utf-8")
    assert driver.mcp_failed(log_path) is None, "no init event yet is not the same as a good one"
    log_path.write_text(init_line("connected"), encoding="utf-8")
    assert driver.mcp_failed(log_path) is False
    log_path.write_text(init_line("failed"), encoding="utf-8")
    assert driver.mcp_failed(log_path) is True


def test_only_so_many_agents_start_at_once(monkeypatch, tmp_path):
    """The whole point: 120 pool threads must not put 120 python3 servers on one node at once.

    The stand-in blocks inside the spawn, which is where the real one sits too -- the gate is held
    across the spawn AND the wait for the init event, so a slow startup holds its slot.
    """
    driver = load_driver(monkeypatch, AGENT_START_CONCURRENCY="3")
    release = threading.Event()
    spawned: list[int] = []
    lock = threading.Lock()

    class BlockingPopen:

        def __init__(self, command, cwd=None, env=None, stdout=None, stderr=None):
            with lock:
                spawned.append(1)
            self.returncode = None
            release.wait(timeout=30)
            stdout.write(init_line("connected"))
            stdout.flush()

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = 0
            return 0

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(driver.subprocess, "Popen", BlockingPopen)
    threads = []
    for index in range(12):
        workdir = tmp_path / str(index)
        workdir.mkdir()

        def run(workdir=workdir):
            log_path = workdir / "claude.log"
            with log_path.open("w", encoding="utf-8") as log:
                driver.start_agent(["claude"], workdir, {}, log, log_path)

        threads.append(threading.Thread(target=run))
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + 5
    while len(spawned) < 3 and time.monotonic() < deadline:
        time.sleep(0.05)
    held = len(spawned)
    release.set()
    for thread in threads:
        thread.join(timeout=30)
    assert held == 3, f"{held} agents were inside startup at once, cap is 3"
    assert len(spawned) == 12, "every agent must still get its turn"
