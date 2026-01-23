#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

PCER_PATTERN = "pcer_*.log"


# -----------------------------
# Helpers
# -----------------------------
def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_timestamp_str() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_name_for_dir(s: str) -> str:
    s = s.strip().replace("\\", "/")
    s = s.replace("/", "__")
    s = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in s)
    return s


def now_str() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_line(msg: str, fp: Optional[object] = None, also_print: bool = True) -> None:
    line = f"[{now_str()}] {msg}\n"
    if fp is not None:
        fp.write(line)
        fp.flush()
    if also_print:
        sys.stdout.write(line)
        sys.stdout.flush()


def resolve_config_path(cli_value: Optional[str]) -> Path:
    if cli_value:
        return Path(cli_value)
    script_dir = Path(__file__).resolve().parent
    return script_dir / "bench_config.json"


def list_pcer_files() -> List[Path]:
    # 在当前工作目录（项目根目录）匹配
    return sorted(Path(".").glob(PCER_PATTERN))


def get_target_timeout_seconds(cfg: Dict, tcfg: Dict) -> int:
    t = tcfg.get("timeout_seconds")
    if isinstance(t, int) and t > 0:
        return t
    t2 = cfg.get("default_timeout_seconds")
    if isinstance(t2, int) and t2 > 0:
        return t2
    return 60


def get_target_field_target(tcfg: Dict) -> str:
    t = tcfg.get("target")
    if not isinstance(t, str) or not t.strip():
        raise ValueError("Each targets[] entry must have non-empty string field 'target'")
    return t.strip()


def get_target_field_name(tcfg: Dict) -> str:
    n = tcfg.get("name")
    if isinstance(n, str):
        return n.strip()
    return ""


def make_target_output_dirname(target: str, name: str, ts: str) -> str:
    if name:
        return f"{target}_{name}_{ts}"
    return f"{target}_{ts}"


def resolve_pcer_analyzer_path() -> Optional[Path]:
    # 优先使用与 bench_runner.py 同目录的 pcer_analyzer.py
    local = Path(__file__).resolve().parent / "pcer_analyzer.py"
    if local.is_file():
        return local
    return None


# -----------------------------
# Process runner with watchdog timeout
# -----------------------------
def run_command_with_timeout(cmd: List[str], timeout_s: int, cwd: Path) -> Tuple[int, str, float]:
    """
    运行命令，捕获 stdout+stderr。
    超时 kill 整个进程组（SIGTERM -> 等2秒 -> SIGKILL）
    返回 (exit_code, combined_output, elapsed_seconds)
      exit_code: 0 success, 124 timeout, else subprocess returncode
    """
    start_time = time.time()

    p = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,  # 新进程组
    )

    output_chunks: List[str] = []
    try:
        while True:
            line = p.stdout.readline() if p.stdout else ""
            if line:
                output_chunks.append(line)

            ret = p.poll()
            if ret is not None:
                if p.stdout:
                    rest = p.stdout.read()
                    if rest:
                        output_chunks.append(rest)
                elapsed = time.time() - start_time
                return ret, "".join(output_chunks), elapsed

            if time.time() - start_time >= timeout_s:
                # 超时：TERM -> 2s -> KILL
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                time.sleep(2)
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

                if p.stdout:
                    try:
                        rest = p.stdout.read()
                        if rest:
                            output_chunks.append(rest)
                    except Exception:
                        pass

                elapsed = time.time() - start_time
                return 124, "".join(output_chunks), elapsed

            time.sleep(0.01)
    finally:
        try:
            if p.stdout:
                p.stdout.close()
        except Exception:
            pass


def move_new_pcer_logs(before: List[Path], after: List[Path], pcer_dir: Path) -> int:
    """
    仅移动本次新增的 pcer_*.log（用 before/after diff）
    """
    before_set = {p.resolve() for p in before}
    moved = 0
    for f in after:
        if f.resolve() in before_set:
            continue
        ensure_dir(pcer_dir)
        dest = pcer_dir / f.name
        try:
            shutil.move(str(f), str(dest))
            moved += 1
        except Exception:
            pass
    return moved


def run_pcer_analyzer(
    pcer_dir: Path,
    kernel_out_dir: Path,
    project_root: Path,
    trl_fp: Optional[object],
) -> Dict:
    """
    若 pcer_dir 中存在 pcer_*.log，则执行:
      python3 pcer_analyzer.py <pcer_dir>
    输出保存到 kernel_out_dir/pcer_analyzer.log
    """
    result = {
        "attempted": False,
        "script": None,
        "exit_code": None,
        "elapsed_seconds": None,
        "log_file": None,
        "reason": None,
    }

    if not pcer_dir.is_dir():
        result["reason"] = "pcer_dir_missing"
        return result

    if not any(pcer_dir.glob(PCER_PATTERN)):
        result["reason"] = "no_pcer_logs"
        return result

    analyzer_path = resolve_pcer_analyzer_path()
    script_to_run = str(analyzer_path) if analyzer_path else "pcer_analyzer.py"

    out_log = kernel_out_dir / "pcer_analyzer.log"
    result["attempted"] = True
    result["script"] = script_to_run
    result["log_file"] = str(out_log)

    cmd = ["python3", script_to_run, str(pcer_dir)]
    log_line(f"    PCER analyzer: {' '.join(cmd)}", trl_fp)

    start = time.time()
    try:
        cp = subprocess.run(
            cmd,
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        elapsed = time.time() - start
        out_log.write_text(cp.stdout or "", encoding="utf-8", errors="replace")
        result["exit_code"] = cp.returncode
        result["elapsed_seconds"] = round(elapsed, 3)

        if cp.returncode == 0:
            log_line(f"    ✅ PCER analyzer OK elapsed={elapsed:.3f}s log={out_log}", trl_fp)
        else:
            log_line(f"    ❌ PCER analyzer FAIL exit={cp.returncode} elapsed={elapsed:.3f}s log={out_log}", trl_fp)
    except FileNotFoundError as e:
        msg = f"pcer analyzer not found/executable: {e}"
        out_log.write_text(msg + "\n", encoding="utf-8")
        result["exit_code"] = 127
        result["elapsed_seconds"] = round(time.time() - start, 3)
        result["reason"] = "analyzer_not_found"
        log_line(f"    ❌ PCER analyzer ERROR: {msg}", trl_fp)
    except Exception as e:
        msg = f"pcer analyzer exception: {e}"
        out_log.write_text(msg + "\n", encoding="utf-8")
        result["exit_code"] = 1
        result["elapsed_seconds"] = round(time.time() - start, 3)
        result["reason"] = "analyzer_exception"
        log_line(f"    ❌ PCER analyzer ERROR: {msg}", trl_fp)

    return result


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=None,
        help="Path to bench_config.json (default: bench_config.json next to this script)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned tasks only; do not execute benchmark or write any files",
    )
    args = ap.parse_args()

    config_path = resolve_config_path(args.config)
    if not config_path.is_file():
        print(f"ERROR: config file not found: {config_path}", file=sys.stderr)
        return 2

    cfg = load_config(config_path)

    results_dir = Path(cfg.get("results_dir", "results"))
    timestamp = get_timestamp_str()
    project_root = Path(".").resolve()

    gapy_cfg = cfg.get("gapy", {}) or {}
    gapy_path = gapy_cfg.get("path", "./install/bin/gapy")
    platform = gapy_cfg.get("platform", "gvsoc")
    target_dir = gapy_cfg.get("target_dir", "install/generators")
    model_dir = gapy_cfg.get("model_dir", "install/models")

    targets = cfg.get("targets", [])
    if not isinstance(targets, list) or not targets:
        print("ERROR: config.targets must be a non-empty list", file=sys.stderr)
        return 2

    if args.dry_run:
        print(f"[DRY-RUN] Config   : {config_path.resolve()}")
        print(f"[DRY-RUN] Root     : {project_root}")
        print(f"[DRY-RUN] Results  : {results_dir}")
        print(f"[DRY-RUN] Timestamp: {timestamp}")
        print(f"[DRY-RUN] PCER     : {PCER_PATTERN} (if exists -> run pcer_analyzer.py <pcer_dir>)")
        print("")

    any_timeout_or_fail = False

    for tcfg in targets:
        try:
            target = get_target_field_target(tcfg)
        except ValueError as e:
            print(f"WARN: {e}", file=sys.stderr)
            any_timeout_or_fail = True
            continue

        name = get_target_field_name(tcfg)  # optional
        dirname = make_target_output_dirname(target, name, timestamp)
        target_out_dir = results_dir / dirname

        binary_dir_val = tcfg.get("binary_dir")
        if not isinstance(binary_dir_val, str) or not binary_dir_val.strip():
            print(f"WARN: target {target} missing binary_dir, skipped", file=sys.stderr)
            any_timeout_or_fail = True
            continue
        binary_dir = Path(binary_dir_val).resolve()

        binaries = tcfg.get("binaries", [])
        if not isinstance(binaries, list) or not binaries:
            print(f"WARN: target {target} has no binaries, skipped", file=sys.stderr)
            continue

        extra_args = tcfg.get("extra_args", [])
        if extra_args is None:
            extra_args = []
        if not isinstance(extra_args, list):
            raise ValueError(f"targets[].extra_args must be list, got {type(extra_args)}")

        timeout_s = get_target_timeout_seconds(cfg, tcfg)

        # ---------------- Dry run ----------------
        if args.dry_run:
            print(f"[DRY-RUN] TARGET: {target}")
            print(f"  name       : {name!r} (optional)")
            print(f"  output_dir : {target_out_dir}")
            print(f"  binary_dir : {binary_dir}")
            print(f"  timeout_s  : {timeout_s}")
            print(f"  runner_log : {target_out_dir / 'runner.log'}")
            print(f"  summary    : {target_out_dir / 'summary.json'}")
            print("")
            for b in binaries:
                if not isinstance(b, str) or not b.strip():
                    continue
                b = b.strip()
                binary_path = (binary_dir / b).resolve()
                kernel_out_dir = target_out_dir / safe_name_for_dir(b)
                out_log = kernel_out_dir / "terminal_output.log"
                pcer_dir = kernel_out_dir / "pcer"

                cmd = [
                    gapy_path,
                    f"--platform={platform}",
                    f"--target-dir={target_dir}",
                    f"--model-dir={model_dir}",
                    f"--target={target}",
                    "--binary",
                    str(binary_path),
                    "run",
                ] + extra_args

                print(f"  - KERNEL: {b}")
                print(f"    binary_path : {binary_path}")
                print(f"    out_dir     : {kernel_out_dir}")
                print(f"    stdout_log  : {out_log}")
                print(f"    pcer_dir    : {pcer_dir}")
                print(f"    cmd         : {' '.join(cmd)}")
                print(f"    on_pcer     : move {PCER_PATTERN} -> {pcer_dir} then run: python3 pcer_analyzer.py {pcer_dir}")
                print("")
            continue

        # ---------------- Real execution ----------------
        ensure_dir(results_dir)
        ensure_dir(target_out_dir)

        target_runner_log_path = target_out_dir / "runner.log"

        # per-target summary
        t_total = t_ok = t_timeouts = t_fails = t_missing = 0
        t_records: List[Dict] = []

        with target_runner_log_path.open("w", encoding="utf-8", buffering=1) as trl:
            log_line(f"========== TARGET: {target} ==========", trl)
            if name:
                log_line(f"Name      : {name}", trl)
            log_line(f"Output    : {target_out_dir}", trl)
            log_line(f"Binary dir: {binary_dir}", trl)
            log_line(f"Timeout   : {timeout_s}s", trl)
            log_line(f"PCER patt : {PCER_PATTERN}", trl)
            log_line(f"GAPY      : {gapy_path}", trl)
            log_line(f"Platform  : {platform}", trl)
            log_line(f"target_dir: {target_dir}", trl)
            log_line(f"model_dir : {model_dir}", trl)
            log_line(f"Config    : {config_path.resolve()}", trl)
            log_line(f"Timestamp : {timestamp}", trl)
            if extra_args:
                log_line(f"extra_args: {extra_args}", trl)
            log_line("", trl)

            for b_raw in binaries:
                if not isinstance(b_raw, str) or not b_raw.strip():
                    continue
                b = b_raw.strip()
                t_total += 1

                binary_path = (binary_dir / b).resolve()
                kernel_out_dir = target_out_dir / safe_name_for_dir(b)
                ensure_dir(kernel_out_dir)

                out_log = kernel_out_dir / "terminal_output.log"
                meta_json = kernel_out_dir / "meta.json"
                pcer_dir = kernel_out_dir / "pcer"

                meta = {
                    "target": target,
                    "name": name if name else None,
                    "timestamp": timestamp,
                    "output_dir": str(target_out_dir),
                    "binary_dir": str(binary_dir),
                    "binary_name": b,
                    "binary_path": str(binary_path),
                    "timeout_seconds": timeout_s,
                    "start_time": dt.datetime.now().isoformat(timespec="seconds"),
                    "command": None,
                    "exit_code": None,
                    "elapsed_seconds": None,
                    "pcer_moved": 0,
                    "pcer_analyzer": {
                        "attempted": False,
                        "script": None,
                        "exit_code": None,
                        "elapsed_seconds": None,
                        "log_file": None,
                        "reason": None,
                    },
                }

                log_line(f"--- Kernel: {b}", trl)
                log_line(f"    binary: {binary_path}", trl)
                log_line(f"    outdir: {kernel_out_dir}", trl)
                log_line(f"    log   : {out_log}", trl)

                if not binary_path.is_file():
                    msg = f"ERROR: binary not found: {binary_path}"
                    log_line(f"    ❌ {msg}", trl)
                    out_log.write_text(msg + "\n", encoding="utf-8")

                    meta["exit_code"] = 127
                    meta["elapsed_seconds"] = 0.0
                    meta["end_time"] = dt.datetime.now().isoformat(timespec="seconds")
                    meta_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

                    t_missing += 1
                    t_records.append(
                        {
                            "binary": b,
                            "status": "MISSING",
                            "exit_code": 127,
                            "elapsed_seconds": 0.0,
                            "output_dir": str(kernel_out_dir),
                            "log_file": str(out_log),
                            "pcer_moved": 0,
                            "pcer_analyzer_exit_code": None,
                            "pcer_analyzer_log_file": None,
                        }
                    )
                    any_timeout_or_fail = True
                    log_line("", trl)
                    continue

                cmd = [
                    gapy_path,
                    f"--platform={platform}",
                    f"--target-dir={target_dir}",
                    f"--model-dir={model_dir}",
                    f"--target={target}",
                    "--binary",
                    str(binary_path),
                    "run",
                ] + extra_args
                meta["command"] = cmd
                log_line(f"    cmd: {' '.join(cmd)}", trl)

                before_pcer = list_pcer_files()
                exit_code, output, elapsed = run_command_with_timeout(cmd, timeout_s, project_root)
                out_log.write_text(output, encoding="utf-8", errors="replace")
                after_pcer = list_pcer_files()
                moved = move_new_pcer_logs(before_pcer, after_pcer, pcer_dir)

                analyzer_info = {
                    "attempted": False,
                    "script": None,
                    "exit_code": None,
                    "elapsed_seconds": None,
                    "log_file": None,
                    "reason": None,
                }
                if pcer_dir.is_dir() and any(pcer_dir.glob(PCER_PATTERN)):
                    analyzer_info = run_pcer_analyzer(
                        pcer_dir=pcer_dir,
                        kernel_out_dir=kernel_out_dir,
                        project_root=project_root,
                        trl_fp=trl,
                    )

                meta["exit_code"] = exit_code
                meta["elapsed_seconds"] = round(elapsed, 3)
                meta["pcer_moved"] = moved
                meta["pcer_analyzer"] = analyzer_info
                meta["end_time"] = dt.datetime.now().isoformat(timespec="seconds")
                meta_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

                if exit_code == 0:
                    t_ok += 1
                    status = "OK"
                    log_line(f"    ✅ OK  elapsed={elapsed:.3f}s  pcer_moved={moved}", trl)
                elif exit_code == 124:
                    t_timeouts += 1
                    status = "TIMEOUT"
                    any_timeout_or_fail = True
                    log_line(f"    ⏱️  TIMEOUT elapsed={elapsed:.3f}s limit={timeout_s}s pcer_moved={moved}", trl)
                else:
                    t_fails += 1
                    status = "FAIL"
                    any_timeout_or_fail = True
                    log_line(f"    ❌ FAIL exit={exit_code} elapsed={elapsed:.3f}s pcer_moved={moved}", trl)

                t_records.append(
                    {
                        "binary": b,
                        "status": status,
                        "exit_code": exit_code,
                        "elapsed_seconds": round(elapsed, 3),
                        "output_dir": str(kernel_out_dir),
                        "log_file": str(out_log),
                        "pcer_moved": moved,
                        "pcer_analyzer_exit_code": analyzer_info.get("exit_code"),
                        "pcer_analyzer_log_file": analyzer_info.get("log_file"),
                    }
                )

                log_line("", trl)

            target_summary_path = target_out_dir / "summary.json"
            target_summary_path.write_text(
                json.dumps(
                    {
                        "target": target,
                        "name": name if name else None,
                        "timestamp": timestamp,
                        "config_path": str(config_path.resolve()),
                        "output_dir": str(target_out_dir),
                        "binary_dir": str(binary_dir),
                        "timeout_seconds": timeout_s,
                        "total": t_total,
                        "ok": t_ok,
                        "timeouts": t_timeouts,
                        "fails": t_fails,
                        "missing": t_missing,
                        "records": t_records,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            log_line("============== TARGET SUMMARY ==============", trl)
            log_line(f"total    : {t_total}", trl)
            log_line(f"ok       : {t_ok}", trl)
            log_line(f"timeouts : {t_timeouts}", trl)
            log_line(f"fails    : {t_fails}", trl)
            log_line(f"missing  : {t_missing}", trl)
            log_line(f"summary  : {target_summary_path}", trl)
            log_line("Target done.", trl)

    return 0 if not any_timeout_or_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
