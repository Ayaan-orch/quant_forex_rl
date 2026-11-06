"""
24/5 Production Watchdog & Health Supervisor.

Monitors `multi_asset_execution.py` continuously. If an unhandled exception or
broker socket disconnect causes the execution process to terminate, the supervisor
dispatches an emergency Discord alert, applies exponential backoff, and automatically
restarts the engine without human intervention.
"""

import sys
import time
import subprocess
import os
from pathlib import Path
from datetime import datetime, timezone
import requests

ROOT_DIR = Path(__file__).resolve().parent.parent
EXEC_SCRIPT = ROOT_DIR / "src" / "multi_asset_execution.py"
PYTHON_EXE = sys.executable

# Load webhook from .env if present
env_path = ROOT_DIR / ".env"
DISCORD_WEBHOOK_URL = ""
if env_path.exists():
    with open(env_path, "r") as f:
        for line in f:
            if line.strip().startswith("DISCORD_WEBHOOK_URL="):
                DISCORD_WEBHOOK_URL = line.strip().split("=", 1)[1].strip()


def send_supervisor_alert(title: str, message: str, color: int = 15548997):
    if not DISCORD_WEBHOOK_URL:
        return
    payload = {
        "embeds": [{
            "title": f"🛡️ [WATCHDOG] {title}",
            "description": message,
            "color": color,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }]
    }
    try:
        requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=5)
    except Exception:
        pass


def run_supervisor(max_consecutive_crashes: int = 10):
    print("=" * 65)
    print("  QUANT FOREX RL — 24/5 PRODUCTION EXECUTION SUPERVISOR")
    print(f"  Target Script: {EXEC_SCRIPT.name}")
    print("=" * 65)

    crash_count = 0
    backoff_delay = 5

    send_supervisor_alert("Daemon Started", "Production watchdog initialized. Monitoring execution daemon 24/5.", color=5763719)

    while True:
        start_time = time.time()
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Launching multi-asset execution daemon...")

        try:
            process = subprocess.Popen(
                [PYTHON_EXE, str(EXEC_SCRIPT)],
                cwd=str(ROOT_DIR)
            )
            return_code = process.wait()

            runtime = time.time() - start_time

            # If it ran for more than 15 minutes before exiting, reset crash counter
            if runtime > 900:
                crash_count = 0
                backoff_delay = 5

            if return_code == 0:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Daemon terminated cleanly (Code 0).")
                send_supervisor_alert("Daemon Stopped", "Multi-asset daemon terminated cleanly.", color=3447003)
                break
            else:
                crash_count += 1
                msg = f"Daemon crashed with exit code {return_code}. Uptime: {runtime:.1f}s. Restarting #{crash_count}..."
                print(f"\n⚠️ {msg}")
                send_supervisor_alert("Crash Detected", msg, color=15548997)

                if crash_count >= max_consecutive_crashes:
                    emergency_msg = f"Fatal: Reached {max_consecutive_crashes} consecutive crashes. Halting supervisor to prevent spam."
                    print(f"🛑 {emergency_msg}")
                    send_supervisor_alert("Supervisor Halting", emergency_msg, color=10038562)
                    break

                time.sleep(backoff_delay)
                backoff_delay = min(120, backoff_delay * 2)

        except KeyboardInterrupt:
            print("\nSupervisor stopped by user. Terminating daemon...")
            if process.poll() is None:
                process.terminate()
            break
        except Exception as e:
            print(f"Supervisor error: {e}")
            time.sleep(10)


if __name__ == "__main__":
    run_supervisor()
