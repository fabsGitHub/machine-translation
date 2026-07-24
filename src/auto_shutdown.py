"""
Stops this RunPod pod via the RunPod REST API once the pipeline finishes, to
avoid paying for idle GPU time after a long unattended run. Safe to use with
data on a persistent Network Volume - stopping (not terminating) the pod has
no effect on anything under /workspace, since that storage is not tied to the
pod's lifecycle.

Reads RUNPOD_API_KEY and RUNPOD_POD_ID from the environment - RunPod injects
both into every pod automatically, so no manual credential setup is needed.
Never pass the key on a command line (visible in `ps`) or hardcode it here.
"""
import os
import sys
import time
import json
import urllib.request
import urllib.error


def stop_this_pod(delay_seconds=45):
    """Stops the current pod. Prints a countdown first so a human watching the
    log has a window to Ctrl+C the pipeline process if they want to cancel."""
    api_key = os.environ.get("RUNPOD_API_KEY")
    pod_id = os.environ.get("RUNPOD_POD_ID")

    if not api_key or not pod_id:
        print("⚠️  Auto-shutdown skipped: RUNPOD_API_KEY or RUNPOD_POD_ID not set "
              "in the environment (not running on a RunPod pod?).")
        return False

    print(f"\n🛑 Pipeline complete. Stopping pod {pod_id} in {delay_seconds}s to save "
          f"GPU cost (data on the network volume is unaffected - Ctrl+C now to cancel).")
    for remaining in range(delay_seconds, 0, -5):
        print(f"   ...{remaining}s")
        time.sleep(5)

    url = f"https://rest.runpod.io/v1/pods/{pod_id}/stop"
    req = urllib.request.Request(url, method="POST", headers={
        "Authorization": f"Bearer {api_key}",
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            print(f"✅ Stop request accepted (HTTP {resp.status}): {body}")
            return True
    except urllib.error.HTTPError as e:
        print(f"⚠️  Stop request failed (HTTP {e.code}): {e.read().decode('utf-8', errors='replace')}")
        return False
    except Exception as e:
        print(f"⚠️  Stop request failed: {e}")
        return False


if __name__ == "__main__":
    delay = int(sys.argv[1]) if len(sys.argv) > 1 else 45
    stop_this_pod(delay_seconds=delay)
