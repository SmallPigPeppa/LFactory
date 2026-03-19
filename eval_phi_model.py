import argparse
import json
from pathlib import Path
from urllib import error, request

from phi_runtime import PhiRuntime


def score_reply(reply: str, scoring: dict) -> dict:
    """Score a reply against required and bad term lists.

    Returns a dict with:
      score          float 0.0-1.0 (required hit rate, penalised for bad hits)
      pass           bool (score >= pass_threshold AND no bad hits)
      required_hits  list of required terms found
      required_misses list of required terms not found
      bad_hits       list of bad terms found
    """
    text = reply.lower()
    required = scoring.get("required", [])
    bad = scoring.get("bad", [])
    threshold = scoring.get("pass_threshold", 0.5)

    def normalize_required_term(term) -> tuple[str, tuple[str, ...]]:
        if isinstance(term, str):
            return term, (term.lower(),)
        options = tuple(option.lower() for option in term)
        label = " / ".join(term)
        return label, options

    normalized_required = [normalize_required_term(term) for term in required]
    required_hits = [label for label, options in normalized_required if any(option in text for option in options)]
    required_misses = [label for label, options in normalized_required if not any(option in text for option in options)]
    bad_hits = [t for t in bad if t.lower() in text]

    base_score = len(required_hits) / len(normalized_required) if normalized_required else 1.0
    penalty = 0.4 * len(bad_hits)
    score = round(max(0.0, base_score - penalty), 3)
    passed = score >= threshold and len(bad_hits) == 0

    return {
        "score": score,
        "pass": passed,
        "required_hits": required_hits,
        "required_misses": required_misses,
        "bad_hits": bad_hits,
    }


EVAL_CASES = [
    {
        "id": "compliance_bitlocker_70005",
        "category": "Intune Compliance Failures",
        "prompt": (
            "A Windows 11 device is showing non-compliant for BitLocker in Intune "
            "with error 0x80070005. TPM 2.0 is present and BitLocker is already enabled. "
            "What should L1 support check first?"
        ),
        # 0x80070005 = access denied. Model should frame this as a permissions /
        # reporting issue, not as BitLocker-not-enabled or TPM-misconfigured.
        "scoring": {
            "required": ["permission", "access", "reporting", "intune"],
            "bad": [],
            "pass_threshold": 0.5,
        },
    },
    {
        "id": "update_component_store_73712",
        "category": "Windows Update Errors",
        "prompt": (
            "A device fails cumulative updates with Windows Update error 0x80073712. "
            "Give a concise diagnosis and first remediation steps."
        ),
        # 0x80073712 = component store corruption. Disk-space framing is the known bad answer.
        "scoring": {
            "required": ["component store", "dism", "sfc", "corruption"],
            "bad": ["disk space", "insufficient disk", "low disk", "storage space"],
            "pass_threshold": 0.5,
        },
    },
    {
        "id": "baseline_gpo_conflict",
        "category": "Security Baseline Misconfigurations",
        "prompt": (
            "An Intune security baseline is not applying because local or domain GPO settings "
            "appear to be overriding it. How should support confirm that?"
        ),
        # Must mention diagnostic tooling. DCPromo is wrong context (DC promotion tool).
        "scoring": {
            "required": ["gpresult", "mdm", "policy", "baseline"],
            "bad": ["dcpromo"],
            "pass_threshold": 0.5,
        },
    },
    {
        "id": "enrollment_80180014",
        "category": "Device Enrollment & Sync Issues",
        "prompt": (
            "Enrollment fails with Intune error 80180014 on a corporate laptop. "
            "Explain what the error usually means and what to verify."
        ),
        # 80180014 = device enrollment limit. Hardware-hash framing is wrong.
        # dsregcmd /getdeviceinfo for hardware hashes is an invented check.
        "scoring": {
            "required": ["enrollment limit", "device limit", "enrolled", "retire", "stale"],
            "bad": ["hardware hash", "dsregcmd /getdeviceinfo"],
            "pass_threshold": 0.4,
        },
    },
    {
        "id": "app_timeout_87d1041c",
        "category": "App Deployment Failures",
        "prompt": (
            "A required Win32 app install is failing with error 0x87D1041C. "
            "What are the likely causes and the fastest checks?"
        ),
        # 0x87D1041C = Win32 app timeout / detection failure via IME.
        # appxbundle / certificate / dsregcmd /verify framing is wrong.
        "scoring": {
            "required": ["management extension", "ime", "timeout", "detection"],
            "bad": ["dsregcmd /verify", "appxbundle", "certificate trust"],
            "pass_threshold": 0.5,
        },
    },
    {
        "id": "autopilot_800705b4",
        "category": "Autopilot Provisioning Problems",
        "prompt": (
            "Autopilot pre-provisioning is timing out at the app install stage with 0x800705B4. "
            "What should the technician inspect?"
        ),
        # 0x800705B4 = timeout. Key checks: ESP phase, IME logs, required app assignments.
        "scoring": {
            "required": ["esp", "enrollment status", "ime", "app", "timeout"],
            "bad": [],
            "pass_threshold": 0.4,
        },
    },
    {
        "id": "conditional_access_block",
        "category": "Conditional Access Policy Blocks",
        "prompt": (
            "A user says their device is compliant in Intune but SharePoint access is still blocked "
            "by Conditional Access. What should support validate end to end?"
        ),
        # Must steer toward sign-in logs and CA policy inspection in Entra.
        "scoring": {
            "required": ["sign-in log", ("entra", "azure ad"), "compliance", "policy"],
            "bad": [],
            "pass_threshold": 0.5,
        },
    },
    {
        "id": "driver_surface_wifi",
        "category": "Driver & Hardware Conflicts",
        "prompt": (
            "Surface devices lost Wi-Fi after a recent Windows update. "
            "Provide a short L1 troubleshooting plan."
        ),
        # Driver rollback via Device Manager is the core first step.
        "scoring": {
            "required": ["driver", "rollback", "device manager"],
            "bad": [],
            "pass_threshold": 0.5,
        },
    },
    {
        "id": "ime_service_failure",
        "category": "App Deployment Failures",
        "prompt": (
            "The Intune Management Extension service is not starting on several devices. "
            "What logs, services, and local checks should support run first?"
        ),
        # Key checks: ProgramData IME log folder, services.msc, event viewer.
        "scoring": {
            "required": ["programdata", "managementextension", "services"],
            "bad": [],
            "pass_threshold": 0.5,
        },
    },
    {
        "id": "wufb_conflict",
        "category": "Windows Update Errors",
        "prompt": (
            "Devices are receiving both WSUS and Windows Update for Business policies, "
            "and update scans are failing. Summarize root cause and remediation."
        ),
        # Must mention scan source conflict. Get-WindowsAutoUpdatePolicy is an invented cmdlet.
        "scoring": {
            "required": ["scan source", "wsus", "windows update for business"],
            "bad": ["get-windowsautoupdatepolicy"],
            "pass_threshold": 0.5,
        },
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a fixed prompt evaluation for the local Phi model.")
    parser.add_argument(
        "--output",
        default="output/phi4-mini-intune-eval.json",
        help="Path for JSON results.",
    )
    parser.add_argument(
        "--api-url",
        help="Base URL for a running serve_phi_api.py instance, e.g. http://127.0.0.1:8008",
    )
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def normalize_api_url(api_url: str) -> str:
    return api_url.rstrip("/")


def call_generate_api(
    api_url: str,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
) -> dict:
    payload = json.dumps(
        {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "do_sample": do_sample,
        }
    ).encode("utf-8")
    req = request.Request(
        f"{normalize_api_url(api_url)}/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API request failed with HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"API request failed: {exc.reason}") from exc


def main() -> None:
    args = parse_args()
    runtime = None if args.api_url else PhiRuntime()
    results = []

    if runtime is not None:
        print("Evaluating model:", runtime.model_path)
        print("Device:", runtime.device_name())
    else:
        print("Evaluating via API:", normalize_api_url(args.api_url))
    print("Cases:", len(EVAL_CASES))

    for idx, case in enumerate(EVAL_CASES, start=1):
        print(f"[{idx}/{len(EVAL_CASES)}] {case['id']} ({case['category']})")
        if runtime is not None:
            result = runtime.generate(
                case["prompt"],
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=False,
            )
        else:
            result = call_generate_api(
                args.api_url,
                case["prompt"],
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=False,
            )
        scoring = score_reply(result["reply"], case.get("scoring", {}))
        status = "PASS" if scoring["pass"] else "FAIL"
        print(f"  {status}  score={scoring['score']:.2f}", end="")
        if scoring["bad_hits"]:
            print(f"  bad={scoring['bad_hits']}", end="")
        if scoring["required_misses"]:
            print(f"  missing={scoring['required_misses']}", end="")
        print()

        result.update(
            {
                "id": case["id"],
                "category": case["category"],
                "prompt": case["prompt"],
                "scoring": scoring,
            }
        )
        results.append(result)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    passed = sum(1 for r in results if r["scoring"]["pass"])
    avg_tps = sum(r["tokens_per_second"] or 0 for r in results) / len(results)
    avg_latency = sum(r["latency_seconds"] for r in results) / len(results)

    print(f"\nResults: {passed}/{len(results)} passed")
    print(f"Average latency: {avg_latency:.2f}s  |  Average tokens/sec: {avg_tps:.2f}")
    print(f"Saved to: {output_path}")

    # Per-case summary table
    print("\n--- Case summary ---")
    for r in results:
        s = r["scoring"]
        status = "PASS" if s["pass"] else "FAIL"
        bad_note = f"  BAD={s['bad_hits']}" if s["bad_hits"] else ""
        miss_note = f"  MISS={s['required_misses']}" if s["required_misses"] else ""
        print(f"  {status}  {r['id']:45s}  score={s['score']:.2f}{bad_note}{miss_note}")


if __name__ == "__main__":
    main()
