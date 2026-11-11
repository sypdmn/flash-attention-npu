
"""
3.2 decode（q_seqlen=1）场景：FA3 / FA4 与 TorchNPU FIA 对比。

Shape 全部使用 TND：
  FA3: flash_attn_npu_3.flash_attn_varlen_func，return_attn_probs=True；
  FA4: flash_attn_npu_4.flash_attn_varlen_func，return_lse=True；
  FIA: torch_npu.npu_fused_infer_attention_score，softmax_lse_flag=True。

用例来自 3.2-性能场景.md 的“典型测试用例明细”，只保留 q_s=1 的 decode 用例。
本脚本按逻辑序列长度测 dense TND，不使用 paged KV / block_table。

--timing event：NPU Event 计时；
--timing kernel：torch_npu.profiler 的 kernel Duration 计时。
外部 msprof 可单独包裹 event 模式，用于采集整次 Python 任务的设备侧时间线。
"""

from __future__ import annotations

import argparse
import csv
import glob
import importlib
import os
import shutil
import sys
import time
from statistics import geometric_mean, median

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

DEFAULT_MD = os.path.join(_ROOT, "3.2-性能场景.md")
DEFAULT_CSV = os.path.join(
    _ROOT, "batchtest", "fa34_fia_tnd_decode_perf.csv"
)
KERNEL_CSV = os.path.join(
    _ROOT, "batchtest", "fa34_fia_tnd_decode_kernel.csv"
)

OPS = ("fa3", "fa4", "fia")


# ============================================================
# 1. 读取性能测试用例
# ============================================================

def load_cases(path: str) -> list[dict]:
    """Read the detail tables under '典型测试用例明细'."""

    model = None
    in_detail = False
    cases = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line.startswith("## "):
                in_detail = "典型测试用例明细" in line
                continue

            if not in_detail:
                continue

            if line.startswith("### "):
                title = line[4:].strip()
                model = title.split(". ", 1)[-1].strip()
                continue

            if not line.startswith("|") or model is None:
                continue

            cells = [
                c.strip()
                for c in line.strip("|").split("|")
            ]

            if len(cells) != 9 or not cells[0].isdigit():
                continue

            (
                case_id,
                batch,
                q_s,
                kv_s,
                nheads,
                nkv,
                dq,
                dv,
                block,
            ) = (int(c) for c in cells)

            cases.append(
                {
                    "case_id": case_id,
                    "model": model,
                    "batch": batch,
                    "q_s": q_s,
                    "kv_s": kv_s,
                    "nheads": nheads,
                    "nkv": nkv,
                    "head_dim": dq,
                    "head_dim_v": dv,
                    "block_size": block,
                    "layout": "TND",
                    "flops": 4.0 * batch * q_s * kv_s * nheads * dq,
                }
            )

    if not cases:
        raise RuntimeError(
            f"未从 {path} 解析到典型用例"
        )

    return cases


# ============================================================
# 2. 构造 TND 输入
# ============================================================

def cu_seqlens(
    batch: int,
    seqlen: int,
    device
) -> torch.Tensor:

    return (
        torch.arange(
            0,
            batch + 1,
            dtype=torch.int32,
            device=device
        ) * seqlen
    )


def build_inputs(
    case: dict,
    device: str,
    dtype: torch.dtype
) -> dict:

    batch = case["batch"]
    q_s = case["q_s"]
    kv_s = case["kv_s"]

    nheads = case["nheads"]
    nkv = case["nkv"]

    dq = case["head_dim"]
    dv = case["head_dim_v"]

    q = torch.randn(
        batch * q_s,
        nheads,
        dq,
        device=device,
        dtype=dtype
    )

    k = torch.randn(
        batch * kv_s,
        nkv,
        dq,
        device=device,
        dtype=dtype
    )

    v = torch.randn(
        batch * kv_s,
        nkv,
        dv,
        device=device,
        dtype=dtype
    )

    cu_q = cu_seqlens(batch, q_s, device)
    cu_k = cu_seqlens(batch, kv_s, device)

    # FIA eager API 使用 Python List[int]。
    # 每个元素为对应 Batch 的累计序列长度，不包含前导 0。
    actual_q = [
        i * q_s
        for i in range(1, batch + 1)
    ]

    actual_k = [
        i * kv_s
        for i in range(1, batch + 1)
    ]

    return {
        "q": q,
        "k": k,
        "v": v,
        "cu_q": cu_q,
        "cu_k": cu_k,
        "actual_q": actual_q,
        "actual_k": actual_k,
        "scale": case["head_dim"] ** -0.5,
    }


# ============================================================
# 3. FA3 / FA4 / FIA 调用入口
# ============================================================

def run_fa3(fn, inp, case):
    """
    FA3 TND varlen forward.
    return_attn_probs=True also requests LSE.
    """

    return fn(
        inp["q"],
        inp["k"],
        inp["v"],
        cu_seqlens_q=inp["cu_q"],
        cu_seqlens_k=inp["cu_k"],
        max_seqlen_q=case["q_s"],
        max_seqlen_k=case["kv_s"],
        softmax_scale=inp["scale"],
        causal=True,
        return_attn_probs=True,
    )


def run_fa4(fn, inp, case):
    """
    FA4 TND varlen forward.
    FA4 uses return_lse rather than return_attn_probs.
    """

    return fn(
        inp["q"],
        inp["k"],
        inp["v"],
        cu_seqlens_q=inp["cu_q"],
        cu_seqlens_k=inp["cu_k"],
        max_seqlen_q=case["q_s"],
        max_seqlen_k=case["kv_s"],
        softmax_scale=inp["scale"],
        causal=True,
        return_lse=True,
    )


def run_fia(inp, case):
    import torch_npu

    return torch_npu.npu_fused_infer_attention_score(
        inp["q"],
        inp["k"],
        inp["v"],
        actual_seq_lengths=inp["actual_q"],
        actual_seq_lengths_kv=inp["actual_k"],
        num_heads=case["nheads"],
        num_key_value_heads=case["nkv"],
        input_layout="TND",
        scale=inp["scale"],
        softmax_lse_flag=True,
    )


# ============================================================
# 4. NPU Event 计时
# ============================================================

def device_median_ms(
    fn,
    warmup: int,
    iters: int
) -> float:

    for _ in range(warmup):
        fn()

    torch.npu.synchronize()

    use_event = hasattr(torch.npu, "Event")
    samples = []

    if use_event:
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)

        for _ in range(iters):
            start.record()
            fn()
            end.record()

            torch.npu.synchronize()

            samples.append(
                float(start.elapsed_time(end))
            )

    else:
        for _ in range(iters):
            torch.npu.synchronize()

            t0 = time.perf_counter()
            fn()
            torch.npu.synchronize()

            samples.append(
                (time.perf_counter() - t0) * 1e3
            )

    return float(median(samples))


# ============================================================
# 5. TorchNPU Profiler
# ============================================================

def _child_dirs(path: str) -> set[str]:

    if not os.path.isdir(path):
        return set()

    return {
        os.path.join(path, name)
        for name in os.listdir(path)
        if os.path.isdir(os.path.join(path, name))
    }


def profile_calls(
    calls: list,
    prof_dir: str
) -> str:
    """
    One profiler step.
    Returns the ascend trace directory.
    """

    import torch_npu

    os.makedirs(prof_dir, exist_ok=True)

    before = _child_dirs(prof_dir)

    experimental_config = (
        torch_npu.profiler._ExperimentalConfig(
            aic_metrics=(
                torch_npu.profiler.AiCMetrics.PipeUtilization
            ),
            profiler_level=(
                torch_npu.profiler.ProfilerLevel.Level1
            ),
            data_simplification=False,
            l2_cache=False,
        )
    )

    prof = torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        record_shapes=False,
        profile_memory=False,
        with_modules=False,
        schedule=torch_npu.profiler.schedule(
            wait=0,
            warmup=0,
            active=1,
            repeat=1,
            skip_first=0
        ),
        experimental_config=experimental_config,
        on_trace_ready=(
            torch_npu.profiler.tensorboard_trace_handler(
                prof_dir,
                analyse_flag=False
            )
        ),
    )

    prof.start()

    for fn in calls:
        fn()
        torch.npu.synchronize()

    prof.step()
    prof.stop()

    created = _child_dirs(prof_dir) - before

    if len(created) != 1:
        raise RuntimeError(
            f"expected 1 profile under {prof_dir}, "
            f"got {sorted(created)}"
        )

    ascend_pt = created.pop()

    from torch_npu.profiler.profiler import analyse

    analyse(
        profiler_path=ascend_pt,
        max_process_number=8,
        export_type="text"
    )

    return ascend_pt


# ============================================================
# 6. 读取 Kernel Duration
# ============================================================

def load_kernel_rows(
    ascend_pt: str
) -> list[tuple[str, float]]:

    csv_paths = glob.glob(
        os.path.join(
            ascend_pt,
            "**/kernel_details.csv"
        ),
        recursive=True
    )

    if not csv_paths:
        raise FileNotFoundError(
            f"kernel_details.csv not found under {ascend_pt}"
        )

    with open(
        max(csv_paths, key=os.path.getmtime),
        newline="",
        encoding="utf-8-sig"
    ) as f:
        rows = list(csv.DictReader(f))

    def duration_ms(row: dict) -> float:

        if row.get("Duration(us)"):
            return float(row["Duration(us)"]) / 1000.0

        if row.get("Duration(ns)"):
            return float(row["Duration(ns)"]) / 1e6

        raise KeyError(
            f"no Duration column in {list(row)}"
        )

    parsed = []

    for row in rows:

        name = (
            row.get("Name")
            or row.get("Op Name")
            or ""
        )

        start = float(
            str(
                row.get("Start Time(us)", "0")
            ).strip() or 0
        )

        parsed.append(
            (start, name, duration_ms(row))
        )

    parsed.sort()

    return [
        (name, ms)
        for _start, name, ms in parsed
    ]


# ============================================================
# 7. Kernel Marker 分段
# ============================================================

def split_by_marker(
    kernels: list[tuple[str, float]],
    marker_name: str
) -> list[list[tuple[str, float]]]:

    segments = []
    current = []

    for name, ms in kernels:

        if name == marker_name:
            segments.append(current)
            current = []
        else:
            current.append((name, ms))

    return segments


def segment_ms(
    segment: list[tuple[str, float]]
) -> float:

    return float(
        sum(ms for _name, ms in segment)
    )


def make_marker(device: str):

    buf = torch.empty(
        1024,
        dtype=torch.int32,
        device=device
    )

    def mark():
        buf.bitwise_not_()

    return mark


def calibrate_marker(
    device: str,
    prof_root: str
) -> str:

    mark = make_marker(device)

    prof_dir = os.path.join(
        prof_root,
        "marker"
    )

    ascend_pt = profile_calls(
        [mark, mark, mark],
        prof_dir
    )

    try:
        kernels = load_kernel_rows(ascend_pt)
    finally:
        shutil.rmtree(
            prof_dir,
            ignore_errors=True
        )

    counts = {}

    for name, _ms in kernels:
        counts[name] = counts.get(name, 0) + 1

    hits = [
        name
        for name, count in counts.items()
        if count == 3
    ]

    if len(hits) != 1:

        detail = ", ".join(
            f"{name}x{count}"
            for name, count in counts.items()
        )

        raise RuntimeError(
            f"marker kernel not unique: {detail}"
        )

    print(
        f"marker kernel: {hits[0]}",
        flush=True
    )

    return hits[0]


# ============================================================
# 8. 按 Kernel 分段采集
# ============================================================

def profile_kernel_chunk(
    prepared: list[dict],
    mark,
    marker_name: str,
    prof_dir: str,
    dump: bool
) -> list[dict]:

    calls = []

    for item in prepared:

        for op in OPS:
            calls.append(item["calls"][op])
            calls.append(mark)

    ascend_pt = profile_calls(
        calls,
        prof_dir
    )

    try:
        kernels = load_kernel_rows(ascend_pt)

        segments = split_by_marker(
            kernels,
            marker_name
        )

    finally:
        shutil.rmtree(
            prof_dir,
            ignore_errors=True
        )

    expect = len(prepared) * len(OPS)

    if len(segments) != expect:
        raise RuntimeError(
            f"expected {expect} kernel segments, "
            f"got {len(segments)}"
        )

    rows = []

    for index, item in enumerate(prepared):

        row = dict(item["case"])
        row["dtype"] = item["dtype"]
        row["timing"] = "kernel"

        for op_index, op in enumerate(OPS):

            segment = segments[
                index * len(OPS) + op_index
            ]

            row[f"{op}_ms"] = segment_ms(segment)

            if dump:
                detail = ", ".join(
                    f"{name}={ms:.4f}ms"
                    for name, ms in segment
                ) or "(none)"

                print(
                    f"    {op} kernels: {detail}",
                    flush=True
                )

        fia = row.get("fia_ms")

        for version in ("fa3", "fa4"):

            fa_ms = row.get(f"{version}_ms")

            row[f"{version}_perf_ratio"] = (
                fia / fa_ms
                if fia and fa_ms
                else None
            )

            row[f"{version}_tflops"] = tflops(
                item["case"]["flops"],
                fa_ms
            )

        row["fia_tflops"] = tflops(
            item["case"]["flops"],
            fia
        )

        row["error"] = ""
        rows.append(row)

    return rows


# ============================================================
# 9. 性能指标计算
# ============================================================

def tflops(
    flops: float,
    ms: float | None
) -> float | None:

    if not ms or ms <= 0:
        return None

    return flops / ms / 1e9


# ============================================================
# 10. 单个 Case 的 Event 测试
# ============================================================

def run_case(
    case,
    device,
    dtype,
    kernels,
    warmup,
    iters
) -> dict:

    inp = build_inputs(
        case,
        device,
        dtype
    )

    calls = {
        "fa3": lambda: run_fa3(
            kernels["fa3"], inp, case
        ),
        "fa4": lambda: run_fa4(
            kernels["fa4"], inp, case
        ),
        "fia": lambda: run_fia(
            inp, case
        ),
    }

    row = dict(case)
    row["dtype"] = str(dtype).replace("torch.", "")
    row["timing"] = "event"

    errors = []

    with torch.no_grad():

        for op, fn in calls.items():

            try:
                row[f"{op}_ms"] = device_median_ms(
                    fn,
                    warmup,
                    iters
                )

            except Exception as exc:

                row[f"{op}_ms"] = None

                errors.append(
                    f"{op}: {exc}"
                )

            finally:
                torch.npu.synchronize()

    fia = row.get("fia_ms")

    for version in ("fa3", "fa4"):

        fa_ms = row.get(f"{version}_ms")

        row[f"{version}_perf_ratio"] = (
            fia / fa_ms
            if fia and fa_ms
            else None
        )

        row[f"{version}_tflops"] = tflops(
            case["flops"],
            fa_ms
        )

    row["fia_tflops"] = tflops(
        case["flops"],
        fia
    )

    row["error"] = " | ".join(errors)

    return row


# ============================================================
# 11. 保存 CSV
# ============================================================

def save_csv(
    rows: list[dict],
    path: str
) -> None:

    fields = [
        "case_id",
        "model",
        "layout",
        "dtype",
        "timing",
        "batch",
        "q_s",
        "kv_s",
        "nheads",
        "nkv",
        "head_dim",
        "head_dim_v",
        "block_size",
        "flops",
        "fa3_ms",
        "fa4_ms",
        "fia_ms",
        "fa3_perf_ratio",
        "fa4_perf_ratio",
        "fa3_tflops",
        "fa4_tflops",
        "fia_tflops",
        "error",
    ]

    os.makedirs(
        os.path.dirname(os.path.abspath(path)) or ".",
        exist_ok=True
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore"
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    k: row.get(k)
                    for k in fields
                }
            )


# ============================================================
# 12. 汇总报告
# ============================================================

def _fmt(value, spec=".3f"):

    return (
        f"{value:{spec}}"
        if isinstance(value, (int, float))
        else "-"
    )


def _ratios(rows, op):

    return [
        r[f"{op}_perf_ratio"]
        for r in rows
        if r.get(f"{op}_perf_ratio")
    ]


def summarize(
    rows: list[dict],
    timing: str
) -> str:

    lines = []

    if timing == "kernel":
        title = (
            "kernel Duration"
            "（该次调用所有 kernel 的 Duration 之和）"
        )
    else:
        title = "NPU event device time"

    lines.append("=" * 108)

    lines.append(
        f"3.2 decode TND：{title}；"
        "性能比 = FIA耗时 / FA耗时"
        "（>1 表示 FA 快于 FIA）"
    )

    lines.append("=" * 108)

    ok = [
        r for r in rows
        if all(
            r.get(f"{op}_ms")
            for op in OPS
        )
    ]

    lines.append(
        f"三路都成功: {len(ok)}/{len(rows)}"
    )

    for version in ("fa3", "fa4"):

        ratios = _ratios(rows, version)

        if ratios:

            lines.append(
                f"  {version}/FIA: "
                f"geomean={geometric_mean(ratios):.3f}x "
                f"median={median(ratios):.3f}x "
                f"min={min(ratios):.3f}x "
                f"max={max(ratios):.3f}x"
            )

        else:
            lines.append(
                f"  {version}/FIA: 无有效性能比"
            )

    models = []

    for row in rows:
        if row["model"] not in models:
            models.append(row["model"])

    lines.append("")

    lines.append(
        f"{'model':<42} "
        f"{'n':>4} "
        f"{'FA3/FIA':>10} "
        f"{'FA4/FIA':>10}"
    )

    for model in models:

        sub = [
            r for r in rows
            if r["model"] == model
        ]

        values = []

        for version in ("fa3", "fa4"):

            ratios = _ratios(sub, version)

            values.append(
                f"{geometric_mean(ratios):.3f}x"
                if ratios
                else "-"
            )

        lines.append(
            f"{model:<42} "
            f"{len(sub):>4} "
            f"{values[0]:>10} "
            f"{values[1]:>10}"
        )

    lines.append("")

    lines.append(
        f"{'id':>4} "
        f"{'model':<30} "
        f"{'B':>4} "
        f"{'kv':>7} "
        f"{'FA3':>9} "
        f"{'FA4':>9} "
        f"{'FIA':>9} "
        f"{'P3':>7} "
        f"{'P4':>7}"
    )

    for row in rows:

        lines.append(
            f"{row['case_id']:>4} "
            f"{row['model']:<30.30} "
            f"{row['batch']:>4} "
            f"{row['kv_s']:>7} "
            f"{_fmt(row.get('fa3_ms')):>9} "
            f"{_fmt(row.get('fa4_ms')):>9} "
            f"{_fmt(row.get('fia_ms')):>9} "
            f"{_fmt(row.get('fa3_perf_ratio'), '.2f'):>7} "
            f"{_fmt(row.get('fa4_perf_ratio'), '.2f'):>7}"
        )

    failed = [
        r for r in rows
        if r.get("error")
    ]

    if failed:

        lines.append("")

        lines.append(
            f"失败 {len(failed)} 条:"
        )

        for row in failed[:30]:
            lines.append(
                f"  case {row['case_id']}: {row['error']}"
            )

        if len(failed) > 30:

            lines.append(
                f"  ... 另有 {len(failed) - 30} 条"
            )

    text = "\n".join(lines)

    print("\n" + text)

    return text


# ============================================================
# 13. 输出 Case 信息
# ============================================================

def _case_banner(
    index: int,
    total: int,
    case: dict
) -> str:

    return (
        f">>> [{index:03d}/{total}] "
        f"case {case['case_id']} "
        f"{case['model']} "
        f"B{case['batch']} "
        f"kv{case['kv_s']} "
        f"H{case['nheads']}/{case['nkv']}"
    )


# ============================================================
# 14. Event 模式全部 Case
# ============================================================

def collect_event(
    cases,
    args,
    dtype,
    kernels
) -> list[dict]:

    rows = []

    for index, case in enumerate(cases, 1):

        print(
            _case_banner(
                index,
                len(cases),
                case
            ),
            flush=True
        )

        try:

            row = run_case(
                case,
                args.device,
                dtype,
                kernels,
                args.warmup,
                args.iters
            )

        except Exception as exc:

            row = {
                **case,
                "dtype": args.dtype,
                "timing": "event",
                "error": str(exc),
                **{
                    f"{op}_ms": None
                    for op in OPS
                }
            }

            print(
                f"    FAILED: {exc}",
                flush=True
            )

        else:

            print(
                "    " + " ".join(
                    f"{op}={_fmt(row.get(op + '_ms'))}ms"
                    for op in OPS
                ),
                flush=True
            )

        rows.append(row)

        save_csv(
            rows,
            args.result_csv
        )

        torch.npu.empty_cache()

    return rows


# ============================================================
# 15. 准备 Kernel Profiling
# ============================================================

def _prepare(
    case,
    device,
    dtype,
    kernels
) -> dict:

    inp = build_inputs(
        case,
        device,
        dtype
    )

    calls = {
        "fa3": lambda: run_fa3(
            kernels["fa3"], inp, case
        ),
        "fa4": lambda: run_fa4(
            kernels["fa4"], inp, case
        ),
        "fia": lambda: run_fia(
            inp, case
        ),
    }

    return {
        "case": case,
        "dtype": str(dtype).replace("torch.", ""),
        "calls": calls,
        "inp": inp,
    }


# ============================================================
# 16. Kernel 模式全部 Case
# ============================================================

def collect_kernel(
    cases,
    args,
    dtype,
    kernels
) -> list[dict]:

    mark = make_marker(args.device)

    marker_name = calibrate_marker(
        args.device,
        args.profile_root
    )

    rows = []

    chunk = max(1, args.chunk_size)
    offset = 0

    while offset < len(cases):

        batch = cases[
            offset:offset + chunk
        ]

        prepared = []
        failed = []

        for case in batch:

            try:

                item = _prepare(
                    case,
                    args.device,
                    dtype,
                    kernels
                )

                with torch.no_grad():

                    for _ in range(args.warmup):

                        for op in OPS:
                            item["calls"][op]()

                        torch.npu.synchronize()

                prepared.append(item)

            except Exception as exc:
                failed.append(
                    (case, str(exc))
                )

        if prepared:

            prof_dir = os.path.join(
                args.profile_root,
                f"chunk_{prepared[0]['case']['case_id']}"
            )

            try:

                with torch.no_grad():

                    measured = profile_kernel_chunk(
                        prepared,
                        mark,
                        marker_name,
                        prof_dir,
                        args.dump_kernels
                    )

            except Exception as exc:

                print(
                    f"    chunk failed ({exc}), "
                    "retry one by one",
                    flush=True
                )

                measured = []

                for item in prepared:

                    one_dir = os.path.join(
                        args.profile_root,
                        f"case_{item['case']['case_id']}"
                    )

                    try:

                        with torch.no_grad():

                            measured.extend(
                                profile_kernel_chunk(
                                    [item],
                                    mark,
                                    marker_name,
                                    one_dir,
                                    args.dump_kernels
                                )
                            )

                    except Exception as one_exc:

                        failed.append(
                            (
                                item["case"],
                                str(one_exc)
                            )
                        )

            for row in measured:

                rows.append(row)

                print(
                    _case_banner(
                        len(rows),
                        len(cases),
                        row
                    )
                    + "\n    "
                    + " ".join(
                        f"{op}={_fmt(row.get(op + '_ms'))}ms"
                        for op in OPS
                    ),
                    flush=True
                )

        for case, exc in failed:

            rows.append(
                {
                    **case,
                    "dtype": args.dtype,
                    "timing": "kernel",
                    "error": exc,
                    **{
                        f"{op}_ms": None
                        for op in OPS
                    }
                }
            )

            print(
                _case_banner(
                    len(rows),
                    len(cases),
                    case
                ),
                flush=True
            )

            print(
                f"    FAILED: {exc}",
                flush=True
            )

        save_csv(
            rows,
            args.result_csv
        )

        torch.npu.empty_cache()
        offset += chunk

    rows.sort(
        key=lambda row: row["case_id"]
    )

    save_csv(
        rows,
        args.result_csv
    )

    return rows


# ============================================================
# 17. 主函数 / 命令行参数
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--device",
        default="npu:0"
    )

    parser.add_argument(
        "--cases-md",
        default=DEFAULT_MD
    )

    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float16")
    )

    parser.add_argument(
        "--case-ids",
        type=int,
        nargs="+"
    )

    parser.add_argument(
        "--models",
        nargs="+",
        help="Substring match against the model name"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Run only the first N selected cases"
    )

    parser.add_argument(
        "--timing",
        choices=("event", "kernel"),
        default="event",
        help=(
            "event: NPU event only; "
            "kernel: profiler kernel Duration only"
        )
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=2
    )

    parser.add_argument(
        "--iters",
        type=int,
        default=5,
        help="event 模式的正式次数"
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4,
        help=(
            "kernel 模式每个 profiler trace "
            "包含的用例数"
        )
    )

    parser.add_argument(
        "--dump-kernels",
        action="store_true",
        help=(
            "打印每个算子的 kernel 名称和 Duration"
        )
    )

    parser.add_argument(
        "--profile-root",
        default="/tmp/fa34_fia_tnd_decode_prof"
    )

    parser.add_argument(
        "--result-csv",
        default=""
    )

    parser.add_argument(
        "--summary-md",
        default=""
    )

    parser.add_argument(
        "--list-only",
        action="store_true"
    )

    args = parser.parse_args()

    if not args.result_csv:

        args.result_csv = (
            KERNEL_CSV
            if args.timing == "kernel"
            else DEFAULT_CSV
        )

    if args.warmup < 0 or args.iters < 1:
        parser.error(
            "--iters must be >= 1"
        )

    # 加载用例
    cases = load_cases(
        args.cases_md
    )

    # 本脚本仅用于单 Token Decode
    cases = [
        case
        for case in cases
        if case["q_s"] == 1
    ]

    if not cases:

        raise RuntimeError(
            f"No decode cases (q_s=1) found in "
            f"{args.cases_md}; "
            "check the 3.2 decode case table."
        )

    # 按模型过滤
    if args.models:

        keys = [
            m.lower()
            for m in args.models
        ]

        cases = [
            c for c in cases
            if any(
                k in c["model"].lower()
                for k in keys
            )
        ]

    # 按用例 ID 过滤
    if args.case_ids:

        wanted = set(
            args.case_ids
        )

        cases = [
            c for c in cases
            if c["case_id"] in wanted
        ]

    # 限制用例数量
    if args.limit:
        cases = cases[:args.limit]

    print(
        f"decode 用例 {len(cases)} 条"
        f"（q_s=1）,来自 {args.cases_md}"
    )

    # 只列出用例
    if args.list_only:

        for case in cases:

            print(
                f"  {case['case_id']:>4} "
                f"{case['model']:<36} "
                f"B{case['batch']} "
                f"q{case['q_s']} "
                f"kv{case['kv_s']} "
                f"H{case['nheads']}/{case['nkv']} "
                f"d{case['head_dim']} "
                f"flops={case['flops']:.3e}"
            )

        return

    # 检查 NPU
    import torch_npu

    if not torch.npu.is_available():
        raise RuntimeError(
            "未检测到可用 NPU"
        )

    torch.npu.set_device(
        args.device
    )

    # 动态导入 FA3/FA4
    modules = {
        "fa3": "flash_attn_npu_3",
        "fa4": "flash_attn_npu_4",
    }

    kernels = {}

    for version, module_name in modules.items():

        module = importlib.import_module(
            module_name
        )

        fn = getattr(
            module,
            "flash_attn_varlen_func",
            None
        )

        if fn is None:
            raise RuntimeError(
                f"{module_name} does not export "
                "flash_attn_varlen_func"
            )

        kernels[version] = fn

    dtype = getattr(
        torch,
        args.dtype
    )

    print(
        f"device={torch.npu.get_device_name(args.device)} "
        f"dtype={args.dtype} "
        f"timing={args.timing} "
        f"warmup={args.warmup} "
        f"torch={torch.__version__} "
        f"torch_npu={torch_npu.__version__}"
    )

    # 选择 Event 或 Kernel 模式
    if args.timing == "event":

        rows = collect_event(
            cases,
            args,
            dtype,
            kernels
        )

    else:

        rows = collect_kernel(
            cases,
            args,
            dtype,
            kernels
        )

    # 性能统计
    text = summarize(
        rows,
        args.timing
    )

    summary_path = (
        args.summary_md
        or os.path.splitext(args.result_csv)[0]
        + "_summary.md"
    )

    os.makedirs(
        os.path.dirname(
            os.path.abspath(summary_path)
        ) or ".",
        exist_ok=True
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "# 3.2 FA3/FA4 vs FIA Decode TND 性能汇总\n\n"
        )

        f.write(
            f"- device: "
            f"{torch.npu.get_device_name(args.device)}\n"
        )

        f.write(
            f"- dtype: {args.dtype}\n"
        )

        f.write(
            f"- timing: {args.timing}\n"
        )

        f.write(
            f"- warmup/iters: "
            f"{args.warmup}/{args.iters}\n"
        )

        f.write(
            f"- csv: `{args.result_csv}`\n\n"
        )

        f.write("```\n")
        f.write(text)
        f.write("\n```\n")

    print(
        f"\n结果 CSV: {args.result_csv}"
    )

    print(
        f"汇总: {summary_path}"
    )


if __name__ == "__main__":
    main()
