import argparse
import torch
import torch_npu

from flash_attn_npu_3 import flash_attn_varlen_func as fa3
from flash_attn_npu_4 import flash_attn_varlen_func as fa4

from benchmark_fa34_fia_tnd_decode import (
    load_cases,
    build_inputs,
    run_fa3,
    run_fa4,
    run_fia,
)

parser = argparse.ArgumentParser()
parser.add_argument("--op", choices=["fa3", "fa4", "fia"], required=True)
parser.add_argument("--case-id", type=int, default=0)
parser.add_argument("--warmup", type=int, default=10)
parser.add_argument("--iters", type=int, default=30)
parser.add_argument("--device", default="npu:0")
args = parser.parse_args()

torch.npu.set_device(args.device)

cases = {
    c["case_id"]: c
    for c in load_cases("3.2-性能场景.md")
}

case = cases[args.case_id]

assert case["q_s"] == 1, "Only decode cases are supported"

inp = build_inputs(case, args.device, torch.float16)

calls = {
    "fa3": lambda: run_fa3(fa3, inp, case),
    "fa4": lambda: run_fa4(fa4, inp, case),
    "fia": lambda: run_fia(inp, case),
}

fn = calls[args.op]

print(
    f"op={args.op}, case={args.case_id}, "
    f"B={case['batch']}, KV={case['kv_s']}, "
    f"H={case['nheads']}/{case['nkv']}",
    flush=True,
)

with torch.no_grad():
    for _ in range(args.warmup):
        result = fn()

    torch.npu.synchronize()

    print("WARMUP_DONE", flush=True)

    for _ in range(args.iters):
        result = fn()

    torch.npu.synchronize()

print("PROFILE_FINISHED", flush=True)
