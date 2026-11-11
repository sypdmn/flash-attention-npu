
"""FA3/FA4 vs FIA, single-token dense TND smoke and rough correctness check on Ascend NPU."""
import math
import torch
import torch_npu
from flash_attn_npu_3 import flash_attn_varlen_func as fa3
from flash_attn_npu_4 import flash_attn_varlen_func as fa4


def first_tensor(result):
    return result[0] if isinstance(result, (tuple, list)) else result


def main():
    torch.npu.set_device('npu:0')
    torch.manual_seed(2026)

    B, SQ, SK, H, HK, D = 1, 1, 128, 8, 8, 128
    dtype = torch.float16

    q = torch.randn(B * SQ, H, D, device='npu:0', dtype=dtype)
    k = torch.randn(B * SK, HK, D, device='npu:0', dtype=dtype)
    v = torch.randn(B * SK, HK, D, device='npu:0', dtype=dtype)

    cq = torch.tensor([0, SQ], device='npu:0', dtype=torch.int32)
    ck = torch.tensor([0, SK], device='npu:0', dtype=torch.int32)

    kwargs = dict(
        cu_seqlens_q=cq,
        cu_seqlens_k=ck,
        max_seqlen_q=SQ,
        max_seqlen_k=SK,
        softmax_scale=1 / math.sqrt(D),
        causal=True
    )

    with torch.no_grad():
        o3 = first_tensor(
            fa3(q, k, v, return_attn_probs=True, **kwargs)
        )

        o4 = first_tensor(
            fa4(q, k, v, return_lse=True, **kwargs)
        )

        oi = first_tensor(
            torch_npu.npu_fused_infer_attention_score(
                q,
                k,
                v,
                actual_seq_lengths=[SQ],
                actual_seq_lengths_kv=[SK],
                num_heads=H,
                num_key_value_heads=HK,
                input_layout='TND',
                scale=1 / math.sqrt(D),
                softmax_lse_flag=True,
            )
        )

        torch.npu.synchronize()

    for name, value in [('FA3', o3), ('FA4', o4), ('FIA', oi)]:
        print(name, 'shape=', tuple(value.shape), 'dtype=', value.dtype)

    for name, value in [('FA3', o3), ('FA4', o4)]:
        diff = (value.float() - oi.float()).abs()
        print(
            f'{name} vs FIA: max_abs_error={diff.max().item():.6f} '
            f'mean_abs_error={diff.mean().item():.6f}'
        )

    print('Smoke test finished (review numerical differences above).')


if __name__ == '__main__':
    main()
