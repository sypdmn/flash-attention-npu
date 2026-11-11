# 3.2 FA3/FA4 vs FIA Decode TND 性能汇总

- device: Ascend910B4-1
- dtype: float16
- timing: event
- warmup/iters: 5/20
- csv: `/data1/songyp/flash-attention-npu/batchtest/fa34_fia_tnd_decode_perf.csv`

```
============================================================================================================
3.2 decode TND：NPU event device time；性能比 = FIA耗时 / FA耗时（>1 表示 FA 快于 FIA）
============================================================================================================
三路都成功: 1/1
  fa3/FIA: geomean=0.256x median=0.256x min=0.256x max=0.256x
  fa4/FIA: geomean=0.279x median=0.279x min=0.279x max=0.279x

model                                         n    FA3/FIA    FA4/FIA
llama2-70B/qwen2-72b                          1     0.256x     0.279x

  id model                             B      kv       FA3       FA4       FIA      P3      P4
   0 llama2-70B/qwen2-72b              1     256     0.960     0.882     0.246    0.26    0.28
```
