# 3.2 FA3/FA4 vs FIA Decode TND 性能汇总

- device: Ascend910B4-1
- dtype: bfloat16
- timing: kernel
- warmup/iters: 5/5
- csv: `batchtest/fa34_case0_kernel.csv`

```
============================================================================================================
3.2 decode TND：kernel Duration（该次调用所有 kernel 的 Duration 之和）；性能比 = FIA耗时 / FA耗时（>1 表示 FA 快于 FIA）
============================================================================================================
三路都成功: 1/1
  fa3/FIA: geomean=0.464x median=0.464x min=0.464x max=0.464x
  fa4/FIA: geomean=0.305x median=0.305x min=0.305x max=0.305x

model                                         n    FA3/FIA    FA4/FIA
llama2-70B/qwen2-72b                          1     0.464x     0.305x

  id model                             B      kv       FA3       FA4       FIA      P3      P4
   0 llama2-70B/qwen2-72b              1     256     0.043     0.066     0.020    0.46    0.30
```
