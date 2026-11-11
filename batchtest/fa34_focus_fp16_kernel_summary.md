# 3.2 FA3/FA4 vs FIA Decode TND 性能汇总

- device: Ascend910B4-1
- dtype: float16
- timing: kernel
- warmup/iters: 10/5
- csv: `batchtest/fa34_focus_fp16_kernel.csv`

```
============================================================================================================
3.2 decode TND：kernel Duration（该次调用所有 kernel 的 Duration 之和）；性能比 = FIA耗时 / FA耗时（>1 表示 FA 快于 FIA）
============================================================================================================
三路都成功: 8/8
  fa3/FIA: geomean=0.719x median=0.764x min=0.453x max=1.020x
  fa4/FIA: geomean=0.997x median=0.752x min=0.334x max=7.934x

model                                         n    FA3/FIA    FA4/FIA
llama2-70B/qwen2-72b                          1     0.453x     0.334x
llama-65B/Qwen-72B                            1     0.641x     0.516x
starcoder-15.5B                               1     0.532x     0.362x
JYXH                                          2     0.906x     6.592x
llama2-7B/baichuan2-7B                        2     1.019x     0.990x
Qwen3-30B                                     1     0.543x     0.368x

  id model                             B      kv       FA3       FA4       FIA      P3      P4
   0 llama2-70B/qwen2-72b              1     256     0.041     0.056     0.019    0.45    0.33
  49 llama-65B/Qwen-72B                1    5120     0.068     0.084     0.044    0.64    0.52
 210 starcoder-15.5B                  16     256     0.048     0.071     0.026    0.53    0.36
 286 JYXH                              1  264144     1.479     0.165     1.310    0.89    7.93
 287 JYXH                              2  264144     1.721     0.291     1.596    0.93    5.48
 404 llama2-7B/baichuan2-7B           64    4096     2.353     2.426     2.397    1.02    0.99
 405 llama2-7B/baichuan2-7B           64    5120     2.940     3.023     2.998    1.02    0.99
 568 Qwen3-30B                         1     512     0.043     0.063     0.023    0.54    0.37
```
