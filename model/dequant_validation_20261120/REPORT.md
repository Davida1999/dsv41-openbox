# FP4/FP8反量化独立抽样核验

日期：2026-11-20。在CANN930容器内使用Python标准库只读检查，未执行模型推理或修改权重。

**抽样通过：34个张量、4624个标量全部与导出BF16逐位一致，差异数为0。** 覆盖{'FP4_E2M1': 24, 'FP8_E4M3FN': 10}；文本层0/10/20/39，原专家0/17/383的w1/w2，attention wq_a/wkv，以及Engram层1/14的embedding。

参考实现直接读取safetensors数据偏移，按E2M1/E4M3FN解码、E8M0尺度、FP32乘积和独立BF16最近偶数舍入计算期望位模式；不调用转换器函数。取样覆盖矩阵首尾、中部、尺度块边界两侧、FP4低/高半字节及固定seed随机坐标。

| 输入 | 精确位置 / 校验 |
|---|---|
| 原量化权重 | `/mnt/share/g00815994/dsv41/Deepseekv4.1_weight_hg` |
| BF16导出 | `/mnt/share/c00954197/model/deepseek-V4.1-Flash-1017-bf16` |
| 原索引SHA256 | `74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8` |
| BF16索引SHA256 | `0418e4757ce4ccc91f12216f5ca8b4135485f67f012c0b0d167bb74c2354520d` |
| seed | 20261120 |

[核验脚本](check_dequant_samples.py) · [逐张量统计](summary.json) · [采样坐标与相等性](sample_coordinates.csv)

本地资产清单在服务器`/mnt/share/c00954197/Workspace/dsv41/basic/asset_checks_20261120/.npu-agent/local_asset_inventory.json`，两个精确路径均唯一FOUND，索引引用分片齐全非空，清单无截断或错误。完整带数值CSV保存在服务器；仓库只存坐标和相等性，不上传模型权重值。

```bash
python3 check_dequant_samples.py \
  --source /mnt/share/g00815994/dsv41/Deepseekv4.1_weight_hg \
  --bf16 /mnt/share/c00954197/model/deepseek-V4.1-Flash-1017-bf16 \
  --output /mnt/share/c00954197/Workspace/dsv41/basic/asset_checks_20261120/recheck
```

这是代表性抽样证据，不表示对全部48496张量逐元素扫描，也不替代完整模型推理精度评测。
