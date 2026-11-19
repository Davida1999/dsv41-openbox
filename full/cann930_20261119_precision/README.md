# CANN930 GPQA 三轮精度归档

2026-11-19 完成的三轮 GPQA Diamond 使用完整 384 专家 BF16＋Engram 模型，输出预算为 65536，reasoning_effort=low，temperature=1.0，top_p=0.95。三轮结果为 180/198、178/198、178/198，平均准确率 **90.235690%**，594 个请求全部成功。

- [正式报告](REPORT.md)
- [原始运行报告与汇总](run/)
- [三轮原始预测、评分和配置](results/)
- [实测服务端、客户端及审计脚本](scripts/)
- [独立 CPU 核对结果](evidence_validation.json)

每轮未解析最终选项的题数依次为 2、1、2，均计错。输出预算命中的题数依次为 1、0、2。原始预测和遥测 JSONL 采用 gzip 无损存放，完整预测的解压后 SHA256 已核对。

在仓库根目录执行以下命令复算保存的分数、校验参数与资源释放记录。

```bash
python3 full/cann930_20261119_precision/scripts/audit_saved_results.py
```
