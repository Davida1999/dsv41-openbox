# E13：E11 纯文本服务配置，单次 curl 实验准备

主节点 177.11.27.1；子节点 177.11.28.1。模型名称 `dsv41-bf16-tp1`；模型及 tokenizer 路径 `/mnt/share/c00954197/workspace/dsv41-flash-bf16`。

沿用 E11 的 TP1/PCP8/DP2、EP、BF16、BF16 KV、FULL_DECODE_ONLY、chunked prefill、异步调度、prefix caching、Engram CPU offload、CPU binding 和 fused MC2 配置。图片输入上限保持为 0。上下文上限 10240，max_num_seqs=32，max_num_batched_tokens=2048。单次 curl 只发送一个请求，无需把服务端 max_num_seqs 改成 1。

## 启动

在两端的 `c00954197_vllm_ascend_a5` 容器中分别执行。启动前确认 NPU 和 8000/12346 端口可用，不与 E11/E12 同时运行。脚本不会停止其他服务。

27.1 主节点：
```bash
bash /workspace/experiments/tp1_dp16_20261029/E13/run_master_full_decode_only.sh
```

28.1 子节点：
```bash
bash /workspace/experiments/tp1_dp16_20261029/E13/run_worker_full_decode_only.sh
```

两端日志分别保存到 E13/server_logs/ 下，每次启动独立 UTC 时间戳文件。容器内服务 PID 分别记录在 E13/master.pid 和 E13/worker.pid，使用前需核对当前进程身份。

## 单次 curl 示例

等待服务就绪，检查健康状态：
```bash
curl --noproxy '*' --fail-with-body -sS http://177.11.27.1:8000/health
```

以下只发送一个纯文本请求，使用与 E11 一致的 Completions 接口和采样参数。它是功能验证示例，并非 2K 输入、6K 输出的定长性能测试；max_tokens 128 仅为示例，可按后续实验修改。

```bash
curl --noproxy '*' --fail-with-body -sS http://177.11.27.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"dsv41-bf16-tp1","prompt":"请简要介绍大语言模型的推理过程。","max_tokens":128,"temperature":1.0,"top_p":1.0,"top_k":-1,"ignore_eos":true,"stream":false}'
```

重复相同 prompt 可能产生前缀缓存命中；比较性能时需明确缓存状态。当前仅完成服务端脚本及语法检查，未启动服务或发送 curl 请求。

## 单 curl 数学题连续十次一致性测试

执行 `python3 /mnt/share/c00954197/workspace/experiments/tp1_dp16_20261029/E13/test_single_curl_math.py`。脚本通过 curl 顺序发送十次相同请求，temperature=0、top_p=1、top_k=-1、seed=42，并请求 logprobs=5。问题为 123+456，标准答案 579。每轮结果在 single_curl_math_results/ 的独立时间戳目录保存，report.txt 为完整文本报告，summary.json 为比较结果，另有每次原始响应、HTTP 头、耗时、实际请求和脚本快照。

已完成运行：20261102T060812Z_erstg3gs。十次均 HTTP 200 且返回 579，生成文本和返回的选中 token/top-5 logprobs 均完全一致，最大选中 token logprob 差值为零。测试保留服务缓存并由 DP 自动路由，验证单请求重复运行一致性，不代表不同 batch 大小之间的一致性。

## 经营场景计算测试

脚本 `test_single_curl_business_math.py`，设计和标准答案见 `BUSINESS_MATH_TEST.txt`，输出保存在 `single_curl_business_math_results/`。采用 Chat Completions 连续十次单请求，记录计算过程、答案和 logprobs。

## 活动票务新题复测

脚本 `test_single_curl_event_math.py`，设计 `EVENT_MATH_TEST.txt`。正式结果：`single_curl_event_math_results/20261102T064156Z_lkysf0bb/`。10次正常完成且13项数值正确，但文本有两个版本、差额解释有符号表述问题；详细表格和分析见analysis.txt。max_tokens=8192，thinking=false，以独立cache_salt隔离历史前缀缓存。
