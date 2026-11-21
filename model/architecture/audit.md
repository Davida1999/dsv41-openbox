# 结构审计与证据边界

日期：2026-11-21。审计基于归档仓库提交 `7e53afe992ca5276ea94155797804e398cb6c2cd` 中的模型资产及运行快照。
逐项值、文件哈希、配置键与源码符号见 [facts.json](facts.json)。未加载权重、未执行模型源码，也未启动 NPU 服务。

## 核对结果

1. 原始 `text_config.num_hidden_layers=40`，权重索引包含且仅包含 `layers.0..39`。
   每个主干层均有编号 `0..383` 的 routed expert 权重；视觉为 `vision.blocks.0..31`，
   草稿为 `mtp.0..2`。这些是索引存在性核验，不涉及张量数值或逐算子精度。
2. `compress_ratios` 有 43 项。实际 `build_layer_plan()` 明确截取前 40 项；主干分布为
   2 层 r=0、18 层 r=2、20 层 r=1。余下 `[0,0,0]` 不纳入主干计数。
   DSpark 源码另将三个 `mtp.*` 块构造为运行层号 40、41、42。
3. KV source 为 `[2,8,14,20]`，Indexer source 为 `[2,8,14,20,24,28,32,36]`。
   消费者使用不晚于本层的最近 source；candidate source=20。源码判断的是
   `index_source > candidate_source`，因此候选筛选从 source=24 开始，而非所有 L20+ 都执行。
4. 文本 Decoder 为 mHC pre → RMSNorm → Attention → mHC post，再经 mHC pre →
   RMSNorm → MoE → mHC post。Engram 在指定 Decoder 之前修改 HC 残差流；最终经过 HC collapse 和 RMSNorm。
5. 视觉分支使用 RMSNorm，不能套用常见 ViT 的 LayerNorm 假设。Aligner 使用补齐后的
   3×3 空间合并和 GELU MLP；视觉 Transformer 的 FFN 为 SwiGLU。图文 embedding
   合并是在占位符位置替换，未画成无条件追加全部图片 token。
6. 950DT 的 cache capability 选择 packed cache：SWA 为 FP8 值与 BF16 scale，
   长程 KV 为 FP4 值与 BF16 scale。BF16 权重或 CLI cache dtype 不能代表全部物理 KV 布局。

## 身份与不确定项

- 主干和图文 wrapper 使用既有 E32 运行快照，并验证了解压后 SHA256；其基线是
  vLLM-Ascend `28e9e97`，含已交付运行补丁。图中的完整模型尺寸仍来自完整资产配置和索引，
  不把运行快照中 E32/Engram off 的实验范围扩展成完整模型实测。
- 视觉、DSpark 和 cache 内部细节使用同一固定公开基线的源码原文；本次未取得这些文件的新运行态指纹。
  不将公开基线核对写成当前服务器逐文件复验。
- 配置没有显式声明 early hash-routing 层数。本图不凭模型名字推断另一个 Hash-MoE 层划分；
  该项在 `facts.json` 中为空并附理由。
- 三个 MTP 草稿层在配置、索引和基线实现中均有依据。已有服务没有显式开启投机解码，
  因此图上单列“结构存在、未启用/未验收”，投机 token 数为空。
- 配置上下文上限为 1,048,576；完整模型归档服务窗口为 40,960/73,728。
  1,000,000 输入测试属于保持 40 层、E32、Engram off 的副本。

## 图示检查

- 中央按从上到下的数据流阅读；左侧视觉和 Engram 连回主干，右侧展开 Attention、MoE 与可选 MTP。
- 逻辑维度和运行并行拓扑分开；共享专家不计入 routed Top-6，MTP 层不计入 40 层主干。
- PNG 实际渲染为 2600×2540；检查中文字体、文字边界和箭头连接，保留可编辑 SVG 与可重跑脚本。
- README、正式交付件及模型分析链接同一套图示和参数依据；原始运行日志与评分不改写。
