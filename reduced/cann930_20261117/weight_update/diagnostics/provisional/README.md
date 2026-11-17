# 早期字段假设记录

本目录记录针对 log2phy 等字段的早期 CPU 试验。该补丁的 V2 实机复验仍出现同样的 32 元素加载配额差。随后依据实际通信初始化代码和模型字段，定位到 expert_ids_per_ep_rank。

正式修复见 [通信映射加载补丁](../../patches/expert_rank_buffer_loading.patch)。
