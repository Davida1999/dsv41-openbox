# 1108 → CANN 9.3.0 容器构建记录

本文记录 2026-11-13 的构建时点。2026-11-17 已在 `177.42.22.1` 的 `c00954197_dsv41_cann930` 完成 Sleep level-2、HCCL 和 NPU IPC 全量更新验证，当前环境与源码补丁见最新报告。[CANN930 当前环境](cann930_environment_20261117.md)

日期：2026-11-13。范围：环境构建、无NPU条件下的编译和静态检查；不执行NPU初始化、算子、通信或推理测试。

## 环境与构建选择

| 项目 | 结果 |
|---|---|
| 新容器名（25、26分别创建） | c00954197_vllm_ascend_cann930 |
| 最终镜像 | c00954197/vllm-ascend:1108-cann930-20261113 |
| 镜像ID | sha256:3614b7afe7000897bf6f1c60a415f2547feb0e033bae0c9bf76c68a76ecf944f |
| Python | 3.12.16 |
| torch | 2.10.0+cpu，保持 |
| torch_npu | 2.10.0.post7，保持 |
| Triton-Ascend | 3.6.0.post2，保持 |
| vLLM | 0.32.0+empty；提交5df63a7203cdb9328d824a27b682e77c7b9928be，保持editable安装 |
| vLLM-Ascend | 0.19.1rc2.dev3401+g28e9e97ba；提交28e9e97ba254b076a88275e8a275556cb9a11664，重新构建并安装 |
| CANN toolkit / 950 ops / NNAL | 均为9.3.0 |
| ATB C++ ABI | 1，与现有torch一致 |
| 编译目标 | ascend950dt_9582 |
| 宿主机驱动 | 保持原25.1.rc3，不升级 |
| 最终运行策略 | 容器只创建、不启动；后续NPU验证另行执行 |

vLLM主体当前为empty后端的editable安装，未发现需随CANN升级重编译的NPU扩展，因此保持原安装。vLLM-Ascend包含C++扩展及自定义算子；旧构建缓存明确引用9.2.1，因此在新镜像中使用新构建目录和缓存，重新编译自定义算子包及vllm_ascend_C扩展。

## 安装与隔离

安装包来源：/mnt/share/a5comm/x00951564/pkgs-930/cann。三个包分别为toolkit、950-ops、nnal的9.3.0 aarch64 run包，SHA256见packages.sha256。

只在25.1的c00954197_cann930_build中安装和编译，限制8 CPU / 32 GiB内存、MAX_JOBS=4；未挂载davinci设备，非privileged。镜像基线来自原1108容器的--pause=false快照，不暂停旧容器。/vllm-workspace源码位于容器层，编译不会改写原1108源码。

实际安装根目录由安装器生成：
- CANN：/usr/local/Ascend/cann-9.3.0/cann-9.3.0
- ATB：/usr/local/Ascend/nnal930/nnal/atb/9.3.0/atb/cxx_abi_1

/usr/local/Ascend/cann、ascend-toolkit/latest和原nnal/atb/latest入口均已在新镜像内指向9.3.0。环境变量由/etc/profile.d/zz-cann930.sh统一配置，并写入镜像ENV，兼容docker exec直接调用与交互shell。Triton、vLLM、Torch扩展缓存使用独立cann930目录。旧9.2.1文件可作为镜像基线保留，但不在激活的搜索路径中。

## 已解决的构建问题

1. NNAL自动探测ABI时，torch自动加载NPU后端产生额外标准输出，误选ABI 0；改为显式source ATB set_env.sh --cxx_abi=1。
2. CMake未找到TorchConfig.cmake；构建时显式设置Torch_DIR及CMAKE_PREFIX_PATH，指向现有torch安装。
3. 无NPU构建环境中的torch后端探测提示混入torch版本号，导致CMake VERSION_EQUAL解析失败；仅构建进程设置TORCH_DEVICE_BACKEND_AUTOLOAD=0。此开关未写入最终镜像运行环境。

最终pip editable构建安装退出码0，无需修改vLLM或vLLM-Ascend源码，也未升级PyTorch依赖。先前失败日志保留在build_logs.tar.gz中。

## 静态验证与限制

- import torch / torch_npu / vllm / vllm_ascend / vllm_ascend.vllm_ascend_C成功。
- 从/proc/self/maps检查实际加载的88个Ascend相关库，无cann-9.2.1路径。
- 包版本与预期基线一致。
- 直接对扩展运行ldd时，libtorch/libc10/libtorch_npu等可能显示not found；Python先导入torch/torch_npu后已成功解析并加载扩展。没有把单独ldd输出等同于完整Python加载结果。
- 设备探测提示来自无设备的构建环境；没有执行torch.npu.set_device、NPU张量、HCCL或服务启动，不声明CANN9.3.0与现有驱动/torch_npu组合已通过设备功能验证。

## 挂载与宿主机保护

最终容器继承1108的host网络、host IPC、设备及必要驱动/URMA/通信挂载；驱动和固件只读。当前保持created状态，因此设备映射不产生运行中的NPU任务。模型和工作区复用共享存储，安装包目录只读挂载。

两台宿主机的/etc/hccl_rootinfo.json源文件已被删除，旧1108仍持有其挂载inode。分别从旧容器读取原内容，保存到：
/workspace/experiments/E23_cann930/configs/node25/hccl_rootinfo.json
/workspace/experiments/E23_cann930/configs/node26/hccl_rootinfo.json

新容器用各自保存文件只读挂载到/etc/hccl_rootinfo.json，不改写宿主机/etc。JSON引用的/usr/local/Ascend/driver/topo/950/atlas_950_1.json在两端均存在，沿用只读驱动挂载。该文件为1108现有配置快照，后续双机功能验证时需一并核验。

## 归档与下一步

远端目录：/workspace/experiments/E23_cann930/（容器视角）；宿主机对应/mnt/share/c00954197/workspace/experiments/E23_cann930/。
本地目录：outputs/cann930/，包含版本/库路径证据、安装包校验值和构建日志。

允许使用设备后，再验证基础NPU初始化、关键算子、双机HCCL/Gloo及E23推理；Gloo继续使用enp39s0f0，HCCL使用enp35s0f2。本次没有升级驱动、没有清理他人任务，也没有启动新推理服务。

## 最终交付核验

25.1与26.1新容器均已创建，状态均为created；镜像ID一致，原1108容器均保持running。新容器运行ENV无9.2.1搜索路径，且未设置TORCH_DEVICE_BACKEND_AUTOLOAD=0。临时构建容器已停止。未执行NPU功能测试。
