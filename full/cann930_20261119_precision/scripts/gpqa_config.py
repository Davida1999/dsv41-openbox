from mmengine.config import read_base

with read_base():
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general_chat import models
    from ais_bench.benchmark.configs.datasets.gpqa.gpqa_gen_0_shot_cot_chat_prompt import gpqa_datasets as datasets

from gpqa_api import GPQAAPIChat

models[0]["type"] = GPQAAPIChat
models[0]["request_timeout"] = 360000
