"""为本实验的每个HTTP请求设置独立超时，单位为秒。"""
import aiohttp

from ais_bench.benchmark.models import VLLMCustomAPIChat


class GPQAAPIChat(VLLMCustomAPIChat):
    def __init__(self, request_timeout, model="", **kwargs):
        super().__init__(model=model, **kwargs)
        self.request_timeout = aiohttp.ClientTimeout(total=request_timeout)

    async def generate(self, input_data, max_out_len, output, session=None, **args):
        async with aiohttp.ClientSession(
            trust_env=True, timeout=self.request_timeout
        ) as request_session:
            return await super().generate(
                input_data, max_out_len, output, session=request_session, **args
            )
