"""Observe real scheduled requests and dummy forwards in each DP worker."""

import json
import time
import types
from pathlib import Path


class RoutingProbe:
    def begin_routing_trace(self, directory):
        root = Path("/mnt/share/c00954197/Workspace/dsv41/basic/routing_replay/runs")
        target = Path(directory).resolve()
        assert target.is_relative_to(root)
        target.mkdir(parents=True, exist_ok=True)
        rank = self.parallel_config.data_parallel_index
        path = target / f"worker_dp{rank}.jsonl"
        self._routing_trace_path = path
        if getattr(self, "_routing_trace_installed", False):
            return {"dp_rank": rank, "path": str(path)}
        owner = self
        original_execute = self.model_runner.execute_model
        original_dummy = self.model_runner._dummy_run

        def record(data):
            with owner._routing_trace_path.open("a") as stream:
                stream.write(json.dumps({"host_epoch": time.time(), "dp_rank": rank, **data}) + "\n")

        def execute(runner, scheduler_output, *args, **kwargs):
            scheduled = getattr(scheduler_output, "num_scheduled_tokens", {})
            record({"event": "execute_model", "requests": scheduled,
                    "num_real_requests": len(scheduled), "tokens": sum(scheduled.values())})
            try:
                return original_execute(scheduler_output, *args, **kwargs)
            except BaseException as error:
                record({"event": "error", "error_type": type(error).__name__, "error": str(error)})
                raise

        def dummy(runner, *args, **kwargs):
            record({"event": "dummy_run", "num_tokens": args[0] if args else kwargs.get("num_tokens"),
                    "uniform_decode": kwargs.get("uniform_decode")})
            return original_dummy(*args, **kwargs)

        self.model_runner.execute_model = types.MethodType(execute, self.model_runner)
        self.model_runner._dummy_run = types.MethodType(dummy, self.model_runner)
        self._routing_trace_installed = True
        return {"dp_rank": rank, "path": str(path)}
