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

    def openbox_memory_probe(self, directory):
        import json,os,torch
        from pathlib import Path
        from vllm_ascend.device_allocator.camem import CaMemAllocator
        target=Path(directory).resolve()
        assert target.is_relative_to(Path('/mnt/share/c00954197/Workspace/dsv41/basic/routing_replay/runs'))
        target.mkdir(parents=True,exist_ok=True)
        torch.npu.synchronize()
        allocator=CaMemAllocator.get_instance()
        rows=[{'tag':d.tag,'bytes':d.handle[1],'asleep':d.is_asleep,'cpu_backup':d.cpu_backup_tensor is not None} for d in allocator.pointer_to_data.values()]
        free,total=torch.npu.mem_get_info()
        result={'pid':os.getpid(),'dp_rank':self.parallel_config.data_parallel_index,'allocated':torch.npu.memory_allocated(),'reserved':torch.npu.memory_reserved(),'free':free,'total':total,'allocations':rows}
        (target/f'dp{result["dp_rank"]}.json').write_text(json.dumps(result,indent=2)+'\n')
        return {k:v for k,v in result.items() if k!='allocations'}

    def openbox_set_host_interleave(self, directory):
        import ctypes,json,os
        from pathlib import Path
        target=Path(directory).resolve()
        assert target.is_relative_to(Path('/mnt/share/c00954197/Workspace/dsv41/basic/routing_replay/runs'))
        target.mkdir(parents=True,exist_ok=True)
        lib=ctypes.CDLL('libnuma.so.1',use_errno=True)
        lib.numa_allocate_nodemask.restype=ctypes.c_void_p
        lib.numa_bitmask_setall.argtypes=[ctypes.c_void_p];lib.numa_bitmask_setall.restype=ctypes.c_void_p
        lib.numa_set_interleave_mask.argtypes=[ctypes.c_void_p]
        lib.numa_get_interleave_mask.restype=ctypes.c_void_p
        lib.numa_bitmask_isbitset.argtypes=[ctypes.c_void_p,ctypes.c_uint]
        lib.numa_bitmask_free.argtypes=[ctypes.c_void_p]
        mask=lib.numa_allocate_nodemask();lib.numa_bitmask_setall(mask);lib.numa_set_interleave_mask(mask);lib.numa_bitmask_free(mask)
        mask=lib.numa_get_interleave_mask();nodes=[i for i in range(lib.numa_num_configured_nodes()) if lib.numa_bitmask_isbitset(mask,i)];lib.numa_bitmask_free(mask)
        result={'pid':os.getpid(),'dp_rank':self.parallel_config.data_parallel_index,'interleaved_nodes':nodes}
        assert len(nodes)>1,result
        (target/f'dp{result["dp_rank"]}.json').write_text(json.dumps(result,indent=2)+'\n')
        return result

    def openbox_begin_hidden_trace(self, directory, phase, positions):
        import json,types,torch,numpy as np
        from pathlib import Path
        target=Path(directory).resolve()
        assert target.is_relative_to(Path('/mnt/share/c00954197/Workspace/dsv41/basic/routing_replay/runs'))
        assert phase.replace('_','').isalnum()
        target.mkdir(parents=True,exist_ok=True)
        if self.parallel_config.data_parallel_index!=0:return {'tracing':False}
        self._openbox_hidden_target=target;self._openbox_hidden_phase=phase;self._openbox_hidden_positions=set(positions)
        if getattr(self,'_openbox_hidden_installed',False):return {'tracing':True,'phase':phase}
        model=self.model_runner.get_model()
        if hasattr(model,'language_model'):model=model.language_model
        if hasattr(model,'model'):model=model.model
        owner=self
        def wrapper(original,idx):
            def forward(layer,positions,hidden,mix,*args,**kwargs):
                pos=positions.detach().cpu().tolist()
                indices=[i for i,p in enumerate(pos) if p in owner._openbox_hidden_positions]
                def snapshot(label,tensor):
                    if not indices:return
                    ids=torch.tensor(indices,device=tensor.device,dtype=torch.int64)
                    data=tensor.detach().index_select(0,ids).float().cpu().numpy()
                    path=owner._openbox_hidden_target/f'{owner._openbox_hidden_phase}_l{idx:02}_{label}.npy'
                    np.save(path,data,allow_pickle=False)
                snapshot('input_hidden',hidden);snapshot('input_mix',mix)
                value=original(positions,hidden,mix,*args,**kwargs)
                snapshot('output_hidden',value[0]);snapshot('output_mix',value[1]);return value
            return forward
        for layer in model.layers:
            if hasattr(layer,'layer_idx'):layer.forward=types.MethodType(wrapper(layer.forward,layer.layer_idx),layer)
        self._openbox_hidden_installed=True
        return {'tracing':True,'phase':phase}


    def openbox_host_policy(self, directory):
        import ctypes,json,os
        from pathlib import Path
        target=Path(directory).resolve()
        assert target.is_relative_to(Path('/mnt/share/c00954197/Workspace/dsv41/basic/routing_replay/runs'))
        target.mkdir(parents=True,exist_ok=True)
        lib=ctypes.CDLL('libnuma.so.1');lib.numa_get_interleave_mask.restype=ctypes.c_void_p
        lib.numa_bitmask_isbitset.argtypes=[ctypes.c_void_p,ctypes.c_uint];lib.numa_bitmask_free.argtypes=[ctypes.c_void_p]
        mask=lib.numa_get_interleave_mask();nodes=[i for i in range(lib.numa_num_configured_nodes()) if lib.numa_bitmask_isbitset(mask,i)];lib.numa_bitmask_free(mask)
        result={'pid':os.getpid(),'dp_rank':self.parallel_config.data_parallel_index,'interleaved_nodes':nodes}
        (target/f'dp{result["dp_rank"]}.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
