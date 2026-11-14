# Read-only worker extension; no direct weight mutation.
import hashlib
class Probe:
 def basic_probe(self,name='',snapshot_dir=''):
  import torch
  model=self.model_runner.get_model()
  params=dict(model.named_parameters())
  result={'rank':self.rank,'allocated':torch.npu.memory_allocated(),'reserved':torch.npu.memory_reserved()}
  if name:
   if name not in params:raise KeyError(f'{name}; norm candidates: {[n for n in params if "norm" in n][-20:]}')
   t=params[name].detach().cpu().contiguous()
   result.update(name=name,shape=list(t.shape),dtype=str(t.dtype),sha256=hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest())
  if snapshot_dir:
   from pathlib import Path
   import json
   target=Path(snapshot_dir).resolve()
   root=Path('/mnt/share/c00954197/workspace/dsv41-kx/basic_feature/results').resolve()
   assert target.is_relative_to(root)
   target.mkdir(parents=True,exist_ok=True)
   result['dp_rank']=self.parallel_config.data_parallel_rank
   (target/f'dp{self.parallel_config.data_parallel_rank}_rank{self.rank}.json').write_text(json.dumps(result))
  return result
