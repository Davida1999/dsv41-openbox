# Four colocated trainer ranks; Gloo carries handles only, NPU IPC carries weights.
import argparse,os,json,struct
from pathlib import Path
os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION']='1'
import torch,torch_npu
import torch.distributed as dist
from safetensors import safe_open
from common import config,request,rpc,save,inputs
from check_inference import generate
from vllm.distributed.weight_transfer.base import WeightSource,ParamMeta
from vllm.distributed.weight_transfer.clients import HTTPVLLMWeightSyncClient
from vllm.distributed.weight_transfer.factory import WeightTransferTrainerFactory
from vllm_ascend.distributed.weight_transfer import register_engine
from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import NPUIPCTrainerInitInfo
p=argparse.ArgumentParser();p.add_argument('case');p.add_argument('out');a=p.parse_args();c=config(a.case)
rank=int(os.environ['LOCAL_RANK']);torch.npu.set_device(c['devices'][rank]);dist.init_process_group('gloo')
assert dist.get_world_size()==len(c['devices'])==c['dp']*c['tp']
root=Path(c['model']);mapping=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
class Source(WeightSource):
 scale=1.
 def metadata(self):
  headers={};result=[]
  for n,f in mapping.items():
   if '.engram.' in n or n.startswith('mtp.'):continue
   if f not in headers:
    with open(root/f,'rb') as h:headers[f]=json.loads(h.read(struct.unpack('<Q',h.read(8))[0]))
   h=headers[f][n];dtype={'BF16':torch.bfloat16,'F32':torch.float32}[h['dtype']]
   result.append(ParamMeta(n,dtype,tuple(h['shape'])))
  return result
 def __iter__(self):
  for index,meta in enumerate(self.metadata()):
   if rank==0 and index % c.get('weight_progress_every',256)==0:print(f'weight tensor {index}: {meta.name}',flush=True)
   with safe_open(root/mapping[meta.name],framework='pt',device='cpu') as f:t=f.get_tensor(meta.name)
   if meta.name==c['probe_checkpoint_name']:t=t*self.scale
   yield meta.name,t.to(f"npu:{c['devices'][rank]}")
def hashes(label):
 target=Path(a.out)/label;rpc(c,'basic_probe',c['probe_runtime_name'],str(target))
 rows=[json.loads(p.read_text()) for p in sorted(target.glob('*.json'))]
 assert len(rows)==c['dp']*c['tp'],(len(rows),c['dp']*c['tp'])
 return rows
source=Source();register_engine()
if rank==0:
 ids=inputs(c,c['prompts'][0]);baseline=generate(c,ids);initial=hashes('ipc_A_initial');save(a.out,'ipc_baseline.json',{'inference':baseline,'hashes':initial})
dist.barrier()
engine=WeightTransferTrainerFactory.trainer_init(NPUIPCTrainerInitInfo(rank=rank,packed=True,packed_buffer_size_bytes=c['weight_buffer_mib']*2**20),client=HTTPVLLMWeightSyncClient(base_url=f"http://{c['host']}:{c['port']}"),source=source)
try:
 for label,scale in [('ipc_B',c['weight_scale']),('ipc_A_restored',1.)]:
  if rank==0:request(c,'/pause?mode=wait&clear_cache=true',{})
  dist.barrier();source.scale=scale;engine.send_weights();dist.barrier()
  if rank==0:
   current=hashes(label)
   for old,new in zip(initial,current):
    assert old['dp_rank']==new['dp_rank']
    assert (old['sha256']==new['sha256'])==(scale==1.)
   request(c,'/resume',{});result=generate(c,ids);save(a.out,label+'.json',{'hashes':current,'inference':result})
   if scale==1.:assert result['response']['choices'][0]['token_ids']==baseline['response']['choices'][0]['token_ids']
  dist.barrier()
finally:
 engine.shutdown();dist.destroy_process_group()
