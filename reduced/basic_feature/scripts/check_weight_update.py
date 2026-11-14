# Uses the installed trainer API. Streams checkpoint tensors; never instantiates a training model.
import argparse,json,hashlib,os
from pathlib import Path
from common import config,request,rpc,save,inputs
from check_inference import generate
p=argparse.ArgumentParser();p.add_argument('case');p.add_argument('out');a=p.parse_args();c=config(a.case)
if c.get('blocked'):raise SystemExit('BLOCKED: '+c['blocked'])
if c['backend']=='ipc':os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION']='1'
import torch,torch_npu
from safetensors import safe_open
from vllm.distributed.weight_transfer.base import WeightSource,ParamMeta
from vllm.distributed.weight_transfer.clients import HTTPVLLMWeightSyncClient
from vllm.distributed.weight_transfer.factory import WeightTransferTrainerFactory
from vllm_ascend.distributed.weight_transfer import register_engine
root=Path(c['model']);mapping=json.loads((root/'model.safetensors.index.json').read_text())['weight_map'];name=c['probe_checkpoint_name'];assert name in mapping
world=request(c,'/get_world_size')['world_size']
if c['backend']=='ipc':assert world==1 and c['devices']==[c['trainer_device']],'IPC requires matching same physical device'
else:assert c['trainer_device'] not in c['devices'],'HCCL sender must use a separate NPU'
torch.npu.set_device(c['trainer_device'])
class Source(WeightSource):
 scale=1.
 def metadata(self):
  import struct
  headers={};result=[]
  for n,f in mapping.items():
   if '.engram.' in n or n.startswith('mtp.'):continue
   if f not in headers:
    with open(root/f,'rb') as h:headers[f]=json.loads(h.read(struct.unpack('<Q',h.read(8))[0]))
   h=headers[f][n];dtype={'BF16':torch.bfloat16,'F32':torch.float32}.get(h['dtype'])
   assert dtype is not None,(n,h['dtype'])
   result.append(ParamMeta(n,dtype,tuple(h['shape'])))
  return result
 def __iter__(self):
  for index,meta in enumerate(self.metadata()):
   if index % c.get('weight_progress_every',256)==0:print(f'weight tensor {index}: {meta.name}',flush=True)
   with safe_open(root/mapping[meta.name],framework='pt',device='cpu') as f:t=f.get_tensor(meta.name)
   if meta.name==name:t=t*self.scale
   yield meta.name,t.to(f"npu:{c['trainer_device']}")
source=Source();register_engine()
if c['backend']=='hccl':
 from vllm_ascend.distributed.weight_transfer.hccl_engine import HCCLTrainerInitInfo
 from vllm.utils.network_utils import get_open_port
 info=HCCLTrainerInitInfo(rank=0,master_address='127.0.0.1',master_port=get_open_port(),world_size=world+1,packed=False,packed_buffer_size_bytes=c['weight_buffer_mib']*2**20)
else:
 from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import NPUIPCTrainerInitInfo
 info=NPUIPCTrainerInitInfo(rank=0,packed=False)
def all_hashes(label):
 target=Path(a.out)/label
 rpc(c,'basic_probe',c['probe_runtime_name'],str(target))
 rows=[json.loads(p.read_text()) for p in sorted(target.glob('*.json'))]
 assert len(rows)==world,(len(rows),world)
 return {'results':rows}
base_hash=all_hashes('weights_A_initial');ids=inputs(c,c['prompts'][0]);base=generate(c,ids);rows=[]
engine=WeightTransferTrainerFactory.trainer_init(info,client=HTTPVLLMWeightSyncClient(base_url=f"http://{c['host']}:{c['port']}"),source=source)
try:
 for scale in [c['weight_scale'],1.]:
  request(c,'/pause?mode=wait&clear_cache=true',{});source.scale=scale;engine.send_weights()
  got=all_hashes('weights_B' if scale!=1. else 'weights_A_restored')
  assert len(got['results'])==len(base_hash['results'])
  for x,y in zip(base_hash['results'],got['results']):
   assert x['rank']==y['rank']
   assert (x['sha256']==y['sha256'])==(scale==1.),'Worker weight did not change/restore as expected'
  request(c,'/resume',{});result=generate(c,ids)
  rows.append({'scale':scale,'worker_hashes':got,'inference':result});save(a.out,'weight_update.json',{'baseline':base,'baseline_hashes':base_hash,'rounds':rows,'scope':'流式发送主模型权重，改变并逐rank校验norm；MoE分片需结合接收日志确认，不能仅凭norm哈希认定所有专家正确更新'})
 assert rows[-1]['inference']['response']['choices'][0]['token_ids']==base['response']['choices'][0]['token_ids']
finally:
 # Never resume after a failed transfer. Stop and reload the server before continuing.
 engine.shutdown()
