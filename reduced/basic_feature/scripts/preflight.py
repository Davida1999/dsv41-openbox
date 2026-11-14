import argparse, json, subprocess, struct, hashlib, re
from pathlib import Path
from common import config,save
p=argparse.ArgumentParser();p.add_argument('case');p.add_argument('out');p.add_argument('--require-free',action='store_true');a=p.parse_args();c=config(a.case)
model=Path(c['model']);hf=json.loads((model/'config.json').read_text());tc=hf['text_config']
assert tc['n_routed_experts']==c['experts'] and tc['engram_layer_ids']==[]
assert hf['dtype']=='bfloat16' and not hf.get('quantization_config')
index=json.loads((model/'model.safetensors.index.json').read_text());dtypes={};sizes={};headers={}
for f in sorted(set(index['weight_map'].values())):
 with open(model/f,'rb') as h:
  size=struct.unpack('<Q',h.read(8))[0];header=json.loads(h.read(size));headers[f]=header
 for n,v in header.items():
  if n=='__metadata__':continue
  dtypes[v['dtype']]=dtypes.get(v['dtype'],0)+1
  group='engram' if 'engram' in n else ('mtp' if n.startswith('mtp.') else 'other')
  sizes[group]=sizes.get(group,0)+v['data_offsets'][1]-v['data_offsets'][0]
for n,f in index['weight_map'].items():assert n in headers[f],(n,f)
raw=subprocess.check_output(['npu-smi','info'],text=True)
versions={}
for name in ['vllm','vllm-ascend']:
 repo='/vllm-workspace/'+name
 versions[name]=subprocess.check_output(['git','-C',repo,'rev-parse','HEAD'],text=True).strip()
 versions[name+'_status']=subprocess.check_output(['git','-C',repo,'status','--short'],text=True)
save(a.out,'preflight.json',dict(config_sha256=hashlib.sha256((model/'config.json').read_bytes()).hexdigest(),versions=versions,dtypes=dtypes,weight_bytes=sizes,npu_smi=raw,probe_checkpoint_present=c['probe_checkpoint_name'] in index['weight_map']))
print(json.dumps({'dtypes':dtypes,'weight_bytes':sizes},indent=2))
if a.require_free:
 # Conservative admission: refuse unknown npu-smi format rather than assume free.
 rows=re.findall(r'(\d+)\s*/\s*(\d+)\s*\|',raw)
 hbm=[(int(u),int(t)) for u,t in rows if int(t)==98304]
 assert len(hbm)==8,'Cannot parse eight 96GiB HBM devices; inspect npu-smi manually'
 for d in c['devices']:assert hbm[d][1]-hbm[d][0]>=c['free_hbm_min_mib'],f'NPU {d} occupied; do not stop unrelated jobs'
