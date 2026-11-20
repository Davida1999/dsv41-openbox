"""Read-only, standard-library scalar reference for exported FP4/FP8 weights."""
import argparse,csv,hashlib,json,math,random,struct
from pathlib import Path
SIZES={'I8':1,'U8':1,'F8_E4M3':1,'F8_E4M3FN':1,'F8_E8M0':1,'BF16':2,'F16':2,'F32':4}

class Store:
 def __init__(self,path):
  self.path=Path(path);raw=(self.path/'model.safetensors.index.json').read_bytes();self.index_sha256=hashlib.sha256(raw).hexdigest();self.index=json.loads(raw)['weight_map'];self.files={}
 def tensor(self,name):
  relative=self.index[name];assert not Path(relative).is_absolute() and '..' not in Path(relative).parts
  if relative not in self.files:
   f=(self.path/relative).open('rb');n=struct.unpack('<Q',f.read(8))[0];assert n<64*2**20;header=json.loads(f.read(n));self.files[relative]=(f,8+n,header)
  f,base,header=self.files[relative];return f,base,header[name]
 def raw(self,name,row,col):
  f,base,t=self.tensor(name);dtype=t['dtype'];size=SIZES[dtype];offset=t['data_offsets'][0]+(row*t['shape'][1]+col)*size;f.seek(base+offset);return dtype,f.read(size)

def e4m3(byte):
 sign=-1.0 if byte&128 else 1.0;exp=(byte>>3)&15;mant=byte&7
 if exp==15 and mant==7:return float('nan')
 return sign*((mant/8)*2**-6 if exp==0 else (1+mant/8)*2**(exp-7))

def scalar(dtype,data):
 if dtype.startswith('F8_E4M3'):return e4m3(data[0])
 if dtype=='F8_E8M0':return 2.0**(data[0]-127) if data[0]!=255 else float('nan')
 if dtype=='F32':return struct.unpack('<f',data)[0]
 if dtype=='F16':return struct.unpack('<e',data)[0]
 if dtype=='BF16':return struct.unpack('<f',b'\0\0'+data)[0]
 if dtype=='U8':return float(data[0])
 raise ValueError(dtype)

def bf16_rne(value):
 bits=struct.unpack('<I',struct.pack('<f',value))[0]
 return ((bits+0x7fff+((bits>>16)&1))>>16)&0xffff

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--bf16',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=True)
 src,dst=Store(a.source),Store(a.bf16);selected=[]
 for layer in [0,10,20,39]:
  for expert in [0,17,383]:
   for matrix in ['w1','w2']:
    key=f'layers.{layer}.ffn.experts.{expert}.{matrix}.weight';assert key in src.index and key in dst.index;selected.append(key)
  for matrix in ['wq_a','wkv']:
   key=f'layers.{layer}.attn.{matrix}.weight';assert key in src.index and key in dst.index;selected.append(key)
 for layer in [1,14]:selected.append(f'layers.{layer}.engram.embed.weight')
 rng=random.Random(20261120);rows=[];tensors=[];mismatches=[]
 for key in selected:
  _,_,st=src.tensor(key);_,_,dt=dst.tensor(key);scale=key[:-7]+'.scale';assert scale in src.index
  _,_,sc=src.tensor(scale);R,C=dt['shape'];assert dt['dtype']=='BF16'
  kind='FP4_E2M1' if st['dtype']=='I8' else 'FP8_E4M3FN';assert st['dtype']=='I8' or st['dtype'].startswith('F8_E4M3')
  if kind=='FP4_E2M1':assert st['shape']==[R,C//2] and sc['shape'][0]==R;group=C//sc['shape'][1]
  else:assert st['shape']==[R,C];br=R//sc['shape'][0];bc=C//sc['shape'][1]
  er=sorted({v for v in [0,1,2,31,32,33,R//2,R-1] if 0<=v<R});ec=sorted({v for v in [0,1,2,3,31,32,33,C//2,C-1] if 0<=v<C});points={(i,j) for i in er for j in ec};points.update((rng.randrange(R),rng.randrange(C)) for _ in range(64))
  good=0;codes=set()
  for i,j in sorted(points):
   _,raw=src.raw(key,i,j//2 if kind=='FP4_E2M1' else j)
   if kind=='FP4_E2M1':
    code=(raw[0]>>(4*(j%2)))&15;value=(-1.0 if code&8 else 1.0)*[0.0,.5,1.0,1.5,2.0,3.0,4.0,6.0][code&7];sd,sb=src.raw(scale,i,j//group)
   else:code=raw[0];value=e4m3(code);sd,sb=src.raw(scale,i//br,j//bc)
   factor=scalar(sd,sb);expected=bf16_rne(value*factor);td,tb=dst.raw(key,i,j);actual=struct.unpack('<H',tb)[0];equal=expected==actual;good+=equal;codes.add(code)
   row={'tensor':key,'format':kind,'row':i,'column':j,'quantized_code':code,'scale':factor,'expected_bf16_bits':expected,'actual_bf16_bits':actual,'equal':equal};rows.append(row)
   if not equal:mismatches.append(row)
  tensors.append({'tensor':key,'format':kind,'shape':[R,C],'source_dtype':st['dtype'],'scale_dtype':sc['dtype'],'scale_shape':sc['shape'],'sampled_values':len(points),'equal_values':good,'distinct_sampled_codes':len(codes)})
 with (a.output/'sample_values.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
 result={'status':'passed' if not mismatches else 'source_export_mismatch_requires_provenance_check','source':a.source,'bf16':a.bf16,'source_index_sha256':src.index_sha256,'bf16_index_sha256':dst.index_sha256,'reference':'Independent scalar bit decode: E2M1/E4M3FN, E8M0 scale, FP32 product and manual BF16 round-to-nearest-even. Converter functions are not imported.','checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'seed':20261120,'tensors_checked':len(tensors),'values_checked':len(rows),'mismatches':mismatches[:20],'mismatch_count':len(mismatches),'scope':'Representative samples across layers0/10/20/39, experts0/17/383 and Engram layers1/14; not exhaustive full-checkpoint numerical alignment','tensors':tensors}
 (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['tensors','mismatches']},indent=2))
 assert not mismatches,'Source/export values differ; do not infer a conversion defect before confirming source version'
if __name__=='__main__':main()
