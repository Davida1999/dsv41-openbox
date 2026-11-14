import argparse,base64,io,json
import numpy as np
from common import config,save
from check_inference import run
p=argparse.ArgumentParser();p.add_argument('case');p.add_argument('out');a=p.parse_args();c=config(a.case);run(c,a.out)
data=json.load(open(a.out+'/inference.json'));checks=[]
for i,row in enumerate(data['serial']+data['parallel']):
 choice=row['response']['choices'][0];value=choice.get('routed_experts');assert value,'Missing routing payload'
 routes=np.load(io.BytesIO(base64.b64decode(value)),allow_pickle=False)
 assert routes.ndim==3 and routes.shape[1:]==(c['layers'],c['top_k']),routes.shape
 assert np.issubdtype(routes.dtype,np.integer) and routes.size and routes.min()>=0 and routes.max()<c['experts']
 total=len(row['input_token_ids'])+len(choice['token_ids'])-1
 assert routes.shape[0]==total,(routes.shape,total)
 np.save(a.out+f'/routing_{i}.npy',routes);checks.append({'shape':list(routes.shape),'min':int(routes.min()),'max':int(routes.max())})
save(a.out,'routing_checks.json',{'checks':checks,'note':'仅验返回路由；结合expert_remap.json检查原专家映射，不代表已验证强制replay。'})
