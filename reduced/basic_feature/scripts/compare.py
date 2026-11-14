import json,sys
left=json.load(open(sys.argv[1]));right=json.load(open(sys.argv[2]));out=[]
for a,b in zip(left['serial'],right['serial']):
 assert a['input_token_ids']==b['input_token_ids'],'Input mismatch: invalid comparison'
 x=a['response']['choices'][0]['token_ids'];y=b['response']['choices'][0]['token_ids']
 out.append({'equal':x==y,'first_difference':next((i for i,(u,v) in enumerate(zip(x,y)) if u!=v),min(len(x),len(y)) if x!=y else None)})
assert len(left['serial'])==len(right['serial'])
print(json.dumps(out,indent=2));sys.exit(0 if all(x['equal'] for x in out) else 2)
