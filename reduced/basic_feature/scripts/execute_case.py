import json,subprocess,sys,time,os
from pathlib import Path
root=Path(__file__).resolve().parents[1];case=sys.argv[1];out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
def stage(name):
 (out/'stage.json').write_text(json.dumps({'stage':name,'time':time.time()}))
def run(args,name):
 stage(name)
 with open(out/(name+'.log'),'w') as log:
  subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True)
try:
 run([sys.executable,str(root/'scripts/serve.py'),case,str(out),'--launch'],'startup')
 run(['bash',str(root/'scripts/run_case.sh'),case,str(out)],'validation')
 if case=='F01_basic':run([sys.executable,str(root/'scripts/benchmark.py'),case,str(out),'--execute'],'benchmark')
 stage('CLIENT_COMPLETE_REVIEW_REQUIRED')
except Exception as e:
 stage('FAILED');(out/'failure.txt').write_text(repr(e));raise
