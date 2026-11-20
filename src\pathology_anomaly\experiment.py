import argparse,json
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score,roc_auc_score

def make_data(n=800,seed=27,shift=.0):
    r=np.random.default_rng(seed); z=r.normal(size=(n,5)); basis=np.random.default_rng(991).normal(size=(5,36)); x=z@basis+r.normal(scale=.35,size=(n,36))+shift; y=r.binomial(1,.3,n); x[y==1]+=r.normal(0,.48,size=((y==1).sum(),36)); return x,y
def evaluate(seed=27):
    train,_=make_data(1000,seed); test,y=make_data(600,seed+1,.2); model=PCA(10,random_state=seed).fit(train); rec=model.inverse_transform(model.transform(test)); score=((test-rec)**2).mean(1); return {"auroc":float(roc_auc_score(y,score)),"average_precision":float(average_precision_score(y,score)),"normal_components":10}
def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=Path("outputs/demo"));p.add_argument("--seed",type=int,default=27);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);r=evaluate(a.seed);(a.output/"metrics.json").write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if __name__=="__main__":main()
