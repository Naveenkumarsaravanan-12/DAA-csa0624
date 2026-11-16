import random, time, json, csv
from daa_algorithms import *
OUT="/home/claude/work/data"
X={}
# Q4 validation of predicted 1-second boundary
pts_rows=[]
for n in (4600,5000):
    pts=[(random.uniform(0,10000),random.uniform(0,10000)) for _ in range(n)]
    t=time.perf_counter(); r=closest_pair_bruteforce(pts); t=time.perf_counter()-t
    pts_rows.append([n,r[2],round(t,3)])
X["q4_validate"]=pts_rows
# Q6 validation of predicted n=22 (and 21)
def inst(n,seed):
    r=random.Random(seed); w=[r.randint(5,40) for _ in range(n)]; v=[r.randint(10,100) for _ in range(n)]
    return w,v,int(0.4*sum(w))
rows=[]
for n in (21,22):
    w,v,c=inst(n,100+n)
    t=time.perf_counter(); bv,_,tot,f=exhaustive_knapsack(w,v,c); t=time.perf_counter()-t
    assert bv==dp_knapsack(w,v,c)
    rows.append([n,tot,f,bv,round(t,3)])
X["q6_validate"]=rows
# Q7: correlated instances (v=w+10), 5 seeds, larger n
def corr(n,seed):
    r=random.Random(seed); w=[r.randint(5,40) for _ in range(n)]; v=[x+10 for x in w]; return w,v,int(0.4*sum(w))
rows=[]
for n in (20,24,28,32,36,40):
    ns=[];ts=[]
    for sd in range(5):
        w,v,c=corr(n,7000+n*10+sd)
        t=time.perf_counter(); b,nd=pruned_search(w,v,c,"C"); ts.append(time.perf_counter()-t); ns.append(nd)
        assert b==dp_knapsack(w,v,c)
    rows.append([n,round(sum(ns)/5),max(ns),round(sum(ts)/5,4),round(max(ts),4)])
X["q7_corr"]=rows
# Q7: adversarial for the bound: even weights, v=w, ODD capacity -> exact fill impossible, bound never tight
rows=[]
for n in (12,16,20,24):
    r=random.Random(n); w=[2*r.randint(3,20) for _ in range(n)]; v=list(w); c=(sum(w)//2)|1
    t=time.perf_counter(); b,nd=pruned_search(w,v,c,"C"); t=time.perf_counter()-t
    _,ndA=pruned_search(w,v,c,"A")
    assert b==dp_knapsack(w,v,c)
    rows.append([n,2**n,ndA,nd,round(t,3)])
X["q7_adv"]=rows
json.dump(X,open(f"{OUT}/extra_results.json","w"),indent=1)
with open(f"{OUT}/extra_validation_and_adversarial.csv","w",newline="") as f:
    w_=csv.writer(f)
    for k,v in X.items():
        w_.writerow([k]); w_.writerows(v); w_.writerow([])
print(json.dumps(X))
