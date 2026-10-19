"""CPU-only controlled reward-hypothesis screening. No CoRe / MetaWorld claims."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np

NCTX, NSTAGE, NACT = 8, 4, 4

def correct_action(ctx,stage): return (ctx*3+stage*5+1)%NACT

def collect_data(seed,n=1200):
    rng=np.random.default_rng(seed)
    samples=[]
    for i in range(n):
        ctx=int(rng.integers(NCTX));traj=[];success=1
        for t in range(NSTAGE):
            a=correct_action(ctx,t) if rng.random()<0.62 else int(rng.integers(NACT))
            traj.append((ctx,t,a))
            if a!=correct_action(ctx,t):success=0;break
        samples.append((traj,success))
    return samples

def pswm_scores(data,noise,seed,mode):
    rng=np.random.default_rng(seed+177)
    pos=np.ones((NCTX,NSTAGE,NACT));neg=np.ones_like(pos)
    for i,(traj,y) in enumerate(data):
        persistent = (rng.random()<noise)
        votes=[]
        for k in range(7):
            flip=persistent or rng.random()<noise/4
            votes.append(1-y if flip else y)
        q={"oracle":float(y),"hard":float(votes[0]),"majority":float(sum(votes)>=4),"soft":sum(votes)/7}[mode]
        for c,t,a in traj:
            pos[c,t,a]+=q;neg[c,t,a]+=1-q
    pos/=pos.sum(axis=2,keepdims=True)
    neg/=neg.sum(axis=2,keepdims=True)
    return np.clip(np.log(pos/neg),-4,4)

def evaluate_action_preferences(scores):
    truth=np.array([[correct_action(c,t) for t in range(NSTAGE)] for c in range(NCTX)])
    return float(np.mean(np.argmax(scores,axis=2)==truth))

def offline_policy_success(scores,seed,episodes=300):
    rng=np.random.default_rng(seed+987)
    succ=0
    for _ in range(episodes):
        ctx=int(rng.integers(NCTX));success=1
        for stage in range(NSTAGE):
            if int(np.argmax(scores[ctx,stage]))!=correct_action(ctx,stage):success=0;break
        succ+=success
    return succ/episodes

def simulate_psvm(seed):
    data=collect_data(seed)
    out=[]
    for noise in [0.0,0.15,0.35]:
        for method in ['oracle','hard','majority','soft']:
            scores=pswm_scores(data,noise,seed,method)
            out.append(dict(idea='psvm',method=method,scenario=f'correlated_noise_{noise}',seed=seed,
                            metric='offline_policy_success',value=offline_policy_success(scores,seed),
                            auxiliary=evaluate_action_preferences(scores),queries=len(data)*7))
    return out

def simulate_facr(seed):
    data=collect_data(seed,1800)
    count=np.ones((NCTX,NSTAGE,NACT))*2
    wins=np.ones_like(count)*0.02
    state_n=np.ones((NCTX,NSTAGE))*NACT*2
    state_w=np.ones((NCTX,NSTAGE))*0.02*NACT
    for traj,y in data:
        for c,t,a in traj:
            count[c,t,a]+=1;wins[c,t,a]+=y;state_n[c,t]+=1;state_w[c,t]+=y
    raw=wins/count
    prior=state_w/state_n
    cf=raw-prior[:,:,None]
    proxy=np.zeros_like(raw);proxy[:,:,0]=0.35
    methods={'proxy_only':proxy,'success_value':raw,'counterfactual_advantage':cf,
             'proxy_plus_value':proxy+raw,'proxy_plus_counterfactual':proxy+cf}
    out=[]
    for name,scores in methods.items():
        out.append(dict(idea='facr',method=name,scenario='tabular_oracle_success_value',seed=seed,
                        metric='offline_policy_success',value=offline_policy_success(scores,seed),
                        auxiliary=evaluate_action_preferences(scores),queries=len(data)))
    return out

def simulate_falsification(seed):
    quality=np.full((NCTX,NACT),0.06)
    for c in range(NCTX):quality[c,correct_action(c,0)]=0.80
    proxy=np.tile(np.linspace(0.70,0.20,NACT),(NCTX,1))
    out=[]
    for method in ['random','on_policy','uncertainty','falsification','hybrid']:
        for budget in [16,32,64,128]:
            rng=np.random.default_rng(seed+5551)
            estimates=proxy.copy();seen=np.zeros_like(estimates,dtype=int);successes=np.zeros_like(estimates)
            for k in range(budget):
                if method=='random':c=int(rng.integers(NCTX));a=int(rng.integers(NACT))
                elif method in ('on_policy','falsification'):
                    c=int(k%NCTX);a=int(np.argmax(estimates[c]))
                elif method=='uncertainty':
                    idx=np.argwhere(seen==seen.min());c,a=idx[int(rng.integers(len(idx)))];c=int(c);a=int(a)
                else:
                    exploration=0.30/np.sqrt(seen+1);score=estimates+exploration
                    c=int(k%NCTX);a=int(np.argmax(score[c]))
                y=int(rng.random()<quality[c,a])
                seen[c,a]+=1;successes[c,a]+=y
                estimates[c,a]=(proxy[c,a]+successes[c,a])/(1+seen[c,a])
            policy=np.argmax(estimates,axis=1)
            val=float(np.mean(quality[np.arange(NCTX),policy]))
            out.append(dict(idea='falsification',method=method,scenario=f'verifications_{budget}',seed=seed,
                            metric='true_policy_success_probability',value=val,
                            auxiliary=float(np.mean(seen>0)),queries=budget))
    return out

def summarize(rows):
    groups={}
    for r in rows:groups.setdefault((r['idea'],r['scenario'],r['method']),[]).append(r['value'])
    result=[]
    for (idea,scen,method),values in sorted(groups.items()):
        a=np.asarray(values)
        result.append(dict(idea=idea,scenario=scen,method=method,n=len(a),
                           mean=round(float(a.mean()),5),sd=round(float(a.std(ddof=1)),5) if len(a)>1 else 0))
    return result

def write_csv(path,rows):
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='results');ap.add_argument('--seeds',type=int,default=30)
    args=ap.parse_args();assert 2<=args.seeds<=1000
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for seed in range(args.seeds):
        rows+=simulate_psvm(seed)+simulate_facr(seed)+simulate_falsification(seed)
    sums=summarize(rows);write_csv(out/'trials.csv',rows);write_csv(out/'summary.csv',sums)
    lines=['# CPU screening decision report','','**Scope:** synthetic finite-state diagnostics, not CoRe / MetaWorld. No robot success rates measured.','','## Aggregate results','','| Hypothesis | Scenario | Method | Trials | Mean score | SD |','|---|---|---|---:|---:|---:|']
    for r in sums:lines.append(f"| {r['idea']} | {r['scenario']} | {r['method']} | {r['n']} | {r['mean']:.3f} | {r['sd']:.3f} |")
    lines+=['','## Interpretation boundaries','','- P-SVM evaluation is an offline policy extracted from visitation ratios, **not** CoRe RRM + SVM RL training.','- FACR implementation is an oracle tabular approximation, **not** a learned neural world model. The counterfactual baseline subtraction preserves the same action ranking as success-value estimation; improvement over it cannot be claimed without an alternative mechanism.','- RLF falsification and on-policy have intentionally identical action acquisition in this naive formulation; equality is expected, demonstrating the lack of a distinguishing exploration objective.','- No statistical significance is implied by scores alone. This suite cannot choose a publication-ready winner.','','## Next decisive experiments','','1. Reproduce CoRe RRM-only on MetaWorld with matched seeds.','2. Oracle SVM as FRM replacement; compare downstream RL success and gap closure.','3. FACR versus plain success-value shaping using equal oracle data and training compute.','4. RLF with genuinely distinct verifier-disagreement search against on-policy hard-negative mining.','','**Decision:** inconclusive for CoRe. Reject claims of demonstrated superiority from this synthetic screen.']
    (out/'DECISION_REPORT.md').write_text('\n'.join(lines)+'\n')
    (out/'metadata.json').write_text(json.dumps({'seeds':args.seeds,'synthetic_only':True,'methods':['psvm','facr','falsification']},indent=2))
    print(f'Wrote {len(rows)} individual results to {out}')
if __name__=='__main__':main()
