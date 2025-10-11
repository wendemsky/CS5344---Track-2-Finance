# =============================================================
# Script 1/3: meta_learner_stack.py
# (tiny meta-learner over simple detectors; CV on valid-train; reports AUPRC/AUROC)
# =============================================================
#!/usr/bin/env python3
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from typing import List
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from features_amortization import FeatureBuilderAmort

rng = np.random.default_rng(42)

# --- helpers ---
def rank01(x):
    s=pd.Series(x); return ((s.rank(method='average')-1)/(len(s)-1+1e-12)).to_numpy()

# base detectors trained on train normals only
def make_global_lof_models(Xtr, ks:List[int]):
    models=[]
    for k in ks:
        m=LocalOutlierFactor(n_neighbors=k, novelty=True, contamination='auto')
        m.fit(Xtr)
        models.append((f'lof_k{k}', m))
    return models

def make_clusterwise_lof(Xtr, Xvtr, yvtr):
    best_km=None; best_auc=-1; best_k=8
    for k in [8,10,12,14]:
        km=KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xtr)
        labs=km.predict(Xvtr)
        d=np.linalg.norm(Xvtr - km.cluster_centers_[labs], axis=1)
        auc=roc_auc_score(yvtr, d)
        if auc>best_auc: best_auc=auc; best_km=km; best_k=k
    km=best_km
    cl_models={}
    labs_tr=km.labels_
    labs_v =km.predict(Xvtr)
    for c in range(best_k):
        idx=(labs_tr==c)
        if idx.sum()<30: continue
        Xc=Xtr[idx]
        best=None; best_ap=-1
        for k in [5,6,7,8,9]:
            m=LocalOutlierFactor(n_neighbors=k, novelty=True, contamination='auto').fit(Xc)
            sel=(labs_v==c)
            if sel.sum()==0: continue
            s=-m.score_samples(Xvtr[sel])
            ap=average_precision_score(yvtr[sel], s)
            if ap>best_ap:
                best_ap=ap; best=m
        if best is not None:
            cl_models[c]=best
    # gate on valid-train
    s_cl=np.zeros(len(Xvtr))
    for i,x in enumerate(Xvtr):
        c=km.predict(x[None,:])[0]
        if c in cl_models:
            s_cl[i]=-cl_models[c].score_samples(x[None,:])[0]
    ap = average_precision_score(yvtr, s_cl)
    return km, cl_models, s_cl, ap


def main():
    # data + features
    fb=FeatureBuilderAmort()
    train=pd.read_csv('Data/loans_train.csv')
    valid=pd.read_csv('Data/loans_valid.csv')
    fb.fit(train)
    Xtr=fb.transform(train)

    y=valid['target'].to_numpy()
    sss=StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    for tri,hoi in sss.split(valid,y):
        vtr=valid.iloc[tri]; vho=valid.iloc[hoi]
    Xvtr=fb.transform(vtr); Xvho=fb.transform(vho)
    yvtr=vtr['target'].to_numpy(); yvho=vho['target'].to_numpy()

    # --- base detectors ---
    ks=[5,6,7,8]
    lof_models=make_global_lof_models(Xtr, ks)
    S_vtr=[]; names=[]
    for name, m in lof_models:
        S_vtr.append(-m.score_samples(Xvtr)); names.append(name)
    # cluster-LOF
    km, cl_models, s_cl, ap_cl = make_clusterwise_lof(Xtr, Xvtr, yvtr)
    # include cluster score only if it beats weakest LOF-k
    weak_ap = min([average_precision_score(yvtr, s) for s in S_vtr]) if len(S_vtr)>0 else -1
    if ap_cl >= weak_ap:
        S_vtr.append(s_cl); names.append('cluster_lof')
        use_cluster=True
    else:
        use_cluster=False
    # amortization scalars (already in features; add a small projection: top 6 most variant dims)
    # simple: take last 6 columns of fb.transform that correspond to amort signals (as appended at end)
    Xvtr_full=fb.transform(vtr)
    Xvho_full=fb.transform(vho)
    # select last 6 columns as a proxy for amort signals
    if Xvtr_full.shape[1]>=6:
        amort_vtr = Xvtr_full[:,-6:]
        amort_h   = Xvho_full[:,-6:]
        for j in range(amort_vtr.shape[1]):
            S_vtr.append(amort_vtr[:,j]); names.append(f'amort_{j}')
    else:
        amort_h=None

    S_vtr=np.column_stack(S_vtr)

    # --- CV on valid-train to pick meta-learner ---
    skf=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_name=None; best_cv=-1; best_model=None

    # candidate 1: Logistic (with simple scaling per feature rank)
    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler()
    S_scaled=scaler.fit_transform(S_vtr)
    lr=LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0, solver='liblinear')
    cv_scores=[]
    for tr, te in skf.split(S_scaled, yvtr):
        lr.fit(S_scaled[tr], yvtr[tr])
        p=lr.predict_proba(S_scaled[te])[:,1]
        cv_scores.append(average_precision_score(yvtr[te], p))
    cv_lr=np.mean(cv_scores)

    # candidate 2: shallow RF
    rf=RandomForestClassifier(n_estimators=200, max_depth=3, min_samples_leaf=20, class_weight='balanced_subsample', random_state=42)
    cv_scores=[]
    for tr, te in skf.split(S_vtr, yvtr):
        rf.fit(S_vtr[tr], yvtr[tr])
        p=rf.predict_proba(S_vtr[te])[:,1]
        cv_scores.append(average_precision_score(yvtr[te], p))
    cv_rf=np.mean(cv_scores)

    if cv_lr>=cv_rf:
        best_name='logistic'; best_cv=cv_lr; best_model=('lr', scaler, lr)
    else:
        best_name='rf'; best_cv=cv_rf; best_model=('rf', None, rf)

    print(f"[Meta] CV(valid-train) best={best_name}  AUPRC={best_cv:.4f}  features={names}")

    # fit best on all valid-train, evaluate holdout
    if best_model[0]=='lr':
        _, sc, model = best_model
        model.fit(S_scaled, yvtr)
        # build holdout stack
        S_h=[]
        for name, m in lof_models:
            S_h.append(-m.score_samples(Xvho))
        if use_cluster:
            s=np.zeros(len(Xvho))
            labs=km.predict(Xvho)
            for i,(x,c) in enumerate(zip(Xvho,labs)):
                if c in cl_models:
                    s[i]=-cl_models[c].score_samples(x[None,:])[0]
            S_h.append(s)
        if amort_h is not None:
            for j in range(amort_h.shape[1]):
                S_h.append(amort_h[:,j])
        S_h=np.column_stack(S_h)
        S_hs=sc.transform(S_h)
        ph=model.predict_proba(S_hs)[:,1]
    else:
        _, _, model = best_model
        model.fit(S_vtr, yvtr)
        S_h=[]
        for name, m in lof_models:
            S_h.append(-m.score_samples(Xvho))
        if use_cluster:
            s=np.zeros(len(Xvho))
            labs=km.predict(Xvho)
            for i,(x,c) in enumerate(zip(Xvho,labs)):
                if c in cl_models:
                    s[i]=-cl_models[c].score_samples(x[None,:])[0]
            S_h.append(s)
        if amort_h is not None:
            for j in range(amort_h.shape[1]):
                S_h.append(amort_h[:,j])
        S_h=np.column_stack(S_h)
        ph=model.predict_proba(S_h)[:,1]

    ap_h=average_precision_score(yvho, ph); auc_h=roc_auc_score(yvho, ph)
    print(f"[Meta Stack] HOLDOUT  AUPRC={ap_h:.4f} AUROC={auc_h:.4f}")

    # test submission
    try:
        test=pd.read_csv('Data/loans_test.csv')
        Xt=fb.transform(test)
        S_t=[]
        for name, m in lof_models:
            S_t.append(-m.score_samples(Xt))
        if use_cluster:
            st=np.zeros(len(Xt)); labs=km.predict(Xt)
            for i,(x,c) in enumerate(zip(Xt,labs)):
                if c in cl_models:
                    st[i]=-cl_models[c].score_samples(x[None,:])[0]
            S_t.append(st)
        if amort_h is not None:
            Xt_full=fb.transform(test)
            for j in range(min(6, Xt_full.shape[1])):
                S_t.append(Xt_full[:,-6+j])
        S_t=np.column_stack(S_t)
        if best_model[0]=='lr':
            S_t = best_model[1].transform(S_t)
        pt = best_model[2].predict_proba(S_t)[:,1]
        idc='Id' if 'Id' in test.columns else 'index'
        pd.DataFrame({idc:test[idc], 'anomaly_score':pt}).to_csv('SUB_meta_stack.csv', index=False)
        print('Saved SUB_meta_stack.csv')
    except Exception as e:
        print('Test skipped:', e)

if __name__=='__main__':
    main()
