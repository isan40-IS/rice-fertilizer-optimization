from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder

from rice_fertilizer.data.feature_engineering import add_npk_features
from rice_fertilizer.evaluation.surrogate_diagnostics import (
    NPK, DERIVED, ScenarioDiagnostics, empirical_pairs, load_frozen_study,
    manual_parameter_evidence,
)


@pytest.fixture
def diagnostics(monkeypatch):
    data = pd.DataFrame({NPK[0]:[0,20,40,60,0,20,40,60],NPK[1]:[5]*8,NPK[2]:[10]*8,
                         'ph':[6,6,np.nan,6,7,7,7,7], 'fertilizer_type':['Inorg']*8,
                         'record_id':range(8),'group':['a']*4+['b']*4,'yield':np.arange(8)*100.})
    data=add_npk_features(data,ratio_pseudocount=1e-5)
    features=['ph','fertilizer_type']+NPK+DERIVED
    numeric=[c for c in features if c!='fertilizer_type']
    pipeline=Pipeline([('preprocessor',ColumnTransformer([
        ('numeric','passthrough',numeric),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['fertilizer_type'])])),
        ('model',DecisionTreeRegressor(random_state=42))])
    pipeline.fit(data[features],data['yield'])
    def forbid(*args,**kwargs):raise AssertionError('Diagnostics must not fit')
    monkeypatch.setattr(pipeline,'fit',forbid)
    bundle={'pipeline':pipeline,'feature_columns':features,'categorical_features':['fertilizer_type'],
            'ratio_pseudocount':1e-5,'target_column':'yield','group_column':'group'}
    study=SimpleNamespace(bundle=bundle,train=data,holdout=data,data=data,
                          configs={'decisions':{'feature_engineering':{'ratio_pseudocount':1e-5}}})
    return ScenarioDiagnostics(study)


def test_scenario_noop_recomputes_and_preserves_state(diagnostics):
    d=diagnostics;b=d.choose_base();original=b.copy()
    assert np.array_equal(d.predict(pd.DataFrame([b])),d.predict(d.prepare(b,{})))
    f=d.prepare(b,{NPK[0]:90})
    assert f.iloc[0]['total_npk']==105
    assert f.iloc[0]['ratio_n_p']==pytest.approx(90/(5+1e-5))
    assert f.iloc[0]['ph']==b.ph
    pd.testing.assert_series_equal(b,original)
    changes,result,domain=d.playground(b,{NPK[0]:90})
    assert result['actual_modified'] is None and result['error_modified'] is None
    assert domain.verified_violation.iloc[0]


@pytest.mark.parametrize('overrides',[{'yield':1},{'record_id':4},{'total_npk':30},{'unknown':2},
                                    {NPK[0]:-1},{NPK[1]:np.nan},{NPK[2]:np.inf},{'ph':'6'},
                                    {'fertilizer_type':'No fertilizer'}])
def test_invalid_overrides(diagnostics,overrides):
    with pytest.raises(ValueError):diagnostics.prepare(diagnostics.choose_base(),overrides)


def test_missing_zero_categories_and_selection(diagnostics):
    d=diagnostics;b=d.choose_base()
    masked=d.prepare(b,{'ph':np.nan})
    assert np.isfinite(d.predict(masked)).all()
    zero=d.prepare(b,{c:0. for c in NPK})
    assert (zero[['prop_n','prop_p','prop_k']]==0).all().all()
    assert np.isfinite(d.predict(zero)).all()
    unknown=d.prepare(b,{'fertilizer_type':'NEW'})
    assert d.domain(unknown).unknown_categories.iloc[0]=='fertilizer_type'
    assert d.scenarios(4,42).equals(d.scenarios(4,42))
    with pytest.raises(ValueError):d.choose_base('record','bad')
    d.study.data=pd.concat([d.study.data,d.study.data.iloc[[0]]])
    with pytest.raises(ValueError):d.choose_base('record',0)


def test_pairing_does_not_assume_comparability(diagnostics):
    pairs,summary=empirical_pairs(diagnostics,[],metadata_complete=False)
    assert len(pairs)>0 and not pairs.eligible.any()
    assert 'insufficient_evidence' in set(summary.status)
    pairs,_=empirical_pairs(diagnostics,[],metadata_complete=True)
    assert pairs.eligible.any()
    assert not pairs.loc[pairs.exclusion_reason.eq('missing_context'),'eligible'].any()


def test_corrupted_derived_features_rejected(diagnostics):
    study=diagnostics.study
    study.train.loc[0,'ratio_n_p']=999
    with pytest.raises(ValueError,match='Derived feature mismatch'):ScenarioDiagnostics(study)


def test_manual_bounds_are_explicit_and_soft_composition_not_hard(diagnostics):
    data={c:{'min':0,'max':100} for c in ['N','P','K']}
    literature={c:{'min':0,'max':80,'unit':'kg/ha_reported_nutrient','reference':'test fixture'} for c in data}
    params={'total_soft':50,'total_max':100,'proportion_bounds':{c:{'min':0.2,'max':0.5} for c in data}}
    bounds,evidence=manual_parameter_evidence(diagnostics.train,data,literature,params)
    assert len(bounds)==3 and len(evidence)==len(diagnostics.train)
    assert 'composition_penalty' in evidence and 'within_composition' not in evidence
    params['total_max']=10
    params['total_soft']=20
    with pytest.raises(ValueError):manual_parameter_evidence(diagnostics.train,data,literature,params)


def test_loader_rejects_hash_before_deserializing(tmp_path,monkeypatch):
    import json
    group=tmp_path/'C';run=group/'optuna_seed_11'
    (run/'config_snapshot').mkdir(parents=True)
    (group/'experiment_manifest.json').write_text(json.dumps({'status':'complete','dataset_sha256':'expected','runs':[{'run_dir':str(run)}]}))
    (group/'frozen_candidate.json').write_text(json.dumps({'candidate':{'run_dir':str(run),'model':'xgboost'}}))
    (run/'training_manifest.json').write_text(json.dumps({'dataset_sha256':'expected'}))
    from rice_fertilizer.evaluation import surrogate_diagnostics as module
    monkeypatch.setattr(module,'load_all_configs',lambda _: {'paths':{'processed_dataset':'x.csv'}})
    monkeypatch.setattr(module,'hash_file',lambda _:'wrong')
    with pytest.raises(ValueError,match='hash mismatch'):load_frozen_study(group)
