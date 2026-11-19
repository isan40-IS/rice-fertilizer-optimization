"""Read-only diagnostics for a frozen Experiment-C surrogate (no training)."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import json
import warnings
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from ..config import load_all_configs, resolve_path
from ..data.feature_engineering import add_npk_features
from ..modeling.serializer import load_model_bundle
from ..modeling.evaluator import evaluate_holdout
from ..utils import hash_file

NPK = ['N.rate.kg_ha', 'P.rate.kg_ha', 'K.rate.kg_ha']
DERIVED = ['total_npk', 'ratio_n_p', 'ratio_n_k', 'ratio_p_k', 'prop_n', 'prop_p', 'prop_k']


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def metadata_dictionary(path):
    """Read metadata without an Excel engine dependency; keep original definitions."""
    ns = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(path) as archive:
        strings = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            strings = [''.join(t.text or '' for t in node.findall('.//m:t', ns))
                       for node in ET.fromstring(archive.read('xl/sharedStrings.xml')).findall('m:si', ns)]
        rows = []
        for row in ET.fromstring(archive.read('xl/worksheets/sheet1.xml')).findall('.//m:row', ns):
            values = []
            for cell in row.findall('m:c', ns):
                node = cell.find('m:v', ns)
                value = node.text if node is not None else ''.join(t.text or '' for t in cell.findall('.//m:t', ns))
                if cell.get('t') == 's':
                    value = strings[int(value)]
                values.append(value)
            rows.append(values)
    return pd.DataFrame(rows)


@dataclass
class FrozenStudy:
    group: Path
    run: Path
    bundle: dict
    configs: dict
    data: pd.DataFrame
    split: pd.DataFrame
    candidate: dict
    source_hashes: dict

    @property
    def train(self):
        return self.data.loc[self.split.loc[self.split.split.eq('train'), 'row_index']]

    @property
    def holdout(self):
        return self.data.loc[self.split.loc[self.split.split.eq('test'), 'row_index']]


def load_frozen_study(group):
    group = Path(group).resolve()
    manifest = read_json(group / 'experiment_manifest.json')
    if manifest['status'] != 'complete':
        raise ValueError('Experiment C must be complete')
    candidate = read_json(group / 'frozen_candidate.json')['candidate']
    if not candidate:
        raise ValueError('No frozen candidate')
    run = Path(candidate['run_dir']).resolve()
    if run.parent != group or str(run) not in [str(Path(r['run_dir']).resolve()) for r in manifest['runs']]:
        raise ValueError('Candidate run is inconsistent with manifest')
    configs = load_all_configs(run / 'config_snapshot')
    provenance = read_json(run / 'training_manifest.json')
    dataset = resolve_path(configs['paths']['processed_dataset'])
    if hash_file(dataset) != manifest['dataset_sha256'] or hash_file(dataset) != provenance['dataset_sha256']:
        raise ValueError('Dataset hash mismatch')
    cv_file = run / 'splits/cv_assignments.json'
    if hash_file(cv_file) != provenance['cv_manifest_sha256']:
        raise ValueError('CV assignment hash mismatch')
    model_path = run / 'models' / f"{candidate['model']}.joblib"
    bundle = load_model_bundle(model_path)
    if bundle['provenance'] != provenance or bundle['feature_columns'] != provenance['feature_columns']:
        raise ValueError('Bundle provenance/schema mismatch')
    data = pd.read_csv(dataset)
    split = pd.read_csv(run / 'splits/split_assignments.csv')
    if (len(split) != len(data) or split.row_index.duplicated().any()
            or set(split.row_index) != set(range(len(data))) or set(split.split) != {'train', 'test'}):
        raise ValueError('Invalid split assignments')
    if data.record_id.isna().any() or data.record_id.duplicated().any():
        raise ValueError('Ambiguous record IDs')
    if 'record_id' in split and not np.array_equal(
            split.record_id.astype(str), data.loc[split.row_index, 'record_id'].astype(str)):
        raise ValueError('Split record IDs mismatch')
    required = bundle['feature_columns'] + [bundle['target_column'], bundle['group_column']]
    if set(required) - set(data):
        raise ValueError('Required schema columns missing')
    actual_names = list(bundle['pipeline'].feature_names_in_)
    if actual_names != bundle['feature_columns']:
        raise ValueError('Fitted input schema mismatch')
    train = data.loc[split.loc[split.split.eq('train'), 'row_index']]
    test = data.loc[split.loc[split.split.eq('test'), 'row_index']]
    group_col = bundle['group_column']
    if set(train[group_col]) & set(test[group_col]):
        raise ValueError('Holdout group overlap')
    cv = read_json(cv_file)
    train_ids = set(train.record_id.astype(str))
    validation_ids = []
    for fold in cv:
        a, b = set(map(str, fold['train_record_ids'])), set(map(str, fold['validation_record_ids']))
        if a & b or a | b != train_ids:
            raise ValueError('Invalid CV membership')
        groups_a = set(train.loc[train.record_id.astype(str).isin(a), group_col])
        groups_b = set(train.loc[train.record_id.astype(str).isin(b), group_col])
        if groups_a & groups_b:
            raise ValueError('CV group overlap')
        validation_ids.extend(b)
    if len(validation_ids) != len(train_ids) or set(validation_ids) != train_ids:
        raise ValueError('CV validation coverage mismatch')
    if bundle['model_name'] != candidate['model'] or bundle['tuning']['search_seed'] != candidate['search_seed']:
        raise ValueError('Frozen model identity mismatch')
    comparison = pd.read_csv(group/'study_comparison.csv')
    matching = comparison.loc[comparison.model.eq(candidate['model'])
                              & comparison.search_seed.eq(candidate['search_seed'])
                              & comparison.tuning_method.eq(candidate['tuning_method'])]
    if len(matching) != 1:
        raise ValueError('Candidate does not match comparison table')
    for metric in ['rmse_mean','mae_mean','r2_mean']:
        if not np.isclose(float(candidate[metric]),float(matching.iloc[0][metric])) or not np.isclose(
                float(candidate[metric]),float(bundle['cv_summary'][metric])):
            raise ValueError('Frozen candidate metric mismatch')
    paths = [dataset, model_path, group/'experiment_manifest.json', group/'frozen_candidate.json',
             group/'study_comparison.csv',run/'training_manifest.json', run/'splits/split_assignments.csv', cv_file]
    paths.extend(sorted((run/'config_snapshot').glob('*.yaml')))
    study = FrozenStudy(group, run, bundle, configs, data, split, candidate,
                        {str(p): hash_file(p) for p in paths})
    study.cv_assignments = cv
    return study


class ScenarioDiagnostics:
    """One shared feature-consistent path for playground, sweeps and exports."""

    def __init__(self, study):
        self.study = study
        self.bundle = study.bundle
        self.pipeline = self.bundle['pipeline']
        self.features = list(self.bundle['feature_columns'])
        self.train = study.train.copy()
        self.categorical = [c for c in self.bundle.get('categorical_features', []) if c in self.features]
        if not self.categorical:
            profile = study.configs['features']['profiles'][study.configs['decisions']['feature_profile']]
            self.categorical = [c for c in profile.get('categorical', []) if c in self.features]
        self.numeric = [c for c in self.features if c not in self.categorical]
        self.state = [c for c in self.features if c not in NPK + DERIVED]
        self.state_numeric = [c for c in self.state if c in self.numeric]
        self.pseudocount = float(self.bundle['ratio_pseudocount'])
        snapshot_pc = study.configs['decisions']['feature_engineering']['ratio_pseudocount']
        if self.pseudocount != float(snapshot_pc):
            raise ValueError('Feature formula snapshot mismatch')
        regenerated = self.recompute(self.train)
        for c in set(DERIVED) & set(self.features):
            if not np.allclose(regenerated[c], self.train[c], rtol=1e-7, atol=1e-8, equal_nan=True):
                raise ValueError(f'Derived feature mismatch: {c}')
        pre = self.pipeline.named_steps['preprocessor']
        self.effective = list(dict.fromkeys(c for _, transformer, cols in pre.transformers_
                                           if not isinstance(transformer, str) or transformer != 'drop'
                                           for c in cols))
        self.transformed_names = list(pre.get_feature_names_out())
        self.encoder_categories = {}
        for _, transformer, cols in pre.transformers_:
            encoder = getattr(transformer, 'named_steps', {}).get('encoder', transformer)
            if hasattr(encoder, 'categories_'):
                self.encoder_categories.update({c:list(values) for c, values in zip(cols, encoder.categories_)})
        self.npk_reference = self.train[NPK].to_numpy(float)
        q = self.train[NPK].quantile([.25, .75])
        iqr = (q.loc[.75] - q.loc[.25]).to_numpy()
        ranges = self.train[NPK].max().to_numpy() - self.train[NPK].min().to_numpy()
        self.scales = np.where(iqr > 0, iqr, np.where(ranges > 0, ranges, 1.0))
        self.constant_npk = [c for c, r in zip(NPK, ranges) if r == 0]

    def recompute(self, frame):
        return add_npk_features(frame, *NPK, ratio_pseudocount=self.pseudocount)

    def choose_base(self, mode='default', record_id=None, row_index=None):
        if mode not in {'record', 'default'}:
            raise ValueError('BASE_MODE must be record or default')
        if row_index is not None:
            if row_index not in self.study.data.index:
                raise ValueError('Unknown row index')
            return self.study.data.loc[row_index].copy()
        if mode == 'record' and record_id is not None:
            hits = self.study.data[self.study.data.record_id.astype(str).eq(str(record_id))]
            if len(hits) != 1:
                raise ValueError('Record ID missing or ambiguous')
            return hits.iloc[0].copy()
        # First maximize observed state dimensions, then minimize scaled median distance.
        cols = self.state_numeric
        if not cols:
            return self.train.iloc[0].copy()
        v = self.train[cols]
        scale = (v.quantile(.75) - v.quantile(.25)).replace(0, np.nan)
        scale = scale.fillna(v.max() - v.min()).replace(0, 1).fillna(1)
        score = ((v - v.median()) / scale).abs().mean(axis=1).fillna(np.inf)
        order = pd.DataFrame({'missing': v.isna().sum(axis=1), 'distance': score}).sort_values(
            ['missing', 'distance'], kind='stable').index
        return self.train.loc[order[0]].copy()

    def scenarios(self, n=8, seed=42):
        candidates = self.train.sample(frac=1, random_state=seed)
        pattern = candidates[self.state].isna().astype(int).astype(str).agg(''.join, axis=1)
        diverse = candidates.assign(_pattern=pattern).drop_duplicates(
            [self.bundle['group_column'], '_pattern']).drop(columns='_pattern')
        first = diverse.drop_duplicates(subset=self.state).head(n)
        rest = diverse.loc[~diverse.index.isin(first.index)].head(max(0, n-len(first)))
        return pd.concat([first, rest]).head(n).copy()

    def prepare(self, base, overrides):
        unknown = set(overrides) - set(self.features)
        if unknown or set(overrides) & set(DERIVED):
            raise ValueError(f'Forbidden/unknown/derived overrides: {sorted(unknown | (set(overrides) & set(DERIVED)))}')
        row = base.copy()
        for c, value in overrides.items():
            if c in self.numeric:
                if isinstance(value, (str, bool)) or (not pd.isna(value) and not np.isfinite(value)):
                    raise ValueError(f'{c}: expected finite numeric or NaN')
                if c in NPK and (pd.isna(value) or value < 0):
                    raise ValueError('NPK must be finite and nonnegative')
            elif not pd.isna(value) and not isinstance(value, str):
                raise ValueError(f'{c}: expected string category or missing')
            if c not in self.effective:
                warnings.warn(f'{c} is not used by the fitted transformer', stacklevel=2)
            row[c] = value
        if not np.isfinite(row[NPK].astype(float)).all() or (row[NPK].astype(float) < 0).any():
            raise ValueError('NPK must be finite and nonnegative')
        if str(row.get('fertilizer_type', '')).strip().lower() == 'no fertilizer' and row[NPK].astype(float).sum() > 0:
            raise ValueError('No fertilizer is a control category: positive NPK requires an explicit compatible fertilizer_type')
        return self.recompute(pd.DataFrame([row]))[self.features]

    def predict(self, frame):
        y = np.asarray(self.pipeline.predict(frame[self.features]), dtype=float)
        if not np.isfinite(y).all():
            raise ValueError('Nonfinite model predictions')
        return y

    def domain(self, frame):
        """Verified violations plus continuous distances; no invented support cutoff."""
        f = frame[self.features]
        details = [[] for _ in range(len(f))]
        quantile_counts = np.zeros(len(f), dtype=int)
        for c in self.numeric:
            ref = self.train[c].dropna()
            if ref.empty:
                continue
            outside = f[c].notna() & ((f[c] < ref.min()) | (f[c] > ref.max()))
            for i in np.flatnonzero(outside.to_numpy()):
                details[i].append(c)
            quantile_counts += (f[c].notna() & ((f[c] < ref.quantile(.05)) | (f[c] > ref.quantile(.95)))).to_numpy(int)
        unknown = [[] for _ in range(len(f))]
        for c in self.categorical:
            known = set(self.train[c].dropna())
            for i, value in enumerate(f[c]):
                if not pd.isna(value) and value not in known:
                    unknown[i].append(c)
        patterns = self.train[self.state].isna().astype(int).astype(str).agg(''.join, axis=1).value_counts()
        missing = f[self.state].isna().astype(int).astype(str).agg(''.join, axis=1)
        distance = cdist(f[NPK].to_numpy(float)/self.scales, self.npk_reference/self.scales)
        nearest = distance.argmin(axis=1)
        out = pd.DataFrame({
            'outside_training_range': [';'.join(v) for v in details],
            'outside_q05_q95_count': quantile_counts,
            'unknown_categories': [';'.join(v) for v in unknown],
            'missing_pattern_train_n': missing.map(patterns).fillna(0).to_numpy(int),
            'npk_distance': distance[np.arange(len(f)), nearest],
            'nearest_train_record_id': self.train.iloc[nearest].record_id.to_numpy(),
            'verified_violation': [bool(a or b) for a,b in zip(details, unknown)],
        })
        for j, c in enumerate(NPK):
            out[f'nearest_{c}'] = self.npk_reference[nearest, j]
        # Only compare observed state dimensions, within the SAME missingness pattern.
        state_dist, dimensions, refcounts = [], [], []
        for _, row in f.iterrows():
            cols = [c for c in self.state_numeric if pd.notna(row[c])]
            same = (self.train[self.state].isna().to_numpy() == row[self.state].isna().to_numpy()).all(axis=1)
            for c in self.categorical:
                same &= (self.train[c].eq(row[c]) | (self.train[c].isna() & pd.isna(row[c]))).to_numpy()
            ref = self.train.loc[same, cols]
            dimensions.append(len(cols)); refcounts.append(len(ref))
            if not cols or ref.empty:
                state_dist.append(np.nan); continue
            scale = self.train[cols].quantile(.75)-self.train[cols].quantile(.25)
            scale = scale.where(scale > 0, self.train[cols].max()-self.train[cols].min()).replace(0, 1).fillna(1)
            state_dist.append(float(np.sqrt((((ref-row[cols].astype(float))/scale)**2).sum(axis=1)).min()))
        out['state_distance'] = state_dist
        out['state_distance_dimensions'] = dimensions
        out['state_reference_rows'] = refcounts
        return out

    def playground(self, base, overrides):
        original = pd.DataFrame([base])[self.features]
        modified = self.prepare(base, overrides)
        y0, y1 = self.predict(original)[0], self.predict(modified)[0]
        before, after = original.iloc[0], modified.iloc[0]
        same = before.eq(after) | (before.isna() & after.isna())
        table = pd.DataFrame({'before': before, 'after': after, 'changed': ~same,
                              'missing_before': before.isna(), 'missing_after': after.isna()})
        table['delta'] = [float(after[c])-float(before[c]) if c in self.numeric else np.nan for c in self.features]
        changed = (~same).any()
        return table, {'yhat_base': y0, 'yhat_modified': y1, 'delta_y': y1-y0,
                       'delta_percent': 100*(y1-y0)/y0 if y0 > 0 else None,
                       'actual_modified': None if changed else float(base[self.bundle['target_column']]),
                       'error_modified': None if changed else float(base[self.bundle['target_column']])-y1}, self.domain(modified)

    def grid(self, column, n=21, baseline=None, include_observed=False):
        observed = self.train[column].dropna()
        if observed.empty:
            return np.array([])
        values = list(np.linspace(observed.quantile(.05), observed.quantile(.95), n))
        if baseline is not None and pd.notna(baseline):
            values.append(float(baseline))
        if (observed == 0).any():
            values.append(0.0)
        if include_observed:
            values.extend(observed.unique())
        return np.unique(values)

    def sweep(self, base, overrides_list):
        frame = pd.concat([self.prepare(base, o) for o in overrides_list], ignore_index=True)
        out = frame.copy()
        out['predicted'] = self.predict(frame)
        out['baseline_prediction'] = self.predict(pd.DataFrame([base])[self.features])[0]
        out['delta_y'] = out.predicted - out.baseline_prediction
        out['scenario_id'] = base.record_id
        return pd.concat([out, self.domain(frame)], axis=1)


def response_summary(curves, tolerance=1e-6):
    rows = []
    for (scenario, nutrient), g in curves.groupby(['scenario_id', 'varied_feature']):
        g = g.sort_values(nutrient)
        delta = np.diff(g.predicted)
        valid = g.loc[~g.verified_violation]
        rows.append(dict(scenario_id=scenario, varied_feature=nutrient,
                         baseline_prediction=g.baseline_prediction.iloc[0],
                         min_prediction=g.predicted.min(), max_prediction=g.predicted.max(),
                         amplitude=g.predicted.max()-g.predicted.min(),
                         n_points=len(g), nonflat_steps=int((np.abs(delta)>tolerance).sum()),
                         flat_fraction=float((np.abs(delta)<=tolerance).mean()),
                         up_fraction=float((delta>tolerance).mean()), down_fraction=float((delta < -tolerance).mean()),
                         maximum_at_endpoint=bool(np.isclose(g.predicted.iloc[[0,-1]],g.predicted.max(),atol=tolerance,rtol=0).any()),
                         no_verified_violation_n=len(valid),
                         no_verified_violation_amplitude=valid.predicted.max()-valid.predicted.min() if len(valid)>1 else np.nan))
    return pd.DataFrame(rows)


def numeric_statistics(frame, columns, group_column, split):
    rows = []
    for c in columns:
        for population in (['all', 'positive'] if c in NPK else ['all']):
            v = pd.to_numeric(frame[c], errors='raise')
            selected = v > 0 if population == 'positive' else v.notna()
            x = v[selected]
            row = dict(split=split, column=c, population=population, n_rows=len(frame),
                       observed_n=int(v.notna().sum()), missing_n=int(v.isna().sum()),
                       missing_pct=float(v.isna().mean()*100), selected_n=len(x),
                       group_n=frame.loc[selected, group_column].nunique(), zero_n=int(v.eq(0).sum()),
                       positive_n=int(v.gt(0).sum()), unique_n=x.nunique(),
                       min=x.min(), max=x.max(), mean=x.mean(), std=x.std(),
                       iqr=x.quantile(.75)-x.quantile(.25))
            row.update({f'q{int(q*100):02}':x.quantile(q) for q in [.01,.05,.25,.5,.75,.9,.95,.99]})
            rows.append(row)
    return pd.DataFrame(rows)


def distance_reference(diagnostics):
    rows = []
    for label, reference in [('all_rows', diagnostics.npk_reference),
                              ('distinct_combinations', np.unique(diagnostics.npk_reference, axis=0))]:
        if len(reference)<2:
            continue
        d = cdist(reference/diagnostics.scales, reference/diagnostics.scales)
        np.fill_diagonal(d, np.inf)
        nn = d.min(axis=1)
        rows.append(dict(reference=label, n=len(reference), zero_distance_n=int((nn==0).sum()),
                         q50=np.quantile(nn,.5), q90=np.quantile(nn,.9), q95=np.quantile(nn,.95),
                         max=nn.max(), scale=json.dumps(dict(zip(NPK, diagnostics.scales.tolist()))),
                         constant_dimensions=';'.join(diagnostics.constant_npk)))
    return pd.DataFrame(rows)


def observed_predictions(study):
    b=study.bundle
    metrics, out = evaluate_holdout(b['pipeline'], study.holdout[b['feature_columns']], study.holdout[b['target_column']])
    if not np.isfinite(out.predicted).all():
        raise ValueError('Nonfinite holdout predictions')
    out['absolute_error'] = out.residual.abs()
    out['squared_error'] = out.residual**2
    out['absolute_percentage_error'] = np.where(out.actual.ne(0), 100*out.absolute_error/out.actual.abs(), np.nan)
    out = study.holdout.drop(columns=[b['target_column']]).join(out)
    out.index.name = 'row_index'
    saved = study.group/'selected_holdout/predictions.csv'
    if saved.exists():
        old = pd.read_csv(saved).set_index('row_index')
        if set(old.index) != set(out.index) or not np.allclose(old.loc[out.index,'predicted'], out.predicted,rtol=1e-6,atol=1e-5):
            raise ValueError('Existing holdout predictions mismatch')
        evaluation = read_json(study.group/'selected_holdout/evaluation.json')
        if evaluation['candidate']['run_dir'] != study.candidate['run_dir']:
            raise ValueError('Existing holdout candidate mismatch')
    return metrics, out


def empirical_pairs(diagnostics, context_columns, metadata_complete=False):
    """Conservative audit. Missing context never establishes a controlled match.

    Candidate pairs are descriptive only until design/context is confirmed. Repeated
    doses are reported and excluded rather than silently aggregated as replicates.
    """
    data = diagnostics.study.holdout
    group = diagnostics.bundle['group_column']
    target = diagnostics.bundle['target_column']
    predicted = pd.Series(diagnostics.predict(data), index=data.index)
    context = list(dict.fromkeys(diagnostics.state + list(context_columns)))
    missing_columns = set(context)-set(data)
    if missing_columns:
        raise ValueError(f'Unknown matching metadata: {missing_columns}')
    rows = []
    for name,g in data.groupby(group):
        repeated = g.duplicated(NPK, keep=False)
        for ia,ib in combinations(g.index,2):
            a,b = g.loc[ia],g.loc[ib]
            changes = sum(not np.isclose(float(a[c]),float(b[c]),rtol=0,atol=1e-8) for c in NPK)
            if not changes:
                continue
            missing = any(pd.isna(a[c]) or pd.isna(b[c]) for c in context)
            mismatch = [c for c in context if pd.notna(a[c]) and pd.notna(b[c]) and a[c]!=b[c]]
            reason = ('different_context' if mismatch else 'missing_context' if missing else
                      'repeated_dose_requires_design' if repeated.loc[ia] or repeated.loc[ib] else
                      'metadata_design_unconfirmed' if not metadata_complete else '')
            da,dp = float(b[target]-a[target]),float(predicted[ib]-predicted[ia])
            sa,sp = np.sign(da) if abs(da)>1e-6 else 0, np.sign(dp) if abs(dp)>1e-6 else 0
            rows.append(dict(group=name,record_a=a.record_id,record_b=b.record_id,
                             nutrient_changes=changes,pair_type='one_nutrient' if changes==1 else 'multi_nutrient',
                             delta_actual=da,delta_predicted=dp,delta_error=dp-da,
                             actual_tie=sa==0,predicted_tie=sp==0,
                             sign_agreement=sa==sp if sa!=0 and sp!=0 else np.nan,
                             eligible=not reason,exclusion_reason=reason,
                             treatment_a=json.dumps(a[NPK].to_dict()),treatment_b=json.dumps(b[NPK].to_dict())))
    cols=['group','record_a','record_b','nutrient_changes','pair_type','delta_actual','delta_predicted',
          'delta_error','actual_tie','predicted_tie','sign_agreement','eligible','exclusion_reason','treatment_a','treatment_b']
    pairs=pd.DataFrame(rows, columns=cols)
    summary=[]
    for label,sub in [('all_candidate_pairs',pairs),('eligible_pairs',pairs.loc[pairs.eligible.astype(bool)])]:
        for kind in ['one_nutrient','multi_nutrient']:
            p=sub.loc[sub.pair_type.eq(kind)]
            summary.append(dict(population=label,pair_type=kind,n_pairs=len(p),n_groups=p.group.nunique(),
                                n_treatments=len(set(p.record_a)|set(p.record_b)),
                                actual_ties=int(p.actual_tie.sum()),predicted_ties=int(p.predicted_tie.sum()),
                                sign_agreement=p.sign_agreement.mean() if len(p) else np.nan,
                                delta_mae=p.delta_error.abs().mean(),delta_rmse=np.sqrt((p.delta_error**2).mean()),
                                status='descriptive_only' if label=='all_candidate_pairs' else ('available' if len(p) else 'insufficient_evidence')))
    return pairs,pd.DataFrame(summary)


def manual_parameter_evidence(train, data_bounds, literature_bounds, parameters):
    """Validate explicitly supplied choices and report coverage, never choose limits."""
    from ..optimization.penalties import total_dose_penalty, extreme_composition_penalty
    for name in ['total_soft','total_max','yield_scale','lambda_total','lambda_composition']:
        value=parameters.get(name)
        if value is not None and (not np.isfinite(value) or value<0 or (name=='yield_scale' and value==0)):
            raise ValueError(f'Invalid {name}')
    soft,maximum=parameters.get('total_soft'),parameters.get('total_max')
    if soft is not None and maximum is not None and soft>=maximum:
        raise ValueError('total_soft must be below total_max')
    rows=[]
    for symbol,column in zip(['N','P','K'],NPK):
        data,lit=data_bounds[symbol],literature_bounds[symbol]
        if any(v is None for v in [data['min'],data['max'],lit['min'],lit['max']]):
            continue
        numbers=np.asarray([data['min'],data['max'],lit['min'],lit['max']],dtype=float)
        if not np.isfinite(numbers).all() or (numbers<0).any() or numbers[0]>numbers[1] or numbers[2]>numbers[3]:
            raise ValueError('Invalid data/literature bounds')
        if lit.get('unit')!='kg/ha_reported_nutrient' or not lit.get('reference'):
            raise ValueError('Confirm nutrient basis, unit and literature reference')
        lo,hi=max(numbers[0],numbers[2]),min(numbers[1],numbers[3])
        if lo>hi:
            raise ValueError('Empty bounds intersection')
        rows.append(dict(nutrient=symbol,column=column,min=lo,max=hi))
    bounds=pd.DataFrame(rows,columns=['nutrient','column','min','max'])
    total=train[NPK].sum(axis=1)
    evidence=pd.DataFrame({'record_id':train.record_id,'total_npk':total})
    if len(bounds)==3:
        if maximum is not None and bounds['min'].sum()>maximum:
            raise ValueError('Total max incompatible with nutrient lower bounds')
        inside=np.ones(len(train),dtype=bool)
        for row in rows:
            inside &= train[row['column']].between(row['min'],row['max']).to_numpy()
        evidence['inside_chosen_bounds']=inside
    if maximum is not None:
        evidence['within_chosen_total_max']=total.le(maximum)
    if soft is not None and maximum is not None:
        evidence['total_penalty']=[total_dose_penalty(x,soft,maximum) for x in train[NPK].to_numpy(float)]
    props=parameters.get('proportion_bounds')
    if props is not None:
        lower=np.array([props[c]['min'] for c in ['N','P','K']],float)
        upper=np.array([props[c]['max'] for c in ['N','P','K']],float)
        if (not np.isfinite(lower).all() or not np.isfinite(upper).all() or (lower<0).any()
                or (upper>1).any() or (lower>upper).any()):
            raise ValueError('Invalid soft composition bounds')
        evidence['composition_penalty']=[extreme_composition_penalty(x,lower,upper) for x in train[NPK].to_numpy(float)]
        evidence['positive_total_for_composition']=total.gt(0)
        # Not a hard exclusion: report whether zero penalty is mathematically possible.
        evidence['soft_zero_violation_possible']=bool(lower.sum()<=1<=upper.sum())
    return bounds,evidence
