"""Read frozen V2 artifacts and publish a standalone research report; no model loading."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
from datetime import datetime, timezone

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CV = ROOT / 'artifacts/cross_validation/4bc978e2773687d3'
PRE = ROOT / 'artifacts/preprocessing/e22890aeb6ffddf6'
FEAT = ROOT / 'artifacts/features/1225e8a76a8ce014'
RESEARCH = ROOT / 'artifacts/research_evaluation/4ad7e153513e7d64'
DIAG = ROOT / 'artifacts/flight66_diagnosis'
FAMILIES = ['increased_resistance', 'progressive', 'sudden_drop']
DEADLINES = [1, 2, 5, 10, 30]
COLORS = ['#087e8b', '#d57930', '#7253a0']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dump(path, data):
    def plain(o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(type(o).__name__)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=plain, allow_nan=False), encoding='utf-8')


def fmt(value, places=6):
    if value is None or pd.isna(value):
        return '—'
    if isinstance(value, (bool, np.bool_)):
        return 'Ya' if value else 'Tidak'
    if isinstance(value, (float, np.floating)):
        return f'{value:,.{places}f}'.replace(',', '_').replace('.', ',').replace('_', '.')
    if isinstance(value, (int, np.integer)):
        return f'{value:,}'.replace(',', '.')
    return str(value).replace('|', '/')


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |'] +
                     ['| ' + ' | '.join(fmt(v) for v in row) + ' |' for row in rows])


def ratio(a, b):
    return a / b if b else None


def summarize(g):
    delay = g.loc[g.detected, 'delay_seconds'].dropna()
    d = dict(events=len(g), detected=int(g.detected.sum()), preexisting=int(g.preexisting.sum()),
             missed=int((~g.detected & ~g.preexisting).sum()), incident_recall=float(g.detected.mean()),
             delay_n=len(delay), delay_mean_seconds=float(delay.mean()) if len(delay) else None,
             delay_median_seconds=float(delay.median()) if len(delay) else None)
    for k in ['tp', 'fp', 'tn', 'fn']:
        d[k] = int(g[k].sum())
    d.update(precision=ratio(d['tp'], d['tp']+d['fp']), recall=ratio(d['tp'], d['tp']+d['fn']),
             f1=ratio(2*d['tp'], 2*d['tp']+d['fp']+d['fn']))
    for t in DEADLINES:
        n = int((g.detected & (g.delay_seconds <= t)).sum())
        d[f'deadline_{t}_successes'] = n
        d[f'deadline_{t}_rate'] = n / len(g)
    return d


def inline(s):
    s = html.escape(s, quote=False)
    s = re.sub(r'`([^`]+)`', r'<code>\1</code>', s)
    s = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', s)
    s = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', s)
    return s


def render(md, css):
    """Render the deliberately small Markdown subset generated below."""
    lines = md.splitlines()
    blocks, toc, i = [], [], 0
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        if s.startswith('```'):
            code = []
            i += 1
            while i < len(lines) and not lines[i].startswith('```'):
                code.append(lines[i]); i += 1
            blocks.append('<pre><code>' + html.escape('\n'.join(code)) + '</code></pre>')
        elif s.startswith('#'):
            n = len(s) - len(s.lstrip('#'))
            title = s[n:].strip()
            ident = f'section-{len(toc)+1}' if n == 2 else f'heading-{i}'
            if n == 2:
                toc.append(f'<a href="#{ident}">{html.escape(title)}</a>')
            blocks.append(f'<h{n} id="{ident}">{inline(title)}</h{n}>')
        elif s.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                cells = lines[i].strip().strip('|').split('|')
                if not all(re.fullmatch(r'[:\-\s]+', c) for c in cells):
                    rows.append([inline(c.strip()) for c in cells])
                i += 1
            blocks.append('<div class="table-wrap"><table><thead><tr>' + ''.join('<th scope="col">'+v+'</th>' for v in rows[0]) +
                          '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+v+'</td>' for v in row)+'</tr>' for row in rows[1:]) + '</tbody></table></div>')
            continue
        elif s.startswith('!['):
            match = re.fullmatch(r'!\[(.*?)\]\((.*?)\)', s)
            if not match:
                raise ValueError(s)
            alt, src = match.groups()
            blocks.append(f'<figure><img loading="lazy" src="{html.escape(src, quote=True)}" alt="{html.escape(alt, quote=True)}"><figcaption>{html.escape(alt)}</figcaption></figure>')
        elif s.startswith('- '):
            items = []
            while i < len(lines) and lines[i].strip().startswith('- '):
                items.append('<li>'+inline(lines[i].strip()[2:])+'</li>'); i += 1
            blocks.append('<ul>'+''.join(items)+'</ul>')
            continue
        else:
            p = [s]; i += 1
            while i < len(lines) and lines[i].strip() and not re.match(r'^(#|\||!\[|```|- )', lines[i].strip()):
                p.append(lines[i].strip()); i += 1
            blocks.append('<p>'+inline(' '.join(p))+'</p>')
            continue
        i += 1
    return '<!doctype html><html lang="id"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HGB V2 — Laporan penelitian baterai UAV</title><style>'+css+'</style></head><body><main><div class="eyebrow">HGB V2 · LAPORAN PENELITIAN · 26 OKTOBER 2026</div><nav aria-label="Daftar isi">'+''.join(toc)+'</nav>'+''.join(blocks)+'</main></body></html>'


def build(out, template_dir):
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Output must be new/empty: {out}')
    out.mkdir(parents=True, exist_ok=True)
    (out/'figures').mkdir(exist_ok=True)
    (out/'tables').mkdir(exist_ok=True)
    used = set()
    def read(p):
        used.add(Path(p)); return load(p)
    def csv(p):
        used.add(Path(p)); return pd.read_csv(p)
    def export(name, frame):
        frame.to_csv(out/'tables'/f'{name}.csv', index=False)
    def link(p):
        return os.path.relpath(p, out).replace('\\', '/')
    checks = []
    before = {}
    # Check entire frozen primary run and upstream outputs, without unpickling estimators.
    for directory in [CV, PRE, FEAT]:
        mpath = directory/'manifest.json'
        m = read(mpath)
        before[mpath] = sha(mpath)
        for rel, digest in m['outputs'].items():
            p = directory / rel
            if sha(p) != digest:
                raise ValueError(f'Frozen output checksum mismatch: {p}')
            before[p] = digest
        checks.append(dict(check='source_manifest', source=str(directory.relative_to(ROOT)), files=len(m['outputs']), status='PASS'))
    manifest = read(CV/'manifest.json')
    protocol = manifest['identity']['protocol']
    results = read(CV/'results.json')
    pooled = results['pooled']
    research = read(RESEARCH/'results.json')
    prep = read(PRE/'summary.json')
    feature = read(FEAT/'summary.json')
    historic = read(ROOT/'artifacts/sprint13/verification.json')
    diag = read(DIAG/'results.json')
    daudit = read(DIAG/'audit_verification.json')
    inv = csv(PRE/'flight_inventory.csv')
    coverage = csv(FEAT/'reference_coverage.csv')
    segments = csv(PRE/'segments.csv')
    health = csv(CV/'healthy_active_per_fold.csv')
    reg = csv(CV/'regression_per_fold.csv')
    calibration = csv(CV/'validation_calibration_per_fold.csv')
    scenarios = csv(CV/'fault_per_scenario.csv')
    assignments = csv(CV/'fold_assignments.csv')
    for name, frame in [('fold_assignments', assignments), ('healthy_metrics', health), ('regression_metrics', reg),
                        ('calibration', calibration), ('scenario_metrics', scenarios), ('reference_coverage', coverage), ('flight_inventory', inv)]:
        export(name, frame)
    healthy_frames, extra = {}, []
    for _, h in health.iterrows():
        df = csv(CV/'folds'/h.fold_id/'healthy_test_scores.csv.gz')
        healthy_frames[h.test_flight] = df
        r = df.loc[df.reference_ready]
        assert len(r) == h.ready_samples
        assert int(r.anomaly_confirmed.sum()) == h.confirmed_alarm_samples
        extra.append(dict(test_flight=h.test_flight, residual_bias_v=float(r.residual_signed_v.mean()),
                          residual_median_v=float(r.residual_signed_v.median()),
                          residual_negative_fraction=float((r.residual_signed_v < 0).mean())))
    merged = health.merge(reg, on=['fold_id', 'test_flight', 'unseen_by_model', 'used_for_threshold']).merge(pd.DataFrame(extra))
    export('healthy_combined', merged)
    # Reaggregate sufficient statistics independently, including global target mean for pooled R2.
    n = int(reg.samples.sum()); sse = float(reg.sum_squared_error_v2.sum())
    calculated_reg = dict(samples=n, mae_v=float(reg.sum_absolute_error_v.sum()/n), rmse_v=math.sqrt(sse/n),
                          r2=1-sse/(reg.sum_squared_target_v2.sum()-reg.sum_target_v.sum()**2/n))
    for k, v in calculated_reg.items():
        assert math.isclose(v, pooled['regression'][k], abs_tol=1e-10), k
    records = []
    selections = []
    for _, s in scenarios.iterrows():
        df = csv(CV/s.scores_csv)
        ready = df.reference_ready.to_numpy(bool)
        support = df.fault_support.to_numpy(bool)
        t = df.TimeUS.to_numpy(np.int64)
        segment = df.segment_id.to_numpy()
        connected = np.r_[False, (np.diff(t) <= 500000) & (np.diff(segment) == 0) & ready[1:] & ready[:-1]]
        healthy = healthy_frames[s.NamaFile]
        # Same X means predictions are exactly reused; effect changes signed residual by +added sag.
        np.testing.assert_allclose(df.loc[ready, 'predicted_voltage_sag'], healthy.loc[ready, 'predicted_voltage_sag'], atol=0, rtol=0)
        np.testing.assert_allclose(df.loc[ready, 'residual_signed_v'], healthy.loc[ready, 'residual_signed_v'] + df.loc[ready, 'added_sag_v'], atol=1e-12, rtol=0)
        for series, col in [('raw', 'anomaly_raw'), ('confirmed', 'anomaly_confirmed')]:
            mask = df[col].to_numpy(bool) & ready
            starts = mask & ~(np.r_[False, mask[:-1]] & connected)
            onset_index = int(np.flatnonzero(support)[0])
            carried = bool(mask[onset_index] and not starts[onset_index])
            fresh = np.flatnonzero(starts & support)
            detected = bool(len(fresh) and not carried)
            delay = float((t[fresh[0]]-s.onset_time_us)/1e6) if detected else None
            rec = dict(series=series, scenario_id=s.scenario_id, fold_id=s.fold_id, flight=s.NamaFile,
                       family=s.fault_family, severity=s.severity, start_position=float(s.start_position),
                       observability=s.observability, detected=detected, preexisting=carried, delay_seconds=delay,
                       tp=int((mask & support & ready).sum()), fp=int((mask & ~support & ready).sum()),
                       tn=int((~mask & ~support & ready).sum()), fn=int((~mask & support & ready).sum()),
                       max_added_sag_v=float(s.max_added_sag_v), mean_added_sag_v=float(s.mean_added_sag_v),
                       effect_threshold_ratio=float(s.max_added_sag_v/s.threshold_v),
                       support_seconds=float(s.fault_duration_seconds), source_scores=s.scores_csv)
            if series == 'confirmed':
                assert detected == bool(s.incident_detected)
                assert carried == bool(s.preexisting_alarm)
                for k in ['tp', 'fp', 'tn', 'fn']:
                    assert rec[k] == int(s[k]), (s.scenario_id, k)
                if detected:
                    assert abs(delay-s.delay_seconds) < 1e-10
            records.append(rec)
    events = pd.DataFrame(records)
    export('event_metrics', events)
    totals = {series: summarize(g) for series, g in events.groupby('series')}
    for k in ['tp', 'fp', 'tn', 'fn']:
        assert totals['confirmed'][k] == pooled['synthetic'][k]
    assert totals['confirmed']['detected'] == pooled['synthetic']['detected_scenarios']
    assert sum(totals['confirmed'][k] for k in ['tp', 'fp', 'tn', 'fn']) == n*27
    assert int(health.confirmed_alarm_samples.sum()) == pooled['healthy_active']['confirmed_alarm_samples']
    checks.append(dict(check='granular_reaggregation_and_incident_replay', scenarios=len(scenarios), series_events=len(events), status='PASS'))
    grouped_rows = []
    for cols in [['series','family'],['series','severity'],['series','family','severity'],['series','fold_id'],['series','start_position'],['series','observability']]:
        for keys, g in events.groupby(cols, sort=True):
            keys = keys if isinstance(keys, tuple) else (keys,)
            grouped_rows.append(dict(grouping='/'.join(cols[1:]), group='/'.join(str(v) for v in keys[1:]), series=keys[0], **summarize(g)))
    grouped = pd.DataFrame(grouped_rows); export('grouped_results', grouped)
    macro = []
    for group, metrics in results['macro_per_fold'].items():
        for metric, values in metrics.items():
            macro.append(dict(scope=group, metric=metric, **values))
    export('macro_results', pd.DataFrame(macro))
    export('pooled_results', pd.DataFrame([dict(series=k, **v) for k,v in totals.items()]))
    compare = csv(RESEARCH/'model_comparison_overview.csv'); export('model_comparison', compare)
    ablation = csv(RESEARCH/'ablation_overview.csv'); export('ablation', ablation)
    low = csv(ROOT/'artifacts/low_load/56b20c4b7936b122/per_model_summary.csv'); export('low_load', low)
    eps = csv(DIAG/'false_alarm_episodes.csv'); export('false_alarm_episodes', eps)
    for name in ['condition_strata', 'feature_distribution', 'segments', 'currtot_current_cross_strata']:
        export('diagnosis_'+name, csv(DIAG/f'{name}.csv'))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.titleweight':'bold','figure.facecolor':'white','axes.facecolor':'#fbfcfd','axes.grid':True,'grid.alpha':.18})
    def save(fig, name):
        fig.savefig(out/'figures'/f'{name}.png', dpi=150, bbox_inches='tight')
        fig.savefig(out/'figures'/f'{name}.svg', bbox_inches='tight')
        plt.close(fig)
    def continuous(ax, df, col, color, label, **kwargs):
        first = True
        for _, block in df.groupby('segment_id', sort=False):
            ax.plot(block.elapsed_time, block[col], color=color, label=label if first else None, **kwargs)
            first = False
    labels = [x[6:8] for x in merged.test_flight]
    x = np.arange(6)
    fig, ax = plt.subplots(2,2,figsize=(12,7),layout='constrained')
    ax[0,0].bar(x-.18,100*merged.raw_fpr,.36,label='Raw',color='#a5cbd1');ax[0,0].bar(x+.18,100*merged.confirmed_fpr,.36,label='Confirmed',color=COLORS[0])
    ax[0,0].set(ylabel='Healthy sample FPR (%)',title='False alarm berbeda antar-flight'); ax[0,0].legend()
    ax[0,1].bar(x-.18,merged.mae_v,.36,label='MAE',color=COLORS[0]);ax[0,1].bar(x+.18,merged.rmse_v,.36,label='RMSE',color=COLORS[1]);ax[0,1].set(ylabel='Error (V)',title='Error regresi');ax[0,1].legend()
    ax[1,0].bar(x,merged.residual_bias_v,color=np.where(merged.residual_bias_v>=0,COLORS[1],COLORS[0]));ax[1,0].axhline(0,color='black',lw=.6);ax[1,0].set(ylabel='Mean signed residual (V)',title='Score positif tidak menangkap bias negatif')
    ax[1,1].bar(x,merged.confirmed_episodes_per_minute,color=COLORS[2]);ax[1,1].set(ylabel='Episodes / ready minute',title='Frekuensi episode, bukan jumlah sampel')
    for a in ax.flat:a.set_xticks(x,labels);a.set_xlabel('Flight')
    save(fig,'healthy_overview')
    fig, ax = plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for j, series in enumerate(['raw','confirmed']):
        gs=[summarize(events[(events.series==series)&(events.family==f)]) for f in FAMILIES]
        bottom=np.zeros(3)
        for key,col in [('detected',COLORS[0]),('missed',COLORS[1]),('preexisting',COLORS[2])]:
            vals=np.array([g[key] for g in gs]);ax[j].bar(np.arange(3),vals,bottom=bottom,label=key,color=col);bottom+=vals
        ax[j].set(xticks=range(3),xticklabels=['Resistance','Progressive','Sudden drop'],ylabel='Jumlah events',title=series.capitalize());ax[j].legend()
    save(fig,'event_outcomes')
    fig, ax = plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for series, color in [('raw','#85aeb8'),('confirmed',COLORS[0])]:
        g=events[(events.series==series)&events.detected]; v=np.sort(g.delay_seconds)
        ax[0].step(v,np.arange(1,len(v)+1)/len(v),where='post',label=f'{series} (n={len(v)})',color=color)
        ax[1].plot(DEADLINES,[100*totals[series][f'deadline_{d}_rate'] for d in DEADLINES],'-o',color=color,label=series)
    ax[0].set(xlabel='Delay termasuk persistence (s)',ylabel='ECDF detected-only',title='Event lambat membentuk ekor distribusi');ax[0].legend()
    ax[1].set(xlabel='Deadline (s)',ylabel='Success / 162 applicable (%)',ylim=(0,105),title='Miss tetap berada dalam denominator');ax[1].legend()
    save(fig,'delays_deadlines')
    conf=events[events.series=='confirmed']
    heat=np.array([[100*conf[(conf.family==f)&(conf.severity==s)].detected.mean() for s in ['mild','moderate','severe']] for f in FAMILIES])
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained');im=ax.imshow(heat,vmin=0,vmax=100,cmap='YlGnBu');ax.set(xticks=range(3),xticklabels=['Mild','Moderate','Severe'],yticks=range(3),yticklabels=FAMILIES,title='Confirmed event recall per family dan severity (18 events/cell)')
    for j in range(3):
        for i in range(3):ax.text(i,j,f'{heat[j,i]:.1f}%',ha='center',va='center',color='white' if heat[j,i]>70 else '#18313d')
    fig.colorbar(im,ax=ax,label='Recall (%)');save(fig,'scenario_grid')
    fig,ax=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for j,f in enumerate(FAMILIES):
        g=conf[conf.family==f];ax[0].scatter(g.effect_threshold_ratio,g.delay_seconds,label=f,color=COLORS[j],alpha=.65,s=24)
        z=g[~g.detected];ax[0].scatter(z.effect_threshold_ratio,np.zeros(len(z)),marker='x',s=65,color='crimson',label='Miss (y=0 marker)' if j==0 else None)
        ax[1].scatter(np.repeat(j,len(g))+np.linspace(-.18,.18,len(g)),g.effect_threshold_ratio,color=np.where(g.detected,COLORS[0],'crimson'),alpha=.6,s=24)
    ax[0].set(xlabel='Event max added sag / threshold',ylabel='Detected delay (s)',title='Besar efek dan delay bukan relasi satu-ke-satu');ax[0].legend(fontsize=7)
    ax[1].axhline(1,color='black',ls='--',lw=.8);ax[1].set(xticks=range(3),xticklabels=['Resistance','Progressive','Sudden'],ylabel='Event max added sag / threshold',title='Merah: missed; hijau: detected')
    save(fig,'effects_threshold')
    for _, h in merged.iterrows():
        df=healthy_frames[h.test_flight];threshold=float(calibration.loc[calibration.fold_id==h.fold_id,'threshold_v'].iloc[0])
        fig,ax=plt.subplots(4,1,figsize=(12,9),sharex=True,layout='constrained')
        continuous(ax[0],df,'Curr',COLORS[0],'Curr');ax[0].set(ylabel='Curr (A)',title=f'{h.test_flight} | {h.fold_id} | healthy-assumed test')
        continuous(ax[1],df,'voltage_sag_actual',COLORS[0],'Sag actual');continuous(ax[1],df,'predicted_voltage_sag',COLORS[1],'Sag predicted');ax[1].set_ylabel('Sag (V)');ax[1].legend(loc='best')
        continuous(ax[2],df,'residual_signed_v',COLORS[2],'Signed residual');ax[2].axhline(threshold,color='crimson',ls='--',label=f'P99={threshold:.6f} V');ax[2].axhline(0,color='black',lw=.6);ax[2].set_ylabel('Residual (V)');ax[2].legend()
        continuous(ax[3],df,'anomaly_raw','#a5cbd1','Raw',drawstyle='steps-post');continuous(ax[3],df,'anomaly_confirmed',COLORS[0],'Confirmed',drawstyle='steps-post');ax[3].set(ylabel='Alarm',xlabel='Elapsed native timestamp (s)',yticks=[0,1]);ax[3].legend()
        save(fig,'healthy_'+h.test_flight[:8])
    # Deterministic selection declared independently of outcome: mild, position 0.5, all six flights/families.
    chosen=scenarios[(scenarios.severity=='mild') & (scenarios.start_position==.5)].sort_values(['NamaFile','fault_family'])
    for _,s in chosen.iterrows():
        df=csv(CV/s.scores_csv);healthy=healthy_frames[s.NamaFile]
        name=f'timeline_{s.NamaFile[:8]}_{s.fault_family}'
        onset=(s.onset_time_us-df.TimeUS.iloc[0])/1e6;end=(s.fault_end_time_us-df.TimeUS.iloc[0])/1e6
        fig,ax=plt.subplots(4,1,figsize=(12,8.5),sharex=True,layout='constrained')
        continuous(ax[0],df,'Curr',COLORS[0],'Curr');ax[0].set(ylabel='Curr (A)',title=f'{s.NamaFile[:8]} | {s.fault_family}, mild, onset 0.5 | {s.evaluation_status}')
        continuous(ax[1],healthy,'voltage_sag_actual','#a5cbd1','Healthy sag');continuous(ax[1],df,'voltage_sag_actual',COLORS[0],'Injected sag');continuous(ax[1],df,'predicted_voltage_sag',COLORS[1],'Prediction');ax[1].set_ylabel('Sag (V)');ax[1].legend(fontsize=8)
        continuous(ax[2],df,'anomaly_score',COLORS[2],'Positive residual');ax[2].axhline(s.threshold_v,color='crimson',ls='--',label='P99');ax[2].set_ylabel('Score (V)');ax[2].legend(fontsize=8)
        continuous(ax[3],df,'anomaly_raw','#a5cbd1','Raw',drawstyle='steps-post');continuous(ax[3],df,'anomaly_confirmed',COLORS[0],'Confirmed',drawstyle='steps-post');ax[3].set(ylabel='Alarm',xlabel='Elapsed native timestamp (s)',yticks=[0,1]);ax[3].legend(fontsize=8)
        for a in ax:a.axvspan(onset,end,color=COLORS[1],alpha=.10);a.axvline(onset,color=COLORS[1],lw=.8)
        ax[-1].set_xlim(max(0,onset-3),end+min(5,max(1,end-onset)*.1))
        save(fig,name)
        selections.append(dict(scenario_id=s.scenario_id, flight=s.NamaFile,family=s.fault_family, severity=s.severity,
                               start_position=float(s.start_position), outcome=s.evaluation_status, figure=f'figures/{name}.png',source=s.scores_csv))
    for name in ['flight66_residual_relationships','flight66_training_feature_ecdf','flight66_vs_control']:
        src=DIAG/'plots'/f'{name}.png';used.add(src);shutil.copyfile(src,out/'figures'/src.name)
    # Report narrative. Template controls presentation, not methodology or metric values.
    parts=[]
    def p(text):parts.append(text.strip())
    def tab(headers, rows):p(table(headers,rows))
    def fig(name, caption):
        p(f'![{caption}](figures/{name}.png)')
        if (out/'figures'/f'{name}.svg').exists():p(f'[Versi SVG](figures/{name}.svg).')
    p('# Deteksi anomali respons tegangan baterai UAV — laporan penelitian HGB V2')
    p('Disusun 26 Oktober 2026 (Asia/Jakarta). Baseline LOAFO `4bc978e2773687d3`; konsolidasi `4ad7e153513e7d64`; diagnosis Flight 66 historis. Struktur dan tampilan mengikuti template yang diberikan; data, rumus, fitur, dan hasil berasal dari artefak HGB V2 aktif. Laporan ini membaca hasil tersimpan dan melakukan re-agregasi, tanpa menjalankan ulang training atau memilih konfigurasi.')
    p('## 1. Hasil utama dan cakupan klaim')
    p('Baseline mendeteksi **156/162 applicable synthetic events (96,296296%)**, tetapi **3.178/26.870 healthy-assumed ready samples (11,827317%)** juga menghasilkan confirmed alarm. Kinerja regresi pooled yang tinggi, R²=0,946481, belum menjamin false alarm rendah atau respons cepat terhadap gangguan gradual. Target abrupt severe ≤1 s tercapai 36/36; target progressive observable ≤5 s hanya 12/51 (23,529412%), sehingga belum tercapai.')
    pr=pooled['regression'];ph=pooled['healthy_active'];ps=pooled['synthetic']
    tab(['Metrik utama','Hasil','Denominator / interpretasi'],[
        ['MAE / RMSE (V)',f"{fmt(pr['mae_v'])} / {fmt(pr['rmse_v'])}",'26.870 unique healthy test ready samples'],
        ['R² pooled',pr['r2'],'Target sag; bukan akurasi deteksi'],
        ['Healthy confirmed FPR (%)',100*ph['confirmed_fpr'],'3.178 / 26.870'],
        ['Episode / menit',ph['confirmed_episodes_per_minute'],'29 episodes / 2.683,400543 observed ready seconds'],
        ['Event recall (%)',100*ps['incident_recall'],'156 / 162; 6 missed; 0 preexisting'],
        ['Synthetic sample precision / recall / F1',f"{fmt(ps['precision'])} / {fmt(ps['recall'])} / {fmt(ps['f1'])}",'Seluruh ready rows dalam 162 salinan'],
        ['Delay median / mean (s)',f"{fmt(ps['delay_median_seconds'])} / {fmt(ps['delay_mean_seconds'])}",'n=156 detected-only; persistence sudah termasuk']])
    p('Enam flight active dan sebelas log low-load telah ditelaah selama pengembangan. Evaluasi ini adalah historical development cross-validation; keenam test flight terpisah dari training dan calibration dalam fold masing-masing, tetapi bukan independent final holdout baru untuk klaim sesudah pemilihan metode. Label healthy berasal dari asumsi sumber, bukan hasil pemeriksaan kesehatan baterai independen. Alarm pada healthy-assumed disebut false positive relatif terhadap label tersebut. Salinan sintetis tidak menambah jumlah penerbangan independen.')
    p('## 2. Data, ekstraksi, preprocessing, dan fitur kausal')
    p('Pipeline V2 dimulai dari `data/raw/dataset_baterai.csv`, bukan decoder log BIN. Nama `00000066.BIN` dan seterusnya adalah identitas sumber yang tercatat di CSV. Repo aktif tidak menyediakan converter BIN ke CSV maupun log BIN asli untuk baseline ini. Script persiapan dataset mengemas CSV yang tersedia; laporan ini tidak mengklaim telah mengekstrak ulang data dari log asli. Schema sumber: NamaFile, TimeUS, Inst, Volt, Curr, CurrTot. TimeUS dinyatakan µs, Volt dalam V, Curr dalam A; unit CurrTot tetap unverified dan tidak dikonversi menjadi SOC atau kapasitas.')
    p(f"Audit tersimpan mempertahankan seluruh {fmt(prep['retained_rows'])} baris / 17 flights, tanpa dropped rows pada dataset ini. Terdapat {prep['segments']} continuous segments dan {prep['gaps']} gaps. Observed exposure {fmt(prep['observed_duration_seconds'])} s; excluded gap duration {fmt(prep['excluded_gap_seconds'])} s. Sampling nominal 10 Hz hanya menjadi konteks; segmentasi dan timer menggunakan timestamp aktual, dengan gap strictly >0,5 s memulai segmen baru.")
    iv=inv.merge(coverage[['NamaFile','ready_samples','warmup_samples','ready_observed_duration_seconds','V_ref']],on='NamaFile')
    tab(['Flight','Kategori arus','Retained','Ready','Warm-up','Ready exposure (s)','V_ref (V)'],[
        [r.NamaFile[:8],r.v2_current_based_label,r.retained_samples,r.ready_samples,r.warmup_samples,r.ready_observed_duration_seconds,r.V_ref] for _,r in iv.iterrows()])
    p('ACTIVE_LOAD ditetapkan ulang dari Curr ≥1 A pada continuous run minimal 1 s dan active exposure minimal 5% dari observed duration. Klasifikasi ini menghasilkan flight 66, 69, 70, 71, 73, 74. Sebelas lainnya LOW_LOAD_OR_IDLE. Kategori arus tidak menentukan airborne/ground; bagian low-current dalam active flight tetap disertakan. Sensitivitas aturan arus 0,5/1/2 A dan durasi 0,5/1/2 s pada audit historis menghasilkan daftar enam active flight yang sama.')
    p('Reference menggunakan median sampel Curr≤1 A dalam fixed warm-up awal sampai elapsed≤2 s, minimum lima sampel. Nilai dikunci pada sampel pertama strictly setelah akhir window. Reference yang terkunci dipertahankan melintasi gap. Tanpa sampel cukup, reference tetap unavailable; pada dataset ini semua 17 flight berhasil mengunci reference. Total 349 warm-up rows tidak ikut fitting maupun denominator alarm. V_ref merupakan reference empiris awal log, bukan estimasi OCV/SOC.')
    tab(['Urutan','Prediktor Battery-D','Definisi dan batas'],[
        [1,'Curr','Arus saat ini (A)'],[2,'CurrTot','Kanal akumulasi sumber; unit/reset fisik belum diverifikasi'],[3,'dI_dt','ΔCurr/Δt native dalam segmen (A/s); awal segmen diisi 0'],
        [4,'Curr_roll_mean_0p5s','Mean aritmetika pada (t−0,5 s,t]'],[5,'Curr_roll_std_0p5s','Std populasi ddof=0 pada window sama'],
        [6,'Curr_roll_mean_1p0s','Mean aritmetika pada (t−1 s,t]'],[7,'Curr_roll_std_1p0s','Std populasi ddof=0 pada window sama']])
    p('Rolling memakai sampel masa lalu dan sampel saat ini; satu sampel menghasilkan std=0. Gap menghapus history rolling/derivative dan mereset persistence, sedangkan fixed V_ref tetap terkunci. Reference-ready adalah dasar scoring baseline; flags count/coverage/derivative readiness adalah diagnostik dan bukan persyaratan full-window baru. Karena itu warm-up ulang satu detik pada setiap segmen dari template tidak diterapkan pada V2. Volt aktual, V_ref, target sag, residual, dV_dt, ID, label dan readiness tidak masuk X HGB-D; elapsed_time hanya ada pada ablation Battery-E. Tidak ada scaler operasional HGB.')
    p('## 3. Split, estimator, threshold, dan rumus injeksi aktif')
    tab(['Fold','Training flights','Validation','Test'],[[a['fold_id'],' / '.join(f[:8] for f in a['train']),a['validation'][0][:8],a['test'][0][:8]] for a in protocol['assignments']])
    p('Setiap fold menggunakan empat flight healthy-assumed untuk fitting, satu flight validation utuh untuk threshold, dan satu flight test utuh. Semua reference-ready rows dari assigned training flights, termasuk low-current, ikut fitting. Enam model dan kebijakan sudah dibekukan sebelum outer scoring. HGB mempelajari target kontinu secara supervised pada data nominal; tahapan deteksi tidak memerlukan contoh fault untuk fitting baseline.')
    params=protocol['model_parameters']
    tab(['Parameter HGB','Nilai frozen'],[[k,params[k] if params[k] is not None else 'None'] for k in ['loss','learning_rate','max_iter','max_leaf_nodes','max_depth','min_samples_leaf','l2_regularization','max_bins','early_stopping','random_state']])
    tab(['Fold','Validation','Ready samples','P99 threshold (V)'],[[r.fold_id,r.validation_flight[:8],r.ready_samples,r.threshold_v] for _,r in calibration.iterrows()])
    p('```text\ny(t) = V_ref − Volt(t)\nr(t) = y(t) − y_predicted(t)\nscore(t) = max(0, r(t))\nraw(t) = reference_ready(t) AND score(t) > P99_validation\nconfirmed(t) = raw true secara kontinu dengan elapsed timestamp ≥0,5 s\n```')
    p('Threshold menggunakan percentile 99 dengan interpolasi linear dari score positive residual validation fold terkait. Run raw dimulai pada timestamp exceedance pertama; false/unready atau boundary segmen memutus run. Persistensi 0,5 s merupakan elapsed time, bukan lima sampel. Onset sintetis tidak mereset timer. Bias residual negatif tetap dihitung pada error regresi, tetapi dipetakan ke score=0 oleh detector satu arah.')
    tab(['Family','Added sag a(t)','Grid dan dukungan'],[
        ['increased_resistance','a(t)=Curr(t)·ΔR','ΔR=0,02 / 0,05 / 0,08 Ω; longest valid active run'],
        ['progressive','a(t)=Curr(t)·ΔR_max·clip((t−t0)/D,0,1)','ΔR_max=0,02 / 0,05 / 0,08 Ω; D=end−onset pada active run'],
        ['sudden_drop','a(t)=ΔV·min(1,(t−t0)/0,4 s)','ΔV=0,5 / 1,0 / 1,5 V; longest continuous ready segment']])
    p('Pada support: Volt_injected=Volt_original−a(t), sag_injected=V_ref−Volt_injected. Arus, fitur arus, CurrTot, V_ref dan readiness tetap berasal dari sumber. Karena X tidak berubah, prediksi HGB-D untuk setiap salinan sama dengan prediksi nominal. dV_dt dihitung ulang untuk kebutuhan direct baselines, termasuk saat kembali ke telemetry asli. Intervensi memodelkan perubahan respons listrik yang terkontrol; tambahan ΔR bukan pengukuran resistansi internal dan bukan simulasi elektrokimia lengkap.')
    p('Untuk setiap family dipilih segmen eligible terpanjang; tie-break timestamp awal. Requested onset berada pada fraksi 0,4/0,5/0,6 durasi segmen, lalu aligned ke sampel pertama at/after timestamp itu. Seluruh tail dari onset sampai sampel terakhir segmen terpilih adalah support inklusif [t0,t_end], minimum post-onset 1 s. Tidak ada durasi event tetap 1/2/4 s pada baseline ini. Sampel onset progressive/sudden ramp bisa memiliki efek nol dan tetap berlabel support. Setelah support, telemetry asli kembali tanpa recovery ramp buatan. Grid menghasilkan 3 families ×3 severities ×3 onsets ×6 active flights=162 events, satu event per salinan.')
    bank=research['fault_bank']
    p(f"Bedakan dari bank generator seluruh 17 flights: {bank['scenarios']} planned, {bank['applicable_scenarios']} applicable, {bank['not_applicable_scenarios']} NOT_APPLICABLE. Outer active baseline hanya mengevaluasi 162 applicable events tersebut. Angka 261 atau 459 bukan denominator recall baseline. NA tidak diinferensikan sebagai kegagalan deteksi maupun keberhasilan.")
    p('## 4. Healthy test: regresi, bias, dan false alarm per flight')
    tab(['Flight','Ready','MAE V','RMSE V','R²','Mean residual V','FP','FPR %','Episodes/min'],[
        [r.test_flight[:8],r.ready_samples,r.mae_v,r.rmse_v,r.r2,r.residual_bias_v,r.confirmed_alarm_samples,100*r.confirmed_fpr,r.confirmed_episodes_per_minute] for _,r in merged.iterrows()])
    fig('healthy_overview','Original healthy test: false alarm, error regresi, signed bias, dan laju episode per flight.')
    p('Healthy denominator menghitung setiap ready test row sekali: 26.870 samples, 2.683,400543 s. Raw FPR=3.607/26.870 (13,423893%); confirmed FPR=3.178/26.870 (11,827317%). Episode rate=29/(2.683,400543/60)=0,648431/min. Interval exposure hanya dijumlah bila dua ujung ready dan berada pada segmen yang sama; warm-up serta gap tidak menjadi waktu observasi detector. FPR berbasis sampel dan laju episode berbasis waktu mengukur aspek yang berbeda.')
    p('Flight 66 menyumbang 3.168/3.178 (99,685337%) confirmed healthy-assumed samples. Lima flight lain bersama-sama menghasilkan sepuluh confirmed samples; statistik ini menunjukkan konsentrasi error dan tidak menjadi alasan menghapus Flight 66. Flight 70 mempunyai mean signed residual −0,079212 V dan nol alarm detector positif: model masih bias meski FPR nol. Pooled R² tinggi juga dipengaruhi variasi target antarflight; per-fold R² dan error tetap disajikan agar variasi domain terlihat.')
    for flight in ['00000066','00000070']:
        fig('healthy_'+flight,f'Timeline {flight}: arus, sag actual/predicted, signed residual dan alarm; garis diputus pada gap.')
    p('## 5. Evaluasi event, sampel, delay, dan deadline')
    tab(['Seri','Applicable','Detected','Missed','Preexisting','Event recall %','Delay n'],[[s,v['events'],v['detected'],v['missed'],v['preexisting'],100*v['incident_recall'],v['delay_n']] for s,v in totals.items()])
    p('Confirmed event sukses memerlukan episode konfirmasi baru di dalam support. Bila alarm confirmed sudah dibawa ke onset tanpa episode start baru di sampel onset, seluruh event PREEXISTING_ALARM dan tidak diberi fresh credit. Miss/preexisting tetap denominator; delay null. Pada run ini preexisting=0 untuk kedua seri. Seri raw tambahan dalam laporan diperoleh secara deskriptif dari mask raw tersimpan dengan aturan episode dan support yang sama; tidak dipromosikan sebagai konfigurasi detector baru.')
    fig('event_outcomes','Status seluruh 162 events: seri raw diagnostik dan confirmed utama, dipisah per family.')
    tab(['Fold / test','Detected/applicable','Missed','Sample F1','Delay n','Median s','Mean s'],[
        [f"{fold} / {g.flight.iloc[0][:8]}",f"{(v:=summarize(g))['detected']}/{v['events']}",v['missed'],v['f1'],v['delay_n'],v['delay_median_seconds'],v['delay_mean_seconds']]
        for fold,g in conf.groupby('fold_id')])
    tab(['Family','Detected/applicable','Missed','Recall %','Delay median s','Delay mean s'],[
        [f,f"{(v:=summarize(conf[conf.family==f]))['detected']}/{v['events']}",v['missed'],100*v['incident_recall'],v['delay_median_seconds'],v['delay_mean_seconds']] for f in FAMILIES])
    fig('scenario_grid','Grid family × severity lengkap; setiap sel memuat 18 events dari enam flight dan tiga onset.')
    for key in ['severity','start_position','observability']:
        rows=[]
        for name,g in conf.groupby(key):
            v=summarize(g);rows.append([name,f"{v['detected']}/{v['events']}",100*v['incident_recall'],v['delay_n'],v['delay_median_seconds'],v['delay_mean_seconds']])
        p(f'Ringkasan berdasarkan {key}; seluruh kelompok, tanpa memilih tingkat yang paling berhasil.')
        tab([key,'Detected/applicable','Recall %','Delay n','Median s','Mean s'],rows)
    tab(['Seri','TP','FP','TN','FN','Precision %','Sample recall %','F1 %'],[[s,v['tp'],v['fp'],v['tn'],v['fn'],100*v['precision'],100*v['recall'],100*v['f1']] for s,v in totals.items()])
    p('Confusion counts dihitung pada seluruh ready rows tiap salinan, termasuk prefix dan telemetry setelah support: confirmed total 725.490 rows, positive support 82.356, negative background 643.134. Sample recall=TP/(TP+FN), precision=TP/(TP+FP), F1=2TP/(2TP+FP+FN). Background yang berulang membuat precision/F1 bergantung desain bank dan durasi support. Synthetic sample FPR=85.257/643.134 (13,256491%) mempunyai scope berbeda dari original healthy FPR. Event recall 96,30% tidak berarti 96,30% sampel gangguan terdeteksi.')
    tab(['Deadline s','Raw success/162','Confirmed success/162','Confirmed %'],[[d,f"{totals['raw'][f'deadline_{d}_successes']}/162",f"{totals['confirmed'][f'deadline_{d}_successes']}/162",100*totals['confirmed'][f'deadline_{d}_rate']] for d in DEADLINES])
    fig('delays_deadlines','Delay ECDF hanya detected events; deadline success tetap memakai seluruh applicable events.')
    p('Mean delay confirmed 5,526932 s jauh di atas median 0,699965 s karena sebagian progressive events baru terkonfirmasi lama setelah onset. Persistence sudah terkandung dalam timestamp first confirmation dan tidak ditambahkan ulang. Tidak ada confidence interval iid antar-sampel: pengamatan dalam flight dan salinan fault saling bergantung; unit domain utama tetap hanya enam flight historis.')
    failed=scenarios[~scenarios.incident_detected]
    tab(['Flight','Family','Severity','Onset fraction','Max added sag V','Threshold V','Reason'],[[r.NamaFile[:8],r.fault_family,r.severity,r.start_position,r.max_added_sag_v,r.threshold_v,r.detection_reason] for _,r in failed.iterrows()])
    p('Keenam misses berasal dari Flight 74 pada mild increased_resistance dan mild progressive, masing-masing tiga onset. Semua tetap dalam denominator. Tidak ada threshold, dosis atau label yang diubah untuk menutup kegagalan tersebut.')
    targets=results['research_target_checks']
    tab(['Target penelitian','Success / denominator','Hasil %','Target %','Status'],[
        ['Abrupt severe ≤1 s','36 / 36',100*targets['abrupt_severe']['rate'],100*targets['abrupt_severe']['target'],targets['abrupt_severe']['status']],
        ['Progressive observable ≤5 s','12 / 51',100*targets['gradual_observable']['rate'],100*targets['gradual_observable']['target'],targets['gradual_observable']['status']]])
    p('Grup abrupt severe dalam protokol mencakup increased_resistance severe dan sudden_drop severe; progressive observable hanya memakai subset progressive kategori MEDIUM/HIGH (max added sag ≥0,20 V). Batas observability: VERY_LOW <0,05 V; LOW ≥0,05 dan <0,20 V; MEDIUM ≥0,20 dan <0,50 V; HIGH ≥0,50 V. Definisi grup mengikuti sumber historis. Eventual recall progressive observable 48/51 tidak menggantikan target deadline 12/51. Kegagalan mencapai deadline merupakan hasil penelitian meskipun software berjalan benar.')
    p('## 6. Pooled dan macro: pengaruh denominator')
    p('Pooled MAE=Σabsolute_error/Σn, pooled RMSE=√(ΣSSE/Σn). Pooled R²=1−ΣSSE/(Σy²−(Σy)²/Σn), sehingga tidak dihitung sebagai rata-rata R² fold. Macro adalah mean aritmetika nilai per fold yang terdefinisi; n_defined disertakan dan undefined tidak diganti nol. Jumlah event sama per fold membuat pooled/macro incident recall sama, tetapi hal ini tidak berlaku bagi healthy FPR, F1, regresi atau mean delay detected-only.')
    tab(['Scope','Metrik','Macro mean','Population std','Median','Min','Max','n_defined'],[[r['scope'],r['metric'],r['mean'],r['std'],r['median'],r['min'],r['max'],r['n_defined']] for r in macro])
    p('Contoh: healthy confirmed FPR pooled 11,827317%, macro 5,640413%, median fold 0,011471%. Flight 66 memiliki banyak sampel dan FPR besar, sehingga bobotnya tinggi pada pooled. Synthetic F1 pooled 0,567735 versus macro 0,796940; keduanya sah untuk estimand berbeda. Perbedaan ini harus dijelaskan dan tidak digunakan untuk memilih angka yang tampak paling baik.')
    p('## 7. Besar efek dan mekanisme error residual')
    effectrows=[]
    for (family,severity),g in conf.groupby(['family','severity']):
        effectrows.append([family,severity,len(g),float(g.max_added_sag_v.median()),float(g.max_added_sag_v.max()),float(g.effect_threshold_ratio.median()),float(g.support_seconds.median())])
    tab(['Family','Severity','Events','Median max added sag V','Max added sag V','Median max effect / T','Median support s'],effectrows)
    fig('effects_threshold','Efek relatif terhadap threshold dan delay; marker missed pada y=0 hanya penempatan visual, bukan delay nol.')
    p('Dengan X dan V_ref tetap, residual_injected=residual_original+added_sag. Karena itu keberhasilan tergantung bukan hanya amplitudo fault tetapi juga bias nominal lokal. Model yang memprediksi sag terlalu besar menghasilkan residual negatif; tambahan sag terlebih dahulu mengurangi bias negatif sebelum score menjadi positif dan melampaui threshold. Jika nominal residual sudah positif mendekati threshold, gangguan kecil dapat lebih cepat menghasilkan exceedance. Ratio max effect/T adalah statistik deskriptif; nilai >1 bukan jaminan fresh confirmation dengan persistence.')
    p('Progressive meningkatkan ΔR secara bertahap sepanjang sisa segmen, sehingga crossing ditentukan arus, kecepatan ramp, posisi onset, nominal error dan threshold fold. Mean delay yang besar konsisten dengan mekanisme gradual tersebut; korelasi deskriptif tidak mengidentifikasi kontribusi kausal masing-masing. Sudden drop memberi perubahan tegangan absolut yang tidak bergantung arus, sedangkan resistance hanya besar ketika arus tinggi. Perbedaan support dan dosis membuat recall antarfamily bukan perbandingan gangguan fisik yang setara.')
    p('## 8. Root cause analysis Flight 66 dan pembanding')
    p('Diagnosis memakai saved model fold_00 asli: training 70/71/73/74, calibration 69, test 66. Flight 66 tidak dipakai untuk training atau calibration model yang menilainya. Prediksi dan timer direproduksi identik pada audit historis. Control Flight 70 memakai fold_02 miliknya sendiri; model itu mencakup Flight 66 sebagai training. Flight 69 dengan model fold_00 adalah konteks calibration, bukan test independen. Perbedaan role ini tetap dipertahankan.')
    tab(['Bukti terukur','Interpretasi yang didukung'],[
        ['3.168 confirmed samples, 26 episodes; 33,326320% FPR','Long positive residual runs melampaui P99=0,111810859 V'],
        ['2.944/3.168 (92,929293%) alarms di atas CurrTot maksimum training','Mayoritas alarm berasosiasi domain counter di luar support training'],
        ['CurrTot train max 1.016,583374; Flight 66 max 1.568,515137','Upper histogram bin model menyatukan seluruh nilai tinggi; tidak mengekstrapolasi tren counter di sana'],
        ['5 episodes >30 s memuat 2.726 confirmed samples','Bias panjang mendominasi jumlah alarm, bukan hanya burst sesaat'],
        ['Episode terpanjang 112,599917 s; arus mean sekitar 0,000300 A','Load transient tidak menjelaskan episode dominan ini'],
        ['2.289 alarms pada Curr≤1 A; 1.604 setelah low-current≥10 s tanpa transient context','Mismatch tetap bertahan setelah perubahan beban mereda'],
        ['V_ref=16,417031288 V, tetap pada 15 segmen','Perubahan level voltage relatif terhadap reference awal masuk target sag'],
        ['Audit 14 gaps; max rolling error 1,78e−15; timer exact','Tidak ditemukan bug segment/reset/rolling pada data tersedia']])
    p('Batas domain CurrTot memberi mekanisme representasi model yang konkret. Namun CurrTot, waktu dan level tegangan berubah bersama; bukti ini belum memisahkan kontribusi kausalnya. Istilah voltage drift pada laporan berarti perubahan level tegangan teramati terhadap initial reference, bukan bukti sensor drift. Sebanyak 224 alarms tetap terjadi saat CurrTot dalam rentang training, sehingga counter tinggi tidak menjelaskan setiap episode. Data CSV tidak menunjukkan numeric/ordering corruption yang menjelaskan alarm; penyebab fisik baterai, pack, temperatur, kalibrasi atau kontinuitas sesi belum terbukti.')
    fig('flight66_residual_relationships','Diagnosis historis: residual versus kondisi listrik; kolom diagnostik tidak dipakai sebagai fitur tambahan.')
    fig('flight66_training_feature_ecdf','Distribusi fitur Flight 66 terhadap assigned healthy training; audit tidak mengubah scaler/model.')
    e66=eps[eps.NamaFile=='00000066.BIN']
    tab(['Episode','Segmen','Start elapsed s','Confirmed span s','Samples','Mean Curr A','CurrTot start','Mean Volt V','Mean residual V','Akhir'],[
        [r.episode_id.rsplit('_',1)[-1],r.segment_id,r.confirmed_start_s,r.confirmed_span_seconds,r.confirmed_samples,r.Curr_mean,r.CurrTot_start,r.Volt_mean,r.residual_mean_v,r.end_reason] for _,r in e66.iterrows()])
    p('Confirmed span=timestamp terakhir confirmed−timestamp pertama confirmed dalam episode, tidak menambah interval nominal. Satu sampel confirmed mempunyai span nol walaupun raw sebelumnya sudah berlangsung ≥0,5 s. Gap memutus episode. Total spans dan sample counts adalah ukuran berbeda; CSV lengkap menyertakan raw start, voltage/current min-mean-max dan kondisi domain. Status healthy-assumed tidak diubah sebagai akibat alarm.')
    fig('flight66_vs_control','Flight 66 test, Flight 70 own-fold test dan Flight 69 fold_00 validation memiliki role serta domain yang berbeda.')
    p('## 9. Ablation, pembanding model, tuning, dan low-load')
    p('Eksperimen berikut dilaporkan sebagai konteks development yang sudah selesai. Kebijakan baseline tetap frozen dan tidak dipilih ulang berdasarkan tabel outer. Ablation membandingkan representasi fitur; family dan estimator memiliki domain score berbeda sehingga threshold masing-masing dikalibrasi pada validation yang ditetapkan.')
    tab(['Feature set','Prediktor'],[['Battery-A','Curr'],['Battery-B','Curr + CurrTot'],['Battery-C','Battery-B + dI_dt'],['Battery-D','Battery-C + current rolling mean/std 0,5 dan 1,0 s'],['Battery-E','Battery-D + elapsed_time; ablation saja']])
    tab(['Candidate','Healthy FPR %','Episodes/min','Detected/applicable','Sample F1','RMSE V'],[[r.candidate,100*r.healthy_confirmed_fpr,r.healthy_episodes_per_minute,f'{r.incident_detected}/{r.applicable}',r.sample_f1,r.regression_rmse_v] for _,r in compare.iterrows()])
    tab(['Ablation','Healthy FPR %','Episodes/min','Detected/applicable','F1','RMSE V'],[[r.candidate,100*r.healthy_confirmed_fpr,r.healthy_episodes_per_minute,f'{r.incident_detected}/{r.applicable}',r.sample_f1,r.regression_rmse_v] for _,r in ablation.iterrows()])
    p('RF Battery-D sedikit lebih tinggi F1 pooled tetapi episodes/min lebih tinggi dan RMSE lebih besar daripada HGB-D; tabel ini tidak menetapkan pemenang baru. IF nol healthy FPR hanya mendeteksi 22/162 events. OCSVM mendeteksi 80/162. FPR rendah saja tidak membuktikan detector sensitif. Direct IF/OCSVM melihat voltage_sag_actual, Curr, CurrTot, dV_dt dan dI_dt; OCSVM memakai scaler healthy-training-only dan capped deterministic training. Battery-E memasukkan elapsed_time dan tetap ablation.')
    policy_rows=[]
    for exp in ['tuning','dual_path']:
        for policy in ['reference','selected']:
            z=research[exp][policy+'_pooled'];h=z['healthy_active'];a=z['synthetic']
            policy_rows.append([exp,policy,100*h['confirmed_fpr'],h['confirmed_episodes_per_minute'],f"{a['detected_scenarios']}/{a['applicable_scenarios']}",a['f1'],a['delay_mean_seconds']])
    tab(['Eksperimen','Policy','FPR %','Episodes/min','Events','F1','Mean delay s'],policy_rows)
    p('Tuning single-path menguji 120 kombinasi percentile/persistence pada validation, kemudian membekukannya sebelum outer. Dual-path menambah slope path dengan pemilihan validation dan refinement terbatas. Dual selected memperoleh eventual recall 162/162 dan F1 lebih tinggi, tetapi mean delay bertambah dan beberapa deadline memburuk. Gate predeclared menggunakan paired arithmetic macro pada fold sama; hasil pooled tidak menggantikannya. Keputusan tetap NOT_PROMOTED dan dual_path.enabled=false.')
    tab(['Gate promosi dual','Reference macro','Selected macro','Lulus'],[[k,v.get('reference'),v.get('selected'),v['passed']] for k,v in research['promotion_decision']['checks'].items() if 'reference' in v])
    tab(['Frozen model','Low-load ready samples','Confirmed samples','Episodes','FPR %','Exposure s'],[[r.fold_id,r.ready_samples,r.confirmed_alarm_samples,r.confirmed_episodes,100*r.confirmed_fpr,r.observed_ready_seconds] for _,r in low.iterrows()])
    p('Sebelas unique low-load flights memberi 5.119 ready samples dan 506,599454 s. Primary fold_00 menghasilkan nol alarm; sensitivitas enam frozen models menghasilkan 150 confirmed samples pada fold_01 dan empat pada fold_02. Menggunakan ulang 11 flight pada enam model memberi 66 model–flight pairs, bukan 66 flight independen. Studi ini hanya mengukur false alarm pada domain low-load dan tidak menguji fault sensitivity low-load.')
    p('## 10. Timeline deterministik dan akses seluruh artefak')
    p('Aturan pemilihan grafik ditentukan dari metadata: severity mild, start_position=0,5, seluruh enam flight dan tiga family. Hasilnya 18 contoh tanpa memilih berdasarkan sukses/gagal. Plot memuat actual/predicted sag, raw score, threshold dan confirmed; warna support menunjukkan interval fault. Garis diputus pada segment boundary, tidak menginterpolasi gaps. Contoh Flight 74 memperlihatkan miss yang tetap terlihat.')
    for name in ['timeline_00000066_increased_resistance','timeline_00000070_progressive','timeline_00000074_progressive']:
        fig(name,'Contoh deterministik '+name.replace('timeline_','').replace('_',' ')+'; pilihan bukan berdasarkan hasil deteksi.')
    p('Seluruh 18 contoh dan enam healthy timelines:')
    p('\n'.join(f"- [{s['flight'][:8]} / {s['family']} / {s['outcome']}]({s['figure']})" for s in selections))
    p('\n'.join(f'- [Healthy flight {f[:8]}](figures/healthy_{f[:8]}.png)' for f in healthy_frames))
    p('Seluruh 162 scenario scores tetap tersedia pada sumber frozen, bukan hanya contoh yang digambar. Tabel indeks di bawah menghubungkan setiap event ke CSV.gz per sampelnya; CSV berisi timestamp native, current, voltage, actual/predicted sag, residual, raw/confirmed, support dan phase. Full-log prefix diproses oleh eksperimen asli; reporter hanya membaca keputusan tersimpan.')
    index=scenarios[['scenario_id','NamaFile','fault_family','severity','start_position','evaluation_status','scores_csv']].copy()
    index['source_link']=[link(CV/x) for x in index.scores_csv];export('scenario_index',index)
    index_rows=[]
    for _,r in index.iterrows():
        cells=[r.NamaFile[:8],r.fault_family,r.severity,str(r.start_position),r.evaluation_status]
        index_rows.append('<tr>'+''.join('<td>'+html.escape(c)+'</td>' for c in cells)+f'<td><a href="{html.escape(r.source_link,quote=True)}">{html.escape(r.scenario_id)} CSV.gz</a></td></tr>')
    (out/'scenario_index.html').write_text('<!doctype html><html lang="id"><head><meta charset="utf-8"><title>Indeks 162 skenario V2</title><style>body{font:15px/1.5 Arial;max-width:1250px;margin:30px auto;padding:20px;color:#18313d}table{border-collapse:collapse;width:100%}th,td{padding:8px;text-align:left;border-bottom:1px solid #d9e2e5}th{background:#edf5f5}input{padding:12px;width:100%;margin:20px 0}a{color:#076774}</style></head><body><a href="research_report.html">Kembali ke laporan</a><h1>Seluruh 162 skenario baseline</h1><p>Tiap link membuka score CSV.gz frozen; berkas berada di direktori sumber V2. Ketik flight, family, severity atau status untuk menyaring tabel.</p><input aria-label="Cari skenario" placeholder="Cari: 00000074 progressive mild UNDETECTED" oninput="let q=this.value.toLowerCase().split(/\\s+/);document.querySelectorAll(\'tbody tr\').forEach(r=>r.hidden=!q.every(w=>r.textContent.toLowerCase().includes(w)))"><table><thead><tr><th>Flight</th><th>Family</th><th>Severity</th><th>Onset</th><th>Status</th><th>Per-sample scores</th></tr></thead><tbody>'+''.join(index_rows)+'</tbody></table></body></html>',encoding='utf-8')
    p('[Buka indeks interaktif seluruh 162 skenario](scenario_index.html): cari berdasarkan flight, family, severity atau status, lalu buka score CSV.gz asli.')
    p('\n'.join(f'- [{name}](tables/{name}.csv)' for name in ['scenario_index','event_metrics','scenario_metrics','grouped_results','pooled_results','macro_results','healthy_combined','fold_assignments','calibration','false_alarm_episodes','model_comparison','ablation','low_load']))
    p('## 11. Provenance, verifikasi, dan reproduksibilitas laporan')
    p('Sumber baseline diverifikasi terhadap seluruh 217 output hashes pada manifest; preprocessing dan feature outputs juga diperiksa. Reporter merekonsiliasi pooled regression dari sufficient statistics, seluruh confirmed confusion/event/delay dari 162 score files, serta membangun raw diagnostic dengan reset segment/gap. Seluruh source baseline files diperiksa lagi setelah pelaporan untuk memastikan tidak berubah. Tidak ada estimator pickle dimuat dan tidak ada fit/predict, bank regeneration, calibration atau model selection oleh reporter.')
    tab(['Peran','Run / sumber'],[['Baseline utama','4bc978e2773687d3'],['Preprocessing','e22890aeb6ffddf6'],['Features','1225e8a76a8ce014'],['Bank seluruh 17 flight','437104037cb0fea5'],['Konsolidasi','4ad7e153513e7d64'],['Replay historis','ee132dc3901fb80a'],['Diagnosis Flight 66','artifacts/flight66_diagnosis']])
    tab(['Pustaka','Versi scoring sumber'],list(protocol['dependency_versions']['packages'].items()))
    p('```text\nRaw CSV SHA256: '+protocol['data_sha256']+'\nFeature manifest SHA256: '+protocol['feature_manifest_sha256']+'\nBaseline config SHA256: '+protocol['config_sha256']+'\nBaseline manifest SHA256: '+sha(CV/'manifest.json')+'\n```')
    p('Audit software Sprint 13 tersimpan mencatat 371 tests, nol failure/error/skip; sebelas kelompok audit dan 103 model bundles diperiksa. Itu bukti historis, bukan pengujian ulang hari ini. Diagnosis Flight 66 memiliki audit PASS dan exact original policy replay. Reporter mencatat pemeriksaan integritas, angka, link dan gambar pada metadata laporan; kelulusan tersebut tidak mengubah hasil penelitian yang belum memenuhi target gradual.')
    p('Replay historis memverifikasi 761.095 rows pada 179 jobs terhadap batch, max numeric error=0 dalam hasil tersimpan. Latency healthy uncached host p50/p95/p99=1,634/2,184580/2,857084 ms; max=21,8337 ms. Pengukuran berjalan bersamaan dengan pytest, sehingga bukan benchmark hardware UAV. Sebanyak 725.490 cached synthetic predictions dipisahkan dari latency uncached. Laporan ini tidak mengukur ulang latency atau menjalankan streaming fisik. Dual tidak dipromosikan dan status replay tetap BASELINE_SOFTWARE_REPLAY_ONLY.')
    p('```powershell\n.\\.venv\\Scripts\\python.exe scripts/build_research_report.py --output artifacts/reports/research_v2_20261026_new\n```')
    p('Perintah memerlukan direktori keluaran baru/kosong dan menulis laporan serta grafik dari artefak yang sudah tersedia. [Angka terstruktur](numbers.json), [aturan pemilihan grafik](figure_selection.json), [metadata dan hasil pemeriksaan](report_metadata.json), dan [manifest output](report_manifest.json) disertakan. SHA256 input serta template dicatat; template berfungsi sebagai referensi penyajian, bukan sumber angka V2.')
    p('## 12. Penilaian akhir dan prioritas penelitian berikutnya')
    p('Baseline berhasil mendeteksi mayoritas intervensi listrik yang ditetapkan, terutama abrupt severe, tetapi false alarm yang terkonsentrasi pada Flight 66 dan respons gradual yang lambat membatasi kegunaannya. Bukti diagnosis mendukung keterbatasan generalisasi model pada late-discharge/low-current setelah beban, dengan representasi CurrTot di luar training support dan fixed initial reference sebagai konteks utama. Tidak ada bukti cukup untuk menyimpulkan kerusakan internal baterai atau mengukur kontribusi kausal masing-masing faktor dari log ini saja.')
    tab(['Prioritas','Langkah penelitian','Dasar bukti / kriteria berikutnya'],[
        [1,'Tambah healthy development lintas pack/session, terutama akhir discharge dan low-current setelah load','92,93% alarms di atas CurrTot train max; protokolkan flight/pack holdout yang belum diperiksa'],
        [2,'Studi terpisah representasi keadaan dan target/reference secara kausal','Bias panjang pada current hampir nol; bandingkan dengan split dan denominator tetap'],
        [3,'Verifikasi unit/reset/continuity CurrTot dan metadata kondisi','Upper-bin saturation teramati; jangan menyebut kanal sebagai SOC tanpa bukti'],
        [4,'Evaluasi calibration transfer dan monitoring domain/uncertainty','Validation residual dan late-test residual berbeda; pertahankan FPR, episode dan deadline sebagai evaluasi bersama'],
        [5,'Teliti recovery/load-response dan deteksi gradual','224 alarms dalam CurrTot range dan progressive target 12/51; tetapkan eksperimen baru sebelum scoring']])
    p('Rekomendasi merupakan rancangan penelitian berikutnya dan belum diterapkan oleh laporan ini. Flight 66 tetap healthy-assumed dan tetap bagian hasil; tidak dihapus atau diubah labelnya karena detector alarm. Jika hasil Flight 66 dipakai untuk refinement, flight itu menjadi development evidence dan tidak boleh diklaim independent holdout sesudahnya. Kesimpulan mencakup software, generalisasi pada telemetry historis dan intervensi sintetis; belum mencakup validasi switching fisik atau deployment keselamatan.')
    p('Sumber utama dapat dibaca langsung: '+', '.join(f'[{label}]({link(path)})' for label,path in [('baseline report',CV/'research_report.md'),('diagnosis Flight 66',DIAG/'ROOT_CAUSE_ANALYSIS.md'),('hasil final Sprint 13',ROOT/'FINAL_RESEARCH_RESULTS.md'),('verifikasi Sprint 13',ROOT/'artifacts/sprint13/verification.json')])+'.')
    md='\n\n'.join(parts)+'\n'
    (out/'research_report.md').write_text(md,encoding='utf-8')
    default_css="body{font:16px/1.65 Arial;color:#18313d;background:#f2f5f6}main{max-width:1180px;margin:32px auto;padding:48px;background:white}table{border-collapse:collapse;width:100%}th,td{padding:8px;border-bottom:1px solid #ddd}img{max-width:100%}.table-wrap{overflow:auto}nav{columns:2}nav a{display:block}h2{margin-top:48px}pre{white-space:pre-wrap;overflow-wrap:anywhere}"
    templates=[]
    css=default_css
    if template_dir:
        for filename in ['research_report.md','research_report.html']:
            file=template_dir/filename
            templates.append(dict(path=str(file),sha256=sha(file)))
        content=(template_dir/'research_report.html').read_text(encoding='utf-8-sig')
        style=re.search(r'<style>(.*?)</style>',content,re.S)
        if style:css=style.group(1)
    css+='\ncode{font:0.86em Consolas,monospace;overflow-wrap:anywhere}main{overflow-wrap:break-word}td{min-width:64px}nav{margin-bottom:30px}nav a{font-size:14px}figure img{border:1px solid #e2e8eb}h3{margin-top:32px}'
    rendered=render(md,css)
    (out/'research_report.html').write_text(rendered,encoding='utf-8')
    dump(out/'figure_selection.json',dict(rule='all flights and families; mild severity; position 0.5; no outcome filter',selections=selections))
    dump(out/'numbers.json',dict(baseline_run=CV.name,regression=calculated_reg,healthy=ph,synthetic=totals,macro=results['macro_per_fold'],targets=targets))
    # Resolve all relative links and image targets in generated HTML/MD.
    for target in re.findall(r'(?:href|src)="([^"]+)"',rendered):
        if target.startswith(('#','https://','http://')):continue
        if target not in {'report_metadata.json','report_manifest.json'} and not (out/html.unescape(target)).resolve().exists():raise ValueError(f'Broken report link: {target}')
    for target in re.findall(r'\]\(([^)]+)\)',md):
        if target not in {'report_metadata.json','report_manifest.json'} and not (out/target).resolve().exists():raise ValueError(f'Broken Markdown link: {target}')
    from PIL import Image
    for file in (out/'figures').glob('*.png'):
        with Image.open(file) as im:im.verify()
    for file,digest in before.items():
        if sha(file)!=digest:raise ValueError(f'Source changed during reporting: {file}')
    checks.extend([dict(check='local_links_and_png_integrity',status='PASS'),dict(check='source_preservation_after_report',files=len(before),status='PASS')])
    metadata=dict(created_utc=datetime.now(timezone.utc).isoformat(),report_date_local='2026-10-26',
                  baseline_run=CV.name,report_generator_sha256=sha(__file__),scope='historical_development_only',
                  model_loading=False,fitting=False,prediction=False,calibration=False,bank_regenerated=False,
                  tests_run_by_report=False,historical_tests=historic['tests'],template_sources=templates,
                  checks=checks,sources=[dict(path=str(f.relative_to(ROOT)),sha256=sha(f)) for f in sorted(used)],
                  images_png=len(list((out/'figures').glob('*.png'))),tables_csv=len(list((out/'tables').glob('*.csv'))))
    dump(out/'report_metadata.json',metadata)
    dump(out/'report_manifest.json',dict(files={str(f.relative_to(out)).replace('\\','/'):sha(f) for f in sorted(out.rglob('*')) if f.is_file() and f.name!='report_manifest.json'}))
    for target in ['report_metadata.json','report_manifest.json']:
        assert (out/target).is_file()
    print(json.dumps(dict(status='PASS',output=str(out),figures=metadata['images_png'],tables=metadata['tables_csv'],checks=checks),indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',default='artifacts/reports/research_v2_20261026')
    ap.add_argument('--template-dir',default=r'C:\Skripsi2\HGB V3\outputs\reports\s6_20261026')
    args=ap.parse_args()
    destination=(ROOT/args.output).resolve()
    if not destination.is_relative_to(ROOT/'artifacts/reports'):
        ap.error('Output must be inside artifacts/reports')
    build(destination,Path(args.template_dir) if args.template_dir else None)
