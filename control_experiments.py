"""Контрольные варианты главы 10. Только стандартная библиотека.
Запуск: python3 control_experiments.py --out control_results
Используется неизменённый генератор corpus_lab.generate.
"""
from __future__ import annotations
import argparse
import json
import random
import statistics
from collections import Counter
from dataclasses import replace
from pathlib import Path
import corpus_lab as lab


def navigation(records, mode, seed):
    if mode == 'cyclic':
        return records
    rng = random.Random(seed + 100003)
    by_family = {}
    output = []
    for record in records:
        if record.family not in by_family:
            domain = rng.choice(lab.NAMES)
            by_family[record.family] = ' '.join(
                rng.choices(lab.DOMAINS[domain], k=40)
            )
        output.append(replace(record, navigation=by_family[record.family]))
    return output


def budget_sample(records, stage, budget, seed):
    """Точная равная экспозиция слов, включая все повторы.
    Поток перемешанных документов обрезается на последнем слове.
    Классы и семейства исходных записей сохраняются.
    """
    rng = random.Random(seed)
    order = list(records)
    rng.shuffle(order)
    output = []
    used, cursor = 0, 0
    while used < budget:
        record = order[cursor % len(order)]
        words = lab.tokens(record.text(stage))
        selected = words[:budget-used]
        if selected:
            output.append(replace(
                record, id=f'budget-{cursor}', body=' '.join(selected),
                navigation='',
            ))
            used += len(selected)
        cursor += 1
    assert sum(len(lab.tokens(r.body)) for r in output) == budget
    return output


def filter_training(seed):
    """Обучение и настройка фильтра на отдельных семействах.
    Цель — несоответствие содержимого метке страницы.
    Признаки — согласованность терминов с исходной областью.
    Известный флаг damaged не является целевой меткой.
    """
    rng = random.Random(seed + 770003)
    rows = []
    for domain in lab.NAMES:
        for i in range(180):
            actual = domain if i % 2 == 0 else rng.choice(
                [x for x in lab.NAMES if x != domain]
            )
            words = rng.choices(lab.DOMAINS[actual], k=rng.randint(3, 9))
            if rng.random() < .45:
                words += rng.choices(lab.DOMAINS[rng.choice(lab.NAMES)],
                                     k=rng.randint(5, 11))
            text = 'Документ. Перед проверкой отключить питание. ' + ' '.join(words)
            rows.append((domain, text, int(actual == domain), f'filter-{seed}-{domain}-{i}'))
    # Генерация последовательна; принадлежность не зависит от цели.
    fit = [row for row in rows if int(row[3].rsplit('-', 1)[1]) % 3 != 0]
    validation = [row for row in rows if int(row[3].rsplit('-', 1)[1]) % 3 == 0]
    assert not {x[3] for x in fit} & {x[3] for x in validation}
    return fit, validation


def features(domain, text):
    ts = lab.tokens(text)
    own = sum(t in lab.DOMAINS[domain] for t in ts)
    foreign = sum(t in values for name, values in lab.DOMAINS.items()
                  if name != domain for t in ts)
    total = max(1, own + foreign)
    return [1.0, own/total, foreign/total, min(own, 12)/12]


class LogisticFilter:
    def __init__(self):
        self.weights = [0.0] * 4
        self.threshold = .5

    def probability(self, domain, text):
        import math
        z = sum(a*b for a,b in zip(self.weights, features(domain, text)))
        return 1 / (1 + math.exp(-max(-35, min(35, z))))

    def fit(self, rows):
        import math
        data=[(features(domain,text),y) for domain,text,y,_ in rows]
        for _ in range(700):
            gradient = [0.0]*4
            for xs,y in data:
                z=sum(a*b for a,b in zip(self.weights,xs))
                residual=1/(1+math.exp(-max(-35,min(35,z))))-y
                for j,x in enumerate(xs):
                    gradient[j]+=residual*x
            for j in range(4):
                penalty=.01*self.weights[j] if j else 0.0
                self.weights[j]-=.4*(gradient[j]/len(data)+penalty)
        return self

    def tune(self, rows):
        # Выбрать порог по сбалансированной ошибке на validation.
        candidates = [i/100 for i in range(20, 81, 5)]
        def score(t):
            tp=tn=pos=neg=0
            for domain,text,y,_ in rows:
                pred=self.probability(domain,text)>=t
                pos+=y;neg+=1-y;tp+=int(pred and y);tn+=int(not pred and not y)
            return (tp/max(1,pos)+tn/max(1,neg))/2
        self.threshold = max(candidates, key=lambda t: (score(t), -abs(t-.5)))
        return self

    def accept(self, record):
        return self.probability(record.domain,record.body)>=self.threshold


def transitions(a,b):
    return {
        'both_correct':sum(x and y for x,y in zip(a,b)),
        'both_wrong':sum(not x and not y for x,y in zip(a,b)),
        'improved':sum(not x and y for x,y in zip(a,b)),
        'regressed':sum(x and not y for x,y in zip(a,b)),
    }


def run(out):
    out.mkdir(parents=True,exist_ok=True)
    rows=[];prediction_files=[];filter_results=[]
    for seed in lab.SEEDS:
        base_train,base_test=lab.generate(seed)
        correctness={}
        for mode in ['cyclic','random']:
            train=navigation(base_train,mode,seed)
            test=navigation(base_test,mode,seed+1)
            prepared={stage:lab.prepare(train,stage) for stage in lab.STAGES}
            budget=min(sum(len(lab.tokens(r.text(stage))) for r in records)
                       for stage,records in prepared.items())
            for equal_budget in [False,True]:
                for stage,records in prepared.items():
                    fitted=(budget_sample(records,stage,budget,seed+500)
                            if equal_budget else records)
                    model=lab.NB().fit(fitted,'extract' if equal_budget else stage)
                    for test_view in ['body','navigation']:
                        pred=[model.predict(r.body if test_view=='body'
                                            else r.text('raw')) for r in test]
                        truth=[r.domain for r in test]
                        m=lab.metrics(truth,pred)
                        row={'seed':seed,'navigation':mode,'equal_budget':equal_budget,
                             'stage':stage,'test_view':test_view,
                             'n_words':sum(len(lab.tokens(r.body if equal_budget
                                                        else r.text(stage)))
                                           for r in fitted),
                             'n_records':len(fitted),**m}
                        rows.append(row)
                        if mode=='cyclic' and not equal_budget and test_view=='body':
                            correctness[stage]=[a==b for a,b in zip(pred,truth)]
                        if seed==17:
                            prediction_files.append({k:v for k,v in row.items()
                                                     if k not in ('matrix','recall')}
                                                    |{'truth':truth,'prediction':pred,
                                                      'ids':[r.id for r in test]})
            # Фильтр имеет независимые fit/validation и итоговый тест модели.
            fit,validation=filter_training(seed)
            filt=LogisticFilter().fit(fit).tune(validation)
            source=prepared['dedup']
            accepted=[r for r in source if filt.accept(r)]
            model=lab.NB().fit(accepted,'extract')
            pred=[model.predict(r.body) for r in test]
            # Независимый контроль фильтра: новые fit/validation семьи.
            _,independent=filter_training(seed+900)
            assert not {r[3] for r in fit+validation} & {r[3] for r in independent}
            tp=fp=tn=fn=0
            for domain,text,y,_ in independent:
                a=filt.probability(domain,text)>=filt.threshold
                tp+=int(a and y);fp+=int(a and not y)
                tn+=int(not a and not y);fn+=int(not a and y)
            filter_results.append({'seed':seed,'navigation':mode,
                'threshold':filt.threshold,'n_accepted':len(accepted),
                'filter_tp':tp,'filter_fp':fp,'filter_tn':tn,'filter_fn':fn,
                'model':lab.metrics([r.domain for r in test],pred)})
        if seed==17:
            paired=transitions(correctness['raw'],correctness['quality'])
            paired_ci=lab.bootstrap_difference(correctness['raw'],correctness['quality'])
    summaries=[]
    for mode in ['cyclic','random']:
        for budget in [False,True]:
            for test_view in ['body','navigation']:
                for stage in lab.STAGES:
                    group=[r for r in rows if r['navigation']==mode and
                           r['equal_budget']==budget and r['test_view']==test_view
                           and r['stage']==stage]
                    summaries.append({'navigation':mode,'equal_budget':budget,
                        'test_view':test_view,'stage':stage,
                        'accuracy':statistics.mean(r['accuracy'] for r in group),
                        'sd':statistics.stdev(r['accuracy'] for r in group),
                        'macro_f1':statistics.mean(r['macro_f1'] for r in group),
                        'words':statistics.mean(r['n_words'] for r in group)})
    result={'runs':rows,'summaries':summaries,'filters':filter_results,
            'paired_seed17':paired,'paired_ci_seed17':paired_ci,
            'predictions_seed17':prediction_files,
            'design':{'synthetic':True,'seeds':lab.SEEDS,'filter_fit_n':600,
                      'filter_validation_n':300,'filter_independent_n':300,
                      'word_unit':'regex Cyrillic token as in corpus_lab'}}
    (out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('control_results'))
    args=parser.parse_args()
    report=run(args.out)
    print(json.dumps(report['summaries'],ensure_ascii=False,indent=2))
    print('PAIRED',report['paired_seed17'],report['paired_ci_seed17'])
