# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
STS-{2012,2013,2014,2015,2016} (unsupervised) and
STS-benchmark (supervised) tasks
'''

from __future__ import absolute_import, division, unicode_literals

import os
import io
import numpy as np
import logging

from scipy.stats import spearmanr, pearsonr

from senteval.utils import cosine
from senteval.sick import SICKEval
import torch



class STSEval(object):
    def loadFile(self, fpath):
        self.data = {}
        self.samples = []

        for dataset in self.datasets:
            sent1, sent2 = zip(*[l.split("\t") for l in
                               io.open(fpath + '/STS.input.%s.txt' % dataset,
                                       encoding='utf8').read().splitlines()])
            raw_scores = np.array([x for x in
                                   io.open(fpath + '/STS.gs.%s.txt' % dataset,
                                           encoding='utf8')
                                   .read().splitlines()])
            not_empty_idx = raw_scores != ''

            gs_scores = [float(x) for x in raw_scores[not_empty_idx]]
            sent1 = np.array([s.split() for s in sent1])[not_empty_idx]
            sent2 = np.array([s.split() for s in sent2])[not_empty_idx]
            # sort data by length to minimize padding in batcher
            sorted_data = sorted(zip(sent1, sent2, gs_scores),
                                 key=lambda z: (len(z[0]), len(z[1]), z[2]))
            sent1, sent2, gs_scores = map(list, zip(*sorted_data))

            self.data[dataset] = (sent1, sent2, gs_scores)
            self.samples += sent1 + sent2

    def do_prepare(self, params, prepare):
        if 'similarity' in params:
            self.similarity = params.similarity
        else:  # Default similarity is cosine
            self.similarity = lambda s1, s2: np.nan_to_num(cosine(np.nan_to_num(s1), np.nan_to_num(s2)))
        return prepare(params, self.samples)

    def run(self, params, batcher):
        results = {}
        all_sys_scores = []
        all_gs_scores = []
        for dataset in self.datasets:
            sys_scores = []
            all_enc1 = []
            all_enc2 = []
            input1, input2, gs_scores = self.data[dataset]
            for ii in range(0, len(gs_scores), params.batch_size):
                batch1 = input1[ii:ii + params.batch_size]
                batch2 = input2[ii:ii + params.batch_size]
                # we assume get_batch already throws out the faulty ones
                if len(batch1) == len(batch2) and len(batch1) > 0:
                    enc1 = batcher(params, batch1)
                    enc2 = batcher(params, batch2)
                    all_enc1.append(enc1.detach())
                    all_enc2.append(enc2.detach())
                    for kk in range(enc2.shape[0]):
                        sys_score = self.similarity(enc1[kk], enc2[kk])
                        sys_scores.append(sys_score)
            all_sys_scores.extend(sys_scores)
            all_gs_scores.extend(gs_scores)
            results[dataset] = {'pearson': pearsonr(sys_scores, gs_scores),
                                'spearman': spearmanr(sys_scores, gs_scores),
                                'nsamples': len(sys_scores)}
            logging.debug('%s : pearson = %.4f, spearman = %.4f' %
                          (dataset, results[dataset]['pearson'][0],
                           results[dataset]['spearman'][0]))
            def _norm(x, eps=1e-8): 
                xnorm = torch.linalg.norm(x, dim=-1)
                xnorm = torch.max(xnorm, torch.ones_like(xnorm) * eps)
                return x / xnorm.unsqueeze(dim=-1)
            # from Wang and Isola (with a bit of modification)
            # only consider pairs with gs > 4 (from footnote 3)
            def _lalign(x, y, ok, alpha=2):
                return ((_norm(x) - _norm(y)).norm(dim=1).pow(alpha) * ok).sum() / ok.sum()
            def _lunif(x, t=2):
                sq_pdist = torch.pdist(_norm(x), p=2).pow(2)
                return sq_pdist.mul(-t).exp().mean().log()
            ok = (torch.Tensor(gs_scores) > 4).int()
            align = _lalign(
                torch.cat(all_enc1), 
                torch.cat(all_enc2), 
                ok).item()

            # consider all sentences (from footnote 3)
            unif = _lunif(torch.cat(all_enc1 + all_enc2)).item()
            logging.info(f'align {align}\t\t uniform {unif}')
            results[dataset]['align_loss'] = align
            results[dataset]['uniform_loss'] = unif
        
        weights = [results[dset]['nsamples'] for dset in results.keys()]
        list_prs = np.array([results[dset]['pearson'][0] for
                            dset in results.keys()])
        list_spr = np.array([results[dset]['spearman'][0] for
                            dset in results.keys()])

        avg_pearson = np.average(list_prs)
        avg_spearman = np.average(list_spr)
        wavg_pearson = np.average(list_prs, weights=weights)
        wavg_spearman = np.average(list_spr, weights=weights)
        all_pearson = pearsonr(all_sys_scores, all_gs_scores)
        all_spearman = spearmanr(all_sys_scores, all_gs_scores)


# {'MSRpar': {'pearson': PearsonRResult(statistic=0.6733981475683452, pvalue=3.198762249504683e-100), 'spearman': SignificanceResult(statistic=0.6676468023454749, pvalue=6.093300436222538e-98), 'nsamples': 750, 'align_loss': 0.1699317842721939, 'uniform_loss': -2.3859200477600098}, 
# 'MSRvid': {'pearson': PearsonRResult(statistic=0.8800875588210584, pvalue=3.613911611192667e-244), 'spearman': SignificanceResult(statistic=0.8811380242198552, pvalue=1.6558562323274917e-245), 'nsamples': 750, 'align_loss': 0.202345609664917, 'uniform_loss': -2.2089102268218994}, 
# 'SMTeuroparl': {'pearson': PearsonRResult(statistic=0.5168575555593943, pvalue=1.030109475296398e-32), 'spearman': SignificanceResult(statistic=0.6069878776145674, pvalue=1.5226329413043184e-47), 'nsamples': 459, 'align_loss': 0.21498706936836243, 'uniform_loss': -2.022261619567871}, 
# 'surprise.OnWN': {'pearson': PearsonRResult(statistic=0.7621589823887025, pvalue=2.1601841431401603e-143), 'spearman': SignificanceResult(statistic=0.715843588172194, pvalue=8.602509932350012e-119), 'nsamples': 750, 'align_loss': 0.21920640766620636, 'uniform_loss': -2.216658592224121}, 
# 'surprise.SMTnews': {'pearson': PearsonRResult(statistic=0.7269180252389671, pvalue=8.758387278252232e-67), 'spearman': SignificanceResult(statistic=0.62866469215841, pvalue=2.8144316628312215e-45), 'nsamples': 399, 'align_loss': 0.20132693648338318, 'uniform_loss': -2.1057519912719727}, 
# 'all': {'pearson': {'all': 0.8215141884315961, 'mean': 0.7118840539152934, 'wmean': 0.728446404972873}, 'spearman': {'all': 0.7355029993736851, 'mean': 0.7000561969021003, 'wmean': 0.7168326766568324}}}

# {'FNWN': {'pearson': PearsonRResult(statistic=0.6477520780115289, pvalue=7.254800080786047e-24), 'spearman': SignificanceResult(statistic=0.648114567737392, pvalue=6.722205036720558e-24), 'nsamples': 189, 'align_loss': nan, 'uniform_loss': -2.0370664596557617}, 
# 'headlines': {'pearson': PearsonRResult(statistic=0.8210644669505425, pvalue=2.6325851206603163e-184), 'spearman': SignificanceResult(statistic=0.8264060147267503, pvalue=9.348696860889902e-189), 'nsamples': 750, 'align_loss': 0.18727648258209229, 'uniform_loss': -2.3574347496032715}, 
# 'OnWN': {'pearson': PearsonRResult(statistic=0.8020027319135983, pvalue=3.336786741455608e-127), 'spearman': SignificanceResult(statistic=0.8026957005811624, pvalue=1.3933022623481177e-127), 'nsamples': 561, 'align_loss': 0.23686152696609497, 'uniform_loss': -1.9427512884140015}, 
# 'all': {'pearson': {'all': 0.7954631224813478, 'mean': 0.7569397589585565, 'wmean': 0.7920980170404097}, 'spearman': {'all': 0.8055184887667044, 'mean': 0.7590720943484349, 'wmean': 0.7950736349156414}}}

        # 读取sub-tasks中的对齐损失和均匀性损失
        all_align = 0
        all_uniform = 0
        all_nsamples = 0
        new_align_loss = []
        new_uniform_loss = []
        for task in results:
            nsamples = results[task]['nsamples']
            align_loss = results[task]['align_loss']
            uniform_loss = results[task]['uniform_loss']
            all_align += align_loss * nsamples
            all_uniform += uniform_loss * nsamples
            all_nsamples += nsamples
            new_align_loss.append(all_align/all_nsamples)
            new_uniform_loss.append(all_uniform/all_nsamples)

            
        # results['all'] = {'pearson': {'all': all_pearson[0],
        #                               'mean': avg_pearson,
        #                               'wmean': wavg_pearson},
        #                   'spearman': {'all': all_spearman[0],
        #                                'mean': avg_spearman,
        #                                'wmean': wavg_spearman}}
        results['all'] = {'pearson': {'all': all_pearson[0],
                                      'mean': avg_pearson,
                                      'wmean': wavg_pearson,
                                      "align_loss":new_align_loss[-1],
                                       "uniform_loss":new_uniform_loss[-1]},
                          'spearman': {'all': all_spearman[0],
                                       'mean': avg_spearman,
                                       'wmean': wavg_spearman,
                                       "align_loss":new_align_loss[-1],
                                       "uniform_loss":new_uniform_loss[-1]}}
        logging.debug('ALL : Pearson = %.4f, \
            Spearman = %.4f' % (all_pearson[0], all_spearman[0]))
        logging.debug('ALL (weighted average) : Pearson = %.4f, \
            Spearman = %.4f' % (wavg_pearson, wavg_spearman))
        logging.debug('ALL (average) : Pearson = %.4f, \
            Spearman = %.4f\n' % (avg_pearson, avg_spearman))
        # print(results)
        return results


class STS12Eval(STSEval):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS12 *****\n\n')
        self.seed = seed
        self.datasets = ['MSRpar', 'MSRvid', 'SMTeuroparl',
                         'surprise.OnWN', 'surprise.SMTnews']
        self.loadFile(taskpath)


class STS13Eval(STSEval):
    # STS13 here does not contain the "SMT" subtask due to LICENSE issue
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS13 (-SMT) *****\n\n')
        self.seed = seed
        self.datasets = ['FNWN', 'headlines', 'OnWN']
        self.loadFile(taskpath)


class STS14Eval(STSEval):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS14 *****\n\n')
        self.seed = seed
        self.datasets = ['deft-forum', 'deft-news', 'headlines',
                         'images', 'OnWN', 'tweet-news']
        self.loadFile(taskpath)


class STS15Eval(STSEval):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS15 *****\n\n')
        self.seed = seed
        self.datasets = ['answers-forums', 'answers-students',
                         'belief', 'headlines', 'images']
        self.loadFile(taskpath)


class STS16Eval(STSEval):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS16 *****\n\n')
        self.seed = seed
        self.datasets = ['answer-answer', 'headlines', 'plagiarism',
                         'postediting', 'question-question']
        self.loadFile(taskpath)


class STSBenchmarkEval(STSEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('\n\n***** Transfer task : STSBenchmark*****\n\n')
        self.seed = seed
        self.samples = []
        train = self.loadFile(os.path.join(task_path, 'sts-train.csv'))
        dev = self.loadFile(os.path.join(task_path, 'sts-dev.csv'))
        test = self.loadFile(os.path.join(task_path, 'sts-test.csv'))
        self.datasets = ['train', 'dev', 'test']
        self.data = {'train': train, 'dev': dev, 'test': test}

    def loadFile(self, fpath):
        sick_data = {'X_A': [], 'X_B': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip().split('\t')
                sick_data['X_A'].append(text[5].split())
                sick_data['X_B'].append(text[6].split())
                sick_data['y'].append(text[4])

        sick_data['y'] = [float(s) for s in sick_data['y']]
        self.samples += sick_data['X_A'] + sick_data["X_B"]
        return (sick_data['X_A'], sick_data["X_B"], sick_data['y'])

class STSBenchmarkFinetune(SICKEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('\n\n***** Transfer task : STSBenchmark*****\n\n')
        self.seed = seed
        train = self.loadFile(os.path.join(task_path, 'sts-train.csv'))
        dev = self.loadFile(os.path.join(task_path, 'sts-dev.csv'))
        test = self.loadFile(os.path.join(task_path, 'sts-test.csv'))
        self.sick_data = {'train': train, 'dev': dev, 'test': test}

    def loadFile(self, fpath):
        sick_data = {'X_A': [], 'X_B': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip().split('\t')
                sick_data['X_A'].append(text[5].split())
                sick_data['X_B'].append(text[6].split())
                sick_data['y'].append(text[4])

        sick_data['y'] = [float(s) for s in sick_data['y']]
        return sick_data
        
class SICKRelatednessEval(STSEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('\n\n***** Transfer task : SICKRelatedness*****\n\n')
        self.seed = seed
        self.samples = []
        train = self.loadFile(os.path.join(task_path, 'SICK_train.txt'))
        dev = self.loadFile(os.path.join(task_path, 'SICK_trial.txt'))
        test = self.loadFile(os.path.join(task_path, 'SICK_test_annotated.txt'))
        self.datasets = ['train', 'dev', 'test']
        self.data = {'train': train, 'dev': dev, 'test': test}
    
    def loadFile(self, fpath):
        skipFirstLine = True
        sick_data = {'X_A': [], 'X_B': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                if skipFirstLine:
                    skipFirstLine = False
                else:
                    text = line.strip().split('\t')
                    sick_data['X_A'].append(text[1].split())
                    sick_data['X_B'].append(text[2].split())
                    sick_data['y'].append(text[3])

        sick_data['y'] = [float(s) for s in sick_data['y']]
        self.samples += sick_data['X_A'] + sick_data["X_B"]
        return (sick_data['X_A'], sick_data["X_B"], sick_data['y'])
