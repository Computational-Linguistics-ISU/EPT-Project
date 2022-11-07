import json, os
import numpy as np
from summ_eval.bleu_metric import BleuMetric
from summ_eval.cider_metric import CiderMetric
from summ_eval.s3_metric import S3Metric    # Use sklearn 0.21.X
from summ_eval.meteor_metric import MeteorMetric, enc
from summ_eval.bert_score_metric import BertScoreMetric
from summ_eval.mover_score_metric import MoverScoreMetric

def load_sum(path):
    with open(path, "r", encoding="UTF-8", errors='ignore') as f:
        txt = ""
        for line in f:
            if "<End Header>" in line:
                break
        for line in f:
            txt += line
    return txt.strip()

def load_ref(path):
    with open(path, "r", encoding="UTF-8") as f:
        txt = f.read()
    return txt.strip()

def main():
    # Fix multi reference for BertScoreMetric & S3Metric
    # Fix model repeatly loading for BertScoreMetric
    WORKERS = 6
    scorers = [CiderMetric(), BleuMetric(n_workers=WORKERS), S3Metric(n_workers=WORKERS), MeteorMetric(), BertScoreMetric(), MoverScoreMetric(version=2)]

    prompts = ["AA", "AB", "TT"]
    doc_base = "Docs"

    for prompt in prompts:
        doc_folder = prompt + " docs"
        ref1 = load_ref(os.path.join(doc_base, doc_folder, "ref1.txt"))
        ref2 = load_ref(os.path.join(doc_base, doc_folder, "ref2.txt"))

        sum_folder = os.path.join("ISU EPT Release 2.2 (January 2018)", "2. Untagged spellchecked", "Split - 1 essay per file", "Essay 1 - Read-Summarize")
        sum_files = os.listdir(sum_folder)
        
        placement2score = {"P": 3, "B": 2, "C": 2, "D": 1}
        Hyps = []
        scores = []
        for file in sum_files:
            filename, ext = file.split('.')
            if not "_" in filename: continue
            placement, stu_level, stu_id, task_type, doc_letter, semester = filename.split('_')
            if doc_letter != prompt: continue

            sum_text = load_sum(os.path.join(sum_folder, file))
            sum_info = {"filename": file, "text": sum_text, "score": placement2score[placement]}
            
            scores.append(sum_info["score"])
            Hyps.append(sum_info["text"])
        Refs = [[ref1, ref2]] * len(Hyps)

        ## Metric evaluation
        prediction_folder = os.path.join("pred_ref", prompt)
        if not os.path.exists(prediction_folder):
            os.makedirs(prediction_folder)
        
        metric_files = os.listdir(prediction_folder)
        metric_scores = {}
        if len(metric_files) == 0:
            for scorer in scorers:
                mscores = scorer.evaluate_batch(Hyps, Refs, aggregate=False)
                scorer_names = list(mscores[0].keys())
                for scorer_name in scorer_names:
                    with open(os.path.join(prediction_folder, "metric_"+scorer_name+".tsv"), "w", encoding="utf-8") as f:
                        for score in mscores:
                            f.write(str(score[scorer_name])+"\n")
                            metric_scores.setdefault(scorer_name, []).append(score[scorer_name])
        else:
            for tsv in metric_files:
                metric_scores[tsv] = []
                with open(os.path.join(prediction_folder, tsv)) as f:
                    for line in f:
                        metric_scores[tsv].append(float(line))
        
        print(prompt)
        from scipy.stats.stats import pearsonr, spearmanr, kendalltau
        for metric in metric_scores:
            print(metric)
            corr = spearmanr(metric_scores[metric], scores)
            print(corr)

if __name__ == '__main__':
    main()