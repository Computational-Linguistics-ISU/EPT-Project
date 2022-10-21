import os

def load_sum(path):
    with open(path, "r", encoding="UTF-8", errors='ignore') as f:
        txt = ""
        for line in f:
            if "<End Header>" in line:
                break
        for line in f:
            txt += line
    return txt.strip()

def load_doc(path):
    with open(path, "r", encoding="UTF-8") as f:
        txt = f.read()
    return txt.strip()

if __name__ == '__main__':
    doc_folder = "AA docs"
    doc1 = load_doc(os.path.join(doc_folder, "doc1.txt"))
    doc2 = load_doc(os.path.join(doc_folder, "doc2.txt"))
    doc_text = doc1 + " \n" + doc2

    sum_folder = os.path.join("ISU EPT Release 2.2 (January 2018)", "2. Untagged spellchecked", "Split - 1 essay per file", "Essay 1 - Read-Summarize")
    sum_files = os.listdir(sum_folder)
    
    placement2score = {"P": 3, "B": 2, "C": 2, "D": 1}
    sums = []
    scores = []
    for file in sum_files:
        filename, ext = file.split('.')
        if not "_" in filename: continue
        placement, stu_level, stu_id, task_type, doc_letter, semester = filename.split('_')
        if doc_letter != 'AA': continue

        sum_text = load_sum(os.path.join(sum_folder, file))
        sum_info = {"filename": file, "text": sum_text, "score": placement2score[placement]}
        
        scores.append(sum_info["score"])
        sums.append(sum_info["text"])
  
    ## Metric evaluation
    prediction_folder = "predictions"
    metric_files = os.listdir(prediction_folder)
    metric_scores = {}
    if len(metric_files) == 0:
        docs = [doc_text] * len(sums)
        from summ_eval.supert_metric import SupertMetric
        from summ_eval.summa_qa_metric import SummaQAMetric
        from summ_eval.blanc_metric import BlancMetric

        scorers = [SupertMetric(), SummaQAMetric(), BlancMetric(inference_batch_size=32)]
        for scorer in scorers:
            mscores = scorer.evaluate_batch(sums, docs, aggregate=False)
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
    
    from scipy.stats.stats import pearsonr, spearmanr, kendalltau

    for metric in metric_scores:
        print(metric)
        corr = spearmanr(metric_scores[metric], scores)
        print(corr)
        