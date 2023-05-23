import json
import fire
from collections import defaultdict
def get_gen_qa(data, mode):
    res = [l.strip() for l in open(f"output/{data}__{mode}/pred.txt").readlines()]
    test = [json.loads(s) for s in open(f"data/{data}/test.json").readlines()]
    q_correct = defaultdict(int)
    q_total = defaultdict(int)
    q_log = defaultdict(list)
    for r, t in zip(res, test):
        if r in [rr.strip() for rr in t['answers']['text']]:
            q_correct[t["question"]] += 1
        q_total[t["question"]] += 1
        q_log[t["question"]].append(r)
        
    q_correct["Total"] = sum(q_correct.values())
    q_total["Total"] = sum(q_total.values())
    accu = { q : round(q_correct[q] / q_total[q], 3) for q in q_total}
    open(f"output/{data}__{mode}/score.json", "w").write(json.dumps(accu))

if __name__ == "__main__":
    fire.Fire(get_gen_qa)
