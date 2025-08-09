import time, json, statistics as stats
from pathlib import Path
from vec_memory import upsert_note, search
from rag_chain import answer

SEED_PATH = Path("eval_seed.jsonl")  # one JSONL per line: {"q":"...", "expect":["keyword1","keyword2"]}

def load_seed():
    if not SEED_PATH.exists():
        raise SystemExit("Create eval_seed.jsonl with lines like: {\"q\":\"What is my 18–24 month objective?\",\"expect\":[\"equity\",\"frontier AI\"]}")
    return [json.loads(l) for l in SEED_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]

def recall_ok(ctx_docs, expects):
    blob = " ".join(ctx_docs).lower()
    return all(x.lower() in blob for x in expects)

def answer_ok(resp, expects):
    text = resp.lower()
    return all(x.lower() in text for x in expects)

def run():
    cases = load_seed()
    results = []
    for i,c in enumerate(cases,1):
        t0 = time.time()
        # retrieval only
        hits = search(c["q"], k=5)
        ctx_docs = [d for _,d,_ in hits]
        recall = recall_ok(ctx_docs, c["expect"])

        # full RAG
        resp, used = answer(c["q"], k=5)
        passed = answer_ok(resp, c["expect"])
        dt = (time.time()-t0)*1000

        results.append({"q":c["q"], "recall": recall, "answer_ok": passed, "latency_ms": round(dt,1), "used": used})
        print(f"[{i}] recall={recall} answer={passed} {round(dt,1)}ms  :: {c['q']}")

    rrate = sum(r["recall"] for r in results)/len(results)
    arate = sum(r["answer_ok"] for r in results)/len(results)
    lat   = [r["latency_ms"] for r in results]
    summary = {
        "n": len(results),
        "recall_rate": round(rrate,3),
        "answer_rate": round(arate,3),
        "latency_ms_avg": round(stats.mean(lat),1),
        "latency_ms_p95": round(stats.quantiles(lat, n=20)[18],1) if len(lat)>5 else max(lat),
    }
    Path("eval_report.json").write_text(json.dumps({"summary":summary,"results":results}, indent=2), encoding="utf-8")
    print("\nSUMMARY:", json.dumps(summary, indent=2))

if __name__ == "__main__":
    run()
