import time

def benchmark_models(models, question):
    results = []
    for name, slm in models.items():
        start = time.time()
        ans = slm.run(question)
        latency = time.time() - start

        results.append({
            "model": name,
            "latency": latency,
            "tokens": ans["total_tokens"]
        })

    return sorted(results, key=lambda x: x["latency"])
