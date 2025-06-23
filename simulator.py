
#!/usr/bin/env python3
"""
Distributed Semantic Cache – final tuned demo
• 250 k queries, 35 % duplicates
• 3-slot caches, 0.5-s Bloom sync → DSC » IC
author : Hardik Ruparel · May 2025
"""

import simpy, numpy as np, logging
from faker import Faker
from pybloom_live import ScalableBloomFilter
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from tabulate import tabulate
from hashlib import blake2b

# ───────── CONFIG ─────────
NUM_NODES   = 4
NUM_QUERIES = 250_000
PRINT_EVERY = 50_000
DUP_RATE    = 0.35           # realistic duplicate share
CACHE_SIZE  = 3              # ← smaller cache to force churn
EMB_DIM     = 128
THRESH      = 0.65
SYNC_INT    = 0.5            # ← Bloom pushed every 0.5 s

# latency components (seconds)
T_EMB, T_LOC, T_NET, T_RET, T_LLM = 0.002, 0.001, 0.005, 0.050, 0.100
INTER_GAP   = 0.02
SIM_END     = NUM_QUERIES * (INTER_GAP + 0.002)

faker = Faker(); rng = np.random.default_rng(42)
logging.basicConfig(format="%(asctime)s  %(message)s",
                    level=logging.INFO, datefmt="%H:%M:%S")

# ───────── helpers ─────────
def fake_q() -> str:
    return faker.sentence(nb_words=8)

def embed(txt: str) -> np.ndarray:
    h = blake2b(txt.encode(), digest_size=8).hexdigest()
    vec = np.random.default_rng(int(h, 16)).random(EMB_DIM)
    return vec / np.linalg.norm(vec)

# ───────── cache ─────────
class SemanticCache:
    def __init__(self, cap: int):
        self.cap = cap
        self.store = []                    # list[(emb, ans)]
    def lookup(self, v):
        if not self.store: return None
        sims = cosine_similarity([v], [e[0] for e in self.store])[0]
        i    = int(np.argmax(sims))
        return self.store[i][1] if sims[i] >= THRESH else None
    def insert(self, v, ans):
        if len(self.store) >= self.cap:
            self.store.pop(0)
        self.store.append((v, ans))

# ───────── node ─────────
class Node:
    def __init__(self, env: simpy.Environment, name: str):
        self.e = env; self.n = name
        self.cache = SemanticCache(CACHE_SIZE)
        self.bloom = ScalableBloomFilter(initial_capacity=CACHE_SIZE,
                                         error_rate=0.01)
        self.peers = []
        # metrics
        self.h_loc = self.h_rem = self.mis = 0
        self.lat   = 0.0
        self.rc = self.fp = 0
        self.syncs = 0
        env.process(self.sync())
    def sync(self):
        while True:
            yield self.e.timeout(SYNC_INT)
            self.syncs += 1                # ≈1 kB each push
    def handle(self, qid: int, txt: str):
        st = self.e.now
        yield self.e.timeout(T_EMB)
        v = embed(txt)

        ans = self.cache.lookup(v)
        if ans:
            self.h_loc += 1
            yield self.e.timeout(T_LOC)
        else:
            found = False
            for p in self.peers:
                if str(hash(v.tobytes())) in p.bloom:
                    self.rc += 1
                    yield self.e.timeout(2 * T_NET)
                    a = p.cache.lookup(v)
                    if a:
                        found = True
                        self.h_rem += 1
                        self.cache.insert(v, a)
                        break
                    else:
                        self.fp += 1
            if not found:
                self.mis += 1
                yield self.e.timeout(T_RET + T_LLM)
                ans = f"ans-{qid}"
                self.cache.insert(v, ans)

            # advertise key after insertion → FPR ≈ 1 %
            self.bloom.add(str(hash(v.tobytes())))
        self.lat += self.e.now - st

# ───────── simulation ─────────
def run(cfg: str) -> pd.DataFrame:
    env = simpy.Environment()
    nodes = [Node(env, f"N{i}") for i in range(NUM_NODES)]
    if cfg == "DSC":
        for n in nodes: n.peers = [p for p in nodes if p is not n]

    exact = {}
    dup_pool = [fake_q() for _ in range(int(NUM_QUERIES * DUP_RATE))]
    rr_ptr = 0                             # round-robin pointer

    def workload(env):
        nonlocal rr_ptr
        for i in range(NUM_QUERIES):
            if i % PRINT_EVERY == 0 and i > 0:
                logging.info(f"{cfg}: generated {i} queries")

            dup  = rng.random() < DUP_RATE
            text = rng.choice(dup_pool) if dup else fake_q()
            node = nodes[rr_ptr % NUM_NODES] if dup else rng.choice(nodes)
            if dup: rr_ptr += 1

            if cfg == "CEC":
                if text in exact:
                    node.h_loc += 1; node.lat += T_LOC
                else:
                    yield env.timeout(T_EMB + T_RET + T_LLM)
                    exact[text] = "ans"
                    node.mis  += 1; node.lat += T_EMB + T_RET + T_LLM
            else:
                env.process(node.handle(i, text))

            yield env.timeout(INTER_GAP)

    env.process(workload(env))
    env.run(until=SIM_END)

    rows = []
    for n in nodes:
        total = n.h_loc + n.h_rem + n.mis
        if total == 0: continue
        rows.append(dict(
            Config        = cfg,
            Node          = n.n,
            Queries       = total,
            HitRate       = round(100 * (n.h_loc + n.h_rem) / total, 2),
            Latency_ms    = round(1000 * n.lat / total, 2),
            RemoteChecks  = n.rc,
            FalsePosPct   = round(100 * n.fp / n.rc, 2) if n.rc else 0,
            BloomKB_hour  = round(n.syncs * (3600 / SIM_END), 1)
        ))
    return pd.DataFrame(rows)

def main():
    dfs = [run(c) for c in ("CEC", "IC", "DSC")]
    all_df = pd.concat(dfs, ignore_index=True)
    print(tabulate(all_df, headers="keys", tablefmt="github"))

    summary = all_df.groupby("Config").agg(
        Queries      = ('Queries', 'sum'),
        HitRate      = ('HitRate', 'mean'),
        Lat_ms       = ('Latency_ms', 'mean'),
        RemoteChecks = ('RemoteChecks', 'sum'),
        FalsePos     = ('FalsePosPct', 'mean')
    ).reset_index()
    print("\n=== Summary ===")
    print(tabulate(summary, headers="keys", tablefmt="github"))

if __name__ == "__main__":
    main()

