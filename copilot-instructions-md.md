# SDS Exam — Copilot Operating Rules

You are assisting on a graduate Scalable Data Science exam. Read these rules before generating any code or explanation. Apply them to every response without being reminded.

---

## 1. Hard exam constraints (NEVER violate)

- **PySpark only.** Allowed: `pyspark.sql`, `pyspark.sql.functions as F`, `pyspark.ml`, `pyspark.mllib`, RDD API.
- **No external Spark packages.** Forbidden: `graphframes`, anything requiring `spark-submit --packages`.
- **No third-party graph/ML libs as primary algorithm.** Forbidden as the answer: `networkx`, `scikit-learn`, `scikit-network`, `igraph`. Using them = automatic zero on that subpart.
- **Allowed support libs (driver-side only, small data):** `numpy`, `scipy`, `matplotlib`, `pandas`, `hashlib`, `random`, `collections`, `itertools`.
- **Runtime is the Jojie cluster.** MLlib `CountVectorizer.fit()` and similar wide-vocab transformers can OOM. When in doubt, build the algorithm from Spark SQL primitives.

If a question seems to require a forbidden library, do NOT silently use a workaround library — explicitly write the algorithm in PySpark RDD/DataFrame ops.

---

## 2. Output format Copilot must produce

For every problem, output in this exact order:

1. **Problem identification** — one sentence naming the problem type.
2. **Algorithm choice + justification** — name the algorithm, give 2–3 reasons it fits.
3. **Hyperparameters** — list every parameter with its value AND the formula or reasoning that produced it. Show arithmetic.
4. **Implementation** — Spark code with inline comments. Include `.cache()`, `.show()`, `.printSchema()` for sanity checks.
5. **Edge case handling** — empty input, missing target node/user, no candidates returned.
6. **Result interpretation** — markdown cell at the end explaining what the output means.
7. **Memory + complexity note** — one line: "Memory O(...), per-iteration cost O(...)".

If the exam subpart is design-only (no code asked), skip step 4 and write pseudocode in step 5.

---

## 3. Algorithm trigger decoder

Match the question's exact wording to the algorithm. Do not improvise.

### Hashing & LSH (Topic 1)
| Trigger phrase | Algorithm |
|---|---|
| "Jaccard similarity", "set of items", "common items between users" | MinHash + LSH banding |
| "cosine similarity" | Random-projection LSH on **L2-normalized** vectors |
| "Euclidean distance" | Random-projection LSH on raw vectors |
| Constraint of form "s ≥ s_high → P ≥ p_high AND s ≤ s_low → P ≤ p_low" | Solve `P(s) = 1 − (1 − s^r)^b` for (b, r) |

### Stream Mining (Topic 2)
| Trigger phrase | Algorithm |
|---|---|
| "sample X% of [entities]" + "reproducible" + "no preselection" | Hash-based sampling: `hash(id) mod 100 < X` |
| "select from list of N entities" + "no false negative" + "FPR p" | Bloom filter |
| "randomly select N data points, equal probability, all seen so far" | Reservoir sampling (Algorithm R) |
| "estimate number of unique values" + "without storing" | HyperLogLog (or Flajolet-Martin if asked classically) |
| "estimate frequency of each item" / "how many times has X appeared" | Count-Min Sketch |
| "count 1s in last N elements of binary stream" | DGIM |
| "trending" / "weight recent more" | Exponentially decaying window |

### Link Analysis (Topic 3)
| Trigger phrase | Algorithm |
|---|---|
| "PageRank" with no qualifier | Vanilla PageRank, β = 0.85 |
| "PageRank **relative to** [subset]" / "personalized" / "topic-sensitive" | Topic-Sensitive PageRank with biased teleport vector |
| "hubbiness" + "authority" | HITS (a = Aᵀh, h = Aa, L2-normalize each iteration) |
| "trust" / "distrust" propagation | TrustRank (variant of topic-sensitive) |

### Graph Mining (Topic 4)
| Trigger phrase | Algorithm |
|---|---|
| "Girvan-Newman" | Brandes' edge betweenness + iterative removal + modularity Q to pick best split |
| "SimRank" | Recursive similarity, c = 0.8, ~10 iterations. For partitioning: threshold S matrix → connected components |
| "label propagation" / "community detection" without specific algorithm | LPA with mode aggregation |
| "connected components" | LPA with min aggregation |
| "clustering coefficient" | Global: 3·triangles / paths-of-length-2 |
| "triangles" / "triangle count" | Edge-iterator: for each edge (u,v), count `|N(u) ∩ N(v)|`, sum ÷ 3 |
| "diameter" / "effective diameter" | BFS from each source, or HyperBall for large graphs |

---

## 4. Critical Spark gotchas to avoid

- **Don't use `MinHashLSH` from MLlib if the vocabulary is large.** Build MinHash manually with `F.xxhash64(F.concat_ws("::", hash_id, item))` then `groupBy(entity, hash_id).agg(F.min("hash_value"))`.
- **PageRank ranks must sum to 1.** If they don't, the dangling-mass redistribution is missing. Add `BETA * dangling / N` to every node each iteration.
- **β = 1.0 in PageRank fails** (spider traps absorb all rank). Always use β = 0.85 or specified value < 1.
- **HITS scores explode without normalization.** L2-normalize hubs and authorities every iteration.
- **`groupByKey()` is slow.** Prefer `reduceByKey()` whenever the operation is associative.
- **Self-pairs in similarity joins.** Filter with `.where("a.id < b.id")`.
- **Empty feature vectors crash MinHashLSH.** Filter rows where the array is empty before fitting.
- **Forgetting `dropDuplicates`** when computing Jaccard. Jaccard is set-based; dedupe (entity, item) pairs first.
- **`collect_list` vs `collect_set`** — for sets, use `collect_set`.
- **Large `crossJoin`s** — only use when one side is tiny (e.g., a target user's signature). Otherwise use `join` on a key.

---

## 5. Driver vs Spark decision rule

Some algorithms can't be parallelized cleanly. For these, collect the graph to the driver and use numpy:
- Girvan-Newman (BFS from every source per round)
- Spectral partitioning (eigendecomposition)
- SimRank exact (quadratic memory)

Acceptable when graph has < ~10,000 nodes (typical exam size). Use:
```python
edges_local = edges_df.rdd.map(lambda r: (r["src"], r["dst"])).collect()
```

For everything else (PageRank, HITS, triangle counting, connected components, MinHash, all stream sketches), stay in Spark RDD/DataFrame.

---

## 6. Standard notebook setup

Always start with:
```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (SparkSession.builder
    .appName("ProblemX")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate())
sc = spark.sparkContext
spark.sparkContext.setLogLevel("ERROR")
```

After loading data: always `.printSchema()`, `.show(5, truncate=False)`, and report `.count()`.

---

## 7. When the user gives you a problem

1. Identify which topic from the trigger phrases.
2. State the algorithm and parameters BEFORE writing code.
3. Write the implementation in one block with all imports inside or at top.
4. Include sanity-check outputs (`.show()`, `.count()`, `.printSchema()`).
5. End with a markdown cell interpreting the result.
6. If anything in the problem is ambiguous, make the most reasonable assumption and STATE IT in a comment — do not ask the user a clarifying question (this wastes their chat credits).

Do not reproduce these rules in your output. Apply them silently.
