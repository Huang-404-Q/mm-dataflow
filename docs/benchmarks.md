# 吞吐基准

所有数字由 `mmdataflow/bench/throughput.py` 实测产出，不是估算。复现方式：

```bash
python scripts/make_synthetic.py --n 4000 --out-dir data/bench
python scripts/inject_noise.py --input data/bench/clean.jsonl --image-root data/bench/images \
    --output data/bench/mixed.jsonl --noise-image-dir data/bench/images_noise --scale 0.15
python -m mmdataflow.bench.throughput --config configs/pipeline_bench.yaml \
    --workers 1,2,4,8 --repeat 3
```

测试环境：macOS (Darwin 25.6)，10 核，Python 3.9，CPU-only。数据集 5500 条合成样本
（4000 干净 + 1500 注入噪声）。每格取 3 次的中位数。

---

## 1. 规则算子多进程并行

进程池常驻整个 run（下节解释为什么）：

| operator | 1w (samples/s) | 2w | 4w | 8w | speedup (8w) |
|---|---|---|---|---|---|
| `image_resolution_filter` | 16,897 | 28,081 | 44,318 | 48,416 | 2.87x |
| `text_quality_filter` | 199,996 | 273,332 | 404,413 | 410,450 | 2.05x |
| `lang_id_filter` | 194,218 | 259,656 | 403,586 | 404,487 | 2.08x |
| `image_blur_filter` | 1,789 | 3,267 | 5,903 | 8,262 | **4.62x** |
| **规则阶段合计** | 1,592 | 2,864 | 5,078 | 6,822 | **4.29x** |

端到端（`python -m mmdataflow run configs/pipeline_bench.yaml --workers N`）：

| | 规则阶段 | phash 去重（全局算子，不并行） | 总计 |
|---|---|---|---|
| `--workers 1` | 3.5s | 2.4s | 6.5s |
| `--workers 8` | 0.9s | 2.4s | 4.0s |

**加速比受限于算子本身做了多少 Python 层的活。** `image_blur_filter` 要解码 + 做
Laplacian 卷积，接近 CPU 密集，拿到 4.62x；`image_resolution_filter` 只读图像文件头，
是 I/O 密集，2.87x 就到顶；两个纯文本算子本身单核就有 20 万条/秒，跨进程传输样本的
开销占了相当比例，2x 左右。

并行结果与串行**逐条一致**（5500 条样本的 `(id, keep, drop_reason)` 完全相同，
`tests/test_parallel.py` 对此有断言）。这一条是硬要求：下游 A/B/C 微调对照实验
默认「清洗后的数据集只是配置的函数」，如果并行会改变结果，整个对照就不成立。

---

## 2. 两个由实测（而非直觉）决定的设计

### 2.1 进程池常驻整个 run，而不是每个算子一个

第一版给每个算子单独开池。macOS/Windows 用 spawn 启动子进程，每个池要付
~150–300ms 的解释器启动成本，一个规则阶段就付了 4 次。实测对比（`--cold-pool`
把启动成本摊到每格）：

| operator | 1w | 8w（每算子新建池） | 8w（常驻池） |
|---|---|---|---|
| `image_resolution_filter` | 17,103 | 23,717 (1.39x) | 48,416 (2.87x) |
| `text_quality_filter` | 191,095 | 55,932 (**0.29x**) | 410,450 (2.05x) |
| `lang_id_filter` | 195,656 | 55,530 (**0.28x**) | 404,487 (2.08x) |
| `image_blur_filter` | 1,695 | 6,666 (3.93x) | 8,262 (4.62x) |
| **规则阶段合计** | 1,518 | 4,385 (2.89x) | 6,822 (**4.29x**) |

两个廉价的文本算子在冷池下比串行还**慢 3 倍多** —— 它们本身耗时不到 30ms，
完全被池启动淹没。改成常驻池后规则阶段整体从 2.89x 提到 4.29x。

### 2.2 低于 2000 条不开池

最早的基准跑在 349 条样本上，结果是**每一个算子并行都比串行慢**（规则阶段 0.44x）。
与其悄悄发一个负优化，`should_parallelise()` 直接在样本数低于
`MIN_SAMPLES_FOR_POOL = 2000` 时拒绝开池并走串行。这也是为什么 `configs/pipeline_dev.yaml`
的冒烟测试始终是串行的 —— 那个规模下串行本来就更快。

---

## 3. 共享 embedding 缓存

`clip_score_filter` 计算一次图像 embedding 写入 `ctx.embeddings`，
`aesthetic_score_filter` 与 `semantic_dedup` 直接复用，三次 CLIP 前向变一次。

> 该项待 GPU 环境实测后补入具体数字。当前仅有结构保证：
> `tests/test_ops_week2.py::test_scores_come_from_the_shared_embedding_cache`
> 断言美学算子运行后 `ctx.embeddings.misses == 0`，即没有发生任何重新编码。

---

## 4. 已知未优化项

- **`phash_dedup` 是全局算子，不参与并行**，5500 条耗时 2.4s，现在是端到端的
  最大单项。它需要跨样本状态（分带索引），拆批会漏掉落在不同批次的重复对。
  可优化方向：分带索引的构建本身可以并行，合并阶段串行。
- 8w 相对 4w 的边际收益已经很小（规则阶段 5,078 → 6,822），10 核机器上
  `default_workers()` 取 9 是合理默认。
