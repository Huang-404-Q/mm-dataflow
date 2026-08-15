# mm-dataflow

多模态（图文）数据处理流水线 —— 配置驱动的算子库 + 带真值的算子质量评估 + VLM 微调数据质量对照实验。

> A configuration-driven multimodal data pipeline whose operators are measured,
> not asserted: noise is injected with ground-truth labels, so every operator's
> precision and recall is exactly computable.

**当前状态**：Week 2 进行中 —— 核心框架 + 9 个算子 + 噪声注入 + 算子评估闭环 + 多进程并行执行
已跑通（51 个单测全绿）。Week 3 做微调对照实验。

---

## 为什么这样设计

多数数据清洗项目只能说"我写了 N 个算子"，无法回答"算子好不好"。本项目的做法是：

**噪声由我们自己注入，每条脏数据带 `noise_type` 真值标签** → 算子的精确率/召回率变成可精确计算的数字，
阈值可以从 P/R 曲线上选，而不是拍脑袋定。

这个设计在开发第一天就产生了回报，见下面「已发现的问题」。

---

## 架构

```
raw dataset (clean + labelled noise)
        │
        ▼
┌────────────────────────────────────────────────┐
│ Pipeline Runner  —  YAML 配置驱动               │
│                                                │
│ stage1 rule        分辨率 / 文本质量 / 语种      │  CPU，读文件头，最便宜
│ stage2 rule(pixel) 模糊度 (Laplacian 方差)      │  解码像素
│ stage3 dedup       pHash 去重 (图文对级)        │  全局算子
│ stage4 perception  CLIP 图文相似度              │  GPU，填充共享 embedding 缓存
│ stage5 (Week2)     美学打分 / 语义去重 / OCR     │  复用上面的缓存
└────────────────────┬───────────────────────────┘
                     │
   ┌─────────────────┼──────────────────┐
   ▼                 ▼                  ▼
cleaned.jsonl   annotated.jsonl     report.md
(用于微调)      (含所有分数和丢弃原因)  (每算子丢弃率/耗时/分布)
                     │
                     ▼
              scripts/eval_ops.py
              → 算子 P/R、按噪声类型分解、阈值扫描
```

算子按成本递增排序：每一级都缩小工作集，让最贵的 CLIP 前向只处理存活样本。

---

## 快速开始

无需 GPU、无需下载任何模型，5 分钟跑通全链路：

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 1. 生成合成数据（COCO 13GB 下载期间可先用它验证框架）
.venv/bin/python scripts/make_synthetic.py --n 200 --out-dir data/synthetic

# 2. 注入带真值标签的噪声
.venv/bin/python scripts/inject_noise.py \
    --input data/synthetic/clean.jsonl --image-root data/synthetic/images \
    --output data/synthetic/mixed.jsonl \
    --noise-image-dir data/synthetic/images_noise --scale 0.015

# 3. 跑流水线
.venv/bin/python -m mmdataflow run configs/pipeline_dev.yaml

# 4. 评估算子质量
.venv/bin/python scripts/eval_ops.py --input outputs/dev/annotated.jsonl
```

产物：`outputs/dev/report.md`（流水线报告）、`outputs/dev/op_eval.md`（算子 P/R）。

其他命令：

```bash
.venv/bin/python -m mmdataflow list-ops                       # 列出已注册算子
.venv/bin/python -m mmdataflow run <cfg> --limit 500          # 抽样试跑
.venv/bin/python -m mmdataflow run <cfg> --resume             # 从断点续跑
.venv/bin/python -m mmdataflow run <cfg> --workers auto       # 规则算子多进程
.venv/bin/python -m pytest tests/ -q                          # 51 tests
```

阈值扫描（从 P/R 曲线选阈值，而非拍脑袋）：

```bash
.venv/bin/python scripts/eval_ops.py --input outputs/dev/annotated.jsonl \
    --sweep clip_score --sweep-target mismatch
```

---

## 算子

| 算子 | stage | 目标噪声 | 状态 |
|---|---|---|---|
| `image_resolution_filter` | rule | 低质图像 | ✅ |
| `text_quality_filter` | rule | 乱码/截断/n-gram 循环 | ✅ |
| `lang_id_filter` | rule | 错误语种 | ✅ fastText + 启发式回退 |
| `image_blur_filter` | rule | 模糊图像 | ✅ Laplacian 方差 |
| `phash_dedup` | dedup | 重复/近似重复 | ✅ 分带索引 + 图文对级判定 |
| `clip_score_filter` | perception | 图文错配 | ✅ 代码就绪，待 GPU 验证 |
| `aesthetic_score_filter` | perception | 低美学质量 | ✅ LAION MLP，复用 CLIP embedding |
| `semantic_dedup` | dedup | 语义重复 | ✅ faiss/numpy 双后端 + 并查集 |
| `ocr_density_filter` | perception | 截图/文档扫描 | ✅ PaddleOCR 检测框面积占比 |

新增算子只需继承 `Operator` / `ScoreFilter` 并加 `@register_op("name")`，YAML 里即可引用。
`parallel_safe = True` 的算子会自动进入进程池（见下面「性能」）。

---

## 已发现的问题（评估体系的价值验证）

这三个问题都是在 Week 1 由评估脚本和单测直接暴露的，而不是靠事后猜测：

**1. 阈值不能凭直觉设。** 初始 `min_variance=100` 的模糊度阈值精确率只有 44%，误杀 77 张正常图。
阈值扫描显示合成图上退化图的方差上限是 2.0、正常图下限是 4.2 —— 改成 3.0 后精确率 100%。
（该阈值仅适用于合成数据；真实照片纹理丰富得多，必须在 COCO 上重新扫描。）

**2. 图像级去重会破坏指令数据。** LLaVA-Instruct-150K 同一张 COCO 图对应多条不同对话，
纯图像 pHash 去重会把合法样本当重复删掉；同时它会把「图文错配」样本误判为重复
（错配样本复用了原图），抢走本应由 `clip_score_filter` 判定的样本。
改为**图文对级**去重（pHash 命中 + 文本 Jaccard ≥ 0.9）后，该算子精确率从 52.9% → 100%。

**3. 去重不能用样本级 P/R 衡量。** 一对重复样本坍缩为一个存活者就是正确的，无论存活的是哪一个；
但真值只把副本标为噪声，于是约一半的正确操作被记为「漏检 + 误杀」，把召回率压在 50% 附近。
改用**组级指标**后真实表现是：45 个重复组中 82.2% 正确坍缩为单一存活者，0 例过度删除；
剩余 8 例是缩放+裁剪的近似重复（pHash 汉明距离超阈值），正是 Week 2 `semantic_dedup` 的目标。

已知局限（有单测固定）：pHash 基于灰度 DCT，对颜色不敏感 —— 几何结构相同、仅颜色不同的图会碰撞。
这是廉价首过滤器的可接受代价，语义去重会覆盖这一层。

---

## 性能设计

- **共享 embedding 缓存**：CLIP 图像 embedding 由 `clip_score_filter` 计算一次写入 `ctx.embeddings`，
  美学打分和语义去重直接复用 —— 三次前向变一次。缓存落盘（npz），重跑可跳过整个 CLIP 阶段。
- **算子成本递增排序**：读文件头 → 文本规则 → 解码像素 → GPU 前向 → 全局去重。
- **pHash 分带索引**：25k 样本的朴素两两比较是 3.12 亿次；按鸽巢原理将 64 位哈希切成 4 段
  16 位分带，汉明距离 ≤3 的两个哈希必有一段完全相同，用分带取候选再精确校验。
- **规则算子多进程并行**：`parallel_safe = True` 的算子进入常驻进程池。5500 条样本实测
  规则阶段 **4.29x**（`image_blur_filter` 单算子 4.62x），端到端 6.5s → 4.0s，
  且并行与串行**逐条结果一致**（有单测断言）。
- **断点续跑**：每个算子结束后写 checkpoint，`--resume` 跳过已完成阶段。

两个反直觉的结论都来自实测，细节见 [`docs/benchmarks.md`](docs/benchmarks.md)：

1. **每个算子单独开进程池是负优化。** macOS/Windows 用 spawn，池启动约 150–300ms，
   规则阶段要付 4 次；两个廉价文本算子在冷池下比串行还慢 3 倍多（0.29x）。
   改成整个 run 共用一个常驻池后，规则阶段从 2.89x 提到 4.29x。
2. **小数据集不该开池。** 最初在 349 条样本上测，每个算子并行都比串行慢（合计 0.44x）。
   现在 `should_parallelise()` 在样本数 < 2000 时直接拒绝开池走串行。

```bash
.venv/bin/python -m mmdataflow run <cfg> --workers auto     # cpu_count - 1
.venv/bin/python -m mmdataflow.bench.throughput --config configs/pipeline_bench.yaml \
    --workers 1,2,4,8 --repeat 3                            # 复现上面的表
```

---

## 项目结构

```
mmdataflow/core/      Sample / Context / Operator / Pipeline / Registry / Parallel
mmdataflow/ops/       算子实现，一个算子一个文件
mmdataflow/report/    统计报告生成（JSON + Markdown）
mmdataflow/bench/     吞吐基准（写入文档的数字必须来自这里）
configs/              pipeline_dev.yaml（冒烟）、pipeline_bench.yaml（性能）、pipeline_full.yaml（25k 全量）
scripts/              make_synthetic / prepare_data / inject_noise / eval_ops
docs/                 benchmarks.md
tests/                51 个单测，每算子含正负样例
```

完整的 4 周计划、实验设计与预算见 [`../PROJECT_PLAN.md`](../PROJECT_PLAN.md)。
