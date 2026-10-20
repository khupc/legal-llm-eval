# legal-llm-eval · 大模型分类评测设计 Skill

![tests](https://github.com/khupc/legal-llm-eval/actions/workflows/test.yml/badge.svg)

一个 Claude Code Skill，加一套零依赖的 Python 统计脚本，用来设计、执行和审查「让大模型做分类或判断」的评测：法律文本分类、内容审核、风险标注等，单标签和多标签都支持。

很多评测只报一个总分。总分高，不代表每一类都可靠；总分没变，也不代表没出问题。这个工具回答三个问题：

1. **哪里不好**：按类别（多标签就按每个标签）拆开看，找出被系统性少判的类别
2. **为什么**：对最差的类别逐一排除候选解释，差异给出置信区间
3. **该不该让人接手**：根据分歧和无法解析的输出生成人工复核分流规则，并给出「人审率 vs 准确率」的取舍表

**方法跟着评测设计走。** 每个人的标签结构、测试集、比较目的和错误代价都不一样，所以 Skill 的第 0 步是先诊断评测设计，写成一张设计卡，后面每一步按设计卡选择指标和检验，而不是让所有人走同一套固定流程。

## 适用范围

| 情况 | 支持情况 |
|---|---|
| 单标签分类（任意类别数） | 支持 |
| 多标签（每条 0 个、1 个或多个标签） | 支持：每个标签的 P/R/F1、完全匹配率、每条平均标签数、每个标签的 κ、逐标签投票 |
| 层级标签（大类 → 小类） | 支持：`--groups` 同时报告两个层级，如「违规 vs 正常」的漏判率和误判率 |
| 同一来源的多条数据（一个案件多个段落、一个账号多条帖子） | 支持：`--clusters` 按来源整组重抽样，避免置信区间偏窄 |
| 多个维度 | 每个维度拆成一套文件，分别运行 |
| 有序等级、非配对比较、生成类任务 | 不支持，SKILL.md 里写了应该改用什么方法 |

## 三个示例

| 示例 | 任务 | 展示什么 |
|---|---|---|
| [`examples/`](examples/) | 民事判决案由分类（借贷 / 租赁 / 买卖）：单标签、3 类、均衡集 | 完整的 7 步流程 |
| [`examples/governance/`](examples/governance/) | 内容审核：单标签、4 类、自然分布、层级标签 | [报告](examples/governance/report.md)：用漏放率、误杀率、人审率给出结论和分流建议 |
| [`examples/multilabel/`](examples/multilabel/) | 合同条款风险标注：多标签、按合同聚集 | 多标签指标、按合同整组重抽样、逐标签投票分流 |

全部是合成数据。

## 步骤与对应脚本

| 步骤 | 做什么 | 脚本 |
|---|---|---|
| 0 评测设计诊断 | 标签结构、测试集、比较目的、错误代价、结果用途 → 设计卡 | （由 Skill 引导） |
| 1 定义任务与标签 | 边界、判断标准、兜底类、层级 | （由 Skill 引导） |
| 2 构建测试集 | 均衡集或自然分布集，固定种子，记录来源 | （由 Skill 引导） |
| 3 人工标注与一致性 | 原始一致率、Fleiss' κ、Cohen's κ、每个标签的 κ；待仲裁清单 | `agreement.py` |
| 4 冻结配置 | SHA-256 加锁；按冻结规则解析输出（支持多标签） | `freeze.py` `parse_outputs.py` |
| 5 打分与统计 | 按设计卡选指标；McNemar + Holm；类别间或配置间的 Bootstrap 区间；门槛由用户预先设定 | `evaluate.py` `compare.py` |
| 6 失败诊断 | 定位方向，列候选解释，逐一设计检验并排除 | `evaluate.py` `route.py` |
| 7 人工复核分流 | F1 最低且分歧集中的类别转人工；逐级分流的取舍表 | `route.py` |

方法说明见 [SKILL.md](SKILL.md)，报告模板见 [templates/report.md](templates/report.md)。

## 快速开始

需要 Python ≥ 3.8，只用标准库。所有命令都在项目根目录运行（配置里的提示词路径相对于根目录）。

```bash
python examples/make_toy_data.py          # 生成合成示例数据
python scripts/freeze.py lock   config/frozen_config.example.json
python scripts/freeze.py verify config/frozen_config.example.json
python scripts/parse_outputs.py config/frozen_config.example.json examples/raw_outputs.csv examples/parsed_demo.csv
python scripts/agreement.py examples/annotations.csv
python scripts/evaluate.py  examples/gold.csv examples/predictions.csv
python scripts/compare.py   examples/gold.csv examples/predictions.csv --contrast classes --threshold 0.10 --boot 10000 --seed 42
python scripts/route.py     examples/gold.csv examples/members.csv
```

另外两个示例：

```bash
python scripts/evaluate.py examples/governance/gold.csv examples/governance/predictions.csv --groups examples/governance/groups.csv
python scripts/compare.py  examples/multilabel/gold.csv examples/multilabel/predictions.csv --contrast configs --threshold 0.10 --clusters examples/multilabel/clusters.csv
python scripts/route.py    examples/multilabel/gold.csv examples/multilabel/members.csv
```

`parse_outputs.py` 这一步只演示「原始输出 → 标签」的解析，结果写到单独的 `parsed_demo.csv`；不要写到 `predictions.csv`，否则会覆盖后面几步用的合成预测数据。

示例输出（合成数据）：

```
mode: single-label

== few_shot_cot   macro-F1 0.864   unparseable 0/90   missing 0/90
class               P      R     F1  pred share  gold share
lease           0.880  0.733  0.800        0.28        0.33  <- under-predicted
loan            0.875  0.933  0.903        0.36        0.33
sale            0.848  0.933  0.889        0.37        0.33

== Paired configuration comparison (exact McNemar, Holm-corrected)
  few_shot_cot vs zero_shot          only-left  22  only-right   7  p 0.0081  Holm p 0.0081

    F1(lease) - F1(sale) = -0.089  95% CI [-0.192, -0.008]  -> no difference found (not 'equivalent')
```

## 数据格式

| 文件 | 列 | 说明 |
|---|---|---|
| `gold.csv` | item_id, label | 金标准 |
| `annotations.csv` | item_id, annotator, label | 每位标注者一行；空 = 弃权 |
| `raw_outputs.csv` | item_id, config, raw_output | 模型原始输出 |
| `predictions.csv` | item_id, config, label | 解析后的标签；空 = 无法解析；gold 里有但这里没有的条目 = 漏跑，单独报警 |
| `members.csv` | item_id, member, label | 投票成员的标签（多个模型，或同一模型的多次采样） |
| `groups.csv`（可选） | label, group | 小类到大类的映射 |
| `clusters.csv`（可选） | item_id, cluster | 每条数据的来源（案件、合同、账号） |

标签单元格：`a` 表示一个标签，`a|b` 表示多个标签，`none` 表示没有任何标签适用，空着表示无法解析。只要 gold 里有一条不是恰好一个标签，就自动按多标签处理。

## 设计取舍

- **无法解析不算「没答」，单列统计**：它不产生误分类，正确率看不到它，但上线后就是故障。
- **单标签和多标签共用一套代码**：每条的标签在内部统一当成集合处理，单标签就是只有一个元素的集合，所以两种任务的指标定义完全一致。
- **配对 Bootstrap，所有配置用同一次重抽样**：单标签按类别分层，多标签按条目抽，有聚集就按来源整组抽。区间才真正对应「这两者差多少」。
- **门槛没有默认值**：多大差距才算有意义取决于用途，必须在看结果前由用户设定。未达门槛只报「未发现差异」，不报「等效」。
- **冻结靠哈希而不是自觉**：主实验期间提示词改一个字，`verify` 就会报错。

## 测试

```bash
python -m unittest discover -s tests
```

统计函数对照公开参考值校验（如 Fleiss 1971 的示例 κ = 0.210、精确 McNemar 的二项分布值、Holm 逐步校正），每次提交由 GitHub Actions 自动运行。

## 来源

方法来自作者对大模型法律分类任务的评测实践。仓库只含通用方法与合成数据，不含任何真实标注或研究数据。

---

Cheryl
