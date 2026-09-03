# ResearchPilot 初版测评集

版本：`v0.1`

本测评集基于 `data/documents/` 中已经索引的 6 篇 HRRP 文献人工整理，共 24 个案例：

- `qa_cases.jsonl`：18 个文献问答案例。
- `experiment_cases.jsonl`：6 个实验分析与跨文献比较案例。

旧的 `questions.jsonl` 存在一行非法 JSON、重复 `case_id` 和金标准过于宽松的问题，已由上述两份文件取代。

## 文献 ID

| document_id | 文献 |
|---|---|
| `11919883312919513682` | 基于HRRP特征统计特性的雷达目标识别方法 |
| `16541814952559140607` | 基于注意力机制和双向GRU模型的雷达HRRP目标识别 |
| `18059459046969102268` | 采用CNN-SSD的雷达HRRP小样本目标识别方法 |
| `534399252737282530` | 采用双向LSTM模型的雷达HRRP目标识别 |
| `8466653773977333372` | 仿真数据辅助的雷达HRRP小样本目标识别方法 |
| `9220639651122286494` | 基于多球体空间拓扑约束的雷达目标HRRP少样本开集识别方法 |

## 标签含义

- `scope`：`single_document` 或 `multi_document`。
- `answerability`：
  - `full`：当前文献能够完整回答。
  - `partial`：可以回答部分事实，但不能完成排名、外推或总体判断。
  - `none`：当前文献没有回答问题所需的证据。
- `expected_sufficient_evidence`：与当前服务的布尔字段对应。`full` 为 `true`，`partial` 和 `none` 为 `false`。
- `gold_points`：正确回答必须覆盖的原子事实，不要求逐字匹配。
- `gold_evidence`：事实对应的稳定定位信息。不要把运行时动态生成的 `evidence_id` 写入测评集。
- `acceptable_pages`：PDF 文件从 1 开始计算的物理页码，与系统返回的 `page_start/page_end` 一致。
- `anchor_terms`：用于辅助判断召回片段是否命中目标证据；匹配时应忽略空白、全角半角和大小写差异。
- `expected_limitations`：回答必须主动说明的证据边界。
- `allowed_context`：原问题无法回答时，仍可以补充的相关已知事实。
- `forbidden_claims`：出现这些含义时，应视为无依据结论。

## 案例分布

| 范围与可回答性 | 数量 |
|---|---:|
| 单篇文献、完整可回答 | 8 |
| 跨文献、完整可回答 | 8 |
| 单篇文献、证据不足 | 3 |
| 跨文献、部分可回答 | 4 |
| 跨文献、证据不足 | 1 |

## 使用规则

1. `/api/v1/qa/ask` 使用 `qa_cases.jsonl` 中的 `question`、`document_ids`。
2. `/api/v1/analysis/experiments` 使用 `experiment_cases.jsonl` 中的 `focus`、`document_ids`。
3. 固定模型、提示词版本、`top_k` 等参数后再比较多次运行结果。
4. 先评估召回和引用，再评估回答内容，避免把“没有召回”误判成“模型不会回答”。
5. 对 `partial` 案例，理想结果是回答有证据的部分，同时拒绝无依据排名或外推。
6. 对 `none` 案例，理想措辞是“当前文献或当前证据未提供”，而不是断言“论文绝对没有”。

