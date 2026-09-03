# ResearchPilot 批量语义测评报告

## 汇总

- 已评测案例：24/24
- 语义通过率：87.5%
- 标准事实覆盖率：94.6%
- 必要限制覆盖率：100.0%
- 禁止结论安全率：100.0%
- Judge 失败数：0

## 按可回答性划分

| 类型 | 案例数 | 通过数 | 通过率 |
|---|---:|---:|---:|
| 完整可回答 | 16 | 14 | 87.5% |
| 部分可回答 | 4 | 3 | 75.0% |
| 不可回答 | 4 | 4 | 100.0% |

## 案例明细

| case_id | 类型 | 状态 | 事实覆盖 | 限制覆盖 | 禁止结论安全 |
|---|---|---:|---:|---:|---:|
| qa_single_method_statistical_001 | full | PASS | 100.0% | N/A | 100.0% |
| qa_single_result_statistical_001 | full | PASS | 100.0% | 100.0% | 100.0% |
| qa_single_dataset_abigru_001 | full | PASS | 100.0% | N/A | 100.0% |
| qa_single_result_abigru_001 | full | PASS | 100.0% | 100.0% | 100.0% |
| qa_single_result_cnnssd_001 | full | PASS | 100.0% | N/A | 100.0% |
| qa_single_method_blstm_001 | full | PASS | 100.0% | N/A | 100.0% |
| qa_single_method_simassist_001 | full | PASS | 100.0% | N/A | 100.0% |
| qa_single_setting_fsosr_001 | full | PASS | 100.0% | N/A | 100.0% |
| qa_cross_architecture_rnn_001 | full | FAIL | 50.0% | 100.0% | 100.0% |
| qa_cross_fewshot_strategy_001 | full | PASS | 100.0% | 100.0% | 100.0% |
| qa_cross_task_definition_001 | full | PASS | 100.0% | N/A | 100.0% |
| qa_cross_open_set_metrics_001 | full | PASS | 100.0% | 100.0% | 100.0% |
| qa_none_hardware_jetson_001 | none | PASS | N/A | 100.0% | 100.0% |
| qa_none_extrapolation_snr_001 | none | PASS | N/A | 100.0% | 100.0% |
| qa_none_training_power_abigru_001 | none | PASS | N/A | 100.0% | 100.0% |
| qa_partial_cross_accuracy_001 | partial | PASS | 100.0% | 100.0% | 100.0% |
| qa_partial_cross_noise_rnn_001 | partial | PASS | 100.0% | 100.0% | 100.0% |
| qa_none_cross_local_gpu_001 | none | PASS | N/A | 100.0% | 100.0% |
| exp_cross_rnn_architecture_001 | full | PASS | 100.0% | 100.0% | 100.0% |
| exp_cross_fewshot_strategy_001 | full | PASS | 100.0% | 100.0% | 100.0% |
| exp_cross_closed_vs_open_001 | full | FAIL | 80.0% | 100.0% | 100.0% |
| exp_cross_simulation_roles_001 | full | PASS | 100.0% | 100.0% | 100.0% |
| exp_partial_accuracy_efficiency_001 | partial | FAIL | 75.0% | 100.0% | 100.0% |
| exp_partial_noise_robustness_001 | partial | PASS | 100.0% | 100.0% | 100.0% |

## 失败诊断

- `qa_cross_architecture_rnn_001`：未覆盖 p1：答案未明确表达两者都利用前向与后向时序相关性并处理平移敏感性这一共同点。答案仅分别描述了两方法，未将两者共同点作为整体事实陈述。；未覆盖 p4：答案在limitations中明确表示无法确认两篇论文是否使用相同实测数据设置，未表达使用安-26、奖状、雅克-42及7800/5124/256维等具体事实。
- `exp_cross_closed_vs_open_001`：未覆盖 p4：答案仅提及FSOSR采用ACC和AUROC，并给出5-shot/10-shot下ACC 94.48%、AUROC 95.63%，但未提及FPR95指标，也未报告相对次优方法的具体提升数值（如ACC提升6.17、2.94个百分点等），因此未完整表达p4。
- `exp_partial_accuracy_efficiency_001`：未覆盖 p3：答案未提及仿真数据辅助论文报告的单批次训练26.9 ms和单样本推理0.359 ms，也未说明与CNN-SSD时间统计口径不同。
