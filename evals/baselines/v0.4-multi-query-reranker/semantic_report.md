# ResearchPilot 批量语义测评报告

## 汇总

- 已评测案例：24/24
- 语义通过率：91.7%
- 标准事实覆盖率：95.9%
- 必要限制覆盖率：100.0%
- 禁止结论安全率：100.0%
- Judge 失败数：0

## 按可回答性划分

| 类型 | 案例数 | 通过数 | 通过率 |
|---|---:|---:|---:|
| 完整可回答 | 16 | 15 | 93.8% |
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
| exp_cross_closed_vs_open_001 | full | PASS | 100.0% | 100.0% | 100.0% |
| exp_cross_simulation_roles_001 | full | PASS | 100.0% | 100.0% | 100.0% |
| exp_partial_accuracy_efficiency_001 | partial | FAIL | 75.0% | 100.0% | 100.0% |
| exp_partial_noise_robustness_001 | partial | PASS | 100.0% | 100.0% | 100.0% |

## 失败诊断

- `qa_cross_architecture_rnn_001`：未覆盖 p1：答案未提及两者都利用前向与后向时序相关性，也未提及两者都处理平移敏感性；仅分别描述各自方法，未表达共同点。；未覆盖 p4：答案仅说明两篇论文均使用实测数据，但未提及安-26、奖状、雅克-42、7800训练样本、5124测试样本或256维距离像等具体设置。
- `exp_partial_accuracy_efficiency_001`：未覆盖 p3：答案未提及仿真数据辅助论文报告的26.9 ms训练时间和0.359 ms推理时间，也未说明与CNN-SSD时间统计口径不同。
