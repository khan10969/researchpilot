# ResearchPilot 测评报告：20260903_111741

## 汇总

- 案例数：24
- 请求成功率：100.0%
- 案例通过率：58.3%
- 证据充分性准确率：62.5%
- 引用有效率：100.0%
- 文档覆盖率：95.0%
- 标准证据页召回率：93.3%
- 锚点证据召回率：88.3%
- 实验分析结构完整性：100.0%
- 平均耗时：7091 ms
- P50 / P95：4946 / 17150 ms

## 案例明细

| case_id | 状态 | HTTP | 充分性 | 文档覆盖 | 页召回 | 引用有效 | 耗时 |
|---|---:|---:|---:|---:|---:|---:|---:|
| qa_single_method_statistical_001 | PASS | 200 | Y | 100.0% | 100.0% | 100.0% | 5840 ms |
| qa_single_result_statistical_001 | PASS | 200 | Y | 100.0% | 100.0% | 100.0% | 2222 ms |
| qa_single_dataset_abigru_001 | FAIL | 200 | N | 100.0% | 100.0% | 100.0% | 4946 ms |
| qa_single_result_abigru_001 | FAIL | 200 | N | 100.0% | 100.0% | 100.0% | 2307 ms |
| qa_single_result_cnnssd_001 | PASS | 200 | Y | 100.0% | 100.0% | 100.0% | 4578 ms |
| qa_single_method_blstm_001 | PASS | 200 | Y | 100.0% | 100.0% | 100.0% | 7003 ms |
| qa_single_method_simassist_001 | PASS | 200 | Y | 100.0% | 100.0% | 100.0% | 6824 ms |
| qa_single_setting_fsosr_001 | PASS | 200 | Y | 100.0% | 100.0% | 100.0% | 3486 ms |
| qa_cross_architecture_rnn_001 | FAIL | 200 | N | 100.0% | 100.0% | 100.0% | 5149 ms |
| qa_cross_fewshot_strategy_001 | FAIL | 200 | N | 50.0% | 50.0% | 100.0% | 3513 ms |
| qa_cross_task_definition_001 | FAIL | 200 | N | 100.0% | 100.0% | 100.0% | 6864 ms |
| qa_cross_open_set_metrics_001 | FAIL | 200 | N | 100.0% | 66.7% | 100.0% | 5075 ms |
| qa_none_hardware_jetson_001 | PASS | 200 | Y | N/A | N/A | 100.0% | 3509 ms |
| qa_none_extrapolation_snr_001 | PASS | 200 | Y | N/A | N/A | 100.0% | 2950 ms |
| qa_none_training_power_abigru_001 | PASS | 200 | Y | N/A | N/A | 100.0% | 1834 ms |
| qa_partial_cross_accuracy_001 | FAIL | 200 | Y | 50.0% | 50.0% | 100.0% | 4288 ms |
| qa_partial_cross_noise_rnn_001 | PASS | 200 | Y | 100.0% | 100.0% | 100.0% | 3769 ms |
| qa_none_cross_local_gpu_001 | PASS | 200 | Y | N/A | N/A | 100.0% | 3820 ms |
| exp_cross_rnn_architecture_001 | FAIL | 200 | N | 100.0% | 100.0% | 100.0% | 16083 ms |
| exp_cross_fewshot_strategy_001 | PASS | 200 | Y | 100.0% | 100.0% | 100.0% | 17150 ms |
| exp_cross_closed_vs_open_001 | FAIL | 200 | N | 100.0% | 100.0% | 100.0% | 15833 ms |
| exp_cross_simulation_roles_001 | FAIL | 200 | N | 100.0% | 100.0% | 100.0% | 18583 ms |
| exp_partial_accuracy_efficiency_001 | PASS | 200 | Y | 100.0% | 100.0% | 100.0% | 11174 ms |
| exp_partial_noise_robustness_001 | PASS | 200 | Y | 100.0% | 100.0% | 100.0% | 13381 ms |

## 失败诊断

- `qa_single_dataset_abigru_001`：证据充分性判断与预期不一致
- `qa_single_result_abigru_001`：证据充分性判断与预期不一致
- `qa_cross_architecture_rnn_001`：证据充分性判断与预期不一致
- `qa_cross_fewshot_strategy_001`：证据充分性判断与预期不一致；缺少文档 ['18059459046969102268']；未命中 gold_evidence 序号 [1]
- `qa_cross_task_definition_001`：证据充分性判断与预期不一致
- `qa_cross_open_set_metrics_001`：证据充分性判断与预期不一致；未命中 gold_evidence 序号 [2]
- `qa_partial_cross_accuracy_001`：缺少文档 ['18059459046969102268']；未命中 gold_evidence 序号 [1]
- `exp_cross_rnn_architecture_001`：证据充分性判断与预期不一致
- `exp_cross_closed_vs_open_001`：证据充分性判断与预期不一致
- `exp_cross_simulation_roles_001`：证据充分性判断与预期不一致
