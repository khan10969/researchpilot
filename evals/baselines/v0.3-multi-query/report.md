# ResearchPilot 测评报告：20260903_163502

## 汇总

- 案例数：24
- 请求成功率：100.0%
- 案例通过率：95.8%
- 证据充分性准确率：95.8%
- 引用有效率：100.0%
- 文档覆盖率：100.0%
- 标准证据页召回率：100.0%
- 锚点证据召回率：98.3%
- 实验分析结构完整性：100.0%
- 平均耗时：7724 ms
- P50 / P95：6724 / 14688 ms

## 案例明细

| case_id | 状态 | HTTP | 充分性 | 文档覆盖 | 页召回 | 引用有效 | 耗时 |
|---|---:|---:|---:|---:|---:|---:|---:|
| qa_single_method_statistical_001 | PASS | 200 | Y | 100.0% | 100.0% | 100.0% | 12943 ms |
| qa_single_result_statistical_001 | PASS | 200 | Y | 100.0% | 100.0% | 100.0% | 2916 ms |
| qa_single_dataset_abigru_001 | PASS | 200 | Y | 100.0% | 100.0% | 100.0% | 2913 ms |
| qa_single_result_abigru_001 | PASS | 200 | Y | 100.0% | 100.0% | 100.0% | 3009 ms |
| qa_single_result_cnnssd_001 | PASS | 200 | Y | 100.0% | 100.0% | 100.0% | 5149 ms |
| qa_single_method_blstm_001 | PASS | 200 | Y | 100.0% | 100.0% | 100.0% | 5715 ms |
| qa_single_method_simassist_001 | PASS | 200 | Y | 100.0% | 100.0% | 100.0% | 7610 ms |
| qa_single_setting_fsosr_001 | PASS | 200 | Y | 100.0% | 100.0% | 100.0% | 3620 ms |
| qa_cross_architecture_rnn_001 | FAIL | 200 | N | 100.0% | 100.0% | 100.0% | 6724 ms |
| qa_cross_fewshot_strategy_001 | PASS | 200 | Y | 100.0% | 100.0% | 100.0% | 7790 ms |
| qa_cross_task_definition_001 | PASS | 200 | Y | 100.0% | 100.0% | 100.0% | 8768 ms |
| qa_cross_open_set_metrics_001 | PASS | 200 | Y | 100.0% | 100.0% | 100.0% | 8281 ms |
| qa_none_hardware_jetson_001 | PASS | 200 | Y | N/A | N/A | 100.0% | 2732 ms |
| qa_none_extrapolation_snr_001 | PASS | 200 | Y | N/A | N/A | 100.0% | 3072 ms |
| qa_none_training_power_abigru_001 | PASS | 200 | Y | N/A | N/A | 100.0% | 2249 ms |
| qa_partial_cross_accuracy_001 | PASS | 200 | Y | 100.0% | 100.0% | 100.0% | 5795 ms |
| qa_partial_cross_noise_rnn_001 | PASS | 200 | Y | 100.0% | 100.0% | 100.0% | 6503 ms |
| qa_none_cross_local_gpu_001 | PASS | 200 | Y | N/A | N/A | 100.0% | 6928 ms |
| exp_cross_rnn_architecture_001 | PASS | 200 | Y | 100.0% | 100.0% | 100.0% | 14688 ms |
| exp_cross_fewshot_strategy_001 | PASS | 200 | Y | 100.0% | 100.0% | 100.0% | 15401 ms |
| exp_cross_closed_vs_open_001 | PASS | 200 | Y | 100.0% | 100.0% | 100.0% | 12965 ms |
| exp_cross_simulation_roles_001 | PASS | 200 | Y | 100.0% | 100.0% | 100.0% | 14600 ms |
| exp_partial_accuracy_efficiency_001 | PASS | 200 | Y | 100.0% | 100.0% | 100.0% | 13494 ms |
| exp_partial_noise_robustness_001 | PASS | 200 | Y | 100.0% | 100.0% | 100.0% | 11518 ms |

## 失败诊断

- `qa_cross_architecture_rnn_001`：证据充分性判断与预期不一致
