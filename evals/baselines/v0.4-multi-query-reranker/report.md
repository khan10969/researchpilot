# ResearchPilot 测评报告：20260903_201027

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
- 平均耗时：7508 ms
- P50 / P95：6980 / 14230 ms

## 案例明细

| case_id | 状态 | HTTP | 充分性 | 文档覆盖 | 页召回 | 引用有效 | 耗时 |
|---|---:|---:|---:|---:|---:|---:|---:|
| qa_single_method_statistical_001 | PASS | 200 | Y | 100.0% | 100.0% | 100.0% | 5919 ms |
| qa_single_result_statistical_001 | PASS | 200 | Y | 100.0% | 100.0% | 100.0% | 2963 ms |
| qa_single_dataset_abigru_001 | PASS | 200 | Y | 100.0% | 100.0% | 100.0% | 3495 ms |
| qa_single_result_abigru_001 | PASS | 200 | Y | 100.0% | 100.0% | 100.0% | 2716 ms |
| qa_single_result_cnnssd_001 | PASS | 200 | Y | 100.0% | 100.0% | 100.0% | 5505 ms |
| qa_single_method_blstm_001 | PASS | 200 | Y | 100.0% | 100.0% | 100.0% | 4658 ms |
| qa_single_method_simassist_001 | PASS | 200 | Y | 100.0% | 100.0% | 100.0% | 4919 ms |
| qa_single_setting_fsosr_001 | PASS | 200 | Y | 100.0% | 100.0% | 100.0% | 4263 ms |
| qa_cross_architecture_rnn_001 | FAIL | 200 | N | 100.0% | 100.0% | 100.0% | 7929 ms |
| qa_cross_fewshot_strategy_001 | PASS | 200 | Y | 100.0% | 100.0% | 100.0% | 9366 ms |
| qa_cross_task_definition_001 | PASS | 200 | Y | 100.0% | 100.0% | 100.0% | 8145 ms |
| qa_cross_open_set_metrics_001 | PASS | 200 | Y | 100.0% | 100.0% | 100.0% | 9588 ms |
| qa_none_hardware_jetson_001 | PASS | 200 | Y | N/A | N/A | 100.0% | 4184 ms |
| qa_none_extrapolation_snr_001 | PASS | 200 | Y | N/A | N/A | 100.0% | 2791 ms |
| qa_none_training_power_abigru_001 | PASS | 200 | Y | N/A | N/A | 100.0% | 2032 ms |
| qa_partial_cross_accuracy_001 | PASS | 200 | Y | 100.0% | 100.0% | 100.0% | 8551 ms |
| qa_partial_cross_noise_rnn_001 | PASS | 200 | Y | 100.0% | 100.0% | 100.0% | 6980 ms |
| qa_none_cross_local_gpu_001 | PASS | 200 | Y | N/A | N/A | 100.0% | 9013 ms |
| exp_cross_rnn_architecture_001 | PASS | 200 | Y | 100.0% | 100.0% | 100.0% | 14230 ms |
| exp_cross_fewshot_strategy_001 | PASS | 200 | Y | 100.0% | 100.0% | 100.0% | 13118 ms |
| exp_cross_closed_vs_open_001 | PASS | 200 | Y | 100.0% | 100.0% | 100.0% | 12554 ms |
| exp_cross_simulation_roles_001 | PASS | 200 | Y | 100.0% | 100.0% | 100.0% | 15869 ms |
| exp_partial_accuracy_efficiency_001 | PASS | 200 | Y | 100.0% | 100.0% | 100.0% | 11336 ms |
| exp_partial_noise_robustness_001 | PASS | 200 | Y | 100.0% | 100.0% | 100.0% | 10068 ms |

## 失败诊断

- `qa_cross_architecture_rnn_001`：证据充分性判断与预期不一致
