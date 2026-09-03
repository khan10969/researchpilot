# ResearchPilot 批量语义测评报告

## 汇总

- 已评测案例：23/24
- 语义通过率：78.3%
- 标准事实覆盖率：87.3%
- 必要限制覆盖率：100.0%
- 禁止结论安全率：100.0%
- Judge 失败数：1

## 按可回答性划分

| 类型 | 案例数 | 通过数 | 通过率 |
|---|---:|---:|---:|
| 完整可回答 | 15 | 11 | 73.3% |
| 部分可回答 | 4 | 3 | 75.0% |
| 不可回答 | 4 | 4 | 100.0% |

## 案例明细

| case_id | 类型 | 状态 | 事实覆盖 | 限制覆盖 | 禁止结论安全 |
|---|---|---:|---:|---:|---:|
| qa_single_method_statistical_001 | full | judge_failed | N/A | N/A | N/A |
| qa_single_result_statistical_001 | full | PASS | 100.0% | 100.0% | 100.0% |
| qa_single_dataset_abigru_001 | full | PASS | 100.0% | N/A | 100.0% |
| qa_single_result_abigru_001 | full | PASS | 100.0% | 100.0% | 100.0% |
| qa_single_result_cnnssd_001 | full | PASS | 100.0% | N/A | 100.0% |
| qa_single_method_blstm_001 | full | PASS | 100.0% | N/A | 100.0% |
| qa_single_method_simassist_001 | full | PASS | 100.0% | N/A | 100.0% |
| qa_single_setting_fsosr_001 | full | PASS | 100.0% | N/A | 100.0% |
| qa_cross_architecture_rnn_001 | full | FAIL | 50.0% | 100.0% | 100.0% |
| qa_cross_fewshot_strategy_001 | full | FAIL | 75.0% | 100.0% | 100.0% |
| qa_cross_task_definition_001 | full | PASS | 100.0% | N/A | 100.0% |
| qa_cross_open_set_metrics_001 | full | FAIL | 25.0% | 100.0% | 100.0% |
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
| exp_partial_accuracy_efficiency_001 | partial | FAIL | 50.0% | 100.0% | 100.0% |
| exp_partial_noise_robustness_001 | partial | PASS | 100.0% | 100.0% | 100.0% |

## 失败诊断

- `qa_single_method_statistical_001`：ModelOutputError: 语义评测连续 2 次返回无效的结构化结果。
- `qa_cross_architecture_rnn_001`：未覆盖 p3：答案说明BLSTM提取平移稳健序列特征、使用双向LSTM并以softmax和投票融合，符合主要事实，但gold point要求表述'提取平移稳健序列特征，使用双向LSTM，并结合Softmax和投票策略融合结果'，答案完整表达了各要素。；未覆盖 p4：答案明确表示检索证据未提供数据设置一致性的直接阐述，且未能确认两篇论文采用安-26、奖状、雅克-42以及7800/5124/256等具体设置，未表达该事实。
- `qa_cross_fewshot_strategy_001`：未覆盖 p3：答案只提到均值方差补偿扩充实测域和GAN/PN域类对齐，但未明确表述‘先用’均值方差补偿、‘再用’域对齐和类对齐的先后顺序。
- `qa_cross_open_set_metrics_001`：未覆盖 p1：答案未提及χp方法使用6种标准目标和2种库外目标，也未说明通过ROC上的TPR与FPR阈值选择阈值。；未覆盖 p3：答案提到多球体方法面向少样本开集识别且采用元学习，但未说明是5-way 5/10-shot设置，也未提及查询集中有5个已知类和2个未知类。；未覆盖 p4：答案提到多球体用ACC衡量已知分类和AUROC衡量未知类识别，但未提及FPR95。
- `exp_cross_closed_vs_open_001`：未覆盖 p4：答案仅提及ACC和AUROC指标及定性改进，未报告具体提升数值（如6.17/13.1/19.72等），也未提及FPR95指标。
- `exp_partial_accuracy_efficiency_001`：未覆盖 p2：答案提及仿真数据辅助方法在1/5/10-shot准确率为74.94%、85.78%、90.22%，但未提及SAMPLE车辆标准条件中的69.52%、77.46%、82.74%。；未覆盖 p3：答案中未提及仿真数据辅助方法单批次训练26.9 ms和单样本推理0.359 ms，因此未报告这些数值。
