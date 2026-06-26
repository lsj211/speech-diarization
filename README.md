# Speech Major Speaker Diarization Project

本项目用于语音信息处理大作业的“说话人日志”方向。当前保留的官方 baseline 是 NVIDIA NeMo 的 speaker diarization 推理相关文件，自己的数据准备、实验脚本和结果请放在外层目录中。

## 目录结构

```text
baseline/NeMo/
  examples/speaker_tasks/diarization/   # NeMo 说话人日志推理脚本和配置
  tutorials/speaker_tasks/              # NeMo 说话人日志教程
  LICENSE                               # NeMo Apache-2.0 许可

data/
  wavs/                                 # 音频数据
  rttm/                                 # 标注 RTTM
  manifests/                            # NeMo manifest jsonl

scripts/                                # 自己写的数据处理、运行、评测脚本
results/                                # 实验输出和截图
docs/                                   # proposal、实验记录、报告素材
models/                                 # 下载的 .nemo/.ckpt 等模型文件
```

## Baseline 入口

聚类式说话人日志 baseline：

```powershell
cd H:\Speech_Major\baseline\NeMo\examples\speaker_tasks\diarization

python clustering_diarizer\offline_diar_infer.py `
  diarizer.manifest_filepath=H:\Speech_Major\data\manifests\test_manifest.json `
  diarizer.out_dir=H:\Speech_Major\results\nemo_cluster_baseline `
  diarizer.vad.model_path=vad_multilingual_marblenet `
  diarizer.speaker_embeddings.model_path=titanet_large `
  diarizer.speaker_embeddings.parameters.save_embeddings=False
```

manifest 示例：

```json
{"audio_filepath": "H:/Speech_Major/data/wavs/example.wav", "offset": 0, "duration": null, "label": "infer", "text": "-", "num_speakers": null, "rttm_filepath": "H:/Speech_Major/data/rttm/example.rttm", "uem_filepath": null}
```

## 当前建议路线

1. 准备 5 到 10 段会议语音和对应 RTTM。
2. 生成 `data/manifests/test_manifest.json`。
3. 先跑 `offline_diar_infer.py` 得到 RTTM 输出和 DER。
4. 对比不同 VAD 阈值、是否使用已知说话人数、不同配置文件的 DER。
5. 把结果表格、时间轴可视化和失败案例写入最终报告。

