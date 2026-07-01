# Scripts 使用流程

本项目当前使用 AISHELL-4 做说话人日志实验，主线是 NVIDIA NeMo clustering diarizer：

1. 将 AISHELL-4 8 通道音频拆成单通道。
2. 对每个通道分别运行 NeMo VAD。
3. 将 8 个通道的 VAD 结果取并集，生成 `external_vad_manifest`。
4. 使用 NeMo 的 speaker embedding + clustering 完成说话人日志。

以下命令默认在 WSL2 Ubuntu 中执行，项目目录为：

```bash
/home/enovo/Speech_Major
```

## 1. 激活环境

```bash
cd ~/Speech_Major
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

检查 CUDA：

```bash
python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 2. 准备 1 条测试 Manifest

用于最终 diarization 的主 manifest，需要包含音频路径、参考 RTTM 和真实说话人数。

```bash
cd ~/Speech_Major

python3 scripts/prepare_aishell4_manifest.py \
  --audio-dir /home/enovo/Speech_Major/data/wavs_mono \
  --rttm-dir /home/enovo/Speech_Major/data/rttm \
  --limit 1 \
  --with-num-speakers \
  --output /home/enovo/Speech_Major/data/manifests/aishell4_test_manifest_one_mono_wsl.json
```

## 3. 抽取 8 个单通道音频

`convert_to_mono.py` 默认可以平均通道，也可以用 `--channel` 指定通道。这里分别抽取 ch0 到 ch7。

```bash
cd ~/Speech_Major

for ch in 0 1 2 3 4 5 6 7; do
  python3 scripts/convert_to_mono.py \
    --input-dir data/wavs \
    --output-dir data/wavs_ch${ch} \
    --channel ${ch}
done
```

## 4. 为每个通道生成测试 Manifest

每个通道只取同一条测试音频。

```bash
cd ~/Speech_Major

for ch in 0 1 2 3 4 5 6 7; do
  python3 scripts/prepare_aishell4_manifest.py \
    --audio-dir /home/enovo/Speech_Major/data/wavs_ch${ch} \
    --rttm-dir /home/enovo/Speech_Major/data/rttm \
    --limit 1 \
    --with-num-speakers \
    --output /home/enovo/Speech_Major/data/manifests/aishell4_test_one_ch${ch}.json
done
```

确认每个 manifest 只有 1 行：

```bash
wc -l data/manifests/aishell4_test_one_ch*.json
```

## 5. 对每个通道只运行 NeMo VAD

`vad_only_infer.py` 只生成 `vad_outputs`，不进行 embedding 和 clustering。

```bash
cd ~/Speech_Major
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/Speech_Major/baseline/NeMo/examples/speaker_tasks/diarization/clustering_diarizer

for ch in 0 1 2 3 4 5 6 7; do
  python3 -u vad_only_infer.py \
    --config-name=diar_infer_aishell4_local \
    diarizer.manifest_filepath=/home/enovo/Speech_Major/data/manifests/aishell4_test_one_ch${ch}.json \
    diarizer.out_dir=/home/enovo/Speech_Major/results/vad_ch${ch} \
    diarizer.vad.model_path=/home/enovo/Speech_Major/models/nemo_cache/vad_multilingual_marblenet/670f425c7f186060b7a7268ba6dfacb2/vad_multilingual_marblenet.nemo \
    diarizer.vad.external_vad_manifest=null \
    batch_size=2 \
    device=cuda
done
```

输出示例：

```text
results/vad_ch0/vad_outputs/L_R003S01C02.txt
results/vad_ch1/vad_outputs/L_R003S01C02.txt
...
results/vad_ch7/vad_outputs/L_R003S01C02.txt
```

## 6. 合并 8 通道 VAD 并集

`merge_vad_union.py` 会读取各通道的 `vad_outputs/*.txt`，对 speech 区间取并集并合并相邻短间隔，输出 NeMo 可用的 `external_vad_manifest`。

```bash
cd ~/Speech_Major
source .venv/bin/activate

python3 scripts/merge_vad_union.py \
  --vad-dirs \
    results/vad_ch0/vad_outputs \
    results/vad_ch1/vad_outputs \
    results/vad_ch2/vad_outputs \
    results/vad_ch3/vad_outputs \
    results/vad_ch4/vad_outputs \
    results/vad_ch5/vad_outputs \
    results/vad_ch6/vad_outputs \
    results/vad_ch7/vad_outputs \
  --audio-dir /home/enovo/Speech_Major/data/wavs_mono \
  --output /home/enovo/Speech_Major/data/manifests/aishell4_external_vad_union_one.json \
  --merge-gap 0.2 \
  --min-duration 0.05
```

说明：

- `--vad-dirs`：8 个通道各自的 VAD 输出目录。
- `--audio-dir`：后续 speaker embedding 使用的单通道音频目录。当前使用平均 mono，也可以换成表现最好的 `data/wavs_chX`。
- `--merge-gap`：两个 speech 段间隔小于该值时合并。
- `--min-duration`：丢弃过短 speech 段。

## 7. 使用外部 VAD 并集运行 NeMo Diarization

配置文件：

```text
baseline/NeMo/examples/speaker_tasks/diarization/conf/inference/diar_infer_aishell4_local.yaml
```

当前关键配置应为：

```yaml
diarizer:
  manifest_filepath: /home/enovo/Speech_Major/data/manifests/aishell4_test_manifest_one_mono_wsl.json
  out_dir: /home/enovo/Speech_Major/results/nemo_cluster_one_external_vad_union_yaml
  oracle_vad: False

  vad:
    model_path: null
    external_vad_manifest: /home/enovo/Speech_Major/data/manifests/aishell4_external_vad_union_one.json
```

运行：

```bash
cd ~/Speech_Major
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/Speech_Major/baseline/NeMo/examples/speaker_tasks/diarization/clustering_diarizer

python3 -u offline_diar_infer.py --config-name=diar_infer_aishell4_local
```

## 8. 查看结果

预测 RTTM：

```bash
ls -lh /home/enovo/Speech_Major/results/nemo_cluster_one_external_vad_union_yaml/pred_rttms
head -5 /home/enovo/Speech_Major/results/nemo_cluster_one_external_vad_union_yaml/pred_rttms/L_R003S01C02.rttm
```

重点关注日志中的指标：

```text
DER
MISS
FA
CER
Spk. Count Acc.
```

当前实验主要目标是降低 `MISS`，验证多通道 VAD 并集是否能缓解 AISHELL-4 远场音频中的 VAD 漏检问题。

## 9. 常用对照实验

### 内置 NeMo VAD

将配置改回：

```yaml
vad:
  model_path: /home/enovo/Speech_Major/models/nemo_cache/vad_multilingual_marblenet/670f425c7f186060b7a7268ba6dfacb2/vad_multilingual_marblenet.nemo
  external_vad_manifest: null
```

### Oracle VAD 上限

```bash
python3 -u offline_diar_infer.py \
  --config-name=diar_infer_aishell4_local \
  diarizer.out_dir=/home/enovo/Speech_Major/results/nemo_cluster_one_oracle_vad \
  diarizer.oracle_vad=True \
  diarizer.vad.model_path=null \
  diarizer.vad.external_vad_manifest=null
```

### 自行估计说话人数

```bash
python3 -u offline_diar_infer.py \
  --config-name=diar_infer_aishell4_local \
  diarizer.clustering.parameters.oracle_num_speakers=False \
  diarizer.clustering.parameters.max_num_speakers=8 \
  diarizer.out_dir=/home/enovo/Speech_Major/results/nemo_cluster_one_predict_spk
```
