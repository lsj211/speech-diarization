# Scripts

放置本项目的数据准备、运行 baseline、评测 DER、画时间轴等脚本。

## 1. 启动环境

每次新开 PowerShell 后，先激活 conda 环境：

```powershell
conda activate speech_diar
```

如果已经在 `baseline/NeMo` 执行过 `pip install -e ".[asr]"`，不需要设置 `PYTHONPATH`。可以用下面命令确认当前环境导入的是项目内源码：

```powershell
python -c "import nemo; print(nemo.__file__)"
```

只有在没有安装 editable 包、只是直接使用 `baseline/NeMo/nemo` 源码时，才需要临时设置：

```powershell
$env:PYTHONPATH="H:\Speech_Major\baseline\NeMo;$env:PYTHONPATH"
```

## 2. 设置模型缓存目录

默认情况下，NeMo 模型会下载到用户目录下的 `.cache`。为了让模型都放在项目目录中，每次运行前设置：

```powershell
mkdir H:\Speech_Major\models\nemo_cache
mkdir H:\Speech_Major\models\hf_cache

$env:NEMO_CACHE_DIR="H:\Speech_Major\models\nemo_cache"
$env:HF_HOME="H:\Speech_Major\models\hf_cache"
```

确认设置：

```powershell
echo $env:NEMO_CACHE_DIR
echo $env:HF_HOME
```

说明：

```text
NEMO_CACHE_DIR  控制 NeMo/NGC 模型缓存，如 vad_multilingual_marblenet、titanet_large
HF_HOME         控制 HuggingFace 模型缓存，如 Sortformer
```

## 3. 生成 AISHELL-4 Manifest

当前默认从 `data/wavs` 和 `data/rttm` 读取 AISHELL-4 test 数据。

如果原始音频是多通道，先转成单通道：

```powershell
cd H:\Speech_Major
pip install soundfile
python scripts\convert_to_mono.py --input-dir data\wavs --output-dir data\wavs_mono
```

生成全量 manifest：

```powershell
cd H:\Speech_Major
python scripts\prepare_aishell4_manifest.py
```

生成 3 条小样本 manifest：

```powershell
cd H:\Speech_Major
python scripts\prepare_aishell4_manifest.py `
  --audio-dir data\wavs_mono `
  --rttm-dir data\rttm `
  --limit 3 `
  --output data\manifests\aishell4_test_manifest_small.json
```

生成带真实说话人数的全量 manifest：

```powershell
cd H:\Speech_Major
python scripts\prepare_aishell4_manifest.py `
  --with-num-speakers `
  --output data\manifests\aishell4_test_manifest_oracle_spk.json
```

生成带真实说话人数的 3 条小样本 manifest：

```powershell
cd H:\Speech_Major
python scripts\prepare_aishell4_manifest.py `
  --limit 3 `
  --with-num-speakers `
  --output data\manifests\aishell4_test_manifest_small_oracle_spk.json
```

## 4. 小规模 Baseline 测试

先跑 3 条样本，确认环境、模型下载、manifest、输出目录都没问题。

```powershell
conda activate speech_diar

$env:NEMO_CACHE_DIR="H:\Speech_Major\models\nemo_cache"
$env:HF_HOME="H:\Speech_Major\models\hf_cache"

cd H:\Speech_Major\baseline\NeMo\examples\speaker_tasks\diarization

python clustering_diarizer\offline_diar_infer.py `
  diarizer.manifest_filepath=H:\Speech_Major\data\manifests\aishell4_test_manifest_small_mono.json `
  diarizer.out_dir=H:\Speech_Major\results\nemo_cluster_small `
  diarizer.vad.model_path=vad_multilingual_marblenet `
  diarizer.speaker_embeddings.model_path=titanet_large `
  diarizer.speaker_embeddings.parameters.save_embeddings=False
```

查看输出：

```powershell
dir H:\Speech_Major\results\nemo_cluster_small
dir H:\Speech_Major\results\nemo_cluster_small\pred_rttms
```

## 5. 已知说话人数对比实验

先生成小样本 oracle manifest：

```powershell
cd H:\Speech_Major
python scripts\prepare_aishell4_manifest.py `
  --limit 3 `
  --with-num-speakers `
  --output data\manifests\aishell4_test_manifest_small_oracle_spk.json
```

再运行 NeMo，并打开 `oracle_num_speakers`：

```powershell
conda activate speech_diar

$env:NEMO_CACHE_DIR="H:\Speech_Major\models\nemo_cache"
$env:HF_HOME="H:\Speech_Major\models\hf_cache"

cd H:\Speech_Major\baseline\NeMo\examples\speaker_tasks\diarization

python clustering_diarizer\offline_diar_infer.py `
  diarizer.manifest_filepath=H:\Speech_Major\data\manifests\aishell4_test_manifest_small_oracle_spk.json `
  diarizer.out_dir=H:\Speech_Major\results\nemo_cluster_small_oracle_spk `
  diarizer.vad.model_path=vad_multilingual_marblenet `
  diarizer.speaker_embeddings.model_path=titanet_large `
  diarizer.speaker_embeddings.parameters.save_embeddings=False `
  diarizer.clustering.parameters.oracle_num_speakers=True
```

## 6. 多通道 VAD 投票融合

第一步先做 k-of-n 投票融合：只有至少 k 个通道同时检测为 speech，才保留该时间段。

当前 1 条样本 `L_R003S01C02` 上的最佳参数：

```text
vote_threshold      2
merge_gap           0.2
min_duration        0.05
pad_onset           0.2
pad_offset          0.2
external manifest   data/manifests/aishell4_external_vad_vote2_pad02_one.json
output dir          results/nemo_cluster_one_external_vad_vote2_pad02
```

当前对比结果：

```text
原始 union external VAD     DER 17.92% | FA 57.66 | MISS 177.00 | CONF 69.28
vote2, no padding           DER 21.84% | FA 36.03 | MISS 260.89 | CONF 73.40
vote2 + pad 0.2/0.2         DER 11.11% | FA 61.79 | MISS 78.43  | CONF 48.16
```

说明：`vote2` 可以降低误报，但不加边界补偿会明显增加漏检；加入 0.2s 前后 padding 后，漏检大幅下降，是当前第一阶段最佳配置。

### 6.1 生成 1 条样本最佳投票融合 manifest

在 WSL 项目目录中运行，使用项目虚拟环境：

```bash
cd /home/enovo/Speech_Major
source .venv/bin/activate

python scripts/merge_vad_vote.py \
  --vad-dirs \
    results/vad_ch0/vad_outputs \
    results/vad_ch1/vad_outputs \
    results/vad_ch2/vad_outputs \
    results/vad_ch3/vad_outputs \
    results/vad_ch4/vad_outputs \
    results/vad_ch5/vad_outputs \
    results/vad_ch6/vad_outputs \
    results/vad_ch7/vad_outputs \
  --audio-dir data/wavs_mono \
  --output data/manifests/aishell4_external_vad_vote2_pad02_one.json \
  --vote-threshold 2 \
  --merge-gap 0.2 \
  --min-duration 0.05 \
  --pad-onset 0.2 \
  --pad-offset 0.2
```

## 7. 动态 top-k 通道融合

第二步尝试基于 `.frame` VAD 置信度做动态通道选择：每个 10ms 帧读取 8 个通道分数，选择分数最高的 top-k 个通道求平均，再用阈值判定是否为 speech。该方法用于模拟“当前时间帧选择更可靠/更近的麦克风通道”。

脚本：

```text
scripts/merge_vad_topk.py
```

当前 1 条样本测试结果：

```text
top_k=2, threshold=0.50, pad 0.2/0.2  DER 25.37% | FA 21.26 | MISS 351.22 | CONF 57.80
top_k=2, threshold=0.35, pad 0.2/0.2  DER 18.52% | FA 34.25 | MISS 218.34 | CONF 61.54
```

阶段结论：`top-2` 平均分数过于保守，虽然 FA 较低，但 MISS 明显偏高，当前不如第 6 节的 `vote2 + pad 0.2/0.2`。后续如果继续优化 top-k，可以尝试 `top_k=1`、更低阈值，或改成 max/加权融合。

生成 `top_k=2, threshold=0.35` 的 1 条样本 manifest：

```bash
cd /home/enovo/Speech_Major
source .venv/bin/activate

python scripts/merge_vad_topk.py \
  --vad-dirs \
    results/vad_ch0/vad_outputs \
    results/vad_ch1/vad_outputs \
    results/vad_ch2/vad_outputs \
    results/vad_ch3/vad_outputs \
    results/vad_ch4/vad_outputs \
    results/vad_ch5/vad_outputs \
    results/vad_ch6/vad_outputs \
    results/vad_ch7/vad_outputs \
  --audio-dir data/wavs_mono \
  --output data/manifests/aishell4_external_vad_topk2_thr035_pad02_one.json \
  --top-k 2 \
  --score-threshold 0.35 \
  --frame-shift 0.01 \
  --merge-gap 0.2 \
  --min-duration 0.05 \
  --pad-onset 0.2 \
  --pad-offset 0.2
```

运行 diarization：

```bash
cd /home/enovo/Speech_Major/baseline/NeMo/examples/speaker_tasks/diarization

python clustering_diarizer/offline_diar_infer.py \
  --config-name=diar_infer_aishell4_local \
  diarizer.manifest_filepath=/home/enovo/Speech_Major/data/manifests/aishell4_test_manifest_one_mono_wsl.json \
  diarizer.vad.external_vad_manifest=/home/enovo/Speech_Major/data/manifests/aishell4_external_vad_topk2_thr035_pad02_one.json \
  diarizer.out_dir=/home/enovo/Speech_Major/results/nemo_cluster_one_external_vad_topk2_thr035_pad02
```

运行 diarization：

```bash
cd /home/enovo/Speech_Major/baseline/NeMo/examples/speaker_tasks/diarization

python clustering_diarizer/offline_diar_infer.py \
  --config-name=diar_infer_aishell4_local \
  diarizer.manifest_filepath=/home/enovo/Speech_Major/data/manifests/aishell4_test_manifest_one_mono_wsl.json \
  diarizer.vad.external_vad_manifest=/home/enovo/Speech_Major/data/manifests/aishell4_external_vad_vote2_pad02_one.json \
  diarizer.out_dir=/home/enovo/Speech_Major/results/nemo_cluster_one_external_vad_vote2_pad02
```

### 6.2 生成 3 条样本投票融合 manifest

把 1 条样本最佳参数扩展到 3 条样本：

```bash
cd /home/enovo/Speech_Major
source .venv/bin/activate

.venv/bin/python scripts/merge_vad_vote.py \
  --vad-dirs \
    results/vad_3_ch0/vad_outputs \
    results/vad_3_ch1/vad_outputs \
    results/vad_3_ch2/vad_outputs \
    results/vad_3_ch3/vad_outputs \
    results/vad_3_ch4/vad_outputs \
    results/vad_3_ch5/vad_outputs \
    results/vad_3_ch6/vad_outputs \
    results/vad_3_ch7/vad_outputs \
  --audio-dir data/wavs_mono \
  --output data/manifests/aishell4_external_vad_vote2_pad02_3.json \
  --vote-threshold 2 \
  --merge-gap 0.2 \
  --min-duration 0.05 \
  --pad-onset 0.2 \
  --pad-offset 0.2
```

## 8. 通道质量加权融合

第三步改成 Channel Reliability Aware VAD Fusion。核心思想不是简单认为每个通道同等可靠，而是先给每个通道估计一个可靠性权重，再做加权投票：

```text
quality_i =
  margin_weight * VAD_confidence_margin_i
  + speech_ratio_weight * speech_ratio_i
  + stability_weight * VAD_stability_i
  + energy_weight * RMS_energy_i
  + snr_weight * SNR_i

weight_i = quality_i / sum_j quality_j

speech(t) = 1, if sum_i weight_i * active_i(t) >= threshold
```

其中：

- `VAD_confidence_margin`：该通道 `.frame` 分数中高置信度均值和低置信度均值的差，近似表示语音/噪声可分性。
- `speech_ratio`：该通道被 VAD 判为 speech 的比例，避免极端静默或过度激活通道。
- `VAD_stability`：相邻帧分数变化越小，说明该通道 VAD 输出越稳定。
- `RMS_energy`：从原始多通道音频 `data/wavs` 计算通道能量。
- `SNR`：用该通道 VAD 段作为 speech 区间，非 speech 区间作为 noise 区间，计算语音/非语音能量比。

脚本：

```text
scripts/merge_vad_weighted_vote.py
```

推荐先跑 1 条样本，参数沿用当前最佳的 `pad 0.2/0.2 + merge_gap 0.2`，额外打开多通道音频质量分。当前验证最优阈值为 `active_weight_threshold=0.23`：

```bash
cd /home/enovo/Speech_Major
source .venv/bin/activate

python scripts/merge_vad_weighted_vote.py \
  --vad-dirs \
    results/vad_ch0/vad_outputs \
    results/vad_ch1/vad_outputs \
    results/vad_ch2/vad_outputs \
    results/vad_ch3/vad_outputs \
    results/vad_ch4/vad_outputs \
    results/vad_ch5/vad_outputs \
    results/vad_ch6/vad_outputs \
    results/vad_ch7/vad_outputs \
  --frame-dirs \
    results/vad_ch0/vad_outputs \
    results/vad_ch1/vad_outputs \
    results/vad_ch2/vad_outputs \
    results/vad_ch3/vad_outputs \
    results/vad_ch4/vad_outputs \
    results/vad_ch5/vad_outputs \
    results/vad_ch6/vad_outputs \
    results/vad_ch7/vad_outputs \
  --audio-dir data/wavs_mono \
  --multichannel-audio-dir data/wavs \
  --output data/manifests/aishell4_external_vad_weighted_vote_audio_thr023_pad02_one.json \
  --weights-output results/weights_weighted_vote_audio_thr023_one.json \
  --active-weight-threshold 0.23 \
  --merge-gap 0.2 \
  --min-duration 0.05 \
  --pad-onset 0.2 \
  --pad-offset 0.2 \
  --margin-weight 1.0 \
  --speech-ratio-weight 0.3 \
  --stability-weight 0.3 \
  --energy-weight 0.3 \
  --snr-weight 0.5
```

评测命令：

```bash
cd /home/enovo/Speech_Major/baseline/NeMo/examples/speaker_tasks/diarization

python clustering_diarizer/offline_diar_infer.py \
  --config-name=diar_infer_aishell4_local \
  diarizer.manifest_filepath=/home/enovo/Speech_Major/data/manifests/aishell4_test_manifest_one_mono_wsl.json \
  diarizer.vad.external_vad_manifest=/home/enovo/Speech_Major/data/manifests/aishell4_external_vad_weighted_vote_audio_thr023_pad02_one.json \
  diarizer.out_dir=/home/enovo/Speech_Major/results/nemo_cluster_one_external_vad_weighted_vote_audio_thr023_pad02
```

一条样本 `L_R003S01C02` 的权重和结果：

```text
weights: ch0=0.130, ch1=0.136, ch2=0.135, ch3=0.124, ch4=0.108, ch5=0.103, ch6=0.135, ch7=0.130

weighted vote audio, thr=0.22  DER 11.08% | FA 56.85 | MISS 82.51  | CONF 48.59
weighted vote audio, thr=0.23  DER 10.88% | FA 52.97 | MISS 84.42  | CONF 47.08
weighted vote audio, thr=0.24  DER 11.75% | FA 46.71 | MISS 102.77 | CONF 49.80
```

阶段说明：这个方法是第 6 节 `vote2 + pad 0.2/0.2` 的自然升级版。目前在一条样本上从 `vote2 + pad 0.2/0.2` 的 DER 11.11% 进一步降到 10.88%，主要收益来自降低 false alarm 和 confusion；阈值继续升高会让 MISS 明显增加。后续扩展到 3 条或全测试集时，优先使用 `active_weight_threshold=0.23`。
