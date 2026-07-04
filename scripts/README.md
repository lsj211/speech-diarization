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
