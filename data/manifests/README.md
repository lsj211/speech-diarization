# Manifests

放置 NeMo manifest JSONL 文件。每行对应一段音频。

当前默认从 `data/wavs` 和 `data/rttm` 生成 AISHELL-4 test manifest：

```powershell
python scripts\prepare_aishell4_manifest.py
python scripts\prepare_aishell4_manifest.py --limit 3 --output data\manifests\aishell4_test_manifest_small.json
python scripts\prepare_aishell4_manifest.py --with-num-speakers --output data\manifests\aishell4_test_manifest_oracle_spk.json
```

如果以后要直接从原始目录生成，可以传入 `--data-root data\raw\AISHELL-4\test`。
