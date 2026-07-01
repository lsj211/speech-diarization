# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from lightning.pytorch import seed_everything
from omegaconf import OmegaConf

from nemo.collections.asr.models import ClusteringDiarizer
from nemo.collections.asr.parts.utils.speaker_utils import audio_rttm_map
from nemo.core.config import hydra_runner
from nemo.utils import logging

"""
Run only the NeMo VAD stage from the clustering diarizer.

This is useful for multi-channel VAD fusion: run VAD once for each extracted
single channel, then merge the resulting vad_outputs/*.txt files into one
external_vad_manifest.
"""

seed_everything(42)


@hydra_runner(config_path="../conf/inference", config_name="diar_infer_meeting.yaml")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    sd_model = ClusteringDiarizer(cfg=cfg).to(cfg.device)

    sd_model._out_dir = cfg.diarizer.out_dir
    os.makedirs(sd_model._out_dir, exist_ok=True)
    sd_model._vad_dir = os.path.join(sd_model._out_dir, 'vad_outputs')
    sd_model._vad_out_file = os.path.join(sd_model._vad_dir, 'vad_out.json')
    sd_model.AUDIO_RTTM_MAP = audio_rttm_map(cfg.diarizer.manifest_filepath)

    sd_model._perform_speech_activity_detection()
    logging.info(f'VAD outputs are saved in {os.path.abspath(sd_model._vad_dir)}')


if __name__ == '__main__':
    main()
