# Copyright (c) 2025 Zhipu AI Inc (authors: CogAudio Group Members)
# Modified for the M5 colocated verl GRPO recipe.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""WavLM/ECAPA speaker similarity with the Cosy 24 -> 16 kHz boundary."""
import torch
import torch.nn.functional as F
from torchaudio.transforms import Resample
from .reward_models.ecapa_tdnn import ECAPA_TDNN

resample_func = Resample(orig_freq=24000, new_freq=16000)


def load_wav(wav_arr):
    return resample_func(wav_arr.float())


@torch.no_grad()
def verification2(wav1, wav2, model):
    device = next(model.parameters()).device
    emb1 = model(load_wav(wav1).to(device))
    emb2 = model(load_wav(wav2).to(device))
    return F.cosine_similarity(emb1, emb2)


def get_sim_model(device, checkpoint, wavlm_checkpoint):
    model = ECAPA_TDNN(wavlm_checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'], strict=False)
    return model.to(device).eval().requires_grad_(False)
