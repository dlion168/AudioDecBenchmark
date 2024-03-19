import numpy

from SoundCodec.base_codec.general import save_audio, ExtractedUnit
import torchaudio
import torch
import nlp2
import gdown


class BaseCodec:
    def __init__(self):
        try:
            from vocos import Vocos
        except:
            raise Exception("Please install language codec first. https://github.com/jishengpeng/languagecodec/tree/main")

        self.config()
        self.model = Vocos.from_pretrained0802(self.config_path, self.ckpt_path)
        self.model.eval()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        self.sampling_rate = 24000
        self.bandwidth_id = torch.tensor([3]).to(self.device) # 12 kbps

    def config(self):
        self.setting = "languagecodec_chinese_nq8"
        nlp2.download_file(
            "https://raw.githubusercontent.com/jishengpeng/languagecodec/main/configs/languagecodec.yaml",
            "language_codec")
        self.config_path = "language_codec/languagecodec.yaml"
        gdown.cached_download(
            'https://docs.google.com/uc?export=download&id=18JpINstfF2YrbFg6nqs3BVn0oxdLsuUm',
            path=f"./language_codec/{self.setting}.ckpt")
        self.ckpt_path = f"language_codec/{self.setting}.ckpt"

    @torch.no_grad()
    def synth(self, data, local_save=True):
        extracted_unit = self.extract_unit(data)
        data['unit'] = extracted_unit.unit
        audio_values = self.decode_unit(extracted_unit.stuff_for_synth)
        if local_save:
            audio_path = f"dummy-{self.setting}/{data['id']}.wav"
            save_audio(audio_values, audio_path, self.sampling_rate)
            data['audio'] = audio_path
        else:
            data['audio']['array'] = audio_values
        return data

    @torch.no_grad()
    def extract_unit(self, data):
        wav = torch.tensor(numpy.array([data["audio"]['array']]), dtype=torch.float32).to(self.device)
        _, codes = self.model.encode(wav.to(self.device), bandwidth_id=self.bandwidth_id)
        return ExtractedUnit(
            unit=codes.permute(1, 0, 2).squeeze(0),
            stuff_for_synth=codes
        )

    @torch.no_grad()
    def decode_unit(self, stuff_for_synth):
        codes = stuff_for_synth
        features = self.model.codes_to_features(codes)
        wav = self.model.decode(features, bandwidth_id=self.bandwidth_id)
        wav = wav.detach().cpu().squeeze(0).numpy()
        return wav
