import nlp2
from SoundCodec.base_codec.language_codec import BaseCodec
import gdown

class Codec(BaseCodec):
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
