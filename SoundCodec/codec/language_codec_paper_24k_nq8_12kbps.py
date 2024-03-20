import nlp2
from SoundCodec.base_codec.language_codec import BaseCodec
import gdown

class Codec(BaseCodec):
    def config(self):
        self.setting = "languagecodec_paper_nq8"
        nlp2.download_file(
            "https://raw.githubusercontent.com/jishengpeng/languagecodec/main/configs/languagecodec.yaml",
            "language_codec")
        self.config_path = "language_codec/languagecodec.yaml"
        gdown.cached_download(
            'https://docs.google.com/uc?export=download&id=109ectu4NJWFCpmrqc31wdXvkTI6U2nMA',
            path=f"./language_codec/{self.setting}.ckpt")
        self.ckpt_path = f"language_codec/{self.setting}.ckpt"
