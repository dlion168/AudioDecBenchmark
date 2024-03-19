from SoundCodec.base_codec.audiodec import BaseCodec
import nlp2


class Codec(BaseCodec):
    def config(self):
        self.setting = "audiodec_48k_300d_uni"
        try:
            from AudioDec.utils.audiodec import AudioDec as AudioDecModel, assign_model
        except:
            raise Exception("Please install AudioDec first. pip install git+https://github.com/voidful/AudioDec.git")
        # download encoder
        nlp2.download_file(
            'https://huggingface.co/Codec-SUPERB/AudioDec/resolve/main/autoencoder/symADuniv_vctk_48000_hop300/checkpoint-500000steps.pkl',
            'audiodec_autoencoder_48k_300d_uni')
        nlp2.download_file(
            'https://huggingface.co/Codec-SUPERB/AudioDec/resolve/main/autoencoder/symADuniv_vctk_48000_hop300/config.yml',
            "audiodec_autoencoder_48k_300d_uni")
        self.encoder_config_path = "audiodec_autoencoder_48k_300d_uni/checkpoint-500000steps.pkl"
        # download decoder
        nlp2.download_file(
            'https://huggingface.co/Codec-SUPERB/AudioDec/resolve/main/vocoder/AudioDec_v3_symADuniv_vctk_48000_hop300_clean/checkpoint-500000steps.pkl',
            'audiodec_vocoder_48k_300d_uni')
        nlp2.download_file(
            'https://huggingface.co/Codec-SUPERB/AudioDec/resolve/main/vocoder/AudioDec_v3_symADuniv_vctk_48000_hop300_clean/config.yml',
            "audiodec_vocoder_48k_300d_uni")
        nlp2.download_file(
            "https://github.com/facebookresearch/AudioDec/raw/main/stats/symADuniv_vctk_48000_hop300_clean.npy",
            "audiodec_vocoder_48k_300d_uni"
        )
        self.decoder_config_path = "audiodec_vocoder_48k_300d_uni/checkpoint-500000steps.pkl"
        self.sampling_rate = 48000
        audiodec_model = AudioDecModel(tx_device=self.device, rx_device=self.device)
        audiodec_model.load_transmitter(self.encoder_config_path)
        audiodec_model.load_receiver(self.encoder_config_path, self.decoder_config_path)
        self.model = audiodec_model
