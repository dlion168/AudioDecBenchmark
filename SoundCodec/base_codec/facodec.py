import numpy

from SoundCodec.base_codec.general import save_audio, ExtractedUnit
from huggingface_hub import hf_hub_download
import torch
import nlp2
import gdown


class BaseCodec:
    def __init__(self):
        try:
            from Amphion.models.codec.ns3_codec import FACodecEncoder, FACodecDecoder
        except:
            raise Exception("Please install Amphion first. git clone https://github.com/open-mmlab/Amphion.git && cd Amphion && bash env.sh")

        self.config()
        self.encoder = FACodecEncoder(
            ngf=32,
            up_ratios=[2, 4, 5, 5],
            out_channels=256,
        )

        self.decoder = FACodecDecoder(
            in_channels=256,
            upsample_initial_channel=1024,
            ngf=32,
            up_ratios=[5, 5, 4, 2],
            vq_num_q_c=2,
            vq_num_q_p=1,
            vq_num_q_r=3,
            vq_dim=256,
            codebook_dim=8,
            codebook_size_prosody=10,
            codebook_size_content=10,
            codebook_size_residual=10,
            use_gr_x_timbre=True,
            use_gr_residual_f0=True,
            use_gr_residual_phone=True,
        )
        self.encoder.load_state_dict(torch.load(self.encoder_path))
        self.decoder.load_state_dict(torch.load(self.decoder_path))
        self.encoder.eval()
        self.decoder.eval()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        self.sampling_rate = 16000

    def config(self):
        self.setting = "facodec_v1"
        nlp2.download_file(
            "https://huggingface.co/amphion/naturalspeech3_facodec/resolve/main/ns3_facodec_decoder.bin",
            "facodec")
        self.decoder_path = "facodec/ns3_facodec_decoder.bin"
        nlp2.download_file(
            'https://huggingface.co/amphion/naturalspeech3_facodec/resolve/main/ns3_facodec_encoder.bin',
            "facodec")
        self.encoder_path = "facodec/ns3_facodec_encoder.bin"

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
        wav = wav.unsqueeze(0)
        enc_out = self.encoder(wav.to(self.device))
        vq_post_emb, vq_id, _, _, spk_embs = self.decoder(enc_out, eval_vq=False, vq=True)
        
        return ExtractedUnit(
            unit=vq_id.permute(1, 0, 2).squeeze(0),
            stuff_for_synth={"vq_emb": vq_post_emb, "spk": spk_embs}
        )

    @torch.no_grad()
    def decode_unit(self, stuff_for_synth):
        wav = self.decoder.inference(stuff_for_synth['vq_emb'], stuff_for_synth['spk'])
        wav = wav.detach().cpu().squeeze(0).numpy()
        return wav[0]
