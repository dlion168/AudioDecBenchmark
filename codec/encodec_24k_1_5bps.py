from base_codec.encodec import BaseCodec

class Codec(BaseCodec):
    def config(self):
        self.model.set_target_bandwidth(1.5)
        self.setting = "encodec_24khz_1_5"
        self.sampling_rate = 24_000
