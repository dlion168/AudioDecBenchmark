import os


def load_codec(codec_name):
    module = __import__(f"SoundCodec.codec.{codec_name}", fromlist=[codec_name])
    return module.Codec()


def list_codec(ignore_list=['__init__.py', 'general.py', 'dac_44k.py', 'academicodec_hifi_16k_320d.py', 'academicodec_hifi_24k_320d.py', 'funcodec_en_libritts_16k_gr1nq32ds320.py', 'funcodec_en_libritts_16k_gr8nq32ds320.py']):
    dataset_dir = os.path.dirname(os.path.abspath(__file__))
    files = os.listdir(dataset_dir)
    py_files = filter(lambda x: x.endswith('.py') and x not in ignore_list, files)
    py_files = filter(lambda x: os.path.isfile(os.path.join(dataset_dir, x)), py_files)
    py_files = map(lambda x: os.path.splitext(x)[0], py_files)
    return sorted(py_files)
