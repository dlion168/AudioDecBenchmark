from .PODCAST import load_data as load_data_self

def load_data():
    return load_data_self(data_path="/mnt/data/ycevan/SpEAT/s3prl/s3prl/data/IMPROV/Audios", split="train", writer_batch_size=1060)

if __name__=='__main__':
    load_data()