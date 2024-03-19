from .PODCAST import load_data as load_data_self

def load_data():
    return load_data_self(data_path="/mnt/data/ycevan/AudioDecBenchmark/downloaded_data/tasks/gunshot_triangulation-v1.0-full/48000", split="train", writer_batch_size=88)

if __name__=='__main__':
    load_data()