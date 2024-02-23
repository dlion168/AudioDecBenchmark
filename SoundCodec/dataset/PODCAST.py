from datasets import load_dataset
from pathlib import Path

def load_data(data_path = "/mnt/data/ycevan/SpEAT/s3prl/s3prl/data/PODCAST/Audios", split="train+validation+test", writer_batch_size=100, data_files=None):
    dataset = load_dataset("audiofolder", data_dir = data_path, split=split, cache_dir="/mnt/data/ycevan/.cache/datasets", data_files=data_files)
    print(dataset)
    if 'label' in dataset[0].keys():
        dataset = dataset.remove_columns('label')
    dataset = dataset.map(lambda example: {"id": Path(example["audio"]["path"]).stem}, writer_batch_size=writer_batch_size, num_proc=2)
    assert 'id' in dataset[0].keys()
    return dataset

if __name__=='__main__':
    load_data()