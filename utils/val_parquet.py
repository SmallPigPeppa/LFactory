from datasets import load_dataset, Image

ds = load_dataset(
    "parquet",
    data_files="/ppio_net0/datasets/parquet/llava_779k_demo_10000/*.parquet",
    split="train"
)

ds = ds.cast_column("image", Image(decode=False))
print(ds[0])
print(ds[0]["image"]["path"])
print(len(ds[0]["image"]["bytes"]))
print(ds[0]["conversations"][0])