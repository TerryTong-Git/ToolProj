import logging
import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from datasets import Dataset, load_dataset
from fastembed import TextEmbedding
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

BASE = "/nlpgpu/data/terry/ToolProj/src/exps_pret_corr/data/"

dclm_path = os.path.join(BASE, "dclm/main/")
starcoder_path = os.path.join(BASE, "starcoder/main/")


def load_datasets() -> dict:
    starcoder_data = load_dataset("parquet", data_dir=starcoder_path)
    # starcoder_data_iter = starcoder_data.to_iterable_dataset()
    dclm_data = load_dataset("text", data_dir=dclm_path)
    # dclm_data_iter = dclm_data.to_iterable_dataset()
    data = {"STARCODER": starcoder_data["train"]["content"], "DCLM": dclm_data["train"]["text"]}
    return data


def serialize_filtered_datasets(data: Dataset, datapath: Path) -> None:
    data.save_to_disk(datapath)
    logging.info(f"Saved Dataset to: {datapath.name}")


def embed_func(model: TextEmbedding, data: List[str], batch_size: int, parallel: int) -> List:
    """
    Transforms a list of inputs into a list of text embedding vectors
    """
    return list([e for e in tqdm(model.embed(data, batch_size=batch_size, parallel=parallel), total=len(data))])


def kmeans(embeddings: List) -> List[int]:
    model = KMeans(n_clusters=2)
    model.fit(embeddings)
    return model.labels_


def tsne_visualize(embeddings, labels):
    tsne = TSNE(n_components=2)
    X = tsne.fit_transform(np.array(embeddings))
    plt.plot((8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", marker="o")
    plt.show()
    plt.savefig("tsne.png")


def get_data_statistics():
    pass


def get_data_per_label():
    pass


def get_total_tokens():
    pass


def train_logistic():
    pass


def evaluate():
    pass


def main():
    data = load_datasets()
    device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    logger.info("Starting to Load Text Model ... ")
    embed_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", cuda=True, device_ids=device_ids, lazy_load=True)
    logger.info("Finished Loading Text Embedding model ... ")
    embedded_text = embed_func(embed_model, data["STARCODER"], 64, len(device_ids))
    labels = kmeans(embedded_text)
    tsne_visualize(embedded_text, labels)


if __name__ == "__main__":
    main()
