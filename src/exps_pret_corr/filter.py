### Speed Profile:
import cProfile
import io
import logging
import multiprocessing
import os
import pstats
from pathlib import Path
from pstats import SortKey
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from datasets import Dataset, load_dataset
from fastembed import TextEmbedding
from joblib import Parallel, delayed
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from tqdm import tqdm
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

BASE = "/nlpgpu/data/terry/ToolProj/src/exps_pret_corr/data/"

dclm_path = os.path.join(BASE, "dclm/main/")
starcoder_path = os.path.join(BASE, "starcoder/main/")

MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]


def add_task(text_splitter: RecursiveCharacterTextSplitter, doc: Document) -> None:
    return text_splitter.split_documents([doc])


def load_datasets(tokenizer_name: str) -> dict:
    starcoder_data = load_dataset("parquet", data_dir=starcoder_path)
    # starcoder_data_iter = starcoder_data.to_iterable_dataset()
    dclm_data = load_dataset("text", data_dir=dclm_path)
    # dclm_data_iter = dclm_data.to_iterable_dataset()
    pr = cProfile.Profile()
    pr.enable()

    # RAW_KNOWLEDGE_BASE= Parallel(n_jobs=-1)(
    #     delayed(Document)(page_content=doc["content"]) for doc in tqdm(starcoder_data['train'])
    # )

    text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=1000,
        chunk_overlap=100,
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )

    RAW_KNOWLEDGE_BASE: List[Document] = [
        Document(page_content=doc["content"]) for doc in tqdm(starcoder_data["train"])
    ]  ## so far this is faster, what if we scale up?

    cpus = multiprocessing.cpu_count()
    logger.info(f"Utilizing {cpus} CPUS in splitting text")
    processed_docs: List[Document] = Parallel(jobs=cpus, backend="threading")(
        delayed(add_task)(text_splitter, doc) for doc in tqdm(RAW_KNOWLEDGE_BASE, desc="splitting documents")
    )

    # processed_docs: List[Document] = []
    # with concurrent.futures.ProcessPoolExecutor(max_workers = cpus) as executor:
    #     task = partial(add_task, text_splitter = text_splitter)
    #     for result in tqdm(executor.map(task, RAW_KNOWLEDGE_BASE), desc="splitting documents"):
    #         processed_docs += result

    unique_texts = {}
    dedupped_docs = []
    for doc in tqdm(processed_docs, desc="deduplicating"):
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            dedupped_docs.append(doc)
    logger.info(
        f"Number of docs before: {len(RAW_KNOWLEDGE_BASE)}, \
        Number of docs after: {len(processed_docs)}, \
        Number of docs after dedup: {len(dedupped_docs)}"
    )

    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    stats = s.getvalue()

    with open("stats.txt", "w+") as f:
        f.write(stats)
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
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    data = load_datasets(model_name)
    device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    logger.info("Starting to Load Text Model ... ")
    embed_model = TextEmbedding(model_name=model_name, cuda=True, device_ids=device_ids, lazy_load=True)
    logger.info("Finished Loading Text Embedding model ... ")
    embedded_text = embed_func(embed_model, data["STARCODER"], 64, len(device_ids))

    ##TODO retrieve topk documents about algorithm X. Use that.
    labels = kmeans(embedded_text)
    tsne_visualize(embedded_text, labels)


if __name__ == "__main__":
    main()
