### Speed Profile:
import cProfile
import io
import logging
import os
import pstats
from pathlib import Path
from pstats import SortKey
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray
from datasets import Dataset, load_dataset
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import logging as tlogging

tlogging.set_verbosity_error()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
ray.init()
BASE = "/nlpgpu/data/terry/ToolProj/src/exps_pret_corr/data/"

dclm_path = os.path.join(BASE, "dclm_small/main/")
starcoder_path = os.path.join(BASE, "starcoder_small/main/")

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

QUERIES = [
    "rod cutting",
    "longest common subsequence",
    "knapsack",
    "addition",
    "subtraction",
    "multiplication",
    "integer linear programming production problem",
    "integer linear programming assignment problem",
    "integer linear programming partition problem",
]


class DataRecord(BaseModel):
    name: str
    path: Path  # something like text, parquet etc.
    file_type: str
    column_name: str
    source_path: str
    source_path_pattern: str


def add_task(text_splitter: RecursiveCharacterTextSplitter, doc: Document) -> Document:
    return text_splitter.split_documents([doc])[0]


def convert_to_docs(data: Dataset, column_name, text_splitter) -> List[Document]:
    RAW_KNOWLEDGE_BASE: List[Document] = [
        Document(page_content=doc[column_name]) for doc in tqdm(data)
    ]  ## so far this is faster, what if we scale up?
    RAW_KNOWLEDGE_BASE = RAW_KNOWLEDGE_BASE[:10]  # debugging
    # cpus = multiprocessing.cpu_count()
    # logger.info(f"Utilizing {cpus} CPUS in splitting text")
    # processed_docs: List[Document] = Parallel(jobs=cpus, backend="threading")(
    #     delayed(add_task)(text_splitter, doc) for doc in tqdm(RAW_KNOWLEDGE_BASE, desc="splitting documents")
    # )
    # logger.info(
    #     f"Number of docs before: {len(RAW_KNOWLEDGE_BASE)}, \
    #     Number of docs after: {len(processed_docs)}"
    # )
    processed_docs = RAW_KNOWLEDGE_BASE
    return processed_docs


def convert_to_str(data: Dataset, column_name: str, *args) -> List[str]:
    return data[column_name]


def convert_to_ray(dataset: Dataset, column_name) -> ray.data.Dataset:
    ds = ray.data.from_huggingface(dataset)
    return ds.select_columns(column_name)


class Embed:
    def __init__(self):
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.transformer = SentenceTransformer(model_name, device="cuda", cache_folder="../models")

    def __call__(self, text_batch: Dict[str, List[str]]):
        assert isinstance(text_batch, Dict), f"not a dict, type {type(text_batch)}, with {text_batch}"
        embedding: List[float] = self.transformer.encode(text_batch["content"], batch_size=64, show_progress_bar=True, device="cuda").tolist()

        return dict(results=list(zip(text_batch["content"], embedding)))


def dedup_docs(processed_docs: List[Document]) -> List[Document]:
    unique_texts = {}
    dedupped_docs = []
    for doc in tqdm(processed_docs, desc="deduplicating"):
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            dedupped_docs.append(doc)
    return dedupped_docs


def load_datasets(to_load: List[DataRecord], tokenizer: AutoTokenizer, profile: bool = False):
    for dataRecord in tqdm(to_load, desc="loading data"):
        dataset = load_dataset(dataRecord.file_type, data_dir=dataRecord.path)

        if profile:
            pr = cProfile.Profile()
            pr.enable()

        # processed_docs: List[str] = convert_to_docs(dataset["train"], dataRecord.column_name, text_splitter)
        # processed_docs : List[str] = convert_to_str(dataset['train'], dataRecord.column_name)
        logger.info("Beginning to convert to ray")
        processed_docs: ray.data.Dataset = convert_to_ray(dataset["train"], dataRecord.column_name)
        processed_docs = processed_docs.materialize()  # Pin to mem
        processed_docs = processed_docs.repartition(8)
        logger.info("Beginning to map batches")
        ds_iter = processed_docs.map_batches(Embed, batch_size=8192, concurrency=16, num_gpus=1)
        text_and_emb = [tne["results"] for tne in tqdm(ds_iter.iter_rows(), total=len(dataset["train"]), desc="Iterating Rows")]
        logger.info(f"Example Result: {text_and_emb[0]}")

        logger.info("Beginning Model Embedding")

        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        EMBEDDING_MODEL = HuggingFaceEmbeddings(
            model_name=model_name,
            multi_process=True,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 1024},
        )
        logger.info("Beginning Loading Embeddings into FAISS")

        VECTORDB = FAISS.from_embeddings(text_and_emb, embedding=EMBEDDING_MODEL)
        logger.info("Beginning saving VECTORDB to local")

        VECTORDB.save_local("faiss_index")
        # import pdb; pdb.set_trace() # sus doc types

        # deduped_docs: List[Document] = dedup_docs(processed_docs)
        # logger.info(
        #     f"Number of docs before dedup: {len(processed_docs)}, \
        #     Number of docs after dedup: {len(deduped_docs)} "
        # )
        # deduped_docs = processed_docs

        if profile:
            pr.disable()
            s = io.StringIO()
            sortby = SortKey.CUMULATIVE
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            stats = s.getvalue()

            with open(f"stats_{dataRecord.name}.txt", "w+") as f:
                f.write(stats)
        logger.info("Finished Processing")

        # result[dataRecord.name] = deduped_docs
    # return result


def serialize_filtered_datasets(data: Dataset, datapath: Path) -> None:
    data.save_to_disk(datapath)
    logging.info(f"Saved Dataset to: {datapath.name}")


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


def plot_length_distribution(tokenizer: AutoTokenizer, data: Dict[str, List[Document]]) -> None:
    for name, documents in data.items():
        lengths = [len(tokenizer.encode(doc.page_content)) for doc in tqdm(documents, desc="processing_docs")]
        pd.Series(lengths).hist()
        plt.title(f"{name} plot of length distribution")
        plt.show()
        plt.savefig(f"{name}_lengths.png")
        plt.close()


# add types and docstrings
def query_viz(vectordb: FAISS, queries: List[str], processed_docs: List[Document], embedding_model: HuggingFaceEmbeddings) -> None:
    """
    Creates a visualization with the query as a star to see where it lives in the embeddings

    Params:
        vectordb: FAISS approximate nearest neighbor algorithm
        queries: list of algorithms we are trying to retrieve
        processed_docs: List[Document] of documents we have chunked and are ready to retrieve
        embedding_model: HuggingFaceEmbeddings, defaults to sentence-transformers minilm

    Returns:
        None. However, as a side effect, generates a visualization of queries alongside other documents
    """
    import pacmap
    import plotly.express as px

    embedding_proj = pacmap.PaCMAP(n_components=2, n_neighbors=2, MN_ratio=0.5, FP_ratio=0.5, random_state=1)

    logger.info("Starting Embedding Query")
    embedded_queries = [embedding_model.embed_query(query) for query in queries]
    logger.info("Starting to index")
    embeddings_2d = [list(vectordb.index.reconstruct_n(idx, 1)[0]) for idx in range(len(processed_docs))] + [query for query in embedded_queries]
    logger.info("Starting to project")
    proj_docs = embedding_proj.fit_transform(np.array(embeddings_2d), init="pca")
    logger.info("Plotting")
    # import pdb; pdb.set_trace()
    df = [
        {
            "x": proj_docs[i, 0],
            "y": proj_docs[i, 1],
            "source": "",
            "extract": processed_docs[i].page_content[:100] + "...",
            "symbol": "circle",
            "size_col": 4,
        }
        for i in range(len(processed_docs))
    ] + [
        {
            "x": proj_docs[len(processed_docs) + i, 0],
            "y": proj_docs[len(processed_docs) + i, 1],
            "source": queries[i],
            "extract": queries[i],
            "symbol": "star",
            "size_col": 25,
        }
        for i in range(len(queries))
    ]
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="source",
        hover_data="extract",
        size="size_col",
        symbol="symbol",
        color_discrete_map={x: "black" for x in queries},
        width=1000,
        height=700,
    )
    fig.update_traces(marker=dict(opacity=1, line=dict(width=0, color="DarkSlateGrey")), selector=dict(mode="markers"))
    # fig.show()
    fig.write_image("queryscatter.png")
    logger.info("Finished saving")


def search(embedding_model: HuggingFaceEmbeddings, data: Dict[str, List[Document]]) -> None:
    for name, documents in data.items():
        VECTORDB = FAISS.from_documents(documents, embedding_model, distance_strategy=DistanceStrategy.COSINE)
        logger.info(f"Creating Visualizaiton for {name}")
        query_viz(VECTORDB, QUERIES, documents, embedding_model)
        VECTORDB.save_local("faiss_index")


def main() -> None:
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    to_load = [
        DataRecord(
            name="starcoder",
            path=Path(starcoder_path),
            column_name="content",
            source_path="max_stars_repo_path",
            source_path_pattern=r"",
            file_type="parquet",
        ),
        DataRecord(
            name="dclm",
            path=Path(dclm_path),
            column_name="text",
            source_path="fasttext_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train_prob",
            source_path_pattern=r"(0.\d{2})",
            file_type="text",
        ),
    ]
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_name)
    load_datasets(to_load, tokenizer, profile=True)

    # plot_length_distribution(tokenizer, data)

    # EMBEDDING_MODEL = HuggingFaceEmbeddings(
    #     model_name=model_name,
    #     multi_process=True,
    #     model_kwargs={"device": "cuda"},
    #     encode_kwargs={"normalize_embeddings": True, "batch_size": 1024},
    # )
    # search(EMBEDDING_MODEL, data)


if __name__ == "__main__":
    main()
