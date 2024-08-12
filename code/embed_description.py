import os
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI, AzureOpenAI

from code.config import cfg, update_cfg
from code.data_utils.dataset import DatasetLoader
from code.data_utils.utils import (load_description, save_similarity_matrix,
                                   save_open_ai_embedding, load_openai_embedding)
from code.LLMs.utils import num_tokens_from_string


def get_embedding(client, text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def main(cfg):

    # Set up OpenAI API
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version="2023-12-01-preview",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    ) if cfg.llm.provider == 'aoai' else OpenAI(api_key=cfg.OPENAI_API_KEY)

    # Preprocess data
    dataloader = DatasetLoader(name=cfg.dataset, text='raw')
    dataset, smiles = dataloader.dataset, dataloader.text
    description = load_description(dataset_name=cfg.dataset, demo_test=cfg.demo_test)

    if cfg.llm.emb.data == "des":
        text_list = description
    elif cfg.llm.emb.data == "smi":
        text_list = smiles
    elif cfg.llm.emb.data == "smi-des":
        text_list =[]
        for index, (smi, des) in enumerate(zip(smiles, description)):
            _text = "The SMILES string of molecule-{} is {}. {}".format(index, smi, des)
            text_list.append(_text)
    else:
        raise ValueError("{} is not a valid option.".format(cfg.llm.emb.data))

    # Generate embeddings
    list_emb = []
    for index, des in enumerate(tqdm(text_list)):
        num_tokens = num_tokens_from_string(string=des)
        if num_tokens < 8191:
            list_emb.append(get_embedding(
                client=client, text=des, model=cfg.llm.emb.model
            ))
        else:
            print("String {} has {} tokens, more than 8191 tokens.".format(index, num_tokens))
            list_emb.append(get_embedding(
                client=client, text=des[:8191], model=cfg.llm.emb.model
            ))

    save_open_ai_embedding(
        dataset_name=cfg.dataset, list_embedding=list_emb,
        embedding_model=cfg.llm.emb.model, embedding_type=cfg.llm.emb.data,
    )

    # Compute distance
    embeddings = load_openai_embedding(
        dataset_name=cfg.dataset,
        embedding_model=cfg.llm.emb.model,
        embedding_type=cfg.llm.emb.data
    )
    cosine_similarity_matrix = cosine_similarity(embeddings)

    save_similarity_matrix(
        dataset_name=cfg.dataset, embedding_model=cfg.llm.emb.model,
        embedding_type=cfg.llm.emb.data, similarity_metric="cosine",
        similarity_matrix=cosine_similarity_matrix
    )


if __name__ == '__main__':
    cfg = update_cfg(cfg)
    main(cfg)


# python -m code.embed_description dataset ogbg-molbace llm.emb.data des llm.emb.model text-embedding-3-small