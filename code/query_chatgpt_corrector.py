import sys
from tqdm import tqdm
from openai import OpenAI, AzureOpenAI
import os

from code.config import cfg, update_cfg
from code.utils import time_logger
from code.LLMs.utils import num_tokens_from_messages, get_context_window_size_limit
from code.LLMs.chatgpt_querier import query_chatgpt_gnn_preds_batch
from code.data_utils.dataset import DatasetLoader
from code.data_utils.utils import (load_message, save_llm_outputs, clean_llm_cache)


@time_logger
def main(cfg):

    # Preprocess data
    dataloader = DatasetLoader(name=cfg.dataset, text='raw')
    dataset, smiles = dataloader.dataset, dataloader.text

    split_idx = dataset.get_idx_split()
    test_indices = split_idx["test"].numpy()

    if cfg.dataset == "ogbg-molbace":
        demo_list = [101, 102, 103, 201, 202, 0, 1, 6, 239, 240]  # bace
    elif cfg.dataset == "ogbg-molbbbp":
        demo_list = [422, 313, 354, 370, 120, 6, 291, 94, 8, 453]  # bbbp
    elif cfg.dataset == "ogbg-molhiv":
        demo_list = [8773, 1975, 3969, 9063, 6750, 7305, 2191, 7171, 2213, 2190]  # hiv
    else:
        demo_list = [101, 102, 103]

    # RPM limit (adjust according to your plan)
    rpm_limit = 3500
    # TPM limit (adjust according to your plan)
    tpm_limit = 60000
    # Context window size
    cws_limit = get_context_window_size_limit(cfg.llm.model.name)

    # Set up OpenAI API
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version="2023-12-01-preview",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    ) if cfg.llm.provider == 'aoai' else OpenAI(api_key=cfg.OPENAI_API_KEY)

    full_message_list = load_message(
        dataset_name=cfg.dataset, message_type=cfg.llm.template,
        gnn_model=cfg.gnn.model.name, seed=cfg.seed,
        demo_test=cfg.demo_test
    )
    message_list = [full_message_list[id] for id in demo_list] if cfg.demo_test \
        else [full_message_list[id] for id in test_indices]

    # Save all queries
    chat_completion_list = []
    response_list = []

    # Run batch queries
    batch_message_list = []
    batch_message_token_num = 0
    batch_start_id = 0
    display = "Query {} {}".format(cfg.dataset, cfg.llm.model.name)
    for message_id, message in enumerate(tqdm(message_list, desc=display)):

        num_tokens = num_tokens_from_messages(
            messages=message, original_model=cfg.llm.model.name
        )
        if num_tokens > tpm_limit:
            sys.exit("Message token number is large than limit {}.".format(tpm_limit))
        while num_tokens >= cws_limit:
            print("Message context length is {}, larger than Context Window Size limit {}.".format(
                num_tokens, cws_limit
            ))
            str_len = len(message[1]["content"])
            message[1]["content"] = (message[1]["content"][:int(str_len / 3)] +
                                     message[1]["content"][-int(str_len / 3):])
            num_tokens = num_tokens_from_messages(
                messages=message, original_model=cfg.llm.model.name
            )
            print("Message token number is reduced to {}.".format(num_tokens))
        batch_message_token_num += num_tokens

        if (batch_message_token_num >= tpm_limit) and (message_id < len(message_list) - 1):

            batch_chat_completion_list, batch_response_list = query_chatgpt_gnn_preds_batch(
                client=client, dataset_name=cfg.dataset,
                llm_model=cfg.llm.model.name, template=cfg.llm.template,
                gnn_model=cfg.gnn.model.name, seed=cfg.seed,
                batch_message_list=batch_message_list, batch_start_id=batch_start_id,
                rpm_limit=rpm_limit, demo_test=cfg.demo_test
            )
            chat_completion_list += batch_chat_completion_list
            response_list += batch_response_list
            batch_message_list = [message]
            batch_message_token_num = num_tokens_from_messages(
                messages=message, original_model=cfg.llm.model.name
            )
            batch_start_id = message_id

        elif message_id == len(message_list) - 1:
            batch_message_list.append(message)

            batch_chat_completion_list, batch_response_list = query_chatgpt_gnn_preds_batch(
                client=client, dataset_name=cfg.dataset,
                llm_model=cfg.llm.model.name, template=cfg.llm.template,
                gnn_model=cfg.gnn.model.name, seed=cfg.seed,
                batch_message_list=batch_message_list, batch_start_id=batch_start_id,
                rpm_limit=rpm_limit, demo_test=cfg.demo_test
            )
            chat_completion_list += batch_chat_completion_list
            response_list += batch_response_list

        else:
            batch_message_list.append(message)

    # Save all chat completion
    save_llm_outputs(
        dataset_name=cfg.dataset, outputs=chat_completion_list, data_format="chat_completion",
        gnn_model=cfg.gnn.model.name, seed=cfg.seed,
        template=cfg.llm.template, llm_model=cfg.llm.model.name,
        demo_test=cfg.demo_test
    )
    # Save all responses
    save_llm_outputs(
        dataset_name=cfg.dataset, outputs=response_list, data_format="response",
        gnn_model=cfg.gnn.model.name, seed=cfg.seed,
        template=cfg.llm.template, llm_model=cfg.llm.model.name,
        demo_test=cfg.demo_test
    )
    # Clean all cache files
    clean_llm_cache(
        dataset_name=cfg.dataset, template=cfg.llm.template,
        gnn_model=cfg.gnn.model.name, seed=cfg.seed,
        llm_model=cfg.llm.model.name, clean_response=True, clean_completion=True,
    )


if __name__ == "__main__":
    cfg = update_cfg(cfg)
    main(cfg)
