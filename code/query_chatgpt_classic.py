import sys
from tqdm import tqdm
from openai import OpenAI, AzureOpenAI
import os

from code.config import cfg, update_cfg
from code.utils import time_logger
from code.LLMs.utils import num_tokens_from_messages, get_context_window_size_limit
from code.LLMs.chatgpt_querier import query_chatgpt_batch
from code.data_utils.utils import (load_message, save_llm_outputs,
                                   clean_llm_cache)


@time_logger
def main(cfg):

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

    message_list = load_message(
        dataset_name=cfg.dataset, message_type=cfg.llm.template,
        demo_test=cfg.demo_test
    )
    if cfg.demo_test:
        message_list = message_list[:cfg.num_sample]

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

            batch_chat_completion_list, batch_response_list = query_chatgpt_batch(
                client=client, dataset_name=cfg.dataset,
                llm_model=cfg.llm.model.name, template=cfg.llm.template,
                batch_message_list=batch_message_list, batch_start_id=batch_start_id,
                rpm_limit=rpm_limit
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

            batch_chat_completion_list, batch_response_list = query_chatgpt_batch(
                client=client, dataset_name=cfg.dataset,
                llm_model=cfg.llm.model.name, template=cfg.llm.template,
                batch_message_list=batch_message_list, batch_start_id=batch_start_id,
                rpm_limit=rpm_limit
            )
            chat_completion_list += batch_chat_completion_list
            response_list += batch_response_list

        else:
            batch_message_list.append(message)

    # Save all chat completion
    save_llm_outputs(
        dataset_name=cfg.dataset, outputs=chat_completion_list, data_format="chat_completion",
        template=cfg.llm.template, llm_model=cfg.llm.model.name,
        demo_test=cfg.demo_test
    )
    # Save all responses
    save_llm_outputs(
        dataset_name=cfg.dataset, outputs=response_list, data_format="response",
        template=cfg.llm.template, llm_model=cfg.llm.model.name,
        demo_test=cfg.demo_test
    )
    # Clean all cache files
    clean_llm_cache(
        dataset_name=cfg.dataset, template=cfg.llm.template,
        llm_model=cfg.llm.model.name, clean_response=True, clean_completion=True,
    )


if __name__ == "__main__":
    cfg = update_cfg(cfg)
    main(cfg)
