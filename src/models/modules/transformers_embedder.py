import torch
import torch.nn as nn
from termcolor import colored
from transformers import BertModel


class TransformersEmbedder(nn.Module):
    """Encode the input with transformers model such as BERT, Roberta, and so on."""

    def __init__(self, transformer_model_name: str, parallel_embedder: bool = False):
        super().__init__()
        output_hidden_states = False  # to use all hidden states or not
        print(
            colored(
                f"[Model Info] Loading pretrained language model {transformer_model_name}",
                "red",
            )
        )

        self.model = BertModel.from_pretrained(
            transformer_model_name,
            output_hidden_states=output_hidden_states,
            return_dict=False,
        )
        self.parallel = parallel_embedder
        if parallel_embedder:
            self.model = nn.DataParallel(self.model)
        """
        use the following line if you want to freeze the model,
        but don't forget also exclude the parameters in the optimizer
        """
        # self.model.requires_grad = False

    def get_output_dim(self):
        # use different model may have different attribute
        # for example, if you are using GPT, it should be self.model.config.n_embd
        # Check out https://huggingface.co/transformers/model_doc/gpt.html
        # But you can directly write it as 768 as well.
        return (
            self.model.config.hidden_size
            if not self.parallel
            else self.model.module.config.hidden_size
        )

    def forward(
        self,
        word_seq_tensor: torch.Tensor,
        orig_to_token_index: torch.LongTensor,  # batch_size * max_seq_leng
        input_mask: torch.LongTensor,
    ) -> torch.Tensor:
        """

        :param word_seq_tensor: (batch_size x max_wordpiece_len x hidden_size)
        :param orig_to_token_index: (batch_size x max_sent_len x hidden_size)
        :param input_mask: (batch_size x max_wordpiece_len)
        :return:
        """
        word_rep, pooler_output = self.model(
            **{"input_ids": word_seq_tensor, "attention_mask": input_mask}
        )
        # exclude the [CLS] and [SEP] token
        # _, _, word_rep = self.model(**{"input_ids": word_seq_tensor, "attention_mask": input_mask})
        # word_rep = torch.cat(word_rep[-4:], dim=2)

        batch_size, _, rep_size = word_rep.size()
        _, max_sent_len = orig_to_token_index.size()
        return (
            torch.gather(
                word_rep[:, :, :],
                1,
                orig_to_token_index.unsqueeze(-1).expand(batch_size, max_sent_len, rep_size),
            ),
            word_rep,
        )
