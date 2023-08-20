import torch
from torch import nn


class CodeBertModel(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super().__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_layer = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids, num_features, labels=None):
        code = self.encoder(input_ids, attention_mask=input_ids.ne(1))
        output = code[0]

        output = output[:, 0, :]  # take <s> token (equiv. to [CLS])
        output = self.dropout(output)
        output = self.dense(output)
        output = torch.tanh(output)
        output = self.dropout(output)
        output = self.out_layer(output)

        logits = output
        prob = torch.softmax(logits, -1)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob
