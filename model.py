from util import *

class BertForModel(BertPreTrainedModel):
    def __init__(self, config, num_labels, use_rp=False, rp_dim=None):
        super(BertForModel, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.use_rp = use_rp
        if use_rp:
            self.W_rand = torch.randn(config.hidden_size, rp_dim)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None,
                feature_ext=False, mode=None, centroids=None):

        encoded_layer_12, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                    output_all_encoded_layers=True)
        pooled_output = self.dense(encoded_layer_12[-1].mean(dim=1))
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)

        if self.use_rp:
            pooled_output = torch.nn.functional.relu(pooled_output @ self.W_rand)

        logits = self.classifier(pooled_output)

        if feature_ext:
            return pooled_output
        else:
            if mode == 'train':
                loss = nn.CrossEntropyLoss()(logits, labels)
                return loss
            else:
                return pooled_output, logits


