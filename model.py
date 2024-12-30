from transformers import AutoTokenizer, BertForMaskedLM, RobertaForMaskedLM, AutoConfig
from transformers.models.bert.modeling_bert import BertForMaskedLM
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import PreTrainedModel
import torch
from torch import nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import torch.nn.functional as F
from typing import Optional

import wandb

import numpy as np


class Composer(nn.Module):
    """
    Parameter-free methods for aggregrating phrase embeddings
    'avg': average of the [CLS] tokens
    'max': max pooling of the [CLS] tokens
    'cat_first_last': concatenation of the first and last halves of the [CLS] tokens - works well for RoBERTa-based models
    """
    def __init__(self, compose_fn, model_name):
        super().__init__()
        self.compose_fn = compose_fn
        if 'base' in model_name:
            self.half_dim = 384
        elif 'large' in model_name:
            self.half_dim = 512
        assert self.compose_fn in ["avg", "max", "cat_first_last"], "unrecognized pooling type %s" % self.compose_fn

    def forward(self, r_left, r_right):
        if self.compose_fn == 'avg':
            r_pos = 0.5 * (r_left + r_right)
            return r_pos
        elif self.compose_fn == 'max':
            r_pos = torch.max(torch.cat([r_left, r_right], dim=1), dim=1)
            return r_pos
        elif self.compose_fn == 'cat_first_last':
            r_pos = torch.cat([r_left[:, :, :self.half_dim], r_right[:, :, self.half_dim:]], dim=-1)
            return r_pos
        else:
            raise NotImplementedError



def gaussian_kl(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)


def get_layer_embedding(outputs,batch_size):
    num_layers = len(outputs.hidden_states)
    for layer_idx, layer_hidden_state in enumerate(outputs.hidden_states):
        sum_hidden_state_start = torch.zeros_like(outputs.hidden_states[0][:batch_size])

        num_layers_to_sum = num_layers // 3
        for layer_idx in range(num_layers_to_sum):
            sum_hidden_state_start += outputs.hidden_states[layer_idx][:batch_size]
        mu1 = sum_hidden_state_start / num_layers_to_sum
        esp1 = mu1.data.new(mu1.size()).normal_(0., 0.01)
        h1 = mu1 + esp1

        sum_hidden_state_middle = torch.zeros_like(outputs.hidden_states[0][:batch_size])
        for layer_idx in range(num_layers_to_sum ,2 * num_layers_to_sum):
            sum_hidden_state_middle += outputs.hidden_states[layer_idx][:batch_size]
        mu2 = sum_hidden_state_middle / num_layers_to_sum
        esp2 = mu2.data.new(mu2.size()).normal_(0., 0.01)
        h2 = mu2+ esp2

        sum_hidden_state_end = torch.zeros_like(outputs.hidden_states[0][:batch_size])
        for layer_idx in range(2 * num_layers_to_sum, num_layers):
            sum_hidden_state_end += outputs.hidden_states[layer_idx][:batch_size]
        mu3 = sum_hidden_state_end / (num_layers - 2 *num_layers_to_sum)
        esp3 = mu3.data.new(mu3.size()).normal_(0., 0.01)
        h3 = mu3 + esp3

        return mu1, mu2, mu3, h1, h2, h3


def InfoNCE(mu, z):
    mu = mu.unsqueeze(0)
    z = z.unsqueeze(1)
    score = -((z-mu)**2).sum(-1)/20.
    lower_bound = -score.logsumexp(dim=1).mean()
    return lower_bound


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def dcm_loss(f_a, f_b):
    f_a_norm = (f_a - f_a.mean(0)) / (f_a.std(0)+1e-6)
    f_b_norm = (f_b - f_b.mean(0)) / (f_b.std(0)+1e-6)
    c = torch.mm(f_a_norm.T, f_b_norm) / f_a_norm.size(0)

    on_diag = torch.diagonal(c).add_(-1).pow_(2).mean()
    off_diag = off_diagonal(c).pow_(2).mean()
    loss = 2 * on_diag + 0.05 * off_diag

    return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device='cuda', temperature=0.05):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))			# 超参数 温度
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())		# 主对角线为0，其余位置全为1的mask矩阵
        
    def forward(self, emb_i, emb_j):		# emb_i, emb_j 是来自同一句子的两种不同的预处理方法得到
        z_i = F.normalize(emb_i, dim=1)     # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)     # (bs, dim)  --->  (bs, dim)

        representations = torch.cat([z_i, z_j], dim=0)          # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)      # simi_mat: (2*bs, 2*bs)
        
        sim_ij = torch.diag(similarity_matrix, self.batch_size)         # bs
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)        # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)                  # 2*bs
        
        nominator = torch.exp(positives / self.temperature)             # 2*bs
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)             # 2*bs, 2*bs
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))        # 2*bs
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss


class MSSEModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.pooler = config.pooler
        self.sent_embedding_projector = nn.Linear(config.hidden_size, config.hidden_size)
        self.decoder = TransformerDecoder(TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.decoder_num_heads, batch_first=True, dropout=0.1), num_layers=config.decoder_num_layers)
        self.decoder_noise_dropout = nn.Dropout(config.decoder_noise_dropout)
        self.sim = nn.CosineSimilarity(dim=-1)
        self.init_weights()
        
        self.mu_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.var_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.encoder_name_or_path)
        if "bert" in config.encoder_name_or_path:
            self.encoder = BertForMaskedLM.from_pretrained(config.encoder_name_or_path)
            self.prediction_head = self.encoder.cls
            self.encoder = self.encoder.bert
        elif "roberta" in config.encoder_name_or_path:
            self.encoder = RobertaForMaskedLM.from_pretrained(config.encoder_name_or_path)
            self.prediction_head = self.encoder.lm_head
            self.encoder = self.encoder
        else:
            raise NotImplementedError
        self.post_init()
        
    def set_trainer_parameters(self, trainer):
            self.global_step = trainer.state.global_step

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def encode(self, sentences, batch_size=32, **kwargs):
        """ Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
        self.eval()
        all_embeddings = []
        length_sorted_idx = np.argsort([len(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
        if self.config.pooler == 'mask':
            prompt_length = len(self.tokenizer(self.config.prompt_format, add_special_tokens=False)['input_ids'])
            sentences_sorted = self.tokenizer.batch_decode(self.tokenizer(sentences_sorted, padding=True, truncation=True, max_length=self.config.max_length, return_tensors='pt').input_ids, skip_special_tokens=True)
            sentences_sorted = [self.config.prompt_format.replace('[X]', s).replace('[MASK]', self.tokenizer.mask_token) for s in sentences_sorted]
        for start_index in range(0, len(sentences), batch_size):
            sentences_batch = sentences_sorted[start_index:start_index+batch_size]
            max_length = self.config.max_length
            if self.config.pooler == "mask":
                max_length=self.config.max_length + prompt_length
            inputs = self.tokenizer(sentences_batch, padding='max_length', truncation=True, return_tensors="pt", max_length=max_length)
            inputs = {k: v.to(self.device) for k,v in inputs.items()}
            with torch.no_grad():
                encoder_outputs = self.encoder(**inputs, output_hidden_states=True, output_attentions=True, return_dict=True)
                last_hidden_state = encoder_outputs.last_hidden_state
                if self.config.pooler == 'cls':
                    embeddings = last_hidden_state[:, 0, :]
                elif self.config.pooler == 'mean':
                    embeddings = (last_hidden_state * inputs['attention_mask'].unsqueeze(-1)).sum(1) / inputs['attention_mask'].sum(-1).unsqueeze(-1)
                elif self.pooler == 'mask':
                    embeddings = last_hidden_state[inputs['input_ids'] == self.tokenizer.mask_token_id]
                else:
                    raise NotImplementedError()
            all_embeddings.extend(embeddings.cpu().numpy())
        all_embeddings = torch.tensor(np.array([all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]))
        return all_embeddings
    

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            positive_input_ids: Optional[torch.LongTensor] = None,
            positive_attention_mask: Optional[torch.LongTensor] = None,
            negative_input_ids: Optional[torch.LongTensor] = None,
            negative_attention_mask: Optional[torch.LongTensor] = None,
            global_step: Optional[int] = None,
            max_steps: Optional[int] = None,
    ):
        batch_size = input_ids.size(0)
        if negative_input_ids is not None:
            encoder_input_ids = torch.cat([input_ids, positive_input_ids, negative_input_ids], dim=0).to(self.device)
            encoder_attention_mask = torch.cat([attention_mask, positive_attention_mask, negative_attention_mask], dim=0).to(self.device)
        elif positive_input_ids is not None:
            encoder_input_ids = torch.cat([input_ids, positive_input_ids], dim=0).to(self.device)
            encoder_attention_mask = torch.cat([attention_mask, positive_attention_mask], dim=0).to(self.device)
        elif self.config.do_contrastive:
            encoder_input_ids = torch.cat([input_ids, input_ids], dim=0).to(self.device)
            encoder_attention_mask = torch.cat([attention_mask, attention_mask], dim=0).to(self.device)
        elif self.config.do_generative and not self.config.do_contrastive:
            encoder_input_ids = input_ids.to(self.device)
            encoder_attention_mask = attention_mask.to(self.device)
        else:
            raise NotImplementedError()
        
        torch.cuda.empty_cache()
        encoder_outputs = self.encoder(input_ids=encoder_input_ids, attention_mask=encoder_attention_mask, return_dict=True, output_hidden_states=True, output_attentions=True)
        if self.pooler == 'cls':
            # encoder_output:[32,42,768] 
            sent_embedding = encoder_outputs.last_hidden_state[:, 0, :]
        elif self.pooler == 'mean':
            sent_embedding = ((encoder_outputs.last_hidden_state * encoder_attention_mask.unsqueeze(-1)).sum(1) / encoder_attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler == 'mask':
             sent_embedding = encoder_outputs.last_hidden_state[encoder_input_ids == self.tokenizer.mask_token_id]
        else:
            raise NotImplementedError()

        if negative_input_ids is not None:
            num_sent = 3
        elif positive_input_ids is not None:
            num_sent = 2
        else:
            num_sent = 1
    
        pooler_embedding = encoder_outputs.last_hidden_state[:, 0, :].view(batch_size, num_sent, -1)
        embedding_ori = pooler_embedding[:,0,:]
        embedding_aug = pooler_embedding[:,1,:]
        loss_dcm = dcm_loss(embedding_ori, embedding_aug)
        wandb.log({'train/loss_dcm': loss_dcm})

        mu1, mu2, mu3, h1, h2, h3 = get_layer_embedding(encoder_outputs, batch_size)
        
        mu1, mu2, mu3, h1, h2, h3 = mu1[:,0,:], mu2[:,0,:], mu3[:,0,:], h1[:,0,:], h2[:,0,:], h3[:,0,:]
        MI_estimitor = 0.25 * InfoNCE(mu1, h1) + 0.50 * InfoNCE(mu2, h2) + InfoNCE(mu3, h3)
        wandb.log({'train/MI_estimitor': MI_estimitor})

        sent_embedding = sent_embedding.unsqueeze(1)
        sent_embedding = self.sent_embedding_projector(sent_embedding)

        if self.config.do_generative:
            if positive_input_ids is not None:
                tgt = encoder_outputs.hidden_states[0][batch_size:2*batch_size].detach()
                tgt_key_padding_mask = (positive_input_ids == self.tokenizer.pad_token_id)
                labels = positive_input_ids
            else:
                tgt = encoder_outputs.hidden_states[0][:batch_size].detach()
                tgt_key_padding_mask = (input_ids == self.tokenizer.pad_token_id)
                labels = input_ids
            tgt = self.decoder_noise_dropout(tgt)

            decoder_outputs = self.decoder(tgt=tgt, memory=sent_embedding[:batch_size], tgt_mask=None, tgt_key_padding_mask=tgt_key_padding_mask)

            # tgt = self.decoder_noise_dropout(tgt)
            logits = self.prediction_head(decoder_outputs)
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            icm_loss = loss_fct(logits.view(-1, self.encoder.config.vocab_size), labels.view(-1))
            wandb.log({'train/icm_loss': icm_loss})
        
        if self.config.do_contrastive:
            positive_sim = self.sim(sent_embedding[:batch_size], sent_embedding[batch_size:2*batch_size].transpose(0, 1))
            cos_sim = positive_sim

            cos_sim = cos_sim / self.config.contrastive_temp
        
            contrastive_labels = torch.arange(batch_size, dtype=torch.long, device=self.device)
            contrastive_loss = nn.CrossEntropyLoss()(cos_sim, contrastive_labels)
            wandb.log({'train/contrastive_loss': contrastive_loss.item()})
            logits = None
        
        loss = 0

        if self.config.do_contrastive:
            loss += contrastive_loss
        if self.config.do_generative: 
            loss = loss  + icm_loss 
        
        loss = loss  - 0.8 * MI_estimitor + 0.2 * dcm_loss
      
        wandb.log({'train/loss': loss})
        return TokenClassifierOutput(
            loss=loss,  
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
