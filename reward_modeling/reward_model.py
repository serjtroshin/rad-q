import os
from pathlib import Path
import random
from transformers import AutoModelForCausalLM, GPT2LMHeadModel, GPT2ForSequenceClassification
import torch
from torch import nn
from typing import Any, Dict, List, Optional, Tuple


class GPT2RewardModel(nn.Module):
    def __init__(self, reward_model_name="gpt2", out_features=1, loss_fn="cumulative_mse"):
        super(GPT2RewardModel, self).__init__()
        model = GPT2LMHeadModel.from_pretrained(reward_model_name)
        # model = GPT2ForSequenceClassification.from_pretrained(reward_model_name)
        model.lm_head = nn.Linear(in_features=model.lm_head.in_features, out_features=out_features, bias=True)
        # model.score = nn.Linear(in_features=model.score.in_features, out_features=out_features, bias=True)
        model.config.use_cache = False
        self.model = model
        self.pad_token_id = model.config.eos_token_id
        self.out_features = out_features
        self.loss_fn = get_loss_fn(loss_fn)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_logits: Optional[bool] = False,
    ):
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs['logits']
        # find the last valid token's ids
        sequence_lengths = (torch.ne(input_ids, self.pad_token_id).sum(-1) - 1).to(logits.device)
        # use the last valid token's representation: (batch, max_length, out_features) => (batch, out_features)
        scores = logits[torch.arange(input_ids.shape[0], device=logits.device), sequence_lengths]
        
        loss = None
        if labels is not None:
            loss = self.loss_fn(scores, labels, logits, sequence_lengths+1)
        if use_cache:
            # raise NotImplementedError("use_cache is not implemented")
            past_key_values = outputs['past_key_values']
            return loss, scores, past_key_values
        else:
            if return_logits:
                return loss, scores, logits
            return loss, scores

class EmbeddingHead(nn.Module):
    def __init__(self):
        super(EmbeddingHead, self).__init__()
        self.normalize = False

    def set_normalize(self, normalize):
        self.normalize = normalize

    def embeddings(self, lm_head: nn.Linear, index=None) -> torch.Tensor:
        """
        extracts embedding matrix
        index: torch.Tensor is optional index tensor of shape (batch, L, k)
        return: (batch, L, k, dim) if index is used and (batch, 
        """
        if index is None:
            return lm_head.weight.T
        assert index is not None
        weight=lm_head.weight
        # print("weight.shape", weight.shape)
        # print("index.shape", index.shape)
        # print(" weight[index].permute(0, 1, 3, 2)",  weight[index].permute(0, 1, 3, 2).shape)
        # input()
        """
        weight.shape torch.Size([32000, 2048])
        index.shape torch.Size([25, 11, 1024])
        weight[index].permute(0, 1, 3, 2) torch.Size([25, 11, 2048, 1024])
        """
        return weight[index].permute(0, 1, 3, 2)

    def apply_lm_head(self, h, lm_head, index=None) -> torch.Tensor:
        """
        h: tensor of shape (batch, max_length, out_features, hidden_features)
        lm_head: output embedding table of the GPT model
        """
        embeddings = self.embeddings(lm_head, index=None)
        if self.normalize:
            raise NotImplementedError("normalize is not implemented")
            # embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        result = torch.matmul(h, embeddings)  # (batch, max_length, out_features, vocab_size)
        result = result.gather(3, index.unsqueeze(2).expand(-1, -1, result.shape[2], -1))
        return result

        
# Head maps hidden state to out_features x vocab_size
class ClassifierHead(EmbeddingHead):
    def __init__(self, out_features=1, in_features=None):
        super(ClassifierHead, self).__init__()
        self.lm_projection = nn.Linear(in_features=in_features, out_features=in_features * out_features, bias=False)
        self.out_features = out_features
        self.hidden_features = in_features

    def project_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        h = self.lm_projection(hidden_states)  # (batch, max_length, out_features * hidden_features)
        h = h.reshape(h.shape[0], h.shape[1], self.out_features, self.hidden_features)  # (batch, max_length, out_features, hidden_features)
        return h

    def forward(self, lm_head, hidden_states, index=None) -> torch.Tensor:
        """
        lm_head: output embedding table of the GPT model
        hidden_states: tensor of shape (batch, L, hidden_features)
        index: tensor of shape (batch, L, k), querries output embeddings to make prediction
        """
        raise NotImplementedError("ClassifierHead forward is not implemented")
    

class MLPHeadwithBaseline(ClassifierHead):
    def __init__(self, out_features=1, lm_head=None, dim=764):
        super(MLPHeadwithBaseline, self).__init__(out_features, lm_head.in_features)

        self.w1 = nn.Linear(in_features=out_features * lm_head.out_features, out_features=dim, bias=False)
        self.w2 = nn.Linear(in_features=out_features * dim, out_features=lm_head.out_features, bias=False)
        self.dim = dim
        self.out_features = out_features

        self.prefix_baseline = nn.Linear(in_features=lm_head.in_features, out_features=out_features, bias=False)

    def apply_lm_head(self, h, lm_head, index=None) -> torch.Tensor:
        """
        h: tensor of shape (batch, max_length, out_features, hidden_features)
        lm_head: output embedding table of the GPT model
        """
        embeddings = self.embeddings(lm_head, index=None)
        result = torch.matmul(h, embeddings)  # (batch, max_length, out_features, vocab_size)
        w1 = self.w1.weight.reshape(self.out_features, lm_head.out_features, self.dim)
        w2 = self.w2.weight.reshape(self.out_features, self.dim, lm_head.out_features)
        result = torch.einsum('blij,ijk->blik', result, w1)
        result = torch.einsum('blik,ikj->blij', result, w2)  # shape is again (batch, max_length, out_features, vocab_size)

        result = result.gather(3, index.unsqueeze(2).expand(-1, -1, result.shape[2], -1))
        return result

    def forward(self, lm_head, hidden_states, index=None):
        h = self.project_head(hidden_states)  # (batch, max_length, out_features, hidden_features)
        logits = self.apply_lm_head(h, lm_head, index) # (batch, max_length, out_features, vocab_size)

        prefix_baseline = self.prefix_baseline(hidden_states)
        logits += prefix_baseline.unsqueeze(-1)
        return logits
    
    def neg_prediction(self, lm_head, hidden_states, index=None):
        h = self.project_head(hidden_states)  # (batch, max_length, out_features, hidden_features)
        logits = self.apply_lm_head(h, lm_head, index) # (batch, max_length, out_features, vocab_size)
        return logits


class LinearHead(ClassifierHead):
    def __init__(self, out_features=1, lm_head=None):
        super(LinearHead, self).__init__(out_features, lm_head.in_features)

    def forward(self, lm_head, hidden_states, index=None):
        h = self.project_head(hidden_states)  # (batch, max_length, out_features, hidden_features)
        logits = self.apply_lm_head(h, lm_head, index)
        return logits

    def neg_prediction(self, lm_head, hidden_states, index=None):
        h = self.project_head(hidden_states)
        logits = self.apply_lm_head(h, lm_head, index)
        return logits
    
    def debug(self, lm_head, hidden_states, index=None) -> dict:
        h = self.project_head(hidden_states)
        logits = self.apply_lm_head(h, lm_head, index)
        log = {
            "logits": logits.detach().cpu().tolist(),
            "prefix_baseline": None
        }
        return log 

class LinearHeadWithBaseline(LinearHead):
    def __init__(self, out_features=1, lm_head=None):
        super(LinearHeadWithBaseline, self).__init__(out_features=out_features, lm_head=lm_head)

        self.prefix_baseline = nn.Linear(in_features=lm_head.in_features, out_features=out_features, bias=False)

    def forward(self, lm_head, hidden_states, index=None):
        h = self.project_head(hidden_states)  # (batch, max_length, out_features, hidden_features)
        logits = self.apply_lm_head(h, lm_head, index) # (batch, max_length, out_features, vocab_size)

        prefix_baseline = self.prefix_baseline(hidden_states)
        logits += prefix_baseline.unsqueeze(-1)
        return logits
    
    def neg_prediction(self, lm_head, hidden_states, index=None):
        h = self.project_head(hidden_states)  # (batch, max_length, out_features, hidden_features)
        logits = self.apply_lm_head(h, lm_head, index) # (batch, max_length, out_features, vocab_size)
        return logits
    
    def debug(self, lm_head, hidden_states, index=None) -> dict:
        h = self.project_head(hidden_states)
        logits = self.apply_lm_head(h, lm_head, index)
        prefix_baseline = self.prefix_baseline(hidden_states)
        log = {
            "logits": logits.detach().cpu().tolist(),
            "prefix_baseline": prefix_baseline.detach().cpu().tolist()
        }
        return log 


def get_lm_head(name):
    if name == "linear":
        return LinearHead
    elif name == "linear_with_baseline":
        return LinearHeadWithBaseline
    elif name == "mlp_with_baseline":
        return MLPHeadwithBaseline
    else:
        raise ValueError(f"lm_head name {name} not available")


class GPT2RewardSoftModel(nn.Module):
    _tied_weights_keys = ["model.lm_head.weight"]

    def __init__(self, reward_model_name="gpt2", out_features=1, loss_fn="cumulative_mse", lm_head_name="linear", freeze_rm=False, freeze_embeddings=True, embeddings_reg_loss="none", project_emb_on_sphere=False, n_components=None, regularization=None, prepend_bos_token: bool=None, bos_token_id=None, max_length=1024,
               ):
        super(GPT2RewardSoftModel, self).__init__()
        model = GPT2LMHeadModel.from_pretrained(reward_model_name)

        assert prepend_bos_token is not None
        self.prepend_bos_token = prepend_bos_token
        self.bos_token_id = bos_token_id
        self.max_length = max_length
    
        # model.lm_head_orig = model.lm_head  # save the original lm_head in_features -> vocab_size
        # in_features -> in_features*out_features -> vocab_size * out_features
        self.hidden_features = model.lm_head.in_features
        self.vocab_size = model.config.vocab_size

        model.config.use_cache = True
        self.model = model
        self.pad_token_id = model.config.eos_token_id
        self.out_features = out_features
        self.loss_fn = get_loss_fn(loss_fn)
        self.embeddings_reg_loss = get_emb_loss_fn(embeddings_reg_loss)
        self.old_embeddings = model.lm_head.weight.clone().detach()
        self.regularization = regularization

        kwargs={}
        if n_components is not None:
            kwargs["n_components"] = n_components
        self.head = get_lm_head(lm_head_name)(out_features=out_features, lm_head=model.lm_head, **kwargs)
        print(self.head)
        
        if freeze_rm:
            # freeze all weights of the self.model except last layer
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.lm_head.weight.requires_grad = True

        if freeze_embeddings:
            self.model.lm_head.weight.requires_grad = False
        if project_emb_on_sphere:
            print("projecting on sphere")
            self.head.set_normalize(True)

        # print all trainable parameters
        print("Trainable parameters")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, end="; ")
        print()

        self.debug_head_outputs = os.environ.get("DEBUG_HEAD", None)
        self._debug_log = None

    def forward_GPT(
        _self,
        self: GPT2LMHeadModel,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        if use_cache:
            past_key_values = transformer_outputs['past_key_values']
        else:
            past_key_values = None

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        return hidden_states, past_key_values

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        querry: Optional[torch.Tensor] = None,  # querry for the next token candidates (batch, L, k)
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        neg_targets: Optional[bool] = None,  # used for baseline regularization
    ):

        if self.prepend_bos_token:
            assert input_ids.shape[1] <= self.max_length - 1
            input_ids = torch.cat(
                [torch.full((input_ids.shape[0], 1), self.bos_token_id, device=input_ids.device, dtype=input_ids.dtype), input_ids], dim=1
            )
            attention_mask = torch.cat(
                [torch.ones((input_ids.shape[0], 1), device=input_ids.device, dtype=input_ids.dtype), attention_mask], dim=1
            )

        outputs, past_key_values = self.forward_GPT(self.model,
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.debug_head_outputs is not None:
            log = self.head.debug(self.model.lm_head, outputs[:, :-1], index=input_ids[:, 1:].unsqueeze(-1))
            log['input_ids'] = input_ids.detach().cpu().tolist()
            self._debug_log = log
            print("log", log, flush=True)

        if labels is not None:  # training mode
            
            # 1. select the logits for next tokens
            input_lengths = torch.ne(input_ids, self.pad_token_id).sum(-1)  # -1

            target_ids = input_ids[:, 1:]

            # next_input_ids_for_gather = next_input_ids.view(*next_input_ids.size(), 1, 1).expand(-1, -1, self.out_features, 1)
            target_ids = target_ids.unsqueeze(-1)

            predictions = self.head(self.model.lm_head, outputs[:, :-1], index=target_ids).squeeze(-1)

            # prev_logits = logits[:, :-1, ...]

            scores = predictions[torch.arange(input_ids.shape[0], device=predictions.device), input_lengths-1]

            # gathered_predictions = torch.gather(prev_logits, 3, next_input_ids_for_gather).squeeze(-1)
            loss = self.loss_fn(scores, labels, predictions, input_lengths)

            neg_targets=None
            reg_loss = 0.
            if self.regularization is not None:
                if "baseline_prior" in self.regularization:
                    neg_targets = get_regularization_targets(self.regularization, input_ids[:, :-1], self.vocab_size)
                    for neg_target in neg_targets:
                        neg_prediction = self.head.neg_prediction(self.model.lm_head, outputs[:, :-1], index=neg_target.unsqueeze(-1)).squeeze(-1)
                        # to do add mask
                        # use attention_mask to mask the loss
                        # print("neg_prediction", neg_prediction.shape)
                        # print("attention_mask", attention_mask.shape)
                        mask = attention_mask[:, :-1].unsqueeze(-1).repeat(1, 1, self.out_features).bool()
                        assert mask.shape == neg_prediction.shape
                        assert mask.dtype == torch.bool

                        neg_prediction = neg_prediction[mask]

                        reg_loss += nn.MSELoss()(neg_prediction, torch.zeros_like(neg_prediction))   # regularization coefficient
                elif "weight_norm" in self.regularization:
                    # weight norm regularization
                    reg_loss = self.head.weight_norm() * 0.1
                if random.random() < 0.005:
                    print("reg_loss", reg_loss, "loss", loss.item(), flush=True)

            loss += reg_loss
            scores = predictions[torch.arange(input_ids.shape[0], device=predictions.device), input_lengths-1]
            if self.loss_fn.__name__ == "cumulative_ce_fn":
                scores = nn.functional.sigmoid(scores)
            else:
                pass
            return loss, scores
        else:
            logits = self.head(self.model.lm_head, outputs, index=querry)
            if self.loss_fn.__name__ == "cumulative_ce_fn":
                logits = nn.functional.sigmoid(logits)
            else:
                pass

            if neg_targets is not None:
                if isinstance(neg_targets, torch.Tensor):
                    neg_targets = [neg_targets]
                neg_loss = 0.0
                for neg_target in neg_targets:
                    neg_predictions = self.head.neg_prediction(self.model.lm_head, outputs, index=neg_target.unsqueeze(-1))
                    masked_neg_predictions = neg_predictions
                    neg_loss += nn.MSELoss()(masked_neg_predictions, torch.zeros_like(masked_neg_predictions))
                return logits, neg_loss

        if use_cache:
            return logits, past_key_values
        return logits


class DistillationModel(nn.Module):
    """
    Used only for training the student model with distillation loss
    """
    _tied_weights_keys = ["student_model.model.lm_head.weight"]

    def __init__(self, teacher_model: GPT2RewardModel, student_model: GPT2RewardSoftModel, regularization=None):
        super(DistillationModel, self).__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.pad_token_id = teacher_model.model.config.eos_token_id
        self.regularization = regularization

        self._log = {"reg_loss": 0.0, "loss": 0.0}

    def teacher_loss(self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        # student_logits: torch.Tensor,
    ):
        # teacher forcing
        with torch.no_grad():
            # for teacher model we select teacher_logits[:, 1:, ...] (toxicity of the future token)
            teacher_loss, teacher_scores, teacher_logits = self.teacher_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_logits=True)
            
            # for the studen model we select the teacher forcing logits
            if self.student_model.prepend_bos_token:
                querry = input_ids
            else:
                querry = input_ids[:, 1:]
                teacher_logits = teacher_logits[:, 1:, ...]

        

            # teacher_input_ids_for_gather = teacher_input_ids.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, student_logits.shape[-2], 1)
        # evaluate top-k logits

        # pass inputs to the student model
        neg_targets=None
        reg_loss = 0.
        if self.regularization is not None:
            neg_targets = get_regularization_targets(self.regularization, querry, self.student_model.vocab_size)
            student_logits, reg_loss = self.student_model(input_ids=input_ids[:, :-1], attention_mask=attention_mask[:, :-1], use_cache=False, querry=querry.unsqueeze(-1), neg_targets=neg_targets)
        else:
            student_logits = self.student_model(input_ids=input_ids[:, :-1], attention_mask=attention_mask[:, :-1], use_cache=False, querry=querry.unsqueeze(-1), neg_targets=None)
        student_logits=student_logits.squeeze(-1)

        # for student model we select student_logits[:, :-1, ...] (toxicity of the current prefix for all future tokens)
        # student_logits = student_logits[:, :-1, ...]
        # select the right student scores  (bs, seq_len, out_features, vocab_size) -> (bs, seq_len, out_features)
        # student_logits = torch.gather(student_logits, 3, teacher_input_ids_for_gather).squeeze(-1)

        sequence_lengths = (torch.ne(querry, self.pad_token_id).sum(-1)).to(student_logits.device)
        mask = torch.arange(querry.shape[1], device=student_logits.device).unsqueeze(0) < sequence_lengths.unsqueeze(1)

        # print("student_logits", student_logits.shape, "teacher_logits", teacher_logits.shape, "mask", mask.shape, flush=True)
        student_masked_logits = student_logits[mask]
        teacher_masked_logits = teacher_logits[mask]

        # mse loss
        loss = nn.MSELoss()(student_masked_logits, teacher_masked_logits) + reg_loss
        if random.random() < 0.01:
            if self.regularization is not None:
                print("reg_loss", reg_loss.item(), "loss", loss.item(), flush=True)
        
        return loss, student_logits

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        # student_logits = self.student_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        loss, student_logits = self.teacher_loss(input_ids, attention_mask)
        sequence_lengths = (torch.ne(input_ids, self.pad_token_id).sum(-1) - 1).to(student_logits.device)
        logits_for_last_tokens = student_logits[torch.arange(input_ids.shape[0], device=student_logits.device), sequence_lengths-2]
       
        scores = logits_for_last_tokens
        return loss, scores

def generate_random_like(input_ids, vocab_size):
    # generate random tokens for the student model
    return torch.randint(0, vocab_size, input_ids.shape, device=input_ids.device)

def get_regularization_targets(regularization, input_ids, vocab_size) -> List[torch.Tensor]:
    if regularization == "baseline_prior":
        neg_targets = [generate_random_like(input_ids, vocab_size)]
    else:
        raise ValueError(f"regularization {regularization} not available")
    return neg_targets
            
def get_loss_fn(name):
    if name == "mse":
        def mse_loss_fn(scores, labels, logits, lengths):
            return nn.MSELoss()(scores, labels)
        
        loss_fn = mse_loss_fn

    elif name == "cross_entropy":
        def ce_loss_fn(scores, labels, logits, lengths):
            return nn.CrossEntropyLoss()(scores, labels)    # here score is logits[last_token_id]
        
        loss_fn = ce_loss_fn

    elif name == "cumulative_mse":
        def cumulative_mse_fn(scores, labels, logits, lengths):
            if labels.ndim == 1:
                labels = labels.unsqueeze(-1)  # 1 feature

            s_l = 2. / (lengths * (lengths + 1))
            t = torch.arange(1, lengths.max()+1, device=logits.device).float()
            s_t = s_l.unsqueeze(-1) * t.unsqueeze(0)
            mask = (t.unsqueeze(0) <= lengths.unsqueeze(-1))

            labels = (labels.unsqueeze(1).repeat(1, lengths.max(), 1)).float()

            masked_logits = logits[mask]
            masked_future_labels = labels[mask]
            masked_weights = s_t[mask]
            assert masked_logits.shape == masked_future_labels.shape, f"{masked_logits.shape} != {masked_future_labels.shape}"

            # if prediction is > 1 put higher fine
            loss = nn.MSELoss(reduction="none")(masked_logits, masked_future_labels) * masked_weights[:, None]
            loss = loss.sum() / logits.shape[0]
            return loss
        
        loss_fn = cumulative_mse_fn

    elif name == "last_token_mse":
        def last_token_mse_fn(scores, labels, logits, lengths):
            if labels.ndim == 1:
                labels = labels.unsqueeze(-1)  # 1 feature
            loss = nn.MSELoss(reduction="none")(scores, labels)
            loss = loss.sum() / logits.shape[0]
            return loss
        
        loss_fn = last_token_mse_fn
        
    elif name == "cumulative_ce":
        def cumulative_ce_fn(scores, labels, logits, lengths):
            if labels.ndim == 1:
                labels = labels.unsqueeze(-1)  # 1 feature

            s_l = 2. / (lengths * (lengths + 1))
            t = torch.arange(1, lengths.max()+1, device=logits.device).float()
            s_t = s_l.unsqueeze(-1) * t.unsqueeze(0)
            mask = (t.unsqueeze(0) <= lengths.unsqueeze(-1))

            labels = (labels.unsqueeze(1).repeat(1, lengths.max(), 1)).float()

            masked_logits = logits[mask]
            masked_future_labels = labels[mask]
            masked_weights = s_t[mask]
            assert masked_logits.shape == masked_future_labels.shape, f"{masked_logits.shape} != {masked_future_labels.shape}"

            # if prediction is > 1 put higher fine
            loss = nn.BCEWithLogitsLoss(reduction="none")(masked_logits, masked_future_labels)
            loss = loss * masked_weights[:, None]
            loss = loss.sum() / logits.shape[0]
            return loss
        
        loss_fn = cumulative_ce_fn
        
    else:
        raise ValueError(f"loss function name {name} not available")
    
    return loss_fn


def get_emb_loss_fn(name):
    if name == "none":
        return None
    elif name == "mse_pretrained":
        # MSE(new_emb, old_emb)
        def mse_pretrained_loss_fn(new_emb, old_embeddings):
            return nn.MSELoss()(new_emb, old_embeddings)
        return mse_pretrained_loss_fn
    else:
        raise ValueError(f"loss function name {name} not available")