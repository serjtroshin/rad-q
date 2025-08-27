from pathlib import Path
import random
from transformers import AutoModelForCausalLM, GPT2LMHeadModel, GPT2ForSequenceClassification, LlamaForCausalLM
import torch
from torch import nn
from typing import Any, Dict, List, Optional, Tuple

# add reward_modling to the path
import sys
sys.path.append("reward_modeling")
from reward_model import get_lm_head, get_regularization_targets

COUNTER = 0


class LlamaRewardModel(nn.Module):
    def __init__(self, reward_model_name, out_features, loss_fn="cumulative_mse"):
        super(LlamaRewardModel, self).__init__()
        model = LlamaForCausalLM.from_pretrained(reward_model_name)

        model.lm_head = nn.Linear(in_features=model.lm_head.in_features, out_features=out_features, bias=True)
        model.config.use_cache = False
        
        self.model = model
        self.pad_token_id = model.config.eos_token_id
        self.out_features = out_features
        self.loss_fn = get_loss_fn(loss_fn)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        return_logits: Optional[bool] = False,
    ):

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        logits = outputs['logits']
        # find the last valid token's ids
        if attention_mask is not None:
            sequence_lengths = (attention_mask.sum(-1) - 1).to(logits.device)
        else:
            # assuming full attention mask for decoding with the same prompts
            sequence_lengths = input_ids.new_full((input_ids.shape[0],), input_ids.shape[1]-1)
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

            labels = (labels.unsqueeze(1).repeat(1, mask.shape[1], 1)).float()

            masked_logits = logits[mask]

            masked_future_labels = labels[mask]
            masked_weights = s_t[mask]
            assert masked_logits.shape == masked_future_labels.shape, f"{masked_logits.shape} != {masked_future_labels.shape}"

            # if prediction is > 1 put higher fine
            loss = nn.MSELoss(reduction="none")(masked_logits, masked_future_labels) * masked_weights[:, None]
            loss = loss.sum() / logits.shape[0]
            return loss
        
        loss_fn = cumulative_mse_fn
        
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

# RAD-Q model
class LlamaARMModel(nn.Module):
    _tied_weights_keys = ["model.lm_head.weight"]

    def __init__(self, reward_model_name, out_features=1, loss_fn="cumulative_mse", lm_head_name="linear", freeze_rm=False, freeze_embeddings=True, regularization=None, max_length=4098):
        super(LlamaARMModel, self).__init__()
        model = LlamaForCausalLM.from_pretrained(reward_model_name)

        self.max_length = max_length
    
        # model.lm_head_orig = model.lm_head  # save the original lm_head in_features -> vocab_size
        # in_features -> in_features*out_features -> vocab_size * out_features
        self.hidden_features = model.lm_head.in_features
        self.vocab_size = model.config.vocab_size

        model.config.use_cache = True
        self.config = model.config
        self.model = model
        self.pad_token_id = model.config.eos_token_id
        self.out_features = out_features
        self.loss_fn = get_loss_fn(loss_fn)
        self.regularization = regularization

        self.head = get_lm_head(lm_head_name)(out_features=out_features, lm_head=model.lm_head)
        
        if freeze_embeddings:
            self.model.lm_head.weight.requires_grad = False

        # print all trainable parameters
        print("Trainable parameters")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, end="; ")
        print()

    def forward_Llama(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        return_logits: Optional[bool] = False,
    ):

        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if use_cache:
            past_key_values = outputs['past_key_values']
        else:
            past_key_values = None

        return hidden_states, past_key_values

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        querry: Optional[torch.Tensor] = None,  # querry for the next token candidates (batch, L, k)
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        return_logits: Optional[bool] = False,
        neg_targets: Optional[bool] = None,  # used for baseline regularization
    ):

        # Llama already has bos token
        outputs, past_key_values = self.forward_Llama(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            return_logits=return_logits
        )

        if labels is not None:  # training mode
            # 1. select the logits for next tokens
            # input_lengths = torch.ne(input_ids, self.pad_token_id).sum(-1)  # -1
            if attention_mask is not None:
                input_lengths = (attention_mask.sum(-1)).to(outputs.device) - 1
            else:
                # assuming full attention mask for decoding with the same prompts
                input_lengths = input_ids.new_full((input_ids.shape[0],), input_ids.shape[1]) - 1

            target_ids = input_ids[:, 1:]

            target_ids = target_ids.unsqueeze(-1)

            predictions = self.head(self.model.lm_head, outputs[:, :-1], index=target_ids).squeeze(-1)

            loss = self.loss_fn(None, labels, predictions, input_lengths)

            neg_targets=None
            reg_loss = 0.
            if self.regularization is not None:
                if "baseline_prior" in self.regularization:
                    neg_targets = get_regularization_targets(self.regularization, input_ids[:, :-1], self.vocab_size)
                    for neg_target in neg_targets:
                        neg_prediction = self.head.neg_prediction(self.model.lm_head, outputs[:, :-1], index=neg_target.unsqueeze(-1)).squeeze(-1)
                        mask = attention_mask[:, :-1].unsqueeze(-1).repeat(1, 1, self.out_features).bool()
                        assert mask.shape == neg_prediction.shape
                        assert mask.dtype == torch.bool

                        neg_prediction = neg_prediction[mask]

                        reg_loss += nn.MSELoss()(neg_prediction, torch.zeros_like(neg_prediction))   # regularization coefficient
                elif "weight_norm" in self.regularization:
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

class LlamaDeltaDistillationModel(nn.Module):
    """
    Used only for training the student model with distillation loss
    """
    _tied_weights_keys = ["student_model.model.lm_head.weight"]

    def __init__(self, teacher_model: LlamaRewardModel, student_model: LlamaARMModel, regularization=None):
        super(LlamaDeltaDistillationModel, self).__init__()
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

            ########  delta teacher logits
            prev_logits = torch.cat(
                (teacher_logits.new_zeros(teacher_logits.shape[0], 1, teacher_logits.shape[-1]), teacher_logits[:, :-1, ...]), dim=1
            )
            teacher_logits = teacher_logits - prev_logits
            
            # delta logits
            ########
            
            querry = input_ids[:, 1:]
            teacher_logits = teacher_logits[:, 1:, ...]

        # pass inputs to the student model
        neg_targets=None
        reg_loss = 0.
        if self.regularization is not None:
            neg_targets = get_regularization_targets(self.regularization, querry, self.student_model.vocab_size)
            student_logits, reg_loss = self.student_model(input_ids=input_ids[:, :-1], attention_mask=attention_mask[:, :-1], use_cache=False, querry=querry.unsqueeze(-1), neg_targets=neg_targets)
        else:
            student_logits = self.student_model(input_ids=input_ids[:, :-1], attention_mask=attention_mask[:, :-1], use_cache=False, querry=querry.unsqueeze(-1), neg_targets=None)
        student_logits=student_logits.squeeze(-1)

        sequence_lengths = (attention_mask.sum(-1)).to(student_logits.device)
        mask = torch.arange(querry.shape[1], device=student_logits.device).unsqueeze(0) < sequence_lengths.unsqueeze(1)

        student_masked_logits = student_logits[mask]
        teacher_masked_logits = teacher_logits[mask]

        # mse loss
        loss = nn.MSELoss()(student_masked_logits, teacher_masked_logits) + reg_loss
        if random.random() < 0.01:
            if self.regularization is not None:
                print("reg_loss", reg_loss.item(), "loss", loss.item(), flush=True)

        masked_cum_sum_student_logits = torch.cumsum(student_logits * mask.float().unsqueeze(-1), dim=1)
        return loss, masked_cum_sum_student_logits  # for training use cumulative delta scores

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        loss, student_logits = self.teacher_loss(input_ids, attention_mask)
        sequence_lengths = (attention_mask.sum(-1) - 1).to(student_logits.device)
        logits_for_last_tokens = student_logits[torch.arange(input_ids.shape[0], device=student_logits.device), sequence_lengths-2]
       
        scores = logits_for_last_tokens
        return loss, scores


class LlamaDistillationModel(nn.Module):
    """
    Used only for training the student model with distillation loss
    """
    _tied_weights_keys = ["student_model.model.lm_head.weight"]

    def __init__(self, teacher_model: LlamaRewardModel, student_model: LlamaARMModel, regularization=None):
        super(LlamaDistillationModel, self).__init__()
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
            
            
            querry = input_ids[:, 1:]
            teacher_logits = teacher_logits[:, 1:, ...]

        # pass inputs to the student model
        neg_targets=None
        reg_loss = 0.
        if self.regularization is not None:
            neg_targets = get_regularization_targets(self.regularization, querry, self.student_model.vocab_size)
            student_logits, reg_loss = self.student_model(input_ids=input_ids[:, :-1], attention_mask=attention_mask[:, :-1], use_cache=False, querry=querry.unsqueeze(-1), neg_targets=neg_targets)
        else:
            student_logits = self.student_model(input_ids=input_ids[:, :-1], attention_mask=attention_mask[:, :-1], use_cache=False, querry=querry.unsqueeze(-1), neg_targets=None)
        student_logits=student_logits.squeeze(-1)

        # sequence_lengths = (torch.ne(querry, self.pad_token_id).sum(-1)).to(student_logits.device)
        sequence_lengths = (attention_mask.sum(-1)).to(student_logits.device)
        mask = torch.arange(querry.shape[1], device=student_logits.device).unsqueeze(0) < sequence_lengths.unsqueeze(1)
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
        loss, student_logits = self.teacher_loss(input_ids, attention_mask)
        sequence_lengths = (attention_mask.sum(-1) - 1).to(student_logits.device)
        logits_for_last_tokens = student_logits[torch.arange(input_ids.shape[0], device=student_logits.device), sequence_lengths-2]
       
        scores = logits_for_last_tokens
        return loss, scores