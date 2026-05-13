import os
import torch
from typing import Optional, Tuple, Union, List
from transformers import AutoTokenizer, AutoConfig, PreTrainedTokenizerFast, logging, GemmaForCausalLM, LlamaForCausalLM, MistralForCausalLM, PhiForCausalLM, Qwen2ForCausalLM, Qwen3ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast, CausalLMOutputWithCrossAttentions
from huggingface_hub import hf_hub_download
from sven.hf import CodeGenForCausalLM, XGLMForCausalLM, GPT2LMHeadCustomModel, GPT2CustomConfig


def _load_tokenizer(path):
    # AutoTokenizer mis-resolves deepseek-ai/deepseek-coder-* as LlamaTokenizer,
    # which drops the byte-BPE Ġ/Ċ markers and emits a different (wrong) id set
    # than the model was pretrained on. Detect deepseek-coder and load the raw
    # tokenizer.json from HF via PreTrainedTokenizerFast instead.
    hf_id = None
    if os.path.isdir(path):
        lm_path_file = os.path.join(path, 'lm.txt')
        if os.path.exists(lm_path_file):
            with open(lm_path_file) as f:
                hf_id = f.read().strip()
    else:
        hf_id = path

    if hf_id and 'deepseek-ai/' in hf_id.lower() and 'deepseek-coder' in hf_id.lower():
        tokenizer_file = hf_hub_download(hf_id, 'tokenizer.json')
        return PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_file,
            bos_token='<｜begin▁of▁sentence｜>',
            eos_token='<｜end▁of▁sentence｜>',
            pad_token='<｜end▁of▁sentence｜>',
        )
    return AutoTokenizer.from_pretrained(path)

class CodeGenPrefixCausalLM(CodeGenForCausalLM):
    def __init__(self, config):
        super().__init__(config)

        self.n_embed_per_head = config.n_embd // config.n_head
        self.prefix_params = torch.nn.ParameterList()
        for _ in range(config.n_control):
            for _ in range(config.n_layer):
                for _ in range(2):
                    param_size = (config.n_head, config.n_prefix_token, self.n_embed_per_head)
                    param = torch.nn.Parameter(torch.zeros(param_size, requires_grad=True))
                    self.prefix_params.append(param)
        self.dropout = torch.nn.Dropout(config.prefix_dropout)

    def get_past_from_prefix(self, control_ids):
        past = list()
        for i in range(self.config.n_layer):
            past.append(list())
            key_stack, val_stack = [], []
            for control_id in control_ids:
                key_idx = control_id * self.config.n_layer * 2 + i * 2
                val_idx = key_idx + 1
                key = self.dropout(self.prefix_params[key_idx])
                val = self.dropout(self.prefix_params[val_idx])
                key_stack.append(key)
                val_stack.append(val)
            past[i].append(torch.stack(key_stack))
            past[i].append(torch.stack(val_stack))
        return past

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)
        else:
            control_ids = [kwargs['control_id']] * input_ids.shape[0]
            past = self.get_past_from_prefix(control_ids)

        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": None,
            "attention_mask": None,
            "token_type_ids": token_type_ids,
        }

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        control_id = None, # placeholder for passing checks of huggingface, actually unused in this function
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return super().forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

class IncoderPrefixLM(XGLMForCausalLM):
    def __init__(self, config):
        super().__init__(config)

        self.n_embed_per_head = config.d_model // config.attention_heads
        self.prefix_params = torch.nn.ParameterList()
        for _ in range(config.n_control):
            for _ in range(config.num_layers):
                for _ in range(2):
                    param_size = (config.attention_heads, config.n_prefix_token, self.n_embed_per_head)
                    param = torch.nn.Parameter(torch.zeros(param_size, requires_grad=True))
                    self.prefix_params.append(param)
        self.dropout = torch.nn.Dropout(config.prefix_dropout)

    def get_past_from_prefix(self, control_ids):
        past = list()
        for i in range(self.config.num_layers):
            past.append(list())
            key_stack, val_stack = [], []
            for control_id in control_ids:
                key_idx = control_id * self.config.num_layers * 2 + i * 2
                val_idx = key_idx + 1
                key = self.dropout(self.prefix_params[key_idx])
                val = self.dropout(self.prefix_params[val_idx])
                key_stack.append(key)
                val_stack.append(val)
            past[i].append(torch.stack(key_stack))
            past[i].append(torch.stack(val_stack))
        return past

    def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, use_cache=None, **kwargs):
        if past:
            input_ids = input_ids[:, -1:]
        else:
            control_ids = [kwargs['control_id']] * input_ids.shape[0]
            past = self.get_past_from_prefix(control_ids)
        # first step, decoder_cached_states are empty
        return {
            "input_ids": input_ids,  # encoder_outputs is defined. input_ids not needed
            "attention_mask": None,
            "past_key_values": past,
            "use_cache": use_cache,
        }

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        control_id = None, # placeholder for passing checks of huggingface, actually unused in this function
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        return super().forward(
            input_ids,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            head_mask,
            cross_attn_head_mask,
            past_key_values,
            inputs_embeds,
            labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )

class SantaPrefixLM(GPT2LMHeadCustomModel):
    def __init__(self, config):
        super().__init__(config)

        self.n_embed_per_head = config.n_embd // config.n_head
        self.prefix_params = torch.nn.ParameterList()
        for _ in range(config.n_control):
            for _ in range(config.n_layer):
                # mha
                for _ in range(2):
                    param_size = (config.n_head, config.n_prefix_token, self.n_embed_per_head)
                    param = torch.nn.Parameter(torch.zeros(param_size, requires_grad=True))
                    self.prefix_params.append(param)
        self.dropout = torch.nn.Dropout(config.prefix_dropout)

    def get_past_from_prefix(self, control_ids):
        past = list()
        for i in range(self.config.n_layer):
            past.append(list())
            key_stack, val_stack = [], []
            for control_id in control_ids:
                key_idx = control_id * self.config.n_layer * 2 + i * 2
                val_idx = key_idx + 1
                key = self.dropout(self.prefix_params[key_idx])
                val = self.dropout(self.prefix_params[val_idx])
                key_stack.append(key)
                val_stack.append(val)
            past[i].append(torch.stack(key_stack))
            past[i].append(torch.stack(val_stack))
        return past

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)
        else:
            control_ids = [kwargs['control_id']] * input_ids.shape[0]
            past = self.get_past_from_prefix(control_ids)

        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": None,
            "attention_mask": None,
            "token_type_ids": token_type_ids,
        }

    def forward(
        self,
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
        control_id = None, # placeholder for passing checks of huggingface, actually unused in this function
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        return super().forward(
            input_ids,
            past_key_values,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            encoder_hidden_states,
            encoder_attention_mask,
            labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )

class MistralPrefixCausalLM(MistralForCausalLM):
    def __init__(self, config):
        super().__init__(config)

        # Mistral uses GQA: KV heads (num_key_value_heads=8) != query heads (num_attention_heads=32)
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.prefix_params = torch.nn.ParameterList()
        for _ in range(config.n_control):
            for _ in range(config.num_hidden_layers):
                for _ in range(2):  # key, value
                    param_size = (config.num_key_value_heads, config.n_prefix_token, self.head_dim)
                    param = torch.nn.Parameter(torch.zeros(param_size, requires_grad=True))
                    self.prefix_params.append(param)
        self.dropout = torch.nn.Dropout(config.prefix_dropout)

    def get_past_from_prefix(self, control_ids):
        from transformers import DynamicCache
        # Cast to model dtype (bfloat16) to avoid mismatch with frozen weights
        dtype = self.model.embed_tokens.weight.dtype
        cache = DynamicCache()
        for i in range(self.config.num_hidden_layers):
            key_stack, val_stack = [], []
            for control_id in control_ids:
                key_idx = control_id * self.config.num_hidden_layers * 2 + i * 2
                val_idx = key_idx + 1
                key_stack.append(self.dropout(self.prefix_params[key_idx]))
                val_stack.append(self.dropout(self.prefix_params[val_idx]))
            cache.update(
                torch.stack(key_stack).to(dtype=dtype),
                torch.stack(val_stack).to(dtype=dtype),
                i,
            )
        return cache

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        # Extend the attention mask by prefix_len here so that _expand_inputs_for_generation
        # (a tensor-only expander) broadcasts the correct mask to the full batch size.
        # The DynamicCache prefix is injected in prepare_inputs_for_generation after
        # expansion, where input_ids already has the correct batch dimension.
        if 'control_id' in kwargs:
            prefix_len = self.config.n_prefix_token
            if attention_mask is not None:
                prefix_mask = attention_mask.new_ones(attention_mask.shape[0], prefix_len)
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            else:
                attention_mask = input_ids.new_ones(input_ids.shape[0], prefix_len + input_ids.shape[1])
        return super().generate(input_ids, attention_mask=attention_mask, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        # On the first step, inject the prefix KV cache at the correct batch size.
        # The attention_mask was already extended by prefix_len in generate(), so only
        # inject if the mask isn't yet wider than input_ids (handles the no-prefix path).
        is_first_step = past_key_values is None or past_key_values.get_seq_length() == 0
        if is_first_step and 'control_id' in kwargs:
            control_ids = [kwargs['control_id']] * input_ids.shape[0]
            past_key_values = self.get_past_from_prefix(control_ids)
            # If generate() didn't pre-extend the mask (e.g. direct call without generate()),
            # extend it here.
            prefix_len = past_key_values.get_seq_length()
            if attention_mask is not None and attention_mask.shape[1] == input_ids.shape[1]:
                prefix_mask = attention_mask.new_ones(attention_mask.shape[0], prefix_len)
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            elif attention_mask is None:
                attention_mask = input_ids.new_ones(input_ids.shape[0], prefix_len + input_ids.shape[1])
        return super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, attention_mask=attention_mask, **kwargs
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        control_id=None,  # placeholder, unused in forward
    ):
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class GemmaPrefixCausalLM(GemmaForCausalLM):
    def __init__(self, config):
        super().__init__(config)

        # Gemma carries an explicit head_dim (256) that is consistent across 2B and 7B.
        self.head_dim = config.head_dim
        self.prefix_params = torch.nn.ParameterList()
        for _ in range(config.n_control):
            for _ in range(config.num_hidden_layers):
                for _ in range(2):  # key, value
                    param_size = (config.num_key_value_heads, config.n_prefix_token, self.head_dim)
                    param = torch.nn.Parameter(torch.zeros(param_size, requires_grad=True))
                    self.prefix_params.append(param)
        self.dropout = torch.nn.Dropout(config.prefix_dropout)

    def get_past_from_prefix(self, control_ids):
        from transformers import DynamicCache
        dtype = self.model.embed_tokens.weight.dtype
        cache = DynamicCache()
        for i in range(self.config.num_hidden_layers):
            key_stack, val_stack = [], []
            for control_id in control_ids:
                key_idx = control_id * self.config.num_hidden_layers * 2 + i * 2
                val_idx = key_idx + 1
                key_stack.append(self.dropout(self.prefix_params[key_idx]))
                val_stack.append(self.dropout(self.prefix_params[val_idx]))
            cache.update(
                torch.stack(key_stack).to(dtype=dtype),
                torch.stack(val_stack).to(dtype=dtype),
                i,
            )
        return cache

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        if 'control_id' in kwargs:
            prefix_len = self.config.n_prefix_token
            if attention_mask is not None:
                prefix_mask = attention_mask.new_ones(attention_mask.shape[0], prefix_len)
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            else:
                attention_mask = input_ids.new_ones(input_ids.shape[0], prefix_len + input_ids.shape[1])
        return super().generate(input_ids, attention_mask=attention_mask, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        is_first_step = past_key_values is None or past_key_values.get_seq_length() == 0
        if is_first_step and 'control_id' in kwargs:
            control_ids = [kwargs['control_id']] * input_ids.shape[0]
            past_key_values = self.get_past_from_prefix(control_ids)
            prefix_len = past_key_values.get_seq_length()
            if attention_mask is not None and attention_mask.shape[1] == input_ids.shape[1]:
                prefix_mask = attention_mask.new_ones(attention_mask.shape[0], prefix_len)
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            elif attention_mask is None:
                attention_mask = input_ids.new_ones(input_ids.shape[0], prefix_len + input_ids.shape[1])
        return super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, attention_mask=attention_mask, **kwargs
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        control_id=None,  # placeholder, unused in forward
    ):
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class LlamaPrefixCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)

        # LlamaConfig carries an explicit head_dim in recent versions; fall back to formula.
        self.head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
        self.prefix_params = torch.nn.ParameterList()
        for _ in range(config.n_control):
            for _ in range(config.num_hidden_layers):
                for _ in range(2):  # key, value
                    param_size = (config.num_key_value_heads, config.n_prefix_token, self.head_dim)
                    param = torch.nn.Parameter(torch.zeros(param_size, requires_grad=True))
                    self.prefix_params.append(param)
        self.dropout = torch.nn.Dropout(config.prefix_dropout)

    def get_past_from_prefix(self, control_ids):
        from transformers import DynamicCache
        dtype = self.model.embed_tokens.weight.dtype
        cache = DynamicCache()
        for i in range(self.config.num_hidden_layers):
            key_stack, val_stack = [], []
            for control_id in control_ids:
                key_idx = control_id * self.config.num_hidden_layers * 2 + i * 2
                val_idx = key_idx + 1
                key_stack.append(self.dropout(self.prefix_params[key_idx]))
                val_stack.append(self.dropout(self.prefix_params[val_idx]))
            cache.update(
                torch.stack(key_stack).to(dtype=dtype),
                torch.stack(val_stack).to(dtype=dtype),
                i,
            )
        return cache

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        if 'control_id' in kwargs:
            prefix_len = self.config.n_prefix_token
            if attention_mask is not None:
                prefix_mask = attention_mask.new_ones(attention_mask.shape[0], prefix_len)
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            else:
                attention_mask = input_ids.new_ones(input_ids.shape[0], prefix_len + input_ids.shape[1])
        return super().generate(input_ids, attention_mask=attention_mask, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        is_first_step = past_key_values is None or past_key_values.get_seq_length() == 0
        if is_first_step and 'control_id' in kwargs:
            control_ids = [kwargs['control_id']] * input_ids.shape[0]
            past_key_values = self.get_past_from_prefix(control_ids)
            prefix_len = past_key_values.get_seq_length()
            if attention_mask is not None and attention_mask.shape[1] == input_ids.shape[1]:
                prefix_mask = attention_mask.new_ones(attention_mask.shape[0], prefix_len)
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            elif attention_mask is None:
                attention_mask = input_ids.new_ones(input_ids.shape[0], prefix_len + input_ids.shape[1])
        return super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, attention_mask=attention_mask, **kwargs
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        control_id=None,  # placeholder, unused in forward
    ):
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class PhiPrefixCausalLM(PhiForCausalLM):
    def __init__(self, config):
        super().__init__(config)

        # PhiConfig has no explicit head_dim field; formula is correct (2560//32 = 80).
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.prefix_params = torch.nn.ParameterList()
        for _ in range(config.n_control):
            for _ in range(config.num_hidden_layers):
                for _ in range(2):  # key, value
                    param_size = (config.num_key_value_heads, config.n_prefix_token, self.head_dim)
                    param = torch.nn.Parameter(torch.zeros(param_size, requires_grad=True))
                    self.prefix_params.append(param)
        self.dropout = torch.nn.Dropout(config.prefix_dropout)

    def get_past_from_prefix(self, control_ids):
        from transformers import DynamicCache
        dtype = self.model.embed_tokens.weight.dtype
        cache = DynamicCache()
        for i in range(self.config.num_hidden_layers):
            key_stack, val_stack = [], []
            for control_id in control_ids:
                key_idx = control_id * self.config.num_hidden_layers * 2 + i * 2
                val_idx = key_idx + 1
                key_stack.append(self.dropout(self.prefix_params[key_idx]))
                val_stack.append(self.dropout(self.prefix_params[val_idx]))
            cache.update(
                torch.stack(key_stack).to(dtype=dtype),
                torch.stack(val_stack).to(dtype=dtype),
                i,
            )
        return cache

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        if 'control_id' in kwargs:
            prefix_len = self.config.n_prefix_token
            if attention_mask is not None:
                prefix_mask = attention_mask.new_ones(attention_mask.shape[0], prefix_len)
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            else:
                attention_mask = input_ids.new_ones(input_ids.shape[0], prefix_len + input_ids.shape[1])
        return super().generate(input_ids, attention_mask=attention_mask, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        is_first_step = past_key_values is None or past_key_values.get_seq_length() == 0
        if is_first_step and 'control_id' in kwargs:
            control_ids = [kwargs['control_id']] * input_ids.shape[0]
            past_key_values = self.get_past_from_prefix(control_ids)
            prefix_len = past_key_values.get_seq_length()
            if attention_mask is not None and attention_mask.shape[1] == input_ids.shape[1]:
                prefix_mask = attention_mask.new_ones(attention_mask.shape[0], prefix_len)
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            elif attention_mask is None:
                attention_mask = input_ids.new_ones(input_ids.shape[0], prefix_len + input_ids.shape[1])
        return super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, attention_mask=attention_mask, **kwargs
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        control_id=None,  # placeholder, unused in forward
    ):
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class Qwen2PrefixCausalLM(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)

        self.head_dim = config.hidden_size // config.num_attention_heads
        self.prefix_params = torch.nn.ParameterList()
        for _ in range(config.n_control):
            for _ in range(config.num_hidden_layers):
                for _ in range(2):  # key, value
                    param_size = (config.num_key_value_heads, config.n_prefix_token, self.head_dim)
                    param = torch.nn.Parameter(torch.zeros(param_size, requires_grad=True))
                    self.prefix_params.append(param)
        self.dropout = torch.nn.Dropout(config.prefix_dropout)

    def get_past_from_prefix(self, control_ids):
        from transformers import DynamicCache
        dtype = self.model.embed_tokens.weight.dtype
        cache = DynamicCache()
        for i in range(self.config.num_hidden_layers):
            key_stack, val_stack = [], []
            for control_id in control_ids:
                key_idx = control_id * self.config.num_hidden_layers * 2 + i * 2
                val_idx = key_idx + 1
                key_stack.append(self.dropout(self.prefix_params[key_idx]))
                val_stack.append(self.dropout(self.prefix_params[val_idx]))
            cache.update(
                torch.stack(key_stack).to(dtype=dtype),
                torch.stack(val_stack).to(dtype=dtype),
                i,
            )
        return cache

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        if 'control_id' in kwargs:
            prefix_len = self.config.n_prefix_token
            if attention_mask is not None:
                prefix_mask = attention_mask.new_ones(attention_mask.shape[0], prefix_len)
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            else:
                attention_mask = input_ids.new_ones(input_ids.shape[0], prefix_len + input_ids.shape[1])
        return super().generate(input_ids, attention_mask=attention_mask, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        is_first_step = past_key_values is None or past_key_values.get_seq_length() == 0
        if is_first_step and 'control_id' in kwargs:
            control_ids = [kwargs['control_id']] * input_ids.shape[0]
            past_key_values = self.get_past_from_prefix(control_ids)
            prefix_len = past_key_values.get_seq_length()
            if attention_mask is not None and attention_mask.shape[1] == input_ids.shape[1]:
                prefix_mask = attention_mask.new_ones(attention_mask.shape[0], prefix_len)
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            elif attention_mask is None:
                attention_mask = input_ids.new_ones(input_ids.shape[0], prefix_len + input_ids.shape[1])
        return super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, attention_mask=attention_mask, **kwargs
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        control_id=None,  # placeholder, unused in forward
    ):
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class Qwen3PrefixCausalLM(Qwen3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)

        # Qwen3 uses GQA and an explicit head_dim that differs from
        # hidden_size // num_attention_heads (e.g. 4B: 80 vs 128).
        self.head_dim = config.head_dim
        self.prefix_params = torch.nn.ParameterList()
        for _ in range(config.n_control):
            for _ in range(config.num_hidden_layers):
                for _ in range(2):  # key, value
                    param_size = (config.num_key_value_heads, config.n_prefix_token, self.head_dim)
                    param = torch.nn.Parameter(torch.zeros(param_size, requires_grad=True))
                    self.prefix_params.append(param)
        self.dropout = torch.nn.Dropout(config.prefix_dropout)

    def get_past_from_prefix(self, control_ids):
        from transformers import DynamicCache
        dtype = self.model.embed_tokens.weight.dtype
        cache = DynamicCache()
        for i in range(self.config.num_hidden_layers):
            key_stack, val_stack = [], []
            for control_id in control_ids:
                key_idx = control_id * self.config.num_hidden_layers * 2 + i * 2
                val_idx = key_idx + 1
                key_stack.append(self.dropout(self.prefix_params[key_idx]))
                val_stack.append(self.dropout(self.prefix_params[val_idx]))
            cache.update(
                torch.stack(key_stack).to(dtype=dtype),
                torch.stack(val_stack).to(dtype=dtype),
                i,
            )
        return cache

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        if 'control_id' in kwargs:
            prefix_len = self.config.n_prefix_token
            if attention_mask is not None:
                prefix_mask = attention_mask.new_ones(attention_mask.shape[0], prefix_len)
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            else:
                attention_mask = input_ids.new_ones(input_ids.shape[0], prefix_len + input_ids.shape[1])
        return super().generate(input_ids, attention_mask=attention_mask, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        is_first_step = past_key_values is None or past_key_values.get_seq_length() == 0
        if is_first_step and 'control_id' in kwargs:
            control_ids = [kwargs['control_id']] * input_ids.shape[0]
            past_key_values = self.get_past_from_prefix(control_ids)
            prefix_len = past_key_values.get_seq_length()
            if attention_mask is not None and attention_mask.shape[1] == input_ids.shape[1]:
                prefix_mask = attention_mask.new_ones(attention_mask.shape[0], prefix_len)
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            elif attention_mask is None:
                attention_mask = input_ids.new_ones(input_ids.shape[0], prefix_len + input_ids.shape[1])
        return super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, attention_mask=attention_mask, **kwargs
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        control_id=None,  # placeholder, unused in forward
    ):
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


def model_from_pretrained(lm_path, model_type, config):
    kwargs = dict()
    if lm_path.startswith('Salesforce/codegen-'):
        if model_type == 'lm':
            model_class = CodeGenForCausalLM
        elif model_type == 'prefix':
            model_class = CodeGenPrefixCausalLM
        else:
            assert False
    elif lm_path.startswith('facebook/incoder-'):
        if config is not None:
            config.attention_dropout = 0.0
            config.dropout = 0.0
        if model_type == 'lm':
            model_class = XGLMForCausalLM
        elif model_type == 'prefix':
            model_class = IncoderPrefixLM
        else:
            assert False
    elif lm_path == 'bigcode/santacoder':
        kwargs['revision'] = 'mha'
        if config is not None:
            config.attn_pdrop = 0.0
            config.embd_pdrop = 0.0
            config.resid_pdrop = 0.0
        if model_type == 'lm':
            model_class = GPT2LMHeadCustomModel
        elif model_type == 'prefix':
            model_class = SantaPrefixLM
        else:
            assert False
    elif lm_path.startswith('mistralai/'):
        # Load in bfloat16 to fit 7B weights in GPU memory
        kwargs['torch_dtype'] = torch.bfloat16
        if model_type == 'lm':
            model_class = MistralForCausalLM
        elif model_type == 'prefix':
            model_class = MistralPrefixCausalLM
        else:
            assert False
    elif lm_path.startswith('google/gemma-'):
        kwargs['torch_dtype'] = torch.bfloat16
        if model_type == 'lm':
            model_class = GemmaForCausalLM
        elif model_type == 'prefix':
            model_class = GemmaPrefixCausalLM
        else:
            assert False
    elif (lm_path.startswith('ByteDance-Seed/') or lm_path.startswith('codellama/')
          or lm_path.startswith('deepseek-ai/') or lm_path.startswith('meta-llama/')):
        kwargs['torch_dtype'] = torch.bfloat16
        if model_type == 'lm':
            model_class = LlamaForCausalLM
        elif model_type == 'prefix':
            model_class = LlamaPrefixCausalLM
        else:
            assert False
    elif lm_path == 'microsoft/phi-2':
        kwargs['torch_dtype'] = torch.bfloat16
        if model_type == 'lm':
            model_class = PhiForCausalLM
        elif model_type == 'prefix':
            model_class = PhiPrefixCausalLM
        else:
            assert False
    elif lm_path.startswith('Qwen/Qwen2.5-Coder-'):
        kwargs['torch_dtype'] = torch.bfloat16
        if model_type == 'lm':
            model_class = Qwen2ForCausalLM
        elif model_type == 'prefix':
            model_class = Qwen2PrefixCausalLM
        else:
            assert False
    elif lm_path.startswith('Qwen/Qwen3-'):
        kwargs['torch_dtype'] = torch.bfloat16
        if model_type == 'lm':
            model_class = Qwen3ForCausalLM
        elif model_type == 'prefix':
            model_class = Qwen3PrefixCausalLM
        else:
            assert False
    else:
        assert False

    if config is None:
        model = model_class.from_pretrained(lm_path, **kwargs)
    else:
        model = model_class.from_pretrained(lm_path, **kwargs, config=config)

    return model

def config_from_pretrained(lm_path, path):
    if lm_path == 'bigcode/santacoder':
        return GPT2CustomConfig.from_pretrained(path, revision='mha')
    else:
        return AutoConfig.from_pretrained(path)

def save_model(model, path, args):
    if type(model) in (CodeGenPrefixCausalLM, IncoderPrefixLM, SantaPrefixLM, MistralPrefixCausalLM, GemmaPrefixCausalLM, LlamaPrefixCausalLM, PhiPrefixCausalLM, Qwen2PrefixCausalLM, Qwen3PrefixCausalLM):
        assert (args.pretrain_dir.startswith('Salesforce/codegen-') or args.pretrain_dir.startswith('facebook/incoder-')
                or args.pretrain_dir == 'bigcode/santacoder' or args.pretrain_dir.startswith('mistralai/')
                or args.pretrain_dir.startswith('google/gemma-')
                or args.pretrain_dir.startswith('ByteDance-Seed/') or args.pretrain_dir.startswith('codellama/')
                or args.pretrain_dir.startswith('deepseek-ai/') or args.pretrain_dir.startswith('meta-llama/')
                or args.pretrain_dir == 'microsoft/phi-2'
                or args.pretrain_dir.startswith('Qwen/'))
        config_file = os.path.join(path)
        model.config.save_pretrained(config_file)
        prefix_file = os.path.join(path, 'pytorch_model.bin')
        state_dict = model.prefix_params.state_dict()
        for k, v in state_dict.items():
            state_dict[k] = v.cpu()
        torch.save(state_dict, prefix_file)
        lm_path_file = os.path.join(path, 'lm.txt')
        with open(lm_path_file, 'w') as f:
            f.write(args.pretrain_dir)
    else:
        model.save_pretrained(path)

def load_model(model_type, path, is_training, args):
    logging.set_verbosity_error()
    tokenizer = _load_tokenizer(path)
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = tokenizer.bos_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if model_type == 'lm':
        config = config_from_pretrained(path, path)
        model = model_from_pretrained(path, model_type, config)
    elif model_type == 'prefix':
        if is_training:
            lm_path = path
            lm_config = config_from_pretrained(lm_path, lm_path)
            lm_config.n_prefix_token = args.n_prefix_token
            lm_config.prefix_dropout = args.dropout
            lm_config.n_control = 2
            model = model_from_pretrained(lm_path, model_type, lm_config)
            # from_pretrained uses torch.empty for params absent from checkpoint,
            # leaving garbage/NaN; explicitly zero the prefix params after loading
            if hasattr(model, 'prefix_params'):
                for param in model.prefix_params:
                    torch.nn.init.zeros_(param)
        else:
            lm_path_file = os.path.join(path, 'lm.txt')
            assert os.path.exists(lm_path_file)
            with open(lm_path_file) as f:
                lm_path = f.read()
            prefix_config = config_from_pretrained(lm_path, path)
            lm_config = config_from_pretrained(lm_path, lm_path)
            lm_config.n_prefix_token = prefix_config.n_prefix_token
            lm_config.prefix_dropout = prefix_config.prefix_dropout
            lm_config.n_control = prefix_config.n_control
            model = model_from_pretrained(lm_path, model_type, lm_config)
            prefix_file = os.path.join(path, 'pytorch_model.bin')
            model.prefix_params.load_state_dict(torch.load(prefix_file))
    else:
        assert False

    model.resize_token_embeddings(len(tokenizer))
    input_device = parallelize_model(model, args)
    return tokenizer, model, input_device

def parallelize_model(model, args):
    if args.n_gpu > 1 and hasattr(model, 'parallelize'):
        model.parallelize()
        input_device = model.transformer.first_device
    else:
        model.to(args.device)
        input_device = args.device
    return input_device
