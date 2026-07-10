from __future__ import annotations

import torch
import torch.nn as nn

from src.pcd.encoder import TopKEncoder
from src.subject.model_io import load_model_and_tokenizer


def load_pcd_models(
    model_cfg: dict,
    subject_adapter: str,
    lora: dict,
    device: torch.device | str = "cpu",
):
    from peft import LoraConfig, PeftModel, get_peft_model

    subject, tokenizer, repo = load_model_and_tokenizer(model_cfg)
    subject = PeftModel.from_pretrained(subject, subject_adapter).merge_and_unload()
    subject.to(device).eval().requires_grad_(False)

    dec_base, _, _ = load_model_and_tokenizer(model_cfg)
    dec_base = PeftModel.from_pretrained(dec_base, subject_adapter).merge_and_unload()
    peft_cfg = LoraConfig(
        r=int(lora["r"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora.get("dropout", 0.05)),
        target_modules=list(lora["target_modules"]),
        task_type="CAUSAL_LM",
    )
    decoder = get_peft_model(dec_base, peft_cfg)
    decoder.to(device)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return subject, decoder, tokenizer, repo


@torch.no_grad()
def read_middle_activations(
    subject,
    input_ids: torch.Tensor,
    read_layer: int,
    n_middle: int,
) -> torch.Tensor:
    out = subject(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    h = out.hidden_states[read_layer]
    return h[:, -n_middle:, :].contiguous()


class PCDModel(nn.Module):
    def __init__(
        self,
        encoder: TopKEncoder,
        decoder,
        n_middle: int = 16,
        n_suffix: int = 16,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.n_middle = n_middle
        self.n_suffix = n_suffix
        self.decoder_dtype = next(decoder.parameters()).dtype

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(
        self,
        activations: torch.Tensor,
        suffix_ids: torch.Tensor,
        suffix_labels: torch.Tensor,
    ) -> dict:
        b = activations.shape[0]
        enc = self.encoder.encode(activations, out_dtype=self.decoder_dtype)
        soft = enc.soft_tokens

        embed = self.decoder.get_input_embeddings()
        suffix_embeds = embed(suffix_ids)
        inputs_embeds = torch.cat([soft, suffix_embeds], dim=1)

        soft_labels = suffix_labels.new_full((b, self.n_middle), -100)
        labels = torch.cat([soft_labels, suffix_labels], dim=1)
        attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)

        out = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            labels=labels,
            use_cache=False,
        )
        lm_loss = out.loss
        aux = self.encoder.aux_loss(enc.pre_acts)
        return {
            "loss": lm_loss + aux,
            "lm_loss": lm_loss.detach(),
            "aux_loss": aux.detach(),
            "topk_indices": enc.topk_indices,
        }
