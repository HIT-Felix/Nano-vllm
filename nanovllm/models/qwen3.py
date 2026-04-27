import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from dataclasses import dataclass
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import (
    QKVParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.parallel_state import get_ep_group, get_ep_rank, get_ep_size, get_tp_size


@dataclass(slots=True)
class Qwen3MoeStats:
    num_calls: int = 0
    num_tokens: int = 0
    num_dispatches: int = 0
    expert_histogram: list[int] | None = None


class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        tp_size = get_tp_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3MoeMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_proj = ReplicatedLinear(hidden_size, intermediate_size, bias=False)
        self.up_proj = ReplicatedLinear(hidden_size, intermediate_size, bias=False)
        self.down_proj = ReplicatedLinear(intermediate_size, hidden_size, bias=False)
        assert hidden_act == "silu"
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen3SparseMoeBlock(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = getattr(config, "norm_topk_prob", True)
        self.stats = Qwen3MoeStats(expert_histogram=[0] * self.num_experts)
        self.ep_rank = get_ep_rank()
        self.ep_size = get_ep_size()
        self.gate = ReplicatedLinear(config.hidden_size, self.num_experts, bias=False)
        assert self.num_experts % self.ep_size == 0
        self.local_num_experts = self.num_experts // self.ep_size
        self.expert_start_idx = self.ep_rank * self.local_num_experts
        self.expert_end_idx = self.expert_start_idx + self.local_num_experts
        self.experts = nn.ModuleList([
            Qwen3MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                hidden_act=config.hidden_act,
            )
            for _ in range(self.local_num_experts)
        ])
        self.global_to_local_expert = {
            global_idx: global_idx - self.expert_start_idx
            for global_idx in range(self.expert_start_idx, self.expert_end_idx)
        }

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        if hidden_states.ndim == 2:
            flat_states = hidden_states
        elif hidden_states.ndim == 3:
            flat_states = hidden_states.reshape(-1, hidden_states.size(-1))
        else:
            raise ValueError(f"unexpected hidden_states shape: {orig_shape}")
        hidden_dim = flat_states.size(-1)
        self.stats.num_calls += 1
        self.stats.num_tokens += flat_states.size(0)
        router_logits = self.gate(flat_states)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(flat_states.dtype)
        self.stats.num_dispatches += selected_experts.numel()
        expert_hist = torch.bincount(selected_experts.flatten(), minlength=self.num_experts).tolist()
        for idx, count in enumerate(expert_hist):
            self.stats.expert_histogram[idx] += count

        final_hidden_states = flat_states.new_zeros(flat_states.shape)
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = expert_idx.item()
            local_expert_idx = self.global_to_local_expert.get(expert_idx)
            if local_expert_idx is None:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_hidden_states = flat_states[token_idx]
            current_hidden_states = self.experts[local_expert_idx](current_hidden_states)
            current_hidden_states = current_hidden_states * routing_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states)
        if self.ep_size > 1:
            dist.all_reduce(final_hidden_states, group=get_ep_group())
        return final_hidden_states.reshape(orig_shape)

    def reset_stats(self):
        self.stats = Qwen3MoeStats(expert_histogram=[0] * self.num_experts)

    def get_stats(self) -> Qwen3MoeStats:
        return Qwen3MoeStats(
            num_calls=self.stats.num_calls,
            num_tokens=self.stats.num_tokens,
            num_dispatches=self.stats.num_dispatches,
            expert_histogram=list(self.stats.expert_histogram),
        )


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3MoeDecoderLayer(Qwen3DecoderLayer):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__(config)
        self.mlp = Qwen3SparseMoeBlock(config)


class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3MoeModel(Qwen3Model):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        nn.Module.__init__(self)
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3MoeDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def reset_moe_stats(self):
        for layer in self.layers:
            layer.mlp.reset_stats()

    def get_moe_stats(self) -> dict:
        layer_stats = []
        total_calls = total_tokens = total_dispatches = 0
        aggregate_hist = None
        for layer_idx, layer in enumerate(self.layers):
            stats = layer.mlp.get_stats()
            total_calls += stats.num_calls
            total_tokens += stats.num_tokens
            total_dispatches += stats.num_dispatches
            if aggregate_hist is None:
                aggregate_hist = [0] * len(stats.expert_histogram)
            for idx, count in enumerate(stats.expert_histogram):
                aggregate_hist[idx] += count
            layer_stats.append(
                dict(
                    layer=layer_idx,
                    num_calls=stats.num_calls,
                    num_tokens=stats.num_tokens,
                    num_dispatches=stats.num_dispatches,
                    expert_histogram=stats.expert_histogram,
                )
            )
        return dict(
            num_layers=len(self.layers),
            total_calls=total_calls,
            total_tokens=total_tokens,
            total_dispatches=total_dispatches,
            aggregate_expert_histogram=aggregate_hist,
            layers=layer_stats,
        )


class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)


class Qwen3MoeForCausalLM(Qwen3ForCausalLM):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        nn.Module.__init__(self)
        self.model = Qwen3MoeModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def reset_moe_stats(self):
        self.model.reset_moe_stats()

    def get_moe_stats(self) -> dict:
        return self.model.get_moe_stats()

    def remap_weight_name(self, weight_name: str) -> str | None:
        if ".mlp.experts." not in weight_name:
            return weight_name
        parts = weight_name.split(".")
        expert_pos = parts.index("experts") + 1
        global_expert_idx = int(parts[expert_pos])
        layer = None
        for part in parts:
            if part == "layers":
                break
        local_block = None
        try:
            layer_idx = int(parts[parts.index("layers") + 1])
            local_block = self.model.layers[layer_idx].mlp
        except (ValueError, IndexError, AttributeError):
            return weight_name
        local_expert_idx = local_block.global_to_local_expert.get(global_expert_idx)
        if local_expert_idx is None:
            return None
        parts[expert_pos] = str(local_expert_idx)
        return ".".join(parts)
