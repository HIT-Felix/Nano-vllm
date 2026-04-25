import argparse

from nanovllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test for Qwen3 MoE on nano-vllm.")
    parser.add_argument("--model", required=True, help="Local model directory.")
    parser.add_argument(
        "--prompt",
        default="Hello, introduce yourself in one short paragraph.",
        help="Prompt text.",
    )
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    return parser.parse_args()


def main():
    args = parse_args()
    llm = LLM(
        args.model,
        tensor_parallel_size=1,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
    )
    llm.reset_moe_stats()
    outputs = llm.generate(
        [args.prompt],
        SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens),
        use_tqdm=False,
    )
    print(outputs[0]["text"])
    stats = llm.get_moe_stats()
    if stats is not None:
        top_experts = sorted(
            enumerate(stats["aggregate_expert_histogram"]),
            key=lambda item: item[1],
            reverse=True,
        )[:8]
        print(
            "MoE stats:",
            dict(
                total_calls=stats["total_calls"],
                total_tokens=stats["total_tokens"],
                total_dispatches=stats["total_dispatches"],
                top_experts=top_experts,
            ),
        )
    llm.exit()


if __name__ == "__main__":
    main()
