import os
import time
import argparse
from random import randint, seed
from nanovllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark nano-vllm generation throughput.")
    parser.add_argument(
        "--model",
        default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"),
        help="Local model directory.",
    )
    parser.add_argument("--num-seqs", type=int, default=8, help="Number of concurrent sequences.")
    parser.add_argument("--min-input-len", type=int, default=32, help="Minimum prompt token length.")
    parser.add_argument("--max-input-len", type=int, default=256, help="Maximum prompt token length.")
    parser.add_argument("--min-output-len", type=int, default=32, help="Minimum output token length.")
    parser.add_argument("--max-output-len", type=int, default=128, help="Maximum output token length.")
    parser.add_argument("--max-model-len", type=int, default=2048, help="Maximum model length.")
    parser.add_argument("--max-num-seqs", type=int, default=16, help="Engine max_num_seqs.")
    parser.add_argument("--expert-parallel-size", type=int, default=1, help="Engine expert_parallel_size.")
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=2048,
        help="Engine max_num_batched_tokens.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Force eager mode instead of CUDA graph mode.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seed(0)
    assert args.min_input_len <= args.max_input_len
    assert args.min_output_len <= args.max_output_len

    llm = LLM(
        args.model,
        tensor_parallel_size=1,
        expert_parallel_size=args.expert_parallel_size,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
    )

    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(args.min_input_len, args.max_input_len))]
        for _ in range(args.num_seqs)
    ]
    sampling_params = [
        SamplingParams(
            temperature=0.6,
            ignore_eos=True,
            max_tokens=randint(args.min_output_len, args.max_output_len),
        )
        for _ in range(args.num_seqs)
    ]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = (time.time() - t)
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Model: {args.model}")
    print(
        f"Workload: num_seqs={args.num_seqs}, "
        f"input_len=[{args.min_input_len}, {args.max_input_len}], "
        f"output_len=[{args.min_output_len}, {args.max_output_len}]"
    )
    print(
        f"Engine: max_model_len={args.max_model_len}, "
        f"max_num_seqs={args.max_num_seqs}, "
        f"max_num_batched_tokens={args.max_num_batched_tokens}, "
        f"expert_parallel_size={args.expert_parallel_size}, "
        f"enforce_eager={args.enforce_eager}"
    )
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
