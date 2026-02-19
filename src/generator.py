"""
LLM-based answer generation for the Advanced RAG system.

This module provides:
- LLMGenerator: Wrapper for causal LLMs (Phi-3-mini)
- Prompt templates for RAG
- Response parsing and validation
- Memory-efficient 4-bit quantization via bitsandbytes
"""

from typing import Dict, List, Optional

import torch
from loguru import logger
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from src.utils import GPUManager, Timer


class LLMGenerator:
    """
    LLM-based answer generator for RAG.

    Uses quantized causal LLMs (Phi-3-mini) to generate answers
    from retrieved context chunks.
    """

    # Default prompt template
    DEFAULT_PROMPT = """Answer the question using ONLY the provided context. Be direct and concise (1-3 sentences maximum).

DO NOT:
- Repeat the question
- Say "Based on the context" or similar phrases
- Continue after answering

Context:
{context}

Question: {question}

Direct answer:"""

    def __init__(
        self,
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        device: Optional[str] = None,
        load_in_4bit: bool = True,
        max_new_tokens: int = 80,
        temperature: float = 0.1,
        top_p: float = 0.9,
        prompt_template: Optional[str] = None,
    ):
        """
        Initialize LLM generator.

        Args:
            model_name: HuggingFace model identifier
            device: Device to use (None for auto)
            load_in_4bit: Whether to use 4-bit quantization
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            top_p: Nucleus sampling parameter
            prompt_template: Custom prompt template (uses default if None)
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT

        # Determine device
        if device is None:
            self.device = GPUManager.get_device(prefer_gpu=True)
        else:
            self.device = torch.device(device)

        logger.info(f"Loading LLM: {model_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"4-bit quantization: {load_in_4bit}")

        # Configure quantization
        quantization_config = None
        if load_in_4bit and self.device.type == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            logger.info("Using NF4 4-bit quantization")

        # Load model
        with Timer("LLM model loading"):
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16 if not load_in_4bit else None,
            )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"LLM initialized: {model_name}")
        logger.info(f"Model parameters: {self.model.num_parameters() / 1e9:.2f}B")

    def generate(
        self,
        query: str,
        context_chunks: List[Dict],
        max_chunks: int = 5,
    ) -> Dict:
        """
        Generate answer from query and context chunks.

        Args:
            query: User question
            context_chunks: List of retrieved chunks (dicts with 'content' key)
            max_chunks: Maximum number of chunks to include in context

        Returns:
            Dictionary with 'answer', 'prompt', and metadata
        """
        # Format context
        context = self._format_context(context_chunks[:max_chunks])

        # Build prompt
        prompt = self.prompt_template.format(context=context, question=query)

        # Generate
        with Timer("LLM generation"):
            answer = self._generate_text(prompt)

        # Parse and validate
        answer = self._parse_answer(answer)

        return {
            "answer": answer,
            "prompt": prompt,
            "num_chunks_used": min(len(context_chunks), max_chunks),
            "model": self.model_name,
        }

    def batch_generate(
        self,
        queries: List[str],
        context_chunks_list: List[List[Dict]],
        max_chunks: int = 5,
    ) -> List[Dict]:
        """
        Generate answers for multiple queries.

        Args:
            queries: List of questions
            context_chunks_list: List of context chunks per query
            max_chunks: Maximum chunks per query

        Returns:
            List of answer dictionaries
        """
        results = []

        for query, chunks in zip(queries, context_chunks_list):
            result = self.generate(query, chunks, max_chunks)
            results.append(result)

        return results

    def _format_context(self, chunks: List[Dict]) -> str:
        """
        Format context chunks into a string.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            Formatted context string
        """
        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            content = chunk.get("content", "")
            context_parts.append(f"[{i}] {content}")

        return "\n\n".join(context_parts)

    def _generate_text(self, prompt: str, streamer=None) -> str:
        """
        Generate text from prompt using the LLM.

        Uses chat template when available (instruction-tuned models like Qwen),
        falls back to raw tokenization for base models.

        Args:
            prompt: Input prompt
            streamer: Optional TextIteratorStreamer for token-by-token streaming

        Returns:
            Generated text (empty string when streamer is used, since
            the streamer handles output)
        """
        # Use chat template if available (instruction-tuned models)
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        generate_kwargs = dict(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=True if self.temperature > 0 else False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if streamer is not None:
            generate_kwargs["streamer"] = streamer

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**generate_kwargs)

        if streamer is not None:
            return ""

        # Decode (skip prompt tokens)
        generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return generated_text

    def build_prompt(self, query: str, context_chunks: List[Dict], max_chunks: int = 5) -> str:
        """Build the full prompt for a query (used by streaming endpoint)."""
        context = self._format_context(context_chunks[:max_chunks])
        return self.prompt_template.format(context=context, question=query)

    def _parse_answer(self, generated_text: str) -> str:
        """
        Parse and clean generated answer.

        Args:
            generated_text: Raw LLM output

        Returns:
            Cleaned answer
        """
        answer = generated_text.strip()

        # Remove "Answer:" prefix if present
        if answer.lower().startswith("answer:"):
            answer = answer[7:].strip()

        # CRITICAL: Truncate at double newline FIRST (before sentence processing destroys it)
        if "\n\n" in answer:
            answer = answer.split("\n\n")[0].strip()

        # Stop at common hallucination triggers
        stop_phrases = [
            "Based on the context",
            "Based on the provided",
            "According to the context",
            "Human:",
            "Question:",
            "Can you tell me",
            "Given the",
            "This information",
            "You are an AI",
            "You are a helpful",
            "As an AI",
            "I'm an AI",
            "Note:",
            "Additional",
        ]

        for phrase in stop_phrases:
            if phrase in answer:
                # Keep only text before the phrase
                answer = answer.split(phrase)[0].strip()

        # Remove repetitive sentences
        sentences = [s.strip() + "." for s in answer.split(".") if s.strip()]
        if len(sentences) > 1:
            # Check if any sentence is repeated
            unique_sentences = []
            seen_content = set()

            for sent in sentences:
                # Normalize for comparison
                normalized = sent.lower().replace(" ", "")
                # Check if similar content already exists
                is_duplicate = False
                for seen in seen_content:
                    # If 80%+ overlap, consider duplicate
                    overlap = len(set(normalized) & set(seen))
                    if overlap / max(len(normalized), len(seen)) > 0.8:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    unique_sentences.append(sent)
                    seen_content.add(normalized)

            answer = " ".join(unique_sentences)

        # Clean up trailing artifacts
        if answer.endswith(("no", "no.", "Given", "This")):
            # Likely cut off mid-sentence, remove last word
            words = answer.split()
            if len(words) > 3:
                answer = " ".join(words[:-1])

        return answer.strip()

    def update_prompt_template(self, new_template: str) -> None:
        """
        Update the prompt template.

        Args:
            new_template: New prompt template string
                         Must contain {context} and {question} placeholders
        """
        if "{context}" not in new_template or "{question}" not in new_template:
            raise ValueError("Prompt template must contain {context} and {question} placeholders")

        self.prompt_template = new_template
        logger.info("Prompt template updated")

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get GPU memory usage statistics.

        Returns:
            Dictionary with memory statistics in GB
        """
        if self.device.type != "cuda":
            return {"device": "cpu", "memory_used": 0.0}

        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3

        return {
            "device": "cuda",
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "model_name": self.model_name,
        }


if __name__ == "__main__":
    # Example usage
    from src.utils import LoggerConfig

    LoggerConfig.setup(level="INFO")

    # Initialize generator
    generator = LLMGenerator(
        model_name="microsoft/Phi-3-mini-4k-instruct",
        load_in_4bit=True,
        temperature=0.1,
    )

    # Example query and context
    query = "What is the capital of France?"
    context_chunks = [
        {"content": "Paris is the capital and largest city of France."},
        {"content": "The Eiffel Tower is located in Paris."},
        {"content": "France is a country in Western Europe."},
    ]

    # Generate answer
    result = generator.generate(query, context_chunks)

    logger.info(f"\nQuery: {query}")
    logger.info(f"Answer: {result['answer']}")
    logger.info(f"Chunks used: {result['num_chunks_used']}")

    # Memory usage
    memory = generator.get_memory_usage()
    logger.info(f"\nMemory usage: {memory}")
