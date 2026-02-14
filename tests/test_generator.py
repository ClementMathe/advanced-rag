"""
Unit tests for LLM generator module.

Tests:
- Generator initialization
- Prompt formatting
- Answer parsing
- Memory management
- Error handling

Note: Uses comprehensive mocking to avoid torch version conflicts and memory issues.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from src.generator import LLMGenerator

# Check torch version for compatibility
TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])
TORCH_2_6_AVAILABLE = TORCH_VERSION >= (2, 6)


def create_mock_model():
    """Create a properly mocked model instance."""
    model = MagicMock()
    model.num_parameters = Mock(return_value=3_800_000_000)  # Returns int, not MagicMock
    model.to = Mock(return_value=model)  # Allow .to() calls
    model.half = Mock(return_value=model)  # Allow .half() calls
    return model


def create_mock_tokenizer():
    """Create a properly mocked tokenizer instance."""
    tokenizer = MagicMock()
    tokenizer.pad_token = None
    tokenizer.eos_token = "<eos>"
    tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = 0
    return tokenizer


class TestLLMGenerator:
    """Test suite for LLMGenerator class."""

    @pytest.fixture(autouse=True)
    def mock_cuda_memory(self):
        """Mock CUDA memory functions globally."""
        with (
            patch("torch.cuda.memory_allocated", return_value=1_000_000_000),
            patch("torch.cuda.memory_reserved", return_value=2_000_000_000),
        ):
            yield

    @pytest.fixture
    def mock_generator(self):
        """
        Create a mock generator for testing without loading actual model.

        This avoids memory issues and torch version conflicts.
        """
        with (
            patch("src.generator.AutoModelForCausalLM.from_pretrained") as mock_model_cls,
            patch("src.generator.AutoTokenizer.from_pretrained") as mock_tokenizer_cls,
        ):

            mock_model_cls.return_value = create_mock_model()
            mock_tokenizer_cls.return_value = create_mock_tokenizer()

            generator = LLMGenerator(
                model_name="test/model",
                load_in_4bit=False,
                device="cpu",
                max_new_tokens=50,
            )

            yield generator

    def test_initialization(self, mock_generator):
        """Test generator initialization."""
        assert mock_generator.model is not None
        assert mock_generator.tokenizer is not None
        assert mock_generator.max_new_tokens == 50

    def test_default_prompt_template(self, mock_generator):
        """Test default prompt template contains required placeholders."""
        assert "{context}" in mock_generator.prompt_template
        assert "{question}" in mock_generator.prompt_template

    def test_format_context(self, mock_generator):
        """Test context formatting."""
        chunks = [
            {"content": "Paris is the capital of France."},
            {"content": "France is in Europe."},
        ]

        context = mock_generator._format_context(chunks)

        assert "[1]" in context
        assert "[2]" in context
        assert "Paris is the capital of France." in context
        assert "France is in Europe." in context

    def test_format_context_empty(self, mock_generator):
        """Test context formatting with empty list."""
        context = mock_generator._format_context([])
        assert context == ""

    def test_format_context_max_chunks(self, mock_generator):
        """Test context formatting respects ordering."""
        chunks = [{"content": f"Chunk {i}"} for i in range(5)]

        context = mock_generator._format_context(chunks[:3])

        assert "[1]" in context
        assert "[2]" in context
        assert "[3]" in context
        assert "[4]" not in context

    def test_parse_answer_basic(self, mock_generator):
        """Test answer parsing."""
        raw = "Paris is the capital of France."
        parsed = mock_generator._parse_answer(raw)
        assert parsed == "Paris is the capital of France."

    def test_parse_answer_with_prefix(self, mock_generator):
        """Test answer parsing with 'Answer:' prefix."""
        raw = "Answer: Paris is the capital of France."
        parsed = mock_generator._parse_answer(raw)
        assert parsed == "Paris is the capital of France."

    def test_parse_answer_with_extra_content(self, mock_generator):
        """Test answer parsing truncates at double newline."""
        raw = "Paris is the capital.\n\nExtra content here."
        parsed = mock_generator._parse_answer(raw)
        assert parsed == "Paris is the capital."

    def test_parse_answer_strips_whitespace(self, mock_generator):
        """Test answer parsing strips leading/trailing whitespace."""
        raw = "  Paris is the capital.  "
        parsed = mock_generator._parse_answer(raw)
        assert parsed == "Paris is the capital."

    def test_parse_answer_stops_at_ai_hallucination(self, mock_generator):
        """Test answer parsing stops at 'You are an AI' hallucination."""
        raw = "Paris is the capital. You are an AI assistant designed to help."
        parsed = mock_generator._parse_answer(raw)
        assert parsed == "Paris is the capital."

    def test_parse_answer_stops_at_helpful_hallucination(self, mock_generator):
        """Test answer parsing stops at 'You are a helpful' hallucination."""
        raw = "Paris. You are a helpful assistant that answers questions."
        parsed = mock_generator._parse_answer(raw)
        assert parsed == "Paris."

    def test_parse_answer_stops_at_as_an_ai(self, mock_generator):
        """Test answer parsing stops at 'As an AI' hallucination."""
        raw = "The answer is Paris. As an AI, I cannot verify this."
        parsed = mock_generator._parse_answer(raw)
        assert parsed == "The answer is Paris."

    def test_parse_answer_stops_at_note(self, mock_generator):
        """Test answer parsing stops at 'Note:' trailing content."""
        raw = "Paris is the capital. Note: this is from the context."
        parsed = mock_generator._parse_answer(raw)
        assert parsed == "Paris is the capital."

    def test_generate_structure(self, mock_generator):
        """Test generate returns correct structure."""
        mock_generator._generate_text = Mock(return_value="Paris")

        query = "What is the capital of France?"
        chunks = [{"content": "Paris is the capital of France."}]

        result = mock_generator.generate(query, chunks)

        assert "answer" in result
        assert "prompt" in result
        assert "num_chunks_used" in result
        assert "model" in result
        assert result["num_chunks_used"] == 1

    def test_generate_max_chunks(self, mock_generator):
        """Test max_chunks parameter."""
        mock_generator._generate_text = Mock(return_value="Test answer")

        query = "Test query"
        chunks = [{"content": f"Chunk {i}"} for i in range(10)]

        result = mock_generator.generate(query, chunks, max_chunks=3)

        assert result["num_chunks_used"] == 3
        prompt = result["prompt"]
        assert "[1]" in prompt
        assert "[2]" in prompt
        assert "[3]" in prompt
        assert "[4]" not in prompt

    def test_generate_empty_chunks(self, mock_generator):
        """Test generation with empty chunks list."""
        mock_generator._generate_text = Mock(return_value="Cannot answer")

        query = "Test query"
        chunks = []

        result = mock_generator.generate(query, chunks)

        assert "answer" in result
        assert result["num_chunks_used"] == 0

    def test_batch_generate(self, mock_generator):
        """Test batch generation."""
        mock_generator._generate_text = Mock(side_effect=["Answer 1", "Answer 2"])

        queries = ["Query 1", "Query 2"]
        chunks_list = [
            [{"content": "Context 1"}],
            [{"content": "Context 2"}],
        ]

        results = mock_generator.batch_generate(queries, chunks_list)

        assert len(results) == 2
        assert all("answer" in r for r in results)
        assert results[0]["answer"] == "Answer 1"
        assert results[1]["answer"] == "Answer 2"

    def test_custom_prompt_template(self):
        """Test custom prompt template."""
        custom_template = "Context: {context}\nQ: {question}\nA:"

        with (
            patch("src.generator.AutoModelForCausalLM.from_pretrained") as mock_model_cls,
            patch("src.generator.AutoTokenizer.from_pretrained") as mock_tokenizer_cls,
        ):

            mock_model_cls.return_value = create_mock_model()
            mock_tokenizer_cls.return_value = create_mock_tokenizer()

            generator = LLMGenerator(
                model_name="test/model",
                device="cpu",
                load_in_4bit=False,
                prompt_template=custom_template,
            )

            assert generator.prompt_template == custom_template

    def test_update_prompt_template(self, mock_generator):
        """Test updating prompt template."""
        new_template = "New template with {context} and {question}"

        mock_generator.update_prompt_template(new_template)

        assert mock_generator.prompt_template == new_template

    def test_update_prompt_template_invalid(self, mock_generator):
        """Test updating with invalid template raises error."""
        invalid_template = "Missing placeholders"

        with pytest.raises(ValueError, match="must contain"):
            mock_generator.update_prompt_template(invalid_template)

    def test_update_prompt_template_missing_context(self, mock_generator):
        """Test template with missing context placeholder."""
        invalid_template = "Only has {question}"

        with pytest.raises(ValueError):
            mock_generator.update_prompt_template(invalid_template)

    def test_update_prompt_template_missing_question(self, mock_generator):
        """Test template with missing question placeholder."""
        invalid_template = "Only has {context}"

        with pytest.raises(ValueError):
            mock_generator.update_prompt_template(invalid_template)

    def test_memory_usage_cpu(self, mock_generator):
        """Test memory usage reporting on CPU."""
        mock_generator.device = torch.device("cpu")

        memory = mock_generator.get_memory_usage()

        assert "device" in memory
        assert memory["device"] == "cpu"
        assert memory["memory_used"] == 0.0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_memory_usage_gpu(self):
        """Test memory usage reporting on GPU."""
        with (
            patch("src.generator.AutoModelForCausalLM.from_pretrained") as mock_model_cls,
            patch("src.generator.AutoTokenizer.from_pretrained") as mock_tokenizer_cls,
            patch("torch.cuda.memory_allocated", return_value=1_500_000_000),
            patch("torch.cuda.memory_reserved", return_value=2_000_000_000),
        ):

            mock_model_cls.return_value = create_mock_model()
            mock_tokenizer_cls.return_value = create_mock_tokenizer()

            generator = LLMGenerator(
                model_name="test/model",
                device="cuda",
                load_in_4bit=False,
            )

            generator.device = torch.device("cuda")
            memory = generator.get_memory_usage()

            assert memory["device"] == "cuda"
            assert "allocated_gb" in memory
            assert "reserved_gb" in memory
            assert memory["allocated_gb"] > 0


class TestPromptTemplates:
    """Test suite for prompt template variations."""

    @pytest.fixture(autouse=True)
    def mock_cuda_memory(self):
        """Mock CUDA memory functions."""
        with (
            patch("torch.cuda.memory_allocated", return_value=1_000_000_000),
            patch("torch.cuda.memory_reserved", return_value=2_000_000_000),
        ):
            yield

    def test_few_shot_template(self):
        """Test few-shot prompt template."""
        few_shot_template = """Here are some examples:

Example 1:
Context: The sky is blue.
Question: What color is the sky?
Answer: Blue.

Example 2:
Context: Water freezes at 0°C.
Question: At what temperature does water freeze?
Answer: 0°C.

Now answer this:
Context: {context}
Question: {question}
Answer:"""

        with (
            patch("src.generator.AutoModelForCausalLM.from_pretrained") as mock_model_cls,
            patch("src.generator.AutoTokenizer.from_pretrained") as mock_tokenizer_cls,
        ):

            mock_model_cls.return_value = create_mock_model()
            mock_tokenizer_cls.return_value = create_mock_tokenizer()

            generator = LLMGenerator(
                model_name="test/model",
                device="cpu",
                load_in_4bit=False,
                prompt_template=few_shot_template,
            )

            assert "Example 1:" in generator.prompt_template
            assert "{context}" in generator.prompt_template
            assert "{question}" in generator.prompt_template

    def test_cot_template(self):
        """Test chain-of-thought prompt template."""
        cot_template = """Based on the context, first explain your reasoning, then answer.

Context: {context}
Question: {question}

Reasoning:
Answer:"""

        with (
            patch("src.generator.AutoModelForCausalLM.from_pretrained") as mock_model_cls,
            patch("src.generator.AutoTokenizer.from_pretrained") as mock_tokenizer_cls,
        ):

            mock_model_cls.return_value = create_mock_model()
            mock_tokenizer_cls.return_value = create_mock_tokenizer()

            generator = LLMGenerator(
                model_name="test/model",
                device="cpu",
                load_in_4bit=False,
                prompt_template=cot_template,
            )

            assert "Reasoning:" in generator.prompt_template
            assert "{context}" in generator.prompt_template
            assert "{question}" in generator.prompt_template

    def test_template_formatting(self):
        """Test template can be formatted correctly."""
        with (
            patch("src.generator.AutoModelForCausalLM.from_pretrained") as mock_model_cls,
            patch("src.generator.AutoTokenizer.from_pretrained") as mock_tokenizer_cls,
        ):

            mock_model_cls.return_value = create_mock_model()
            mock_tokenizer_cls.return_value = create_mock_tokenizer()

            generator = LLMGenerator(
                model_name="test/model",
                device="cpu",
                load_in_4bit=False,
            )

            chunks = [{"content": "Test context"}]
            context = generator._format_context(chunks)

            prompt = generator.prompt_template.format(context=context, question="Test question")

            assert "Test context" in prompt
            assert "Test question" in prompt


class TestChatTemplate:
    """Test suite for chat template wrapping in _generate_text."""

    @pytest.fixture(autouse=True)
    def mock_cuda_memory(self):
        """Mock CUDA memory functions globally."""
        with (
            patch("torch.cuda.memory_allocated", return_value=1_000_000_000),
            patch("torch.cuda.memory_reserved", return_value=2_000_000_000),
        ):
            yield

    @pytest.fixture
    def generator_with_chat_template(self):
        """Create a mock generator whose tokenizer has a chat template."""
        with (
            patch("src.generator.AutoModelForCausalLM.from_pretrained") as mock_model_cls,
            patch("src.generator.AutoTokenizer.from_pretrained") as mock_tokenizer_cls,
        ):
            mock_model_cls.return_value = create_mock_model()
            tokenizer = create_mock_tokenizer()
            tokenizer.chat_template = "{% for message in messages %}..."
            # apply_chat_template with tokenize=False returns a formatted string
            tokenizer.apply_chat_template = MagicMock(
                return_value="<|im_start|>user\nWhat is the capital?<|im_end|>\n<|im_start|>assistant\n"
            )
            # Tokenizer call returns proper dict for model.generate
            tokenizer.return_value = {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            }
            mock_tokenizer_cls.return_value = tokenizer

            generator = LLMGenerator(
                model_name="test/model",
                load_in_4bit=False,
                device="cpu",
                max_new_tokens=10,
            )

            # Mock model.generate to return token ids
            generator.model.generate = MagicMock(return_value=torch.tensor([[1, 2, 3, 4, 5]]))
            generator.tokenizer.decode = MagicMock(return_value="Paris")

            yield generator

    @pytest.fixture
    def generator_without_chat_template(self):
        """Create a mock generator whose tokenizer has no chat template."""
        with (
            patch("src.generator.AutoModelForCausalLM.from_pretrained") as mock_model_cls,
            patch("src.generator.AutoTokenizer.from_pretrained") as mock_tokenizer_cls,
        ):
            mock_model_cls.return_value = create_mock_model()
            tokenizer = create_mock_tokenizer()
            tokenizer.chat_template = None
            tokenizer.return_value = {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            }
            mock_tokenizer_cls.return_value = tokenizer

            generator = LLMGenerator(
                model_name="test/model",
                load_in_4bit=False,
                device="cpu",
                max_new_tokens=10,
            )

            generator.model.generate = MagicMock(return_value=torch.tensor([[1, 2, 3, 4, 5]]))
            generator.tokenizer.decode = MagicMock(return_value="Paris")

            yield generator

    def test_chat_template_used_when_available(self, generator_with_chat_template):
        """Test that apply_chat_template is called for instruction-tuned models."""
        generator_with_chat_template._generate_text("What is the capital?")

        generator_with_chat_template.tokenizer.apply_chat_template.assert_called_once()
        call_args = generator_with_chat_template.tokenizer.apply_chat_template.call_args
        messages = call_args[0][0]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "What is the capital?"

    def test_chat_template_add_generation_prompt(self, generator_with_chat_template):
        """Test that add_generation_prompt=True and tokenize=False are passed."""
        generator_with_chat_template._generate_text("Test prompt")

        call_kwargs = generator_with_chat_template.tokenizer.apply_chat_template.call_args[1]
        assert call_kwargs["add_generation_prompt"] is True
        assert call_kwargs["tokenize"] is False

    def test_fallback_tokenization_without_chat_template(self, generator_without_chat_template):
        """Test that raw tokenization is used when no chat template exists."""
        generator_without_chat_template._generate_text("What is the capital?")

        # Should NOT call apply_chat_template
        assert (
            not hasattr(generator_without_chat_template.tokenizer, "apply_chat_template")
            or not generator_without_chat_template.tokenizer.apply_chat_template.called
        )
        # Should call tokenizer directly
        generator_without_chat_template.tokenizer.assert_called()


@pytest.mark.skipif(not TORCH_2_6_AVAILABLE, reason="Torch >= 2.6 required for safe model loading")
class TestRealModelLoading:
    """
    Tests with real (tiny) model loading.

    Only runs if torch >= 2.6 for security.
    """

    @pytest.fixture
    def tiny_generator(self):
        """Create generator with tiny GPT2 model for testing."""
        return LLMGenerator(
            model_name="sshleifer/tiny-gpt2",
            load_in_4bit=False,
            device="cpu",
            max_new_tokens=10,
        )

    def test_real_model_initialization(self, tiny_generator):
        """Test initialization with real tiny model."""
        assert tiny_generator.model is not None
        assert tiny_generator.tokenizer is not None

    def test_real_generation(self, tiny_generator):
        """Test actual generation with tiny model."""
        # Ensure model is on CPU
        tiny_generator.model = tiny_generator.model.to("cpu")
        tiny_generator.device = torch.device("cpu")

        query = "What is the capital?"
        chunks = [{"content": "Paris is the capital of France."}]

        result = tiny_generator.generate(query, chunks)

        assert "answer" in result
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0


if __name__ == "__main__":
    # Print torch version info
    print(f"Torch version: {torch.__version__}")
    print(f"Torch >= 2.6: {TORCH_2_6_AVAILABLE}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    pytest.main([__file__, "-v", "--tb=short"])
