import pytest
import torch
import os
import tempfile
from SGS_Tokenizer import ExtendedGPT2Tokenizer

class TestExtendedGPT2Tokenizer:
    @pytest.fixture
    def tokenizer(self):
        # Initialize with default GPT2 files
        return ExtendedGPT2Tokenizer.from_pretrained('gpt2')

    @pytest.fixture
    def sample_text(self):
        return "Hello, this is a test message."

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            yield tmpdirname

    def test_initialization(self, tokenizer):
        assert tokenizer.p == 4
        assert tokenizer.int_bit == 32
        assert tokenizer.num_bins == 16
        assert isinstance(tokenizer.tensor_0, dict)
        assert isinstance(tokenizer.tensor_1, torch.Tensor)
        assert isinstance(tokenizer.tensor_2, torch.Tensor)

    def test_tokenize_and_update_tensors(self, tokenizer, sample_text):
        # Test tokenization
        token_ids = tokenizer.tokenize_text(sample_text)
        assert isinstance(token_ids, list)
        assert len(token_ids) > 0

        # Test tensor updates
        new_tensor, double_values = tokenizer.update_tensors(token_ids)
        assert isinstance(new_tensor, torch.Tensor)
        assert isinstance(double_values, list)
        assert len(double_values) == len(token_ids)

    def test_tensor_conversions(self, tokenizer, sample_text):
        # Create test data
        token_ids = tokenizer.tokenize_text(sample_text)
        new_tensor, _ = tokenizer.update_tensors(token_ids)

        # Test tensor to HLL conversion
        id, sha1, hlltensor = tokenizer.tensor_to_hlltensor(new_tensor)
        assert isinstance(id, int)
        assert isinstance(sha1, str)
        assert isinstance(hlltensor, torch.Tensor)
        assert hlltensor.size(0) == tokenizer.num_bins

        # Test HLL to tensor conversion
        tensor_slice = tokenizer.hlltensor_to_tensor(hlltensor)
        assert isinstance(tensor_slice, torch.Tensor)
        assert tensor_slice.size(1) == 3  # token_id, hash_value, double_value

    def test_search_functionality(self, tokenizer, sample_text):
        # Prepare data
        token_ids = tokenizer.tokenize_text(sample_text)
        tokenizer.update_tensors(token_ids)

        # Test search
        results = tokenizer.search(sample_text, threshold=0.5)
        assert isinstance(results, list)

        # Test related tokens
        if results:
            tokens = tokenizer.get_related_tokens(results)
            assert isinstance(tokens, list)

    def test_sequence_conversions(self, tokenizer):
        # Test integer to sequence
        test_int = 42
        sequence = tokenizer.sequence_from_integer(test_int)
        assert isinstance(sequence, list)
        assert all(isinstance(x, int) for x in sequence)

        # Test sequence to integer
        result = tokenizer.sequence_to_integer(0, sequence)
        assert isinstance(result, int)

    def test_hash_functions(self, tokenizer):
        # Test single token hash
        token_id = 42
        hash_value = tokenizer.hash_token_id(token_id)
        assert isinstance(hash_value, int)

        # Test multiple token hash
        token_ids = [42, 43, 44]
        hash_values = tokenizer.hash_token_ids(token_ids)
        assert isinstance(hash_values, list)
        assert len(hash_values) == len(token_ids)

    def test_bin_calculations(self, tokenizer):
        hash_value = 12345
        bin_number, trailing_zeros = tokenizer.calculate_bin_and_zeros(hash_value)
        assert isinstance(bin_number, int)
        assert isinstance(trailing_zeros, int)
        assert 0 <= bin_number < tokenizer.num_bins

    def test_save_load_tensors(self, tokenizer, temp_dir):
        # Save tensors
        tokenizer.save_tensors(temp_dir)
        
        # Verify files exist
        assert os.path.exists(os.path.join(temp_dir, 'tensor_0.pt'))
        assert os.path.exists(os.path.join(temp_dir, 'tensor_1.pt'))
        assert os.path.exists(os.path.join(temp_dir, 'tensor_2.pt'))

        # Load new tokenizer with saved tensors
        new_tokenizer = ExtendedGPT2Tokenizer.from_pretrained(
            'gpt2',
            tensor_0_path=os.path.join(temp_dir, 'tensor_0.pt'),
            tensor_1_path=os.path.join(temp_dir, 'tensor_1.pt'),
            tensor_2_path=os.path.join(temp_dir, 'tensor_2.pt')
        )

        # Verify tensors are loaded correctly
        assert len(new_tokenizer.tensor_0) == len(tokenizer.tensor_0)
        assert torch.equal(new_tokenizer.tensor_1, tokenizer.tensor_1)
        assert torch.equal(new_tokenizer.tensor_2, tokenizer.tensor_2)

    def test_error_handling(self, tokenizer):
        # Test invalid token ID
        with pytest.raises(KeyError):
            tokenizer.get_tensor_0("invalid_key")

        # Test invalid search query
        with pytest.raises(ValueError):
            tokenizer.search(None, 0.5)

        # Test invalid threshold
        with pytest.raises(ValueError):
            tokenizer.search("test", -1)

    @pytest.mark.parametrize("text,threshold", [
        ("Hello world", 0.5),
        ("Python programming", 0.7),
        ("Machine learning", 0.3),
    ])
    def test_search_with_different_inputs(self, tokenizer, text, threshold):
        token_ids = tokenizer.tokenize_text(text)
        tokenizer.update_tensors(token_ids)
        results = tokenizer.search(text, threshold)
        assert isinstance(results, list)
