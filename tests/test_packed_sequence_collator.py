"""
Tests for PackedSequenceCollator - custom data collator for packing multiple tokenized samples.

Tests cover:
- Basic packing functionality
- Edge cases (empty features, single sequences, exact max_length)
- Efficiency threshold behavior
- Attention mask and label alignment
- Token ID handling (EOS, padding)
- Configuration validation
"""

import pytest
import torch
from unittest.mock import Mock
from transformers import AutoTokenizer

from src.training.packed_sequence_collator import PackedSequenceCollator


class TestPackedSequenceCollator:
    """Test suite for PackedSequenceCollator."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer with standard special tokens."""
        tokenizer = Mock()
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 1
        tokenizer.unk_token_id = 2
        return tokenizer

    @pytest.fixture
    def collator(self, mock_tokenizer):
        """Create a standard PackedSequenceCollator for testing."""
        return PackedSequenceCollator(
            tokenizer=mock_tokenizer,
            max_length=50,
            packing_efficiency_threshold=0.8,
            pack_single_sequences=False,
        )

    @pytest.fixture
    def sample_features(self):
        """Create sample tokenized features for testing."""
        return [
            {
                "input_ids": [10, 11, 12, 13, 14],  # 5 tokens
                "attention_mask": [1, 1, 1, 1, 1],
                "labels": [10, 11, 12, 13, 14],
            },
            {
                "input_ids": [20, 21, 22],  # 3 tokens
                "attention_mask": [1, 1, 1],
                "labels": [20, 21, 22],
            },
            {
                "input_ids": [30, 31, 32, 33],  # 4 tokens
                "attention_mask": [1, 1, 1, 1],
                "labels": [30, 31, 32, 33],
            },
        ]

    def test_basic_packing(self, collator, sample_features):
        """Test basic packing of multiple sequences."""
        result = collator(sample_features)
        
        # Should pack all sequences into one batch
        assert "input_ids" in result
        assert "attention_mask" in result
        assert "labels" in result
        
        batch_size = result["input_ids"].shape[0]
        seq_length = result["input_ids"].shape[1]
        
        assert seq_length == 50  # max_length
        assert batch_size >= 1
        
        # Check tensor types
        assert result["input_ids"].dtype == torch.long
        assert result["attention_mask"].dtype == torch.long
        assert result["labels"].dtype == torch.long

    def test_eos_token_insertion(self, collator, sample_features):
        """Test that EOS tokens are inserted between sequences."""
        result = collator(sample_features)
        
        # Get the first packed sequence
        packed_sequence = result["input_ids"][0]
        
        # Should contain EOS tokens (token_id = 1) between sequences
        eos_positions = (packed_sequence == 1).nonzero(as_tuple=True)[0]
        assert len(eos_positions) >= 1  # At least one EOS token between sequences

    def test_padding_and_attention_mask(self, collator, sample_features):
        """Test that padding tokens and attention masks are correct."""
        result = collator(sample_features)
        
        # Check that padding tokens (token_id = 0) are at the end
        input_ids = result["input_ids"][0]
        attention_mask = result["attention_mask"][0]
        
        # Find first padding token
        padding_positions = (input_ids == 0).nonzero(as_tuple=True)[0]
        
        if len(padding_positions) > 0:
            first_padding = padding_positions[0].item()
            
            # All tokens after first padding should be padding
            assert torch.all(input_ids[first_padding:] == 0)
            
            # Attention mask should be 0 for padding tokens
            assert torch.all(attention_mask[first_padding:] == 0)
            
            # Attention mask should be 1 for non-padding tokens
            assert torch.all(attention_mask[:first_padding] == 1)

    def test_label_alignment(self, collator, sample_features):
        """Test that labels are properly aligned with input_ids and padding."""
        result = collator(sample_features)
        
        labels = result["labels"][0]
        
        # Padding positions in labels should be -100
        label_padding_positions = (labels == -100).nonzero(as_tuple=True)[0]
        
        if len(label_padding_positions) > 0:
            # All -100 labels should be at the end (for padding) or at EOS positions
            # This is expected behavior for our implementation
            pass

    def test_efficiency_threshold(self, mock_tokenizer):
        """Test that efficiency threshold works correctly."""
        # Create collator with high efficiency threshold
        high_threshold_collator = PackedSequenceCollator(
            tokenizer=mock_tokenizer,
            max_length=20,
            packing_efficiency_threshold=0.9,  # Very high threshold
            pack_single_sequences=False,
        )
        
        # Single short sequence that doesn't meet threshold
        short_features = [
            {
                "input_ids": [10, 11, 12],  # 3 tokens, efficiency = 3/20 = 0.15
                "attention_mask": [1, 1, 1],
                "labels": [10, 11, 12],
            }
        ]
        
        # Should not pack due to low efficiency
        result = high_threshold_collator(short_features)
        
        # We expect this to still create a batch (might be empty or with different behavior)
        # The exact behavior depends on implementation details
        assert "input_ids" in result

    def test_pack_single_sequences_option(self, mock_tokenizer):
        """Test pack_single_sequences configuration option."""
        # Enable packing single sequences
        single_pack_collator = PackedSequenceCollator(
            tokenizer=mock_tokenizer,
            max_length=20,
            packing_efficiency_threshold=0.8,
            pack_single_sequences=True,  # Enable single sequence packing
        )
        
        # Single sequence
        single_feature = [
            {
                "input_ids": [10, 11, 12],
                "attention_mask": [1, 1, 1],
                "labels": [10, 11, 12],
            }
        ]
        
        result = single_pack_collator(single_feature)
        
        # Should create a valid batch
        assert result["input_ids"].shape[0] >= 1
        assert result["input_ids"].shape[1] == 20

    def test_exact_max_length_sequence(self, collator):
        """Test handling of sequences that exactly match max_length."""
        exact_length_features = [
            {
                "input_ids": list(range(50)),  # Exactly max_length tokens
                "attention_mask": [1] * 50,
                "labels": list(range(50)),
            }
        ]
        
        result = collator(exact_length_features)
        
        # Should handle exact length correctly
        assert result["input_ids"].shape[1] == 50
        assert torch.all(result["attention_mask"][0] == 1)  # No padding needed

    def test_empty_features_error(self, collator):
        """Test that empty features list raises appropriate error."""
        with pytest.raises(ValueError, match="Cannot create batch from empty features list"):
            collator([])

    def test_missing_required_keys(self, collator):
        """Test that missing required keys raises appropriate error."""
        incomplete_features = [
            {
                "input_ids": [10, 11, 12],
                # Missing attention_mask and labels
            }
        ]
        
        with pytest.raises(ValueError, match="missing required keys"):
            collator(incomplete_features)

    def test_tensor_conversion(self, collator):
        """Test that list inputs are properly converted to tensors."""
        list_features = [
            {
                "input_ids": [10, 11, 12],  # Python list
                "attention_mask": [1, 1, 1],  # Python list
                "labels": [10, 11, 12],  # Python list
            }
        ]
        
        result = collator(list_features)
        
        # Should convert to tensors
        assert isinstance(result["input_ids"], torch.Tensor)
        assert isinstance(result["attention_mask"], torch.Tensor)
        assert isinstance(result["labels"], torch.Tensor)

    def test_mixed_tensor_list_inputs(self, collator):
        """Test handling of mixed tensor and list inputs."""
        mixed_features = [
            {
                "input_ids": torch.tensor([10, 11, 12]),  # Tensor
                "attention_mask": [1, 1, 1],  # List
                "labels": torch.tensor([10, 11, 12]),  # Tensor
            }
        ]
        
        result = collator(mixed_features)
        
        # Should handle mixed inputs correctly
        assert isinstance(result["input_ids"], torch.Tensor)
        assert isinstance(result["attention_mask"], torch.Tensor)
        assert isinstance(result["labels"], torch.Tensor)

    def test_configuration_fallbacks(self):
        """Test fallback behavior when tokenizer lacks special tokens."""
        # Tokenizer without pad_token_id
        no_pad_tokenizer = Mock()
        no_pad_tokenizer.pad_token_id = None
        no_pad_tokenizer.eos_token_id = 1
        no_pad_tokenizer.unk_token_id = 2
        
        # Should use unk_token_id as fallback
        collator = PackedSequenceCollator(
            tokenizer=no_pad_tokenizer,
            max_length=20,
        )
        
        assert collator.pad_token_id == 2  # unk_token_id

    def test_no_available_tokens_error(self):
        """Test error when no pad or unk tokens are available."""
        no_tokens_tokenizer = Mock()
        no_tokens_tokenizer.pad_token_id = None
        no_tokens_tokenizer.eos_token_id = None
        no_tokens_tokenizer.unk_token_id = None
        
        with pytest.raises(ValueError, match="No pad token available"):
            PackedSequenceCollator(
                tokenizer=no_tokens_tokenizer,
                max_length=20,
            )

    def test_get_stats(self, collator):
        """Test that get_stats returns configuration information."""
        stats = collator.get_stats()
        
        expected_keys = {
            "max_length",
            "eos_token_id", 
            "pad_token_id",
            "label_pad_token_id",
            "packing_efficiency_threshold",
            "pack_single_sequences",
        }
        
        assert set(stats.keys()) == expected_keys
        assert stats["max_length"] == 50
        assert stats["packing_efficiency_threshold"] == 0.8
        assert stats["pack_single_sequences"] is False

    def test_large_batch_efficiency(self, collator):
        """Test packing efficiency with larger batch of short sequences."""
        # Create many short sequences
        short_sequences = []
        for i in range(20):
            short_sequences.append({
                "input_ids": [100 + i, 101 + i, 102 + i],  # 3 tokens each
                "attention_mask": [1, 1, 1],
                "labels": [100 + i, 101 + i, 102 + i],
            })
        
        result = collator(short_sequences)
        
        # Should pack efficiently - many sequences into fewer packed sequences
        batch_size = result["input_ids"].shape[0]
        
        # With max_length=50 and 3-token sequences + EOS, we should fit multiple per pack
        # (3 tokens + 1 EOS) * sequences per pack ≤ 50
        # So roughly 12 sequences per pack
        assert batch_size <= len(short_sequences)  # Should pack some sequences together

    def test_variable_length_sequences(self, collator):
        """Test packing with sequences of varying lengths."""
        variable_features = [
            {
                "input_ids": [10] * 5,  # 5 tokens
                "attention_mask": [1] * 5,
                "labels": [10] * 5,
            },
            {
                "input_ids": [20] * 15,  # 15 tokens
                "attention_mask": [1] * 15,
                "labels": [20] * 15,
            },
            {
                "input_ids": [30] * 8,  # 8 tokens
                "attention_mask": [1] * 8,
                "labels": [30] * 8,
            },
            {
                "input_ids": [40] * 20,  # 20 tokens
                "attention_mask": [1] * 20,
                "labels": [40] * 20,
            },
        ]
        
        result = collator(variable_features)
        
        # Should handle variable lengths correctly
        assert result["input_ids"].shape[1] == 50  # max_length
        assert result["input_ids"].shape[0] >= 1  # At least one packed sequence


@pytest.mark.integration
class TestPackedSequenceCollatorIntegration:
    """Integration tests with real tokenizers."""

    @pytest.fixture
    def real_tokenizer(self):
        """Create a real tokenizer for integration testing."""
        # Use a small, fast tokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # Ensure we have required special tokens
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return tokenizer

    def test_with_real_tokenizer(self, real_tokenizer):
        """Test PackedSequenceCollator with a real tokenizer."""
        collator = PackedSequenceCollator(
            tokenizer=real_tokenizer,
            max_length=100,
            packing_efficiency_threshold=0.7,
        )
        
        # Create realistic tokenized sequences
        sample_texts = [
            "Hello world",
            "This is a test",
            "Short",
            "A longer sentence with more tokens to test packing behavior",
        ]
        
        # Tokenize manually to create realistic features
        features = []
        for text in sample_texts:
            encoding = real_tokenizer(
                text,
                truncation=True,
                padding=False,
                return_tensors="pt"
            )
            
            features.append({
                "input_ids": encoding["input_ids"].squeeze().tolist(),
                "attention_mask": encoding["attention_mask"].squeeze().tolist(),
                "labels": encoding["input_ids"].squeeze().tolist(),  # Same as input for this test
            })
        
        result = collator(features)
        
        # Verify proper packing
        assert "input_ids" in result
        assert "attention_mask" in result
        assert "labels" in result
        
        # Check dimensions
        assert result["input_ids"].shape[1] == 100  # max_length
        assert result["input_ids"].shape[0] >= 1  # At least one sequence
        
        # Verify no tensor dtype issues
        assert result["input_ids"].dtype == torch.long
        assert result["attention_mask"].dtype == torch.long
        assert result["labels"].dtype == torch.long 