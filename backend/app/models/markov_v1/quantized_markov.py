# Quantized Markov Chain Implementation
# Uses 16-bit quantization to compress transition probabilities
# Reduces memory usage from ~10GB to ~5GB for 50K states

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from collections import defaultdict, Counter
import pickle
from typing import List, Dict, Tuple, Optional, Union, Any

class QuantizedMarkovChain:
    """
    Memory-efficient Markov Chain using 16-bit quantized probabilities.
    
    Key innovation: Instead of storing float64 probabilities (8 bytes each),
    we quantize to uint16 (2 bytes each), reducing memory by 75%.
    
    For 50K states:
    - Dense matrix: 50K × 50K × 8 bytes = ~20GB (impossible)
    - Sparse float64: ~hundreds of MB to GB depending on sparsity
    - Quantized sparse uint16: ~25-50% of sparse float64 size
    
    Quantization levels: 65536 (2^16) discrete probability values
    Resolution: ~0.0000153 (1/65535) - more than sufficient for recommendations
    """
    
    def __init__(self, states: Optional[List[str]] = None, quantization_bits: int = 8):
        """
        Initialize the Quantized Markov Chain.
        
        Args:
            states: Optional list of state names
            quantization_bits: Number of bits for quantization (default: 8 bits = 256 levels)
                             8 bits:  256 levels,  2.33GB for 50K dense states
                             6 bits:  64 levels,   1.75GB for 50K dense states  
                             4 bits:  16 levels,   1.17GB for 50K dense states
        """
        self.states = states
        self.state_to_idx = {}
        self.idx_to_state = {}
        self.n_states = 0
        self.quantization_bits = quantization_bits
        self.quantization_levels = 2 ** quantization_bits
        self.max_quant_value = self.quantization_levels - 1
        
        # Sparse matrices - using uint8 for quantized probabilities (8-bit = 1 byte)
        self.transition_counts = None  # lil_matrix for construction
        self.quantized_probs = None    # csr_matrix with uint8 values
        
        # Statistics
        self.total_transitions = 0
        self.state_frequencies = None
        self.is_fitted = False
        
        # Quantization parameters
        self._prob_to_quant_scale = self.max_quant_value  # Scale factor for quantization
        self._quant_to_prob_scale = 1.0 / self.max_quant_value  # Scale factor for dequantization
        
    def _build_state_mapping(self, sequences: List[List[str]]) -> None:
        """Build bidirectional mapping between states and indices."""
        if self.states is None:
            # Extract unique states from sequences
            unique_states = set()
            for seq in sequences:
                unique_states.update(seq)
            self.states = sorted(list(unique_states))
        
        self.state_to_idx = {state: idx for idx, state in enumerate(self.states)}
        self.idx_to_state = {idx: state for idx, state in enumerate(self.states)}
        self.n_states = len(self.states)
        
    def _quantize_probability(self, prob: float) -> int:
        """Convert float probability to quantized integer."""
        return int(np.round(prob * self._prob_to_quant_scale))
    
    def _dequantize_probability(self, quant_val: int) -> float:
        """Convert quantized integer back to float probability."""
        return quant_val * self._quant_to_prob_scale
    
    def fit(self, sequences: List[List[str]], verbose: bool = True) -> None:
        """
        Fit the Markov chain from sequence data with quantization.
        
        Args:
            sequences: List of sequences, where each sequence is a list of states
            verbose: Whether to print progress information
        """
        if verbose:
            print(f"Building state mapping from {len(sequences)} sequences...")
        
        self._build_state_mapping(sequences)
        
        if verbose:
            print(f"Found {self.n_states} unique states")
            print(f"Theoretical dense matrix size: {self.n_states**2 * 8 / 1e9:.2f} GB")
            print(f"Building transition counts...")
        
        # Use lil_matrix for efficient construction
        self.transition_counts = lil_matrix((self.n_states, self.n_states), dtype=np.uint32)
        
        # Count transitions
        total_transitions = 0
        for seq_idx, sequence in enumerate(sequences):
            if verbose and seq_idx % 10000 == 0:
                print(f"Processing sequence {seq_idx}/{len(sequences)}")
                
            for i in range(len(sequence) - 1):
                from_state = sequence[i]
                to_state = sequence[i + 1]
                
                # Skip unknown states
                if from_state not in self.state_to_idx or to_state not in self.state_to_idx:
                    continue
                    
                from_idx = self.state_to_idx[from_state]
                to_idx = self.state_to_idx[to_state]
                
                self.transition_counts[from_idx, to_idx] += 1
                total_transitions += 1
        
        self.total_transitions = total_transitions
        
        if verbose:
            print(f"Total transitions: {total_transitions}")
            print(f"Converting to probabilities and quantizing...")
        
        # Convert to CSR for efficient row operations
        counts_csr = self.transition_counts.tocsr()
        
        # Calculate row sums for normalization
        row_sums = np.array(counts_csr.sum(axis=1)).flatten()
        
        # Create quantized probability matrix
        # We'll store quantized values as uint16 in a sparse matrix
        quantized_data = []
        quantized_indices = []
        quantized_indptr = [0]
        
        for row in range(self.n_states):
            start_idx = counts_csr.indptr[row]
            end_idx = counts_csr.indptr[row + 1]
            
            if row_sums[row] == 0:
                # No outgoing transitions - skip this row (will be handled during prediction)
                quantized_indptr.append(len(quantized_data))
                continue
            
            # Get non-zero counts for this row
            row_counts = counts_csr.data[start_idx:end_idx]
            row_indices = counts_csr.indices[start_idx:end_idx]
            
            # Convert to probabilities
            row_probs = row_counts / row_sums[row]
            
            # Quantize probabilities
            quantized_row_probs = np.array([self._quantize_probability(p) for p in row_probs], dtype=np.uint16)
            
            # Ensure at least one non-zero quantized value (handle very small probabilities)
            if np.sum(quantized_row_probs) == 0:
                # If all probabilities quantized to 0, set the highest original probability to 1
                max_idx = np.argmax(row_probs)
                quantized_row_probs[max_idx] = 1
            
            # Store in sparse format
            quantized_data.extend(quantized_row_probs)
            quantized_indices.extend(row_indices)
            quantized_indptr.append(len(quantized_data))
        
        # Create sparse matrix with quantized values
        self.quantized_probs = csr_matrix(
            (quantized_data, quantized_indices, quantized_indptr),
            shape=(self.n_states, self.n_states),
            dtype=np.uint16
        )
        
        # Calculate state frequencies
        self.state_frequencies = np.bincount([self.state_to_idx[state] 
                                           for seq in sequences for state in seq 
                                           if state in self.state_to_idx], 
                                          minlength=self.n_states)
        
        self.is_fitted = True
        
        if verbose:
            nnz = self.quantized_probs.nnz
            sparsity = 1.0 - (nnz / (self.n_states ** 2))
            memory_mb = (nnz * 2 + self.n_states * 4) / 1e6  # uint16 data + uint32 indices
            print(f"Quantized sparse matrix: {nnz:,} non-zero elements ({sparsity:.4f} sparsity)")
            print(f"Estimated memory usage: {memory_mb:.2f} MB")
            print(f"Quantization levels used: {len(np.unique(self.quantized_probs.data))}/{self.quantization_levels}")
    
    def predict_next(self, current_state: str, top_k: int = 5, 
                    use_fallback: bool = True) -> List[Tuple[str, float]]:
        """
        Predict the next most likely states with dequantized probabilities.
        
        Args:
            current_state: Current state
            top_k: Number of top predictions to return
            use_fallback: Whether to use frequency-based fallback for unseen states
            
        Returns:
            List of (state, probability) tuples, sorted by probability (descending)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if current_state not in self.state_to_idx:
            if not use_fallback:
                return []
            # Fallback to most frequent states
            top_indices = np.argsort(self.state_frequencies)[-top_k:][::-1]
            total_freq = np.sum(self.state_frequencies)
            return [(self.idx_to_state[idx], self.state_frequencies[idx] / total_freq) 
                   for idx in top_indices]
        
        current_idx = self.state_to_idx[current_state]
        
        # Get quantized probabilities for current state
        start_idx = self.quantized_probs.indptr[current_idx]
        end_idx = self.quantized_probs.indptr[current_idx + 1]
        
        if start_idx == end_idx:
            # No transitions from this state
            if not use_fallback:
                return []
            # Fallback to most frequent states
            top_indices = np.argsort(self.state_frequencies)[-top_k:][::-1]
            total_freq = np.sum(self.state_frequencies)
            return [(self.idx_to_state[idx], self.state_frequencies[idx] / total_freq) 
                   for idx in top_indices]
        
        # Get quantized values and indices
        quantized_vals = self.quantized_probs.data[start_idx:end_idx]
        col_indices = self.quantized_probs.indices[start_idx:end_idx]
        
        # Dequantize probabilities
        probs = quantized_vals * self._quant_to_prob_scale
        
        # Normalize (in case of quantization errors)
        probs = probs / np.sum(probs)
        
        # Get top-k predictions
        if len(probs) <= top_k:
            # Return all available transitions
            predictions = [(self.idx_to_state[col_indices[i]], probs[i]) 
                          for i in range(len(probs))]
        else:
            # Get top-k indices
            top_indices = np.argsort(probs)[-top_k:][::-1]
            predictions = [(self.idx_to_state[col_indices[i]], probs[i]) 
                          for i in top_indices]
        
        return predictions
    
    def predict_sequence(self, start_state: str, length: int, 
                        method: str = 'sample') -> List[str]:
        """
        Generate a sequence of states using quantized probabilities.
        
        Args:
            start_state: Starting state
            length: Length of sequence to generate
            method: 'sample' for probabilistic sampling, 'greedy' for most likely path
            
        Returns:
            List of states
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if start_state not in self.state_to_idx:
            raise ValueError(f"Unknown state: {start_state}")
        
        sequence = [start_state]
        current_state = start_state
        
        for _ in range(length - 1):
            current_idx = self.state_to_idx[current_state]
            
            # Get quantized probabilities
            start_idx = self.quantized_probs.indptr[current_idx]
            end_idx = self.quantized_probs.indptr[current_idx + 1]
            
            if start_idx == end_idx:
                # No transitions available
                break
            
            quantized_vals = self.quantized_probs.data[start_idx:end_idx]
            col_indices = self.quantized_probs.indices[start_idx:end_idx]
            
            # Dequantize and normalize
            probs = quantized_vals * self._quant_to_prob_scale
            probs = probs / np.sum(probs)
            
            if method == 'sample':
                # Probabilistic sampling
                next_idx_pos = np.random.choice(len(probs), p=probs)
                next_idx = col_indices[next_idx_pos]
            elif method == 'greedy':
                # Most likely next state
                max_pos = np.argmax(probs)
                next_idx = col_indices[max_pos]
            else:
                raise ValueError("Method must be 'sample' or 'greedy'")
            
            next_state = self.idx_to_state[next_idx]
            sequence.append(next_state)
            current_state = next_state
        
        return sequence
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get detailed memory usage statistics in MB."""
        if not self.is_fitted:
            return {"error": "Model not fitted"}
        
        # Quantized probabilities matrix
        data_size = self.quantized_probs.data.nbytes / 1e6
        indices_size = self.quantized_probs.indices.nbytes / 1e6
        indptr_size = self.quantized_probs.indptr.nbytes / 1e6
        
        # State mappings
        states_size = sum(len(s.encode('utf-8')) for s in self.states) / 1e6
        
        # Other arrays
        freq_size = self.state_frequencies.nbytes / 1e6 if self.state_frequencies is not None else 0
        
        total = data_size + indices_size + indptr_size + states_size + freq_size
        
        return {
            "quantized_data_mb": data_size,
            "indices_mb": indices_size,
            "indptr_mb": indptr_size,
            "states_mb": states_size,
            "frequencies_mb": freq_size,
            "total_mb": total,
            "compression_ratio": f"16-bit quantization vs 64-bit float: ~4x smaller"
        }
    
    def save(self, filepath: str) -> None:
        """Save the quantized model to disk."""
        model_data = {
            'states': self.states,
            'state_to_idx': self.state_to_idx,
            'idx_to_state': self.idx_to_state,
            'n_states': self.n_states,
            'quantization_levels': self.quantization_levels,
            'max_quant_value': self.max_quant_value,
            'quantized_probs': self.quantized_probs,
            'total_transitions': self.total_transitions,
            'state_frequencies': self.state_frequencies,
            'is_fitted': self.is_fitted,
            '_prob_to_quant_scale': self._prob_to_quant_scale,
            '_quant_to_prob_scale': self._quant_to_prob_scale
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'QuantizedMarkovChain':
        """Load a quantized model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create new instance
        model = cls()
        
        # Restore all attributes
        for key, value in model_data.items():
            setattr(model, key, value)
        
        return model

# Example usage and testing
if __name__ == "__main__":
    # Test with sample data
    sample_sequences = [
        ['A', 'B', 'C', 'A'],
        ['A', 'C', 'B', 'A'],
        ['B', 'A', 'C'],
        ['C', 'A', 'B', 'C']
    ]
    
    print("Testing Quantized Markov Chain...")
    qmc = QuantizedMarkovChain(quantization_levels=65536)
    qmc.fit(sample_sequences)
    
    print("\nMemory usage:")
    for key, value in qmc.get_memory_usage().items():
        print(f"  {key}: {value}")
    
    print(f"\nPredictions from state 'A':")
    predictions = qmc.predict_next('A', top_k=3)
    for state, prob in predictions:
        print(f"  {state}: {prob:.6f}")
    
    print(f"\nGenerated sequence from 'A':")
    sequence = qmc.predict_sequence('A', length=5, method='sample')
    print(f"  {' -> '.join(sequence)}")
