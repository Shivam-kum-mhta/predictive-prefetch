# Custom Implementation of Markov Chains
# With Sparse Transition Matrix for Large State Spaces

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.sparse.linalg import eigs
from collections import defaultdict, Counter
import pickle
import warnings
from typing import List, Dict, Tuple, Optional, Union, Any

class SparseMarkovChain:
    """
    A memory-efficient Markov Chain implementation using sparse matrices.
    
    Features:
    - Sparse matrix storage for transition probabilities
    - Efficient fitting from sequence data
    - Multiple prediction methods
    - Statistical analysis (steady state, entropy, etc.)
    - Model persistence (save/load)
    - Memory usage tracking
    """
    
    def __init__(self, states: Optional[List[str]] = None):
        """
        Initialize the Markov Chain.
        
        Args:
            states: Optional list of state names. If None, states will be inferred during fitting.
        """
        self.states = states
        self.state_to_idx = {}
        self.idx_to_state = {}
        self.n_states = 0
        
        # Sparse matrices for efficient storage
        self.transition_counts = None  # lil_matrix for efficient construction
        self.transition_probs = None   # csr_matrix for efficient computation
        
        # Statistics
        self.total_transitions = 0
        self.state_frequencies = None
        self.is_fitted = False
        
    def _build_state_mapping(self, sequences: List[List[str]]) -> None:
        """Build mapping between states and indices."""
        if self.states is None:
            # Extract unique states from sequences
            unique_states = set()
            for sequence in sequences:
                unique_states.update(sequence)
            self.states = sorted(list(unique_states))
        
        self.n_states = len(self.states)
        self.state_to_idx = {state: idx for idx, state in enumerate(self.states)}
        self.idx_to_state = {idx: state for idx, state in enumerate(self.states)}
        
        print(f"Initialized with {self.n_states} states")
    
    def _parse_time_segments(self, modifs: Optional[Dict]) -> List[Dict]:
        """
        Parse time segments from modifs dictionary.
        
        Args:
            modifs: Dictionary containing timeSeg modifications
            
        Returns:
            List of parsed time segments with start/end times as minutes since midnight
        """
        if not modifs or 'timeSeg' not in modifs:
            return []
        
        time_segments = []
        for segment in modifs['timeSeg']:
            try:
                # Parse time strings (HH:MM format)
                start_time = segment['timeStart']
                end_time = segment['timeEnd']
                factor = segment['factor']
                
                # Convert to minutes since midnight
                start_hour, start_min = map(int, start_time.split(':'))
                end_hour, end_min = map(int, end_time.split(':'))
                
                start_minutes = start_hour * 60 + start_min
                end_minutes = end_hour * 60 + end_min
                
                time_segments.append({
                    'start_minutes': start_minutes,
                    'end_minutes': end_minutes,
                    'factor': factor
                })
                
            except (KeyError, ValueError, IndexError) as e:
                print(f"Warning: Invalid time segment format: {segment}, error: {e}")
                continue
        
        return time_segments
    
    def _get_time_weight(self, timestamp: str, time_segments: List[Dict]) -> float:
        """
        Get weight factor for a given timestamp based on time segments.
        
        Args:
            timestamp: Timestamp string (assumes format that can be parsed)
            time_segments: List of time segments with start/end times and factors
            
        Returns:
            Weight factor (1.0 if no matching segment)
        """
        if not time_segments:
            return 1.0
        
        try:
            # Extract time from timestamp (assuming format like "2020-01-01 14:30:00")
            # This is a simplified parser - you may need to adjust based on your timestamp format
            if ' ' in timestamp:
                time_part = timestamp.split(' ')[1]  # Get time part
            else:
                time_part = timestamp
            
            hour, minute = map(int, time_part.split(':')[:2])
            current_minutes = hour * 60 + minute
            
            # Check if current time falls within any segment
            for segment in time_segments:
                start_min = segment['start_minutes']
                end_min = segment['end_minutes']
                
                # Handle segments that cross midnight (e.g., 22:00 to 06:00)
                if start_min > end_min:
                    if current_minutes >= start_min or current_minutes <= end_min:
                        return segment['factor']
                else:
                    if start_min <= current_minutes <= end_min:
                        return segment['factor']
            
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not parse timestamp {timestamp}, error: {e}")
        
        return 1.0
    
    def fit(self, sequences: List[List[str]], smoothing: float = 1e-10, modifs: Optional[Dict] = None, 
            timestamps: Optional[List[List[str]]] = None) -> 'SparseMarkovChain':
        """
        Fit the Markov chain from sequence data.
        
        Args:
            sequences: List of sequences, where each sequence is a list of states
            smoothing: Small value to add to avoid zero probabilities (Laplace smoothing)
            modifs: Optional dictionary containing modifications like timeSeg for time-based weighting
                   Format: {
                       "timeSeg": [
                           {
                               "timeStart": "04:00",  # HH:MM format
                               "timeEnd": "09:00",    # HH:MM format
                               "factor": 1.1          # Weight multiplier
                           }
                       ]
                   }
            timestamps: Optional list of timestamp sequences corresponding to each state in sequences
            
        Returns:
            self for method chaining
        """
        print("Fitting Markov Chain...")
        
        # Parse time segments if provided
        time_segments = self._parse_time_segments(modifs)
        if time_segments:
            print(f"Time-based weighting enabled with {len(time_segments)} segments")
            for i, seg in enumerate(time_segments):
                start_hour = seg['start_minutes'] // 60
                start_min = seg['start_minutes'] % 60
                end_hour = seg['end_minutes'] // 60
                end_min = seg['end_minutes'] % 60
                print(f"  Segment {i+1}: {start_hour:02d}:{start_min:02d}-{end_hour:02d}:{end_min:02d} (factor: {seg['factor']})")
        
        # Build state mapping
        self._build_state_mapping(sequences)
        
        # Initialize sparse count matrix (lil_matrix for efficient construction)
        # Use float64 to handle weighted counts
        self.transition_counts = lil_matrix((self.n_states, self.n_states), dtype=np.float64)
        state_counts = Counter()
        
        total_transitions = 0
        processed_sequences = 0
        
        # Count transitions
        for seq_idx, sequence in enumerate(sequences):
            if len(sequence) < 2:
                continue
                
            # Convert states to indices
            idx_sequence = []
            for state in sequence:
                if state in self.state_to_idx:
                    idx_sequence.append(self.state_to_idx[state])
            
            if len(idx_sequence) < 2:
                continue
            
            # Get corresponding timestamps if provided
            timestamp_sequence = None
            if timestamps and seq_idx < len(timestamps):
                timestamp_sequence = timestamps[seq_idx]
            
            # Count state frequencies (with time weighting)
            for i, idx in enumerate(idx_sequence):
                weight = 1.0
                if timestamp_sequence and i < len(timestamp_sequence):
                    weight = self._get_time_weight(timestamp_sequence[i], time_segments)
                state_counts[idx] += weight
            
            # Count transitions (with time weighting)
            for i in range(len(idx_sequence) - 1):
                from_idx = idx_sequence[i]
                to_idx = idx_sequence[i + 1]
                
                # Get weight for the transition (use timestamp of the "to" state)
                weight = 1.0
                if timestamp_sequence and (i + 1) < len(timestamp_sequence):
                    weight = self._get_time_weight(timestamp_sequence[i + 1], time_segments)
                
                self.transition_counts[from_idx, to_idx] += weight
                total_transitions += weight
            
            processed_sequences += 1
            if processed_sequences % 10000 == 0:
                print(f"Processed {processed_sequences} sequences...")
        
        self.total_transitions = total_transitions
        self.state_frequencies = np.array([state_counts.get(i, 0) for i in range(self.n_states)])
        
        print(f"Processed {processed_sequences} sequences with {total_transitions} total transitions")
        print(f"Non-zero transitions: {self.transition_counts.nnz}")
        print(f"Sparsity: {1 - self.transition_counts.nnz / (self.n_states ** 2):.6f}")
        
        # Convert to probabilities
        self._compute_transition_probabilities(smoothing)
        
        self.is_fitted = True
        return self
    
    def _compute_transition_probabilities(self, smoothing: float) -> None:
        """Convert counts to probabilities with optional smoothing."""
        print("Computing transition probabilities...")
        
        # Convert to CSR for efficient row operations
        counts_csr = self.transition_counts.tocsr()
        
        # Calculate row sums (outgoing transition counts per state)
        row_sums = np.array(counts_csr.sum(axis=1)).flatten().astype(np.float64)
        
        # Handle states with no outgoing transitions
        zero_sum_states = np.where(row_sums == 0)[0]
        if len(zero_sum_states) > 0:
            print(f"Found {len(zero_sum_states)} states with no outgoing transitions")
            # Add uniform transitions for isolated states
            for state_idx in zero_sum_states:
                counts_csr[state_idx, :] = smoothing
            
            # Recalculate row sums
            row_sums = np.array(counts_csr.sum(axis=1)).flatten().astype(np.float64)
        
        # Apply smoothing if requested
        if smoothing > 0:
            counts_csr.data = counts_csr.data.astype(np.float64)
            counts_csr.data += smoothing
            row_sums += smoothing * self.n_states
        
        # Convert counts to probabilities
        # Using sparse matrix operations for efficiency
        self.transition_probs = counts_csr.copy().astype(np.float64)
        
        # Normalize rows to sum to 1
        for i in range(self.n_states):
            if row_sums[i] > 0:
                self.transition_probs.data[self.transition_probs.indptr[i]:self.transition_probs.indptr[i+1]] /= row_sums[i]
        
        print(f"Transition matrix: {self.n_states}x{self.n_states} with {self.transition_probs.nnz} non-zero elements")
    
    def predict_next(self, current_state: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Predict the most likely next states.
        
        Args:
            current_state: Current state
            top_k: Number of top predictions to return
            
        Returns:
            List of (state, probability) tuples sorted by probability (descending)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if current_state not in self.state_to_idx:
            raise ValueError(f"Unknown state: {current_state}")
        
        state_idx = self.state_to_idx[current_state]
        
        # Get transition probabilities for this state
        probs = self.transition_probs[state_idx, :].toarray().flatten()
        
        # Get top k predictions
        top_indices = np.argsort(probs)[-top_k:][::-1]
        
        predictions = []
        for idx in top_indices:
            if probs[idx] > 0:
                predictions.append((self.idx_to_state[idx], probs[idx]))
        
        return predictions
    
    def predict_sequence(self, start_state: str, length: int, method: str = 'sample') -> List[str]:
        """
        Generate a sequence of states starting from a given state.
        
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
            probs = self.transition_probs[current_idx, :].toarray().flatten()
            
            if method == 'sample':
                # Probabilistic sampling
                if np.sum(probs) == 0:
                    break
                probs = probs / np.sum(probs)  # Normalize
                next_idx = np.random.choice(self.n_states, p=probs)
            elif method == 'greedy':
                # Most likely next state
                next_idx = np.argmax(probs)
                if probs[next_idx] == 0:
                    break
            else:
                raise ValueError("Method must be 'sample' or 'greedy'")
            
            next_state = self.idx_to_state[next_idx]
            sequence.append(next_state)
            current_state = next_state
        
        return sequence

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the Markov chain.
        
        Returns:
            Dictionary with various statistics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting statistics")
        
        stats = {
            'n_states': self.n_states,
            'total_transitions': self.total_transitions,
            'non_zero_transitions': self.transition_probs.nnz,
            'sparsity': 1 - self.transition_probs.nnz / (self.n_states ** 2),
            'memory_usage_mb': self.get_memory_usage()
        }
        
        # Transition statistics
        row_sums = np.array(self.transition_probs.sum(axis=1)).flatten()
        stats['states_with_no_outgoing'] = int(np.sum(row_sums == 0))
        
        return stats
    
    def get_memory_usage(self) -> float:
        """
        Estimate memory usage in MB.
        
        Returns:
            Memory usage in megabytes
        """
        if not self.is_fitted:
            return 0.0
        
        # Sparse matrix memory usage
        matrix_memory = (
            self.transition_probs.data.nbytes +  # Data array
            self.transition_probs.indices.nbytes +  # Column indices
            self.transition_probs.indptr.nbytes  # Row pointers
        )
        
        # Other arrays
        other_memory = (
            self.state_frequencies.nbytes if self.state_frequencies is not None else 0
        )
        
        total_bytes = matrix_memory + other_memory
        return total_bytes / (1024 * 1024)  # Convert to MB
    
    def print_statistics(self) -> None:
        """Print comprehensive statistics about the Markov chain."""
        if not self.is_fitted:
            print("Model not fitted yet")
            return
        
        stats = self.get_statistics()
        
        print("=" * 50)
        print("SPARSE MARKOV CHAIN STATISTICS")
        print("=" * 50)
        print(f"States: {stats['n_states']:,}")
        print(f"Total transitions: {stats['total_transitions']:,}")
        print(f"Non-zero transitions: {stats['non_zero_transitions']:,}")
        print(f"Sparsity: {stats['sparsity']:.6f}")
        print(f"Memory usage: {stats['memory_usage_mb']:.2f} MB")
        print(f"States with no outgoing transitions: {stats['states_with_no_outgoing']}")
        
        # Memory comparison with dense matrix
        dense_memory_gb = (self.n_states ** 2 * 8) / (1024 ** 3)
        print(f"\nMemory efficiency:")
        print(f"  Sparse matrix: {stats['memory_usage_mb']:.2f} MB")
        print(f"  Dense matrix would be: {dense_memory_gb:.1f} GB")
        print(f"  Memory saved: {dense_memory_gb * 1024 - stats['memory_usage_mb']:.1f} MB")
        print("=" * 50)
    
    def save_model(self, filepath: str) -> None:
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        model_data = {
            'states': self.states,
            'state_to_idx': self.state_to_idx,
            'idx_to_state': self.idx_to_state,
            'n_states': self.n_states,
            'transition_probs': self.transition_probs,
            'total_transitions': self.total_transitions,
            'state_frequencies': self.state_frequencies,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> 'SparseMarkovChain':
        """
        Load a model from a file.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            self for method chaining
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.states = model_data['states']
        self.state_to_idx = model_data['state_to_idx']
        self.idx_to_state = model_data['idx_to_state']
        self.n_states = model_data['n_states']
        self.transition_probs = model_data['transition_probs']
        self.total_transitions = model_data['total_transitions']
        self.state_frequencies = model_data['state_frequencies']
        self.is_fitted = model_data['is_fitted']
        
        
        print(f"Model loaded from {filepath}")
        return self
    
    def validate_model(self) -> Dict[str, bool]:
        """
        Validate the model for correctness.
        
        Returns:
            Dictionary with validation results
        """
        if not self.is_fitted:
            return {'fitted': False}
        
        validation = {'fitted': True}
        
        # Check if transition matrix rows sum to 1
        row_sums = np.array(self.transition_probs.sum(axis=1)).flatten()
        validation['rows_sum_to_one'] = np.allclose(row_sums, 1.0, atol=1e-10)
        
        # Check if all probabilities are non-negative
        validation['non_negative_probs'] = np.all(self.transition_probs.data >= 0)
        
        # Check if all probabilities are <= 1
        validation['probs_le_one'] = np.all(self.transition_probs.data <= 1.0)
        
        # Check state consistency
        validation['state_consistency'] = (
            len(self.states) == self.n_states and
            len(self.state_to_idx) == self.n_states and
            len(self.idx_to_state) == self.n_states
        )
        
        return validation