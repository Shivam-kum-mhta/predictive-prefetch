# Sparse Markov Chain for v2 with Semantic Similarity Smoothing

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import eigs
from collections import defaultdict, Counter
import pickle
import warnings
import time
from typing import List, Dict, Tuple, Optional, Union, Any

class SparseMarkovChain:
    """
    A memory-efficient Markov Chain implementation using sparse matrices.
    Includes optional semantic similarity smoothing for article recommendations.
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
    
    def fit(self, sequences: List[List[str]], smoothing: float = 1e-10) -> 'SparseMarkovChain':
        """
        Fit the Markov chain from sequence data.
        
        Args:
            sequences: List of sequences, where each sequence is a list of states
            smoothing: Small value to add to avoid zero probabilities (Laplace smoothing)
            
        Returns:
            self for method chaining
        """
        print("Fitting Markov Chain...")
        
        # Build state mapping
        self._build_state_mapping(sequences)
        
        # Initialize sparse count matrix (lil_matrix for efficient construction)
        self.transition_counts = lil_matrix((self.n_states, self.n_states), dtype=np.float64)
        state_counts = Counter()
        
        total_transitions = 0
        processed_sequences = 0
        
        # Count transitions
        for sequence in sequences:
            if len(sequence) < 2:
                continue
                
            # Convert states to indices
            idx_sequence = []
            for state in sequence:
                if state in self.state_to_idx:
                    idx_sequence.append(self.state_to_idx[state])
            
            if len(idx_sequence) < 2:
                continue
            
            # Count state frequencies
            for idx in idx_sequence:
                state_counts[idx] += 1
            
            # Count transitions
            for i in range(len(idx_sequence) - 1):
                from_idx = idx_sequence[i]
                to_idx = idx_sequence[i + 1]
                self.transition_counts[from_idx, to_idx] += 1
                total_transitions += 1
            
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
        self.transition_probs = counts_csr.copy().astype(np.float64)
        
        # Normalize rows to sum to 1
        for i in range(self.n_states):
            if row_sums[i] > 0:
                self.transition_probs.data[self.transition_probs.indptr[i]:self.transition_probs.indptr[i+1]] /= row_sums[i]
        
        print(f"Transition matrix: {self.n_states}x{self.n_states} with {self.transition_probs.nnz} non-zero elements")
    
    def apply_semantic_smoothing(self, embeddings_module, similar_articles_n: int = 10, spread_fraction: float = 0.15):
        """
        Apply semantic similarity smoothing to the transition probabilities.
        
        For each article A, find top-N semantically similar articles (A₁, A₂, ...).
        For each outgoing transition from A (say A → B with probability p),
        spread a fraction of p to transitions from similar neighbors (A₁, A₂, ...) to B.
        
        Args:
            embeddings_module: Module with RetrieveSimilarTitlesById() method
            similar_articles_n: Number of similar articles to consider (default: 10)
            spread_fraction: Fraction of probability to spread (default: 0.15 = 15%)
        
        Returns:
            self for method chaining
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before applying semantic smoothing")
        
        print(f"\nApplying semantic similarity smoothing...")
        print(f"  Total states to process: {len(self.states):,}")
        print(f"  Similar articles per state: {similar_articles_n}")
        print(f"  Spread fraction: {spread_fraction}")
        print(f"  Estimated time: 1-2 hours (processing ~{len(self.states):,} articles)")
        
        # Convert to LIL for efficient element-wise modification
        print("\nConverting to LIL format for efficient updates...")
        smoothed_probs = self.transition_probs.tolil()
        print("  Conversion complete!")
        
        total_states = len(self.states)
        processed = 0
        failed_lookups = 0
        start_time = time.time()
        
        for state_idx, article_id in enumerate(self.states):
            if (processed + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                rate = (processed + 1) / elapsed if elapsed > 0 else 0
                remaining = (total_states - processed - 1) / rate if rate > 0 else 0
                print(f"  Progress: {processed + 1:,}/{total_states:,} ({100*(processed+1)/total_states:.1f}%) "
                      f"| Failed: {failed_lookups} | "
                      f"Rate: {rate:.1f} states/sec | "
                      f"ETA: {remaining/60:.1f} min")
            
            # Get similar articles
            try:
                similar_articles = embeddings_module.RetrieveSimilarTitlesById(article_id, k=similar_articles_n)
                if similar_articles is None or len(similar_articles['ids']) == 0:
                    failed_lookups += 1
                    processed += 1
                    continue
                
                similar_ids = similar_articles['ids']
                
                # Get outgoing transitions from current article
                row = smoothed_probs[state_idx, :].toarray().flatten()
                non_zero_cols = np.nonzero(row)[0]
                
                # For each transition from current article
                for col_idx in non_zero_cols:
                    prob = row[col_idx]
                    
                    # Amount to spread
                    spread_amount = prob * spread_fraction
                    
                    # Reduce original transition probability
                    smoothed_probs[state_idx, col_idx] -= spread_amount
                    
                    # Distribute spread_amount equally among similar articles' transitions to the same target
                    if len(similar_ids) > 0:
                        amount_per_similar = spread_amount / len(similar_ids)
                        
                        for similar_id in similar_ids:
                            if similar_id in self.state_to_idx:
                                similar_idx = self.state_to_idx[similar_id]
                                # Add to similar article's transition to same target
                                smoothed_probs[similar_idx, col_idx] += amount_per_similar
                
            except Exception as e:
                failed_lookups += 1
                # Skip if embedding lookup fails
                continue
            
            processed += 1
        
        print(f"Completed processing. Failed lookups: {failed_lookups}/{total_states}")
        
        # Re-normalize all rows to sum to 1
        print("Re-normalizing probabilities...")
        row_sums = np.array(smoothed_probs.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        
        # Convert back to CSR for efficient normalization
        smoothed_probs = smoothed_probs.tocsr()
        
        # Normalize each row
        for i in range(self.n_states):
            start_idx = smoothed_probs.indptr[i]
            end_idx = smoothed_probs.indptr[i + 1]
            if end_idx > start_idx:
                smoothed_probs.data[start_idx:end_idx] /= row_sums[i]
        
        self.transition_probs = smoothed_probs
        print("Semantic smoothing complete")
        
        return self
    
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
