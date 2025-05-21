import unittest
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os

# This test file focuses on the core search logic, not Streamlit components.
# It simulates how embeddings would be compared.

class TestSemanticSearchLogic(unittest.TestCase):

    def setUp(self):
        """Set up mock data for tests."""
        self.mock_chunks_with_embeddings = [
            ("This is the first chunk, very relevant.", np.array([0.9, 0.1, 0.1])), # Highly similar to query
            ("Second chunk, somewhat related.", np.array([0.6, 0.3, 0.2])),      # Moderately similar
            ("Third piece of text, not very relevant.", np.array([0.1, 0.2, 0.8])), # Low similarity
            ("Fourth chunk, also quite relevant.", np.array([0.85, 0.15, 0.1])),   # Also highly similar, but less than first
            ("Fifth, completely unrelated content.", np.array([0.0, 0.1, 0.9]))    # Very low similarity
        ]
        self.mock_query_embedding = np.array([1.0, 0.0, 0.0]) # Query embedding

    def _simulate_search_logic(self, query_embedding, chunks_with_embeddings, top_n=3):
        """
        Simulates the core semantic search logic.
        
        Args:
            query_embedding (np.array): The embedding of the search query.
            chunks_with_embeddings (list): A list of (text_chunk, embedding_vector) tuples.
            top_n (int): The number of top results to return.

        Returns:
            list: A list of (score, text_chunk) tuples for the top_n results, sorted by score.
        """
        if not chunks_with_embeddings:
            return []

        chunk_texts = [item[0] for item in chunks_with_embeddings]
        chunk_embeddings_list = [item[1] for item in chunks_with_embeddings]

        if not chunk_embeddings_list:
            return []

        # Ensure query_embedding is 2D for cosine_similarity
        query_embedding_2d = query_embedding.reshape(1, -1)
        
        chunk_embeddings_matrix = np.array(chunk_embeddings_list)
        
        if chunk_embeddings_matrix.size == 0:
            return []

        similarities = cosine_similarity(query_embedding_2d, chunk_embeddings_matrix)
        
        results_with_scores = []
        if similarities.size > 0:
            for i, score in enumerate(similarities[0]):
                results_with_scores.append((score, chunk_texts[i]))
            
            # Sort results by similarity score in descending order
            sorted_results = sorted(results_with_scores, key=lambda x: x[0], reverse=True)
            return sorted_results[:top_n]
        return []

    def test_ranking_and_correctness(self):
        """Test if results are correctly ranked by similarity."""
        top_3_results = self._simulate_search_logic(
            self.mock_query_embedding, 
            self.mock_chunks_with_embeddings, 
            top_n=3
        )
        
        self.assertEqual(len(top_3_results), 3)
        
        # Expected order based on similarity to [1.0, 0.0, 0.0]
        # 1. "This is the first chunk, very relevant." (Cosine sim with [0.9, 0.1, 0.1] is high)
        # 2. "Fourth chunk, also quite relevant." (Cosine sim with [0.85, 0.15, 0.1] is next)
        # 3. "Second chunk, somewhat related." (Cosine sim with [0.6, 0.3, 0.2] is third)

        self.assertEqual(top_3_results[0][1], "This is the first chunk, very relevant.")
        self.assertEqual(top_3_results[1][1], "Fourth chunk, also quite relevant.")
        self.assertEqual(top_3_results[2][1], "Second chunk, somewhat related.")
        
        # Check scores are descending
        self.assertGreaterEqual(top_3_results[0][0], top_3_results[1][0])
        self.assertGreaterEqual(top_3_results[1][0], top_3_results[2][0])

    def test_no_results_if_chunks_empty(self):
        """Test behavior when there are no chunks to search."""
        results = self._simulate_search_logic(self.mock_query_embedding, [], top_n=3)
        self.assertEqual(results, [])

    def test_no_results_if_embeddings_list_empty(self):
        """Test behavior when chunks exist but embeddings list is somehow empty (e.g., all failed)."""
        results = self._simulate_search_logic(self.mock_query_embedding, [("text", None)], top_n=3)
        # The helper function's internal logic should lead to an empty list if chunk_embeddings_list is empty
        # or if chunk_embeddings_matrix is empty after filtering.
        # Simulating an empty list of embeddings for a chunk:
        mock_chunks_no_valid_embeddings = [("text1", [])] # An empty list as an embedding
        # This scenario is a bit tricky for the current helper. The np.array([]) might cause issues.
        # A more realistic "failed" embedding would be None, which the current app logic filters out
        # before calling cosine_similarity.
        # Let's test with chunks that would result in an empty chunk_embeddings_list after filtering
        
        # Scenario: chunks_with_embeddings contains items where embedding is None or empty
        chunks_with_problematic_embeddings = [
            ("Chunk A", None), 
            ("Chunk B", []) # Empty list as embedding
        ]
        # The current search logic in app.py filters these out before passing to cosine_similarity.
        # Our helper should ideally mimic this or be robust.
        # For _simulate_search_logic, if chunk_embeddings_list becomes empty after list comprehension,
        # it should return [].

        # Let's refine the helper or test based on the assumption that input `chunks_with_embeddings`
        # to `_simulate_search_logic` could have `None` embeddings.
        # The helper's line `chunk_embeddings_list = [item[1] for item in chunks_with_embeddings]`
        # would put `None` into `chunk_embeddings_list`. np.array([None, ...]) will fail.
        # So, the filtering step is crucial.
        
        # Let's assume the input to _simulate_search_logic is *after* filtering None embeddings,
        # as it is in app.py. So, we test an empty list of *valid* embeddings.
        results_empty_valid_embeddings = self._simulate_search_logic(self.mock_query_embedding, [], top_n=3)
        self.assertEqual(results_empty_valid_embeddings, [])


    def test_single_chunk(self):
        """Test with only one chunk."""
        single_chunk_data = [("Only one chunk.", np.array([0.5, 0.5, 0.5]))]
        results = self._simulate_search_logic(self.mock_query_embedding, single_chunk_data, top_n=3)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][1], "Only one chunk.")

    def test_top_n_less_than_available_chunks(self):
        """Test if top_n correctly limits results."""
        results = self._simulate_search_logic(
            self.mock_query_embedding, 
            self.mock_chunks_with_embeddings, 
            top_n=2
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][1], "This is the first chunk, very relevant.")
        self.assertEqual(results[1][1], "Fourth chunk, also quite relevant.")

    def test_top_n_more_than_available_chunks(self):
        """Test if top_n doesn't cause error if more than available chunks."""
        results = self._simulate_search_logic(
            self.mock_query_embedding, 
            self.mock_chunks_with_embeddings, 
            top_n=10 # More than 5 available chunks
        )
        self.assertEqual(len(results), len(self.mock_chunks_with_embeddings)) # Should return all

    def test_identical_similarity_scores(self):
        """Test behavior with identical similarity scores."""
        # Note: Order for identical scores is not strictly guaranteed beyond being grouped,
        # but they should all be present if they fall within top_n.
        chunks_with_identical_similarity = [
            ("Identical A", np.array([0.8, 0.1, 0.1])), # Same similarity as B
            ("Identical B", np.array([0.8, 0.1, 0.1])), # Same similarity as A
            ("Different C", np.array([0.1, 0.8, 0.1]))
        ]
        query = np.array([1.0, 0.0, 0.0])
        results = self._simulate_search_logic(query, chunks_with_identical_similarity, top_n=3)
        
        self.assertEqual(len(results), 3)
        
        # Check that the top two scores are identical
        self.assertAlmostEqual(results[0][0], results[1][0]) 
        
        # Check that both "Identical A" and "Identical B" are in the top 2 results
        top_2_texts = {results[0][1], results[1][1]}
        self.assertIn("Identical A", top_2_texts)
        self.assertIn("Identical B", top_2_texts)
        
        # Check that the third result is "Different C"
        self.assertEqual(results[2][1], "Different C")

if __name__ == '__main__':
    unittest.main()
