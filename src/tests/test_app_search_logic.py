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
            ("This is the first chunk, very relevant.", np.array([0.9, 0.1, 0.1]), "Metadata for chunk 1: Page 1, Signer A"),
            ("Second chunk, somewhat related.", np.array([0.6, 0.3, 0.2]), "Metadata for chunk 2: Page 1, Signer A"),
            ("Third piece of text, not very relevant.", np.array([0.1, 0.2, 0.8]), "Metadata for chunk 3: Page 2, Signer B"),
            ("Fourth chunk, also quite relevant.", np.array([0.85, 0.15, 0.1]), "Metadata for chunk 4: Page 3, Signer C"),
            ("Fifth, completely unrelated content.", np.array([0.0, 0.1, 0.9]), "Metadata for chunk 5: Page 4, Signer D")
        ]
        self.mock_query_embedding = np.array([1.0, 0.0, 0.0]) # Query embedding

    def _simulate_search_logic(self, query_embedding, chunks_with_embeddings_and_metadata, top_n=3):
        """
        Simulates the core semantic search logic.
        
        Args:
            query_embedding (np.array): The embedding of the search query.
            chunks_with_embeddings_and_metadata (list): A list of 
                                                        (text_chunk, embedding_vector, metadata_string) tuples.
            top_n (int): The number of top results to return.

        Returns:
            list: A list of (score, text_chunk, metadata_string) tuples for the top_n results, sorted by score.
        """
        if not chunks_with_embeddings_and_metadata:
            return []

        # Filter out items that don't conform to the expected 3-tuple structure or have invalid embeddings
        valid_items = [
            item for item in chunks_with_embeddings_and_metadata 
            if len(item) == 3 and isinstance(item[1], np.ndarray) and item[1].size > 0
        ]
        
        if not valid_items:
            return []

        chunk_texts = [item[0] for item in valid_items]
        chunk_embeddings_list = [item[1] for item in valid_items]
        chunk_metadata_list = [item[2] for item in valid_items]


        # Ensure query_embedding is 2D for cosine_similarity
        query_embedding_2d = query_embedding.reshape(1, -1)
        
        chunk_embeddings_matrix = np.array(chunk_embeddings_list)
        
        if chunk_embeddings_matrix.ndim == 1: # Handle case of single embedding in list
            chunk_embeddings_matrix = chunk_embeddings_matrix.reshape(1, -1)
            if query_embedding_2d.shape[1] != chunk_embeddings_matrix.shape[1]:
                 # This case should ideally not happen if embeddings are consistently sized
                return []


        if chunk_embeddings_matrix.size == 0:
            return []

        similarities = cosine_similarity(query_embedding_2d, chunk_embeddings_matrix)
        
        results_with_scores_and_metadata = []
        if similarities.size > 0:
            for i, score in enumerate(similarities[0]):
                results_with_scores_and_metadata.append((score, chunk_texts[i], chunk_metadata_list[i]))
            
            # Sort results by similarity score in descending order
            sorted_results = sorted(results_with_scores_and_metadata, key=lambda x: x[0], reverse=True)
            return sorted_results[:top_n]
        return []

    def test_ranking_correctness_and_metadata_passthrough(self):
        """Test if results are correctly ranked and metadata is passed through."""
        top_3_results = self._simulate_search_logic(
            self.mock_query_embedding, 
            self.mock_chunks_with_embeddings, 
            top_n=3
        )
        
        self.assertEqual(len(top_3_results), 3)
        
        # Expected order based on similarity to [1.0, 0.0, 0.0]
        # 1. "This is the first chunk, very relevant."
        # 2. "Fourth chunk, also quite relevant."
        # 3. "Second chunk, somewhat related."

        self.assertEqual(top_3_results[0][1], "This is the first chunk, very relevant.")
        self.assertEqual(top_3_results[0][2], "Metadata for chunk 1: Page 1, Signer A") # Check metadata

        self.assertEqual(top_3_results[1][1], "Fourth chunk, also quite relevant.")
        self.assertEqual(top_3_results[1][2], "Metadata for chunk 4: Page 3, Signer C") # Check metadata

        self.assertEqual(top_3_results[2][1], "Second chunk, somewhat related.")
        self.assertEqual(top_3_results[2][2], "Metadata for chunk 2: Page 1, Signer A") # Check metadata
        
        # Check scores are descending
        self.assertGreaterEqual(top_3_results[0][0], top_3_results[1][0])
        self.assertGreaterEqual(top_3_results[1][0], top_3_results[2][0])


    def test_no_results_if_chunks_empty(self): # Name remains same, underlying data structure handled by helper
        """Test behavior when there are no chunks to search."""
        results = self._simulate_search_logic(self.mock_query_embedding, [], top_n=3)
        self.assertEqual(results, [])

    def test_no_results_if_embeddings_list_empty(self): # Name remains same
        """Test behavior when chunks exist but embeddings list is somehow empty or invalid."""
        # Test with items that would be filtered out by the helper's validation
        chunks_with_invalid_embeddings = [
            ("Chunk A", None, "Meta A"), 
            ("Chunk B", np.array([]), "Meta B"), # Empty embedding array
            ("Chunk C", "not an array", "Meta C"), # Invalid embedding type
            ("Chunk D", np.array([1,2,3]), "Meta D") # Valid one to ensure filter works
        ]
        # The helper should filter these out, and if only invalid ones are present, return empty.
        results_all_invalid = self._simulate_search_logic(
            self.mock_query_embedding, 
            chunks_with_invalid_embeddings[:-1], # Pass only invalid ones
            top_n=3
        )
        self.assertEqual(results_all_invalid, [])

        # Test that valid items are still processed if mixed
        results_mixed_validity = self._simulate_search_logic(
            self.mock_query_embedding,
            chunks_with_invalid_embeddings, # Pass all, including the valid one
            top_n=1
        )
        self.assertEqual(len(results_mixed_validity), 1)
        self.assertEqual(results_mixed_validity[0][1], "Chunk D")


    def test_single_chunk(self): # Name remains same
        """Test with only one chunk."""
        single_chunk_data = [("Only one chunk.", np.array([0.5, 0.5, 0.5]), "Meta for single chunk")]
        results = self._simulate_search_logic(self.mock_query_embedding, single_chunk_data, top_n=3)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][1], "Only one chunk.")
        self.assertEqual(results[0][2], "Meta for single chunk")


    def test_top_n_less_than_available_chunks(self): # Name remains same
        """Test if top_n correctly limits results."""
        results = self._simulate_search_logic(
            self.mock_query_embedding, 
            self.mock_chunks_with_embeddings, 
            top_n=2
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][1], "This is the first chunk, very relevant.")
        self.assertEqual(results[0][2], "Metadata for chunk 1: Page 1, Signer A")
        self.assertEqual(results[1][1], "Fourth chunk, also quite relevant.")
        self.assertEqual(results[1][2], "Metadata for chunk 4: Page 3, Signer C")


    def test_top_n_more_than_available_chunks(self): # Name remains same
        """Test if top_n doesn't cause error if more than available chunks."""
        results = self._simulate_search_logic(
            self.mock_query_embedding, 
            self.mock_chunks_with_embeddings, 
            top_n=10 # More than 5 available chunks
        )
        self.assertEqual(len(results), len(self.mock_chunks_with_embeddings)) # Should return all


    def test_identical_similarity_scores(self): # Name remains same
        """Test behavior with identical similarity scores."""
        chunks_with_identical_similarity = [
            ("Identical A", np.array([0.8, 0.1, 0.1]), "Meta Identical A"),
            ("Identical B", np.array([0.8, 0.1, 0.1]), "Meta Identical B"),
            ("Different C", np.array([0.1, 0.8, 0.1]), "Meta Different C")
        ]
        query = np.array([1.0, 0.0, 0.0])
        results = self._simulate_search_logic(query, chunks_with_identical_similarity, top_n=3)
        
        self.assertEqual(len(results), 3)
        self.assertAlmostEqual(results[0][0], results[1][0]) 
        
        top_2_texts_and_meta = {(res[1], res[2]) for res in results[:2]}
        self.assertIn(("Identical A", "Meta Identical A"), top_2_texts_and_meta)
        self.assertIn(("Identical B", "Meta Identical B"), top_2_texts_and_meta)
        
        self.assertEqual(results[2][1], "Different C")
        self.assertEqual(results[2][2], "Meta Different C")

if __name__ == '__main__':
    unittest.main()
