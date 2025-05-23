// frontend/src/components/Search.jsx
import React, { useState, useEffect } from 'react';
import axios from 'axios';

const Search = ({ 
  docId, 
  onNewEmbeddingsStatus,
  selectedChunks, // Received from App.jsx
  onToggleChunkSelection, // Received from App.jsx
  onClearSelectedChunks, // Received from App.jsx
  onSearchResultsUpdated // Received from App.jsx
}) => {
  const [query, setQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [hasAttemptedEmbed, setHasAttemptedEmbed] = useState(false);

  // Effect to clear local states when docId changes
  useEffect(() => {
    setQuery('');
    setSearchResults([]);
    setError('');
    setHasAttemptedEmbed(false);
    // Selected chunks are cleared by App.jsx when docId changes or search results update
    if (onSearchResultsUpdated) { // Notify App if it needs to clear its own selectedChunks
        onSearchResultsUpdated(); // This effectively clears App's selectedChunks for new searches
    }
  }, [docId]);


  const handleSearch = async () => {
    if (!docId) {
      setError('No document processed or selected.');
      return;
    }
    if (!query.trim()) {
      setError('Please enter a search query.');
      return;
    }

    setIsLoading(true);
    setError('');
    setSearchResults([]); // Clear previous results
    if (onSearchResultsUpdated) { // Notify App to clear its selectedChunks
        onSearchResultsUpdated();
    }


    try {
      if (!hasAttemptedEmbed) {
        try {
          console.log(`Attempting to ensure embeddings for doc_id: ${docId}`);
          const embedResponse = await axios.post('http://localhost:5001/api/embed', { doc_id: docId });
          console.log('Embed API response:', embedResponse.data);
          setHasAttemptedEmbed(true); 
          if (onNewEmbeddingsStatus) { 
            onNewEmbeddingsStatus(embedResponse.data);
          }
        } catch (embedError) {
          console.error('Embedding error:', embedError);
          let embedErrorMsg = 'Failed to ensure document embeddings.';
          if (embedError.response && embedError.response.data && embedError.response.data.error) {
            embedErrorMsg = embedError.response.data.error;
          }
          setError(`Warning: ${embedErrorMsg} Search might be incomplete.`);
          setHasAttemptedEmbed(true); 
        }
      }

      console.log(`Searching for query "${query}" in doc_id: ${docId}`);
      const searchResponse = await axios.post('http://localhost:5001/api/search', {
        doc_id: docId,
        query: query,
      });
      console.log('Search API response:', searchResponse.data);
      setSearchResults(searchResponse.data.search_results || []);
      if ((searchResponse.data.search_results || []).length === 0) {
        setError('No relevant chunks found for your query.');
      }

    } catch (err) {
      console.error('Search error:', err);
      let errorMsg = 'Search failed. Please try again.';
      if (err.response && err.response.data && err.response.data.error) {
        errorMsg = err.response.data.error;
      }
      setError(errorMsg);
      setSearchResults([]);
    } finally {
      setIsLoading(false);
    }
  };

  // toggleChunkSelection is now handled by App.jsx via props.onToggleChunkSelection
  // getSelectedChunkTexts is also removed as App.jsx owns the state

  return (
    <div className="space-y-4">
      <div>
        <label htmlFor="search-query" className="block text-sm font-medium text-gray-700">
          Search Query:
        </label>
        <div className="mt-1 flex rounded-md shadow-sm">
          <input
            type="text"
            id="search-query"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="focus:ring-indigo-500 focus:border-indigo-500 block w-full px-3 py-2 sm:text-sm border-gray-300 rounded-l-md"
            placeholder="Enter search query..."
            disabled={!docId}
          />
          <button
            onClick={handleSearch}
            disabled={isLoading || !docId}
            className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-r-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:bg-gray-400"
          >
            {isLoading ? (hasAttemptedEmbed ? 'Searching...' : 'Embedding & Searching...') : 'Search'}
          </button>
        </div>
        {!docId && <p className="text-xs text-gray-500 mt-1">Upload a document to enable search.</p>}
      </div>

      {error && <p className="text-red-500 text-sm">{error}</p>}

      {/* Display selected chunk count from App's state via props */}
      {Object.keys(selectedChunks).length > 0 && (
        <div className="p-2 bg-indigo-100 rounded-md flex justify-between items-center">
          <p className="text-sm text-indigo-700">
            {Object.keys(selectedChunks).length} chunk(s) selected.
          </p>
          <button 
            onClick={onClearSelectedChunks} // Call prop from App.jsx
            className="text-xs text-indigo-600 hover:text-indigo-800 underline"
          >
            Clear Selections
          </button>
        </div>
      )}

      {searchResults.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-lg font-semibold">Search Results:</h3>
          {searchResults.map((result, index) => (
            <div 
              key={index} 
              className={`p-3 rounded-md shadow-sm flex items-start gap-x-3 ${selectedChunks[index] ? 'bg-blue-100 border border-blue-300' : 'bg-gray-50'}`}
            >
              <input
                type="checkbox"
                id={`chunk-select-${index}`}
                checked={!!selectedChunks[index]} // Use prop from App.jsx
                onChange={() => onToggleChunkSelection(index, result)} // Call prop from App.jsx
                className="mt-1 h-4 w-4 text-indigo-600 border-gray-300 rounded focus:ring-indigo-500"
              />
              <div className="flex-1">
                <label htmlFor={`chunk-select-${index}`} className="cursor-pointer w-full block">
                  <p className="text-xs text-gray-500">
                    Similarity: {result.score.toFixed(4)} | Metadata: {result.metadata || 'N/A'}
                  </p>
                  <p className="text-sm text-gray-800">{result.chunk}</p>
                </label>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default Search;
