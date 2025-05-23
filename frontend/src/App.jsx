// frontend/src/App.jsx
import React, { useState, useEffect } from 'react'; // Added useEffect
import FileUploader from './components/FileUploader';
import MarkdownDisplay from './components/MarkdownDisplay';
import Search from './components/Search'; 
import Chat from './components/Chat'; 

function App() {
  const [uploadedDocData, setUploadedDocData] = useState(null);
  const [embeddingStatus, setEmbeddingStatus] = useState(null); 
  const [selectedChunks, setSelectedChunks] = useState({}); // { [index]: chunkData }

  const handleToggleChunkSelection = (resultIndex, chunkData) => {
    setSelectedChunks(prev => {
      const newSelectedChunks = {...prev};
      if (newSelectedChunks[resultIndex]) {
        delete newSelectedChunks[resultIndex];
      } else {
        newSelectedChunks[resultIndex] = chunkData;
      }
      console.log("Selected chunks in App:", newSelectedChunks); // For debugging
      return newSelectedChunks;
    });
  };

  const handleClearSelectedChunks = () => {
    setSelectedChunks({});
    console.log("Cleared selected chunks in App"); // For debugging
  };
  
  const handleUploadSuccess = (data) => {
    setUploadedDocData(data);
    setEmbeddingStatus(null);
    setSelectedChunks({}); // Clear selections on new upload
  };

  const handleNewEmbeddingsStatus = (statusData) => {
    if (statusData && statusData.error) {
        setEmbeddingStatus(`Embedding issue: ${statusData.error}`);
    } else if (statusData && statusData.message) { 
        setEmbeddingStatus(statusData.message);
    } else if (statusData && typeof statusData.successful_embeddings_count === 'number') {
        setEmbeddingStatus(`Embeddings processed: ${statusData.successful_embeddings_count} of ${statusData.total_chunks_processed} chunks.`);
    } else if (statusData) { // Check if statusData itself is not null/undefined
        setEmbeddingStatus("Embedding status received with unknown format."); 
    } // If statusData is null/undefined, do nothing, keep previous status or null
  };
  
  // Effect to clear selected chunks if the document ID changes (e.g., new file uploaded)
  // This is somewhat redundant with handleUploadSuccess but good for robustness
  useEffect(() => {
    setSelectedChunks({});
  }, [uploadedDocData?.doc_id]);


  return (
    <div className="flex h-screen bg-gray-100">
      {/* Sidebar */}
      <aside className="w-1/4 bg-gray-800 text-white p-4 space-y-4 flex flex-col">
        <h1 className="text-2xl font-semibold">Clinical Assistant</h1>
        
        <FileUploader onUploadSuccess={handleUploadSuccess} />
        
        <div className="bg-gray-700 p-3 rounded-lg mt-4 flex-grow">
          <h2 className="text-lg font-semibold mb-2 text-white">Processed Files</h2>
          {uploadedDocData ? (
            <div>
              <p className="text-sm text-green-300">Last upload: {uploadedDocData.filename}</p>
              <p className="text-xs text-gray-400">Doc ID: {uploadedDocData.doc_id}</p>
            </div>
          ) : (
            <p className="text-sm">History List Placeholder</p>
          )}
          {embeddingStatus && <p className="text-xs text-amber-300 mt-1">{embeddingStatus}</p>}
           {/* Display selected chunk count from App state for debugging/verification */}
           {Object.keys(selectedChunks).length > 0 && (
            <p className="text-xs text-cyan-300 mt-1">
              App State: {Object.keys(selectedChunks).length} chunk(s) selected.
            </p>
          )}
        </div>
      </aside>

      {/* Content Area */}
      <main className="flex-1 p-6 space-y-6 overflow-y-auto">
        <section className="bg-white p-4 rounded-lg shadow">
          <h2 className="text-xl font-semibold mb-2">Document View</h2>
          <MarkdownDisplay markdownText={uploadedDocData ? uploadedDocData.markdown_text : ''} />
        </section>

        <section className="bg-white p-4 rounded-lg shadow">
          <h2 className="text-xl font-semibold mb-2">Search & Chat</h2>
          <Search 
            docId={uploadedDocData ? uploadedDocData.doc_id : null}
            onNewEmbeddingsStatus={handleNewEmbeddingsStatus}
            selectedChunks={selectedChunks} // Pass state down
            onToggleChunkSelection={handleToggleChunkSelection} // Pass handler down
            onClearSelectedChunks={handleClearSelectedChunks} // Pass handler down
            // Prop to notify App.jsx when search results are updated, so it can clear selectedChunks
            onSearchResultsUpdated={() => setSelectedChunks({})} 
          />
          <div className="border-t pt-4 mt-6">
            <Chat 
              docId={uploadedDocData ? uploadedDocData.doc_id : null} 
              selectedChunks={selectedChunks} // Pass selectedChunks to Chat
            />
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
