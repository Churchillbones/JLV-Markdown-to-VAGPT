// frontend/src/components/Chat.jsx
import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';

const Chat = ({ docId, selectedChunks }) => { // selectedChunks received as prop
  const [currentQuestion, setCurrentQuestion] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const chatBodyRef = useRef(null);

  // Scroll to bottom of chat history when it updates
  useEffect(() => {
    if (chatBodyRef.current) {
      chatBodyRef.current.scrollTop = chatBodyRef.current.scrollHeight;
    }
  }, [chatHistory]);

  // Clear chat history if docId changes or selectedChunks are cleared (empty object)
  useEffect(() => {
    setChatHistory([]);
    setError(''); // Also clear any errors from previous context
  }, [docId, selectedChunks]);


  const handleQuestionSubmit = async (e) => {
    e.preventDefault();
    if (!currentQuestion.trim()) return;

    const newQuestion = { type: 'user', message: currentQuestion };
    setChatHistory(prev => [...prev, newQuestion]);
    setCurrentQuestion('');
    setIsLoading(true);
    setError('');

    try {
      const payload = { question: newQuestion.message };
      if (docId) {
        payload.doc_id = docId; // Always send doc_id if available
      }

      // Check for selected chunks and add their text to the payload
      const selectedChunkTexts = selectedChunks && Object.keys(selectedChunks).length > 0 
        ? Object.values(selectedChunks).map(chunkData => chunkData.chunk) 
        : [];

      if (selectedChunkTexts.length > 0) {
        payload.context_chunks = selectedChunkTexts;
        console.log("Chat.jsx: Sending selected chunks to chat API:", selectedChunkTexts);
      } else {
        // If docId is present but no specific chunks are selected, 
        // backend will use full doc context (or general if no docId).
        console.log("Chat.jsx: No specific chunks selected. Using general context (doc_id if available).");
      }

      const response = await axios.post('http://localhost:5001/api/chat', payload);
      const aiResponse = { type: 'ai', message: response.data.answer };
      setChatHistory(prev => [...prev, aiResponse]);

    } catch (err) {
      console.error('Chat API error:', err);
      let errorMsg = 'Failed to get AI response.';
      if (err.response && err.response.data && err.response.data.error) {
        errorMsg = err.response.data.error;
      }
      setError(errorMsg);
      // Optionally add error to chat history:
      // setChatHistory(prev => [...prev, { type: 'ai', message: `Error: ${errorMsg}` }]);
    } finally {
      setIsLoading(false);
    }
  };
  
  const inputPlaceholder = docId 
    ? (selectedChunks && Object.keys(selectedChunks).length > 0 
        ? "Ask about selected chunks..." 
        : "Ask about the document...")
    : "Ask a general question...";

  return (
    <div className="flex flex-col h-full max-h-[400px]"> {/* Added max-h for example */}
      <h3 className="text-lg font-semibold mb-2">Assistant Chat</h3>
      <div ref={chatBodyRef} className="flex-grow bg-gray-50 p-3 rounded-md overflow-y-auto mb-3 space-y-2 border">
        {chatHistory.length === 0 && !isLoading && (
          <p className="text-sm text-gray-400">{inputPlaceholder}</p>
        )}
        {chatHistory.map((entry, index) => (
          <div key={index} className={`flex ${entry.type === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[70%] p-2 rounded-lg ${entry.type === 'user' ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-800'}`}>
              <p className="text-sm">{entry.message}</p>
            </div>
          </div>
        ))}
        {isLoading && chatHistory.length > 0 && chatHistory[chatHistory.length-1].type === 'user' && (
          <div className="flex justify-start">
            <div className="max-w-[70%] p-2 rounded-lg bg-gray-200 text-gray-800 animate-pulse">
              <p className="text-sm">Thinking...</p>
            </div>
          </div>
        )}
      </div>
      {error && <p className="text-red-500 text-sm mb-2">{error}</p>}
      <form onSubmit={handleQuestionSubmit} className="flex">
        <input
          type="text"
          value={currentQuestion}
          onChange={(e) => setCurrentQuestion(e.target.value)}
          className="flex-grow px-3 py-2 border border-gray-300 rounded-l-md focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
          placeholder={inputPlaceholder} // Dynamic placeholder
        />
        <button
          type="submit"
          disabled={isLoading}
          className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-r-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:bg-gray-400"
        >
          {isLoading ? 'Sending...' : 'Send'}
        </button>
      </form>
    </div>
  );
};

export default Chat;
