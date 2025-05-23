// frontend/src/components/FileUploader.jsx
import React, { useState } from 'react';
import axios from 'axios';

const FileUploader = ({ onUploadSuccess }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState(''); // e.g., 'uploading', 'success', 'error'
  const [error, setError] = useState('');

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
    setUploadStatus('');
    setError('');
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please select a file first.');
      return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);

    setUploadStatus('uploading');
    setError('');

    try {
      // Assuming backend is running on port 5001 as per .env file
      const response = await axios.post('http://localhost:5001/api/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setUploadStatus('success');
      console.log('Upload successful:', response.data);
      if(onUploadSuccess) {
        onUploadSuccess(response.data); // Pass data to parent
      }
    } catch (err) {
      setUploadStatus('error');
      let errorMsg = 'Upload failed. Please try again.';
      if (err.response && err.response.data && err.response.data.error) {
        errorMsg = err.response.data.error;
      } else if (err.message) {
        errorMsg = err.message;
      }
      setError(errorMsg);
      console.error('Upload error:', err);
    }
  };

  return (
    <div className="bg-gray-700 p-3 rounded-lg">
      <h2 className="text-lg font-semibold mb-2 text-white">Upload Document</h2>
      <input 
        type="file" 
        onChange={handleFileChange} 
        className="block w-full text-sm text-gray-300 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-500 file:text-white hover:file:bg-blue-600 mb-2"
        accept=".pdf,.docx"
      />
      <button 
        onClick={handleUpload} 
        disabled={uploadStatus === 'uploading'}
        className="w-full bg-green-500 hover:bg-green-600 text-white font-bold py-2 px-4 rounded disabled:bg-gray-500"
      >
        {uploadStatus === 'uploading' ? 'Uploading...' : 'Upload'}
      </button>
      {error && <p className="text-red-400 text-sm mt-2">{error}</p>}
      {uploadStatus === 'success' && selectedFile && <p className="text-green-400 text-sm mt-2">File "{selectedFile.name}" uploaded successfully!</p>}
    </div>
  );
};

export default FileUploader;
