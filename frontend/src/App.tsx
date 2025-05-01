import React, { useState, useRef } from 'react';
import { Histogram } from './Histogram';
import { Operations, OperationConfig } from './Operations';

// Define API base URL
const API_URL = 'http://localhost:5000/api';

// Define interface for the image data
interface ImageData {
  original: string;
  processed?: string;
  histogram?: number[];
}

// Define interface for the parameters
interface Params {
  [key: string]: any;
}

const App: React.FC = () => {
  // State for images
  const [imageData, setImageData] = useState<ImageData | null>(null);
  const [secondImage, setSecondImage] = useState<string | null>(null);
  
  // State for operations
  const [selectedOperation, setSelectedOperation] = useState<string | null>(null);
  const [params, setParams] = useState<Params>({});
  
  // State for UI
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [processStatus, setProcessStatus] = useState<{ type: string; message: string } | null>(null);
  
  // Refs for file inputs
  const fileInputRef = useRef<HTMLInputElement>(null);
  const secondFileInputRef = useRef<HTMLInputElement>(null);
  
  // Function to handle file upload
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>, isSecondImage: boolean = false) => {
    try {
      if (!e.target.files || e.target.files.length === 0) return;
      
      const file = e.target.files[0];
      const formData = new FormData();
      formData.append('file', file);
      
      setLoading(true);
      setError(null);
      
      const response = await fetch(`${API_URL}/upload`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error('Failed to upload image');
      }
      
      const data = await response.json();
      
      if (isSecondImage) {
        setSecondImage(data.base64);
      } else {
        setImageData({
          original: data.base64,
        });
        // Reset processed image when uploading a new image
        setSelectedOperation(null);
      }
    } catch (err) {
      setError((err as Error).message || 'An error occurred during upload');
    } finally {
      setLoading(false);
      // Reset file input
      e.target.value = '';
    }
  };
  
  // Function to trigger file input click
  const triggerFileInput = (isSecondImage: boolean = false) => {
    if (isSecondImage) {
      secondFileInputRef.current?.click();
    } else {
      fileInputRef.current?.click();
    }
  };
  
  // Function to process image
  const processImage = async () => {
    if (!imageData?.original || !selectedOperation) return;
    
    try {
      setLoading(true);
      setError(null);
      
      // Prepare request data
      const requestData: any = {
        image: imageData.original,
        operation: selectedOperation,
        params: params
      };
      
      // Add second image if needed
      if (Operations[selectedOperation]?.requiresSecondImage && secondImage) {
        requestData.image2 = secondImage;
      }
      
      const response = await fetch(`${API_URL}/process`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData),
      });
      
      if (!response.ok) {
        throw new Error('Failed to process image');
      }
      
      const data = await response.json();
      
      // Update image data with processed image
      setImageData(prev => ({
        ...prev!,
        processed: `data:image/jpeg;base64,${data.processed_image}`,
        histogram: data.histogram
      }));
      
      setProcessStatus({ type: 'success', message: 'Görüntü başarıyla işlendi!' });
    } catch (err) {
      setError((err as Error).message || 'İşlem sırasında bir hata oluştu');
      setProcessStatus({ type: 'error', message: (err as Error).message || 'İşlem sırasında bir hata oluştu' });
    } finally {
      setLoading(false);
    }
  };
  
  // Function to handle operation selection
  const handleOperationSelect = (operation: string) => {
    setSelectedOperation(operation);
    
    // Reset params for the new operation
    if (Operations[operation]?.defaultParams) {
      setParams(Operations[operation].defaultParams);
    } else {
      setParams({});
    }
    
    // If operation doesn't require second image, reset second image
    if (!Operations[operation]?.requiresSecondImage) {
      setSecondImage(null);
    }
  };
  
  // Function to handle parameter change
  const handleParamChange = (paramName: string, value: any) => {
    setParams(prev => ({
      ...prev,
      [paramName]: value
    }));
  };
  
  // Function to render parameter inputs based on selected operation
  const renderParamInputs = () => {
    if (!selectedOperation || !Operations[selectedOperation]?.params) {
      return null;
    }
    
    return (
      <div className="params-container">
        <h3>Parameters</h3>
        {Operations[selectedOperation].params.map(param => (
          <div className="param-group" key={param.name}>
            <label htmlFor={param.name}>{param.label}</label>
            {param.type === 'range' ? (
              <div>
                <input
                  type="range"
                  id={param.name}
                  min={param.min}
                  max={param.max}
                  step={param.step}
                  value={params[param.name] || param.defaultValue}
                  onChange={(e) => handleParamChange(param.name, parseFloat(e.target.value))}
                />
                <span>{params[param.name] || param.defaultValue}</span>
              </div>
            ) : param.type === 'select' ? (
              <select
                id={param.name}
                value={params[param.name] || param.defaultValue}
                onChange={(e) => handleParamChange(param.name, e.target.value)}
              >
                {param.options?.map(option => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            ) : (
              <input
                type={param.type}
                id={param.name}
                value={params[param.name] || param.defaultValue}
                onChange={(e) => {
                  const value = param.type === 'number' 
                    ? parseFloat(e.target.value) 
                    : e.target.value;
                  handleParamChange(param.name, value);
                }}
              />
            )}
          </div>
        ))}
        
        {Operations[selectedOperation]?.requiresSecondImage && (
          <div className="second-image-upload">
            <h3>Second Image</h3>
            <div className="upload-area" onClick={() => triggerFileInput(true)}>
              {secondImage ? (
                <img src={secondImage} alt="Second" className="image" style={{ maxHeight: '150px' }} />
              ) : (
                <>
                  <p>Click to upload second image</p>
                  <button className="btn">Select File</button>
                </>
              )}
            </div>
            <input
              type="file"
              accept="image/*"
              ref={secondFileInputRef}
              onChange={(e) => handleFileUpload(e, true)}
              style={{ display: 'none' }}
            />
          </div>
        )}
      </div>
    );
  };
  
  // Loading componentini iyileştirelim
  // Loading gösterirken görünecek bileşen:
  const Loading = () => (
    <div className="loading">
      <div className="spinner"></div>
      <p>İşlem yapılıyor...</p>
    </div>
  );
  
  // Render the application
  return (
    <div className="container">
      <div className="header">
        <h1>Image Processing Application</h1>
        <p>Upload an image and apply various processing operations</p>
      </div>
      
      <div className="main-content">
        <div className="sidebar">
          <h2>Upload Image</h2>
          <div className="upload-area" onClick={() => triggerFileInput()}>
            {imageData?.original ? (
              <img src={imageData.original} alt="Original" className="image" style={{ maxHeight: '150px' }} />
            ) : (
              <>
                <p>Click to upload an image</p>
                <button className="btn">Select File</button>
              </>
            )}
          </div>
          <input
            type="file"
            accept="image/*"
            ref={fileInputRef}
            onChange={handleFileUpload}
            style={{ display: 'none' }}
          />
          
          <h2>Operations</h2>
          <ul className="operation-list">
            {Object.entries(Operations).map(([key, operation]) => (
              <li
                key={key}
                className={`operation-item ${selectedOperation === key ? 'active' : ''}`}
                onClick={() => handleOperationSelect(key)}
              >
                {operation.name}
              </li>
            ))}
          </ul>
          
          {renderParamInputs()}
          
          <button 
            className="btn" 
            style={{ marginTop: '20px', width: '100%' }}
            disabled={!imageData?.original || !selectedOperation || loading}
            onClick={processImage}
          >
            Process Image
          </button>
          
          {error && (
            <div className="error-message">
              {error}
            </div>
          )}
        </div>
        
        <div className="image-container">
          {loading ? (
            <Loading />
          ) : (
            <>
              {imageData && (
                <div className="image-display">
                  <h2>{selectedOperation && imageData.processed ? 'Processed Image' : 'Original Image'}</h2>
                  <div className="image-comparison">
                    {selectedOperation && imageData.processed ? (
                      <>
                        <div className="image-box">
                          <p>Original</p>
                          <img src={imageData.original} alt="Original" className="image" />
                        </div>
                        <div className="image-box">
                          <p>Processed</p>
                          <img src={imageData.processed} alt="Processed" className="image" />
                        </div>
                      </>
                    ) : (
                      imageData.original && (
                        <div className="image-box" style={{ width: '100%' }}>
                          <img src={imageData.original} alt="Original" className="image" />
                        </div>
                      )
                    )}
                  </div>
                </div>
              )}
              
              {imageData?.histogram && (
                <div className="histogram-container">
                  <h2>Histogram</h2>
                  <Histogram data={imageData.histogram} />
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default App; 