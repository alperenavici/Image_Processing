import React, { useState, useRef } from 'react';
import { Histogram } from './Histogram';
import { Operations, OperationConfig } from './Operations';

// Define API base URL
const API_URL = process.env.NODE_ENV === 'production'
  ? '/api'  // Production (Vercel) - relatif yol 
  : 'http://localhost:3000/api';  // Lokal geliştirme

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

// Define interface for selected operations
interface SelectedOp {
  operation: string;
  params: Params;
}

// Define interface for history entry
interface HistoryEntry {
  timestamp: string;
  operation: string | string[];
  processing_time?: number;
  filename?: string;
  success: boolean;
  error?: string;
}

const App: React.FC = () => {
  // State for images
  const [imageData, setImageData] = useState<ImageData | null>(null);
  const [secondImage, setSecondImage] = useState<string | null>(null);

  // State for operations
  const [selectedOperation, setSelectedOperation] = useState<string | null>(null);
  const [params, setParams] = useState<Params>({});

  // State for multi-operations
  const [multiMode, setMultiMode] = useState<boolean>(false);
  const [selectedOperations, setSelectedOperations] = useState<SelectedOp[]>([]);

  // State for UI
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [processStatus, setProcessStatus] = useState<{ type: string; message: string } | null>(null);

  // State for operation history
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [showHistory, setShowHistory] = useState<boolean>(true);

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

      // Update history if available
      if (data.history) {
        setHistory(data.history);
      }

      if (isSecondImage) {
        setSecondImage(data.base64);
      } else {
        setImageData({
          original: data.base64,
        });
        // Reset processed image when uploading a new image
        setSelectedOperation(null);
        setSelectedOperations([]);
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

  // Function to process image with single operation
  const processImage = async () => {
    if (!imageData?.original || (!selectedOperation && selectedOperations.length === 0)) return;

    try {
      setLoading(true);
      setError(null);

      // Prepare request data
      let requestData: any;

      if (multiMode) {
        // Multiple operations mode
        const operations = selectedOperations.map(op => op.operation);
        const opParams = selectedOperations.map(op => op.params);

        requestData = {
          image: imageData.original,
          operation: operations,
          params: opParams
        };

        // Check if any operation requires second image
        const needsSecondImage = selectedOperations.some(op =>
          Operations[op.operation]?.requiresSecondImage
        );

        if (needsSecondImage && secondImage) {
          requestData.image2 = secondImage;
          console.log("Adding second image to multi-operation request");
        }
      } else {
        // Single operation mode
        requestData = {
          image: imageData.original,
          operation: selectedOperation,
          params: params
        };

        // Add second image if needed
        if (selectedOperation && Operations[selectedOperation]?.requiresSecondImage && secondImage) {
          requestData.image2 = secondImage;
          console.log(`Adding second image for operation: ${selectedOperation}`);
        }
      }

      console.log("Request data:", {
        operation: requestData.operation,
        hasImage2: !!requestData.image2,
        params: requestData.params
      });

      // JSON.stringify data to show what's being sent
      console.log("JSON being sent:", JSON.stringify(requestData));

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

      // Update history if available
      if (data.history) {
        setHistory(data.history);
      }

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
    console.log(`Selected operation: ${operation}`);
    console.log(`Requires second image: ${Operations[operation]?.requiresSecondImage}`);

    if (multiMode) {
      // Don't allow duplicate operations in multi-mode
      if (selectedOperations.some(op => op.operation === operation)) {
        return;
      }

      // Add to selected operations list
      const newOperation: SelectedOp = {
        operation,
        params: Operations[operation]?.defaultParams || {}
      };

      setSelectedOperations([...selectedOperations, newOperation]);

      // Check if any operation requires second image
      const needsSecondImage = [...selectedOperations, newOperation].some(
        op => Operations[op.operation]?.requiresSecondImage
      );
      console.log(`Multi-mode needs second image: ${needsSecondImage}`);

      if (needsSecondImage) {
        setProcessStatus({
          type: 'info',
          message: 'Bu işlem için lütfen ikinci bir görüntü ekleyin'
        });
      }
    } else {
      // Single operation mode
      setSelectedOperation(operation);

      // Reset params for the new operation
      if (Operations[operation]?.defaultParams) {
        setParams(Operations[operation].defaultParams);
      } else {
        setParams({});
      }

      // If operation requires second image, show a message
      if (Operations[operation]?.requiresSecondImage) {
        console.log(`Operation ${operation} requires a second image`);
        setProcessStatus({
          type: 'info',
          message: 'Bu işlem için lütfen ikinci bir görüntü ekleyin'
        });
      } else {
        // If operation doesn't require second image, reset second image
        setSecondImage(null);
      }
    }
  };

  // Function to handle parameter change in single operation mode
  const handleParamChange = (paramName: string, value: any) => {
    console.log(`Parameter changed: ${paramName} = ${value}, type: ${typeof value}`);
    setParams(prev => {
      const newParams = {
        ...prev,
        [paramName]: value
      };
      console.log("New params:", newParams);
      return newParams;
    });
  };

  // Function to handle parameter change in multi operation mode
  const handleMultiParamChange = (index: number, paramName: string, value: any) => {
    setSelectedOperations(prev => {
      const newOps = [...prev];
      newOps[index] = {
        ...newOps[index],
        params: {
          ...newOps[index].params,
          [paramName]: value
        }
      };
      return newOps;
    });
  };

  // Function to remove operation from multi-operation list
  const removeOperation = (index: number) => {
    setSelectedOperations(prev => {
      const newOps = [...prev];
      newOps.splice(index, 1);
      return newOps;
    });
  };

  // Function to toggle between single and multi operation modes
  const toggleMode = () => {
    setMultiMode(!multiMode);

    if (!multiMode) {
      // Switching to multi mode
      if (selectedOperation) {
        // Convert current selection to multi-selection
        setSelectedOperations([{
          operation: selectedOperation,
          params: params
        }]);
      } else {
        setSelectedOperations([]);
      }
      setSelectedOperation(null);
    } else {
      // Switching to single mode
      setSelectedOperations([]);
      setSelectedOperation(null);
      setParams({});
    }
  };

  // Function to render parameter inputs based on selected operation
  const renderParamInputs = () => {
    console.log("Rendering param inputs");
    console.log("multiMode:", multiMode);
    console.log("selectedOperation:", selectedOperation);
    console.log("selectedOperations:", selectedOperations);

    if (multiMode) {
      // Check if any operation needs second image
      const needsSecondImage = selectedOperations.some(op => Operations[op.operation]?.requiresSecondImage);
      console.log("Multi mode needs second image:", needsSecondImage);

      // Render multiple operations with parameters
      return (
        <div className="multi-operations-container">
          <h3>Seçilen İşlemler</h3>
          {selectedOperations.length === 0 ? (
            <p>Henüz işlem seçilmedi. Soldaki listeden işlem seçiniz.</p>
          ) : (
            <ul className="selected-operations-list">
              {selectedOperations.map((op, index) => (
                <li key={index} className="selected-operation-item">
                  <div className="selected-operation-header">
                    <span>{Operations[op.operation]?.name || op.operation}</span>
                    <button
                      className="btn-remove"
                      onClick={() => removeOperation(index)}
                      title="Bu işlemi kaldır"
                    >
                      &times;
                    </button>
                  </div>

                  {Operations[op.operation]?.params && (
                    <div className="operation-params">
                      {Operations[op.operation]?.params?.map(param => (
                        <div className="param-group" key={param.name}>
                          <label htmlFor={`${op.operation}-${param.name}-${index}`}>
                            {param.label}
                          </label>
                          {param.type === 'range' ? (
                            <div>
                              <input
                                type="range"
                                id={`${op.operation}-${param.name}-${index}`}
                                min={param.min}
                                max={param.max}
                                step={param.step}
                                value={op.params[param.name] || param.defaultValue}
                                onChange={(e) => handleMultiParamChange(
                                  index,
                                  param.name,
                                  parseFloat(e.target.value)
                                )}
                              />
                              <span>{op.params[param.name] || param.defaultValue}</span>
                            </div>
                          ) : param.type === 'select' ? (
                            <select
                              id={`${op.operation}-${param.name}-${index}`}
                              value={op.params[param.name] || param.defaultValue}
                              onChange={(e) => handleMultiParamChange(
                                index,
                                param.name,
                                e.target.value
                              )}
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
                              id={`${op.operation}-${param.name}-${index}`}
                              value={op.params[param.name] || param.defaultValue}
                              onChange={(e) => {
                                const value = param.type === 'number'
                                  ? parseFloat(e.target.value)
                                  : e.target.value;
                                handleMultiParamChange(index, param.name, value);
                              }}
                            />
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </li>
              ))}
            </ul>
          )}

          {/* Second image section for multi-operations if needed */}
          {needsSecondImage && (
            <div className="second-image-upload">
              <h3>İkinci Görüntü</h3>
              <div className="upload-area" onClick={() => triggerFileInput(true)}>
                {secondImage ? (
                  <>
                    <img src={secondImage} alt="Second" className="image" style={{ maxHeight: '150px' }} />
                    <button
                      className="btn btn-clear-second"
                      onClick={(e) => {
                        e.stopPropagation(); // Prevent triggering file input
                        setSecondImage(null);
                      }}
                    >
                      İkinci Görüntüyü Kaldır
                    </button>
                  </>
                ) : (
                  <>
                    <p>İkinci görüntüyü yüklemek için tıklayın</p>
                    <button className="btn">Dosya Seç</button>
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
    } else {
      // Single operation mode
      if (!selectedOperation) {
        return null;
      }

      console.log("Single mode, operation:", selectedOperation);
      console.log("Requires second image:", Operations[selectedOperation]?.requiresSecondImage);

      return (
        <div className="params-container">
          <h3>Parametreler</h3>
          {Operations[selectedOperation]?.params?.map(param => (
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
              <h3>İkinci Görüntü</h3>
              <div className="upload-area" onClick={() => triggerFileInput(true)}>
                {secondImage ? (
                  <>
                    <img src={secondImage} alt="Second" className="image" style={{ maxHeight: '150px' }} />
                    <button
                      className="btn btn-clear-second"
                      onClick={(e) => {
                        e.stopPropagation(); // Prevent triggering file input
                        setSecondImage(null);
                      }}
                    >
                      İkinci Görüntüyü Kaldır
                    </button>
                  </>
                ) : (
                  <>
                    <p>İkinci görüntüyü yüklemek için tıklayın</p>
                    <button className="btn">Dosya Seç</button>
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
    }
  };

  // Function to continue processing from the processed image
  const continueFromProcessed = () => {
    if (imageData?.processed) {
      setImageData({
        original: imageData.processed,
        processed: undefined,
        histogram: undefined
      });

      // Reset operations after continuing
      setSelectedOperation(null);
      setSelectedOperations([]);
      setProcessStatus(null);
    }
  };

  // Function to toggle history panel
  const toggleHistory = () => {
    setShowHistory(!showHistory);
  };

  // Function to format operation names for display
  const formatOperation = (op: string | string[]): string => {
    if (typeof op === 'string') {
      return Operations[op]?.name || op;
    } else if (Array.isArray(op)) {
      if (op.length === 0) return 'No operation';
      if (op.length === 1) return Operations[op[0]]?.name || op[0];
      return `${op.length} operations`;
    }
    return 'Unknown operation';
  };

  // History component
  const HistoryPanel = () => {
    if (!showHistory || history.length === 0) return null;

    return (
      <div className="history-panel">
        <div className="history-header">
          <h3>İşlem Geçmişi</h3>
          <button className="btn-toggle" onClick={toggleHistory}>
            {showHistory ? 'Gizle' : 'Göster'}
          </button>
        </div>
        <div className="history-content">
          <table className="history-table">
            <thead>
              <tr>
                <th>Zaman</th>
                <th>İşlem</th>
                <th>Süre</th>
                <th>Durum</th>
              </tr>
            </thead>
            <tbody>
              {history.slice().reverse().map((entry, index) => (
                <tr key={index} className={entry.success ? 'success' : 'error'}>
                  <td>{entry.timestamp}</td>
                  <td>{formatOperation(entry.operation)}</td>
                  <td>{entry.processing_time ? `${entry.processing_time}s` : '-'}</td>
                  <td>{entry.success ? '✓' : '✗'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
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

  return (
    <div className="app">
      <header>
        <h1>Görüntü İşleme Uygulaması</h1>
      </header>

      <main>
        <div className="container">
          <div className="image-container">
            <h2>Görüntü</h2>

            {!imageData?.original ? (
              <div className="upload-area" onClick={() => triggerFileInput()}>
                <p>Görüntüyü yüklemek için tıklayın veya sürükleyin</p>
                <button className="btn">Dosya Seç</button>
              </div>
            ) : (
              <div className="images">
                <div className="image-item">
                  <h3>Orijinal Görüntü</h3>
                  <img src={imageData.original} alt="Original" className="image" />
                  <div className="image-actions">
                    <button
                      className="btn btn-change"
                      onClick={() => triggerFileInput()}
                    >
                      Görüntüyü Değiştir
                    </button>
                  </div>
                </div>

                {imageData.processed && (
                  <div className="image-item">
                    <h3>İşlenmiş Görüntü</h3>
                    <img src={imageData.processed} alt="Processed" className="image" />
                    <div className="image-actions">
                      <a
                        href={imageData.processed}
                        download="processed_image.jpg"
                        className="btn btn-download"
                      >
                        İndir
                      </a>
                      <button
                        className="btn btn-continue"
                        onClick={continueFromProcessed}
                      >
                        Bu Görüntüden Devam Et
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )}

            <input
              type="file"
              accept="image/*"
              ref={fileInputRef}
              onChange={(e) => handleFileUpload(e)}
              style={{ display: 'none' }}
            />

            {/* Display histogram if available */}
            {imageData?.histogram && (
              <div className="histogram-container">
                <h3>Histogram</h3>
                <Histogram data={imageData.histogram} />
              </div>
            )}
          </div>

          <div className="operations-container">
            <div className="operations-section">
              <div className="operations-header">
                <h2>İşlemler</h2>
                <div className="mode-toggle">
                  <label>
                    <input
                      type="checkbox"
                      checked={multiMode}
                      onChange={toggleMode}
                    />
                    <span>Çoklu İşlem Modu {multiMode ? "Açık" : "Kapalı"}</span>
                  </label>
                </div>
              </div>

              <div className="operations-list">
                {Object.entries(Operations).map(([key, operation]) => (
                  <div
                    key={key}
                    className={`operation-item ${(multiMode && selectedOperations.some(op => op.operation === key)) ||
                      (!multiMode && selectedOperation === key)
                      ? 'selected'
                      : ''
                      }`}
                    onClick={() => handleOperationSelect(key)}
                  >
                    <h3>{operation.name}</h3>
                    <p>{operation.description}</p>
                  </div>
                ))}
              </div>
            </div>

            <div className="parameters-section">
              {renderParamInputs()}

              {imageData?.original && (
                <div className="process-section">
                  <button
                    className="btn btn-process"
                    onClick={processImage}
                    disabled={loading || (multiMode ? selectedOperations.length === 0 : !selectedOperation)}
                  >
                    {loading ? 'İşleniyor...' : 'Görüntüyü İşle'}
                  </button>

                  {processStatus && (
                    <div className={`status-message ${processStatus.type}`}>
                      {processStatus.message}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* History Panel */}
        <HistoryPanel />
      </main>

      {loading && <Loading />}

      <footer>
        <p>© 2025 Görüntü İşleme Uygulaması</p>
      </footer>
    </div>
  );
}

export default App; 