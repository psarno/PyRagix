// ======================================
// RAG Search Interface TypeScript
// - Modern ES2024 TypeScript implementation
// - Type-safe API communication
// - Responsive UI interactions
// ======================================

// ===============================
// Type Definitions
// ===============================
interface SearchResult {
  source: string;
  chunk_idx: number;
  score: number;
  text: string;
}

interface QueryRequest {
  query: string;
  top_k?: number;
  show_sources?: boolean;
  debug?: boolean;
}

interface QueryResponse {
  answer: string;
  sources: SearchResult[];
  query: string;
  top_k: number;
  success: boolean;
  error?: string;
}

interface HealthResponse {
  status: string;
  version: string;
  rag_loaded: boolean;
  total_documents: number;
  index_type: string;
}

// Visualization interfaces
interface VisualizationRequest {
  query: string;
  method?: string;
  dimensions?: number;
  max_points?: number;
  top_k?: number;
}

interface EmbeddingPoint {
  id: number;
  x: number;
  y: number;
  z?: number;
  source: string;
  chunk_idx: number;
  score: number;
  text: string;
  is_query: boolean;
}

interface VisualizationResponse {
  points: EmbeddingPoint[];
  query: string;
  method: string;
  dimensions: number;
  total_points: number;
  retrieved_count: number;
  success: boolean;
  error?: string;
}

// Plotly type definitions (simplified)
declare global {
  interface Window {
    Plotly: {
      newPlot: (div: string | HTMLElement, data: any[], layout: any, config?: any) => Promise<void>;
      react: (div: string | HTMLElement, data: any[], layout: any, config?: any) => Promise<void>;
    };
  }
}

// ===============================
// DOM Elements
// ===============================
const elements = {
  // Status indicator
  statusIndicator: document.getElementById('statusIndicator') as HTMLDivElement,
  statusDot: document.getElementById('statusDot') as HTMLDivElement,
  statusText: document.getElementById('statusText') as HTMLSpanElement,
  
  // Search form
  searchForm: document.getElementById('searchForm') as HTMLFormElement,
  searchInput: document.getElementById('searchInput') as HTMLInputElement,
  searchButton: document.getElementById('searchButton') as HTMLButtonElement,
  
  // Options
  topKSelect: document.getElementById('topK') as HTMLSelectElement,
  showSourcesCheckbox: document.getElementById('showSources') as HTMLInputElement,
  debugModeCheckbox: document.getElementById('debugMode') as HTMLInputElement,
  
  // Results
  resultsSection: document.getElementById('resultsSection') as HTMLElement,
  queryInfo: document.getElementById('queryInfo') as HTMLDivElement,
  loadingIndicator: document.getElementById('loadingIndicator') as HTMLDivElement,
  answerContainer: document.getElementById('answerContainer') as HTMLDivElement,
  answerContent: document.getElementById('answerContent') as HTMLDivElement,
  sourcesContainer: document.getElementById('sourcesContainer') as HTMLDivElement,
  sourcesList: document.getElementById('sourcesList') as HTMLDivElement,
  errorContainer: document.getElementById('errorContainer') as HTMLDivElement,
  errorContent: document.getElementById('errorContent') as HTMLDivElement,
  
  // Visualization
  visualizationContainer: document.getElementById('visualizationContainer') as HTMLDivElement,
  visualizationPlot: document.getElementById('visualizationPlot') as HTMLDivElement,
  visualizationInfo: document.getElementById('visualizationInfo') as HTMLDivElement,
  vizLoading: document.getElementById('vizLoading') as HTMLDivElement,
  vizMethod: document.getElementById('vizMethod') as HTMLSelectElement,
  vizDimensions: document.getElementById('vizDimensions') as HTMLSelectElement,
  refreshViz: document.getElementById('refreshViz') as HTMLButtonElement,
} as const;

// ===============================
// API Configuration
// ===============================
const API_BASE = window.location.origin;
const ENDPOINTS = {
  health: `${API_BASE}/health`,
  query: `${API_BASE}/query`,
} as const;

// ===============================
// State Management
// ===============================
class AppState {
  private _isLoading = false;
  private _isHealthy = false;
  
  get isLoading(): boolean {
    return this._isLoading;
  }
  
  set isLoading(value: boolean) {
    this._isLoading = value;
    this.updateUI();
  }
  
  get isHealthy(): boolean {
    return this._isHealthy;
  }
  
  set isHealthy(value: boolean) {
    this._isHealthy = value;
    this.updateStatusIndicator();
  }
  
  private updateUI(): void {
    elements.searchButton.disabled = this._isLoading;
    elements.searchInput.disabled = this._isLoading;
    
    if (this._isLoading) {
      showElement(elements.loadingIndicator);
    } else {
      hideElement(elements.loadingIndicator);
    }
  }
  
  private updateStatusIndicator(): void {
    const dot = elements.statusDot;
    const text = elements.statusText;
    
    dot.className = 'status-dot';
    
    if (this._isHealthy) {
      dot.classList.add('healthy');
      text.textContent = 'Ready';
    } else {
      dot.classList.add('error');
      text.textContent = 'Error';
    }
  }
}

const appState = new AppState();

// ===============================
// Utility Functions
// ===============================
function showElement(element: HTMLElement): void {
  element.style.display = '';
  element.classList.add('fade-in');
}

function hideElement(element: HTMLElement): void {
  element.style.display = 'none';
  element.classList.remove('fade-in');
}

function formatPath(path: string): string {
  const parts = path.split(/[/\\]/);
  return parts[parts.length - 1] || path;
}

function escapeHtml(text: string): string {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function formatScore(score: number): string {
  return score.toFixed(3);
}

// ===============================
// API Functions
// ===============================
async function checkHealth(): Promise<HealthResponse> {
  try {
    const response = await fetch(ENDPOINTS.health);
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const data = await response.json() as HealthResponse;
    return data;
    
  } catch (error) {
    console.error('Health check failed:', error);
    throw error;
  }
}

async function performQuery(request: QueryRequest): Promise<QueryResponse> {
  try {
    const response = await fetch(ENDPOINTS.query, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(errorData.detail || `HTTP ${response.status}`);
    }
    
    const data = await response.json() as QueryResponse;
    return data;
    
  } catch (error) {
    console.error('Query failed:', error);
    throw error;
  }
}

// ===============================
// UI Functions
// ===============================
function displayQueryInfo(query: string, topK: number): void {
  elements.queryInfo.textContent = `Searching for: "${query}" (top ${topK} results)`;
  showElement(elements.queryInfo);
}

function displayAnswer(answer: string): void {
  elements.answerContent.textContent = answer;
  showElement(elements.answerContainer);
}

function displaySources(sources: SearchResult[]): void {
  elements.sourcesList.innerHTML = '';
  
  if (sources.length === 0) {
    hideElement(elements.sourcesContainer);
    return;
  }
  
  sources.forEach((source, index) => {
    const sourceElement = document.createElement('div');
    sourceElement.className = 'source-item';
    
    sourceElement.innerHTML = `
      <div class="source-header">
        <div class="source-title">${escapeHtml(formatPath(source.source))} (chunk ${source.chunk_idx})</div>
        <div class="source-score">Score: ${formatScore(source.score)}</div>
      </div>
      <div class="source-text">${escapeHtml(source.text)}</div>
    `;
    
    elements.sourcesList.appendChild(sourceElement);
  });
  
  showElement(elements.sourcesContainer);
}

function displayError(message: string): void {
  elements.errorContent.textContent = message;
  showElement(elements.errorContainer);
}

function clearResults(): void {
  hideElement(elements.answerContainer);
  hideElement(elements.sourcesContainer);
  hideElement(elements.errorContainer);
  hideElement(elements.queryInfo);
  hideElement(elements.visualizationContainer);
}

// ===============================
// Event Handlers
// ===============================
async function handleSearch(event: Event): Promise<void> {
  event.preventDefault();
  
  const query = elements.searchInput.value.trim();
  if (!query) {
    return;
  }
  
  if (!appState.isHealthy) {
    displayError('RAG system is not ready. Please check the server status.');
    return;
  }
  
  // Clear previous results
  clearResults();
  showElement(elements.resultsSection);
  
  // Get form values
  const topK = parseInt(elements.topKSelect.value);
  const showSources = elements.showSourcesCheckbox.checked;
  const debug = elements.debugModeCheckbox.checked;
  
  // Show query info
  displayQueryInfo(query, topK);
  
  // Set loading state
  appState.isLoading = true;
  
  try {
    const request: QueryRequest = {
      query,
      top_k: topK,
      show_sources: showSources,
      debug,
    };
    
    const response = await performQuery(request);
    
    if (response.success) {
      displayAnswer(response.answer);
      
      if (showSources) {
        displaySources(response.sources);
      }
      
      // Generate visualization after successful query
      await generateVisualization(query);
    } else {
      displayError(response.error || 'Query failed for unknown reason');
    }
    
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error occurred';
    displayError(`Request failed: ${message}`);
    
  } finally {
    appState.isLoading = false;
  }
}

// ===============================
// Initialization
// ===============================
async function initializeApp(): Promise<void> {
  console.log('🚀 Initializing RAG Search Interface...');
  
  // Add event listeners
  elements.searchForm.addEventListener('submit', handleSearch);
  
  // Keyboard shortcuts
  elements.searchInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
      event.preventDefault();
      handleSearch(event);
    }
  });
  
  // Visualization controls
  elements.refreshViz.addEventListener('click', async () => {
    const query = elements.searchInput.value.trim();
    if (query) {
      await generateVisualization(query);
    }
  });
  
  // Initial health check
  await performHealthCheck();
  
  // Set up periodic health checks
  setInterval(performHealthCheck, 30000); // Check every 30 seconds
  
  // Focus on search input
  elements.searchInput.focus();
  
  console.log('✅ RAG Search Interface ready!');
}

async function performHealthCheck(): Promise<void> {
  try {
    elements.statusText.textContent = 'Checking...';
    
    const health = await checkHealth();
    
    appState.isHealthy = health.rag_loaded && health.status === 'healthy';
    
    if (appState.isHealthy) {
      elements.statusText.textContent = `Ready (${health.total_documents} docs, ${health.index_type})`;
    } else {
      elements.statusText.textContent = 'RAG system not loaded';
    }
    
  } catch (error) {
    appState.isHealthy = false;
    elements.statusText.textContent = 'Connection failed';
    console.error('Health check failed:', error);
  }
}

// ===============================
// Visualization Functions  
// ===============================
async function fetchVisualization(request: VisualizationRequest): Promise<VisualizationResponse> {
  const response = await fetch(`${API_BASE}/visualize`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || 'Visualization request failed');
  }

  return response.json();
}

function createEmbeddingPlot(vizData: VisualizationResponse): void {
  const { points, method, dimensions } = vizData;
  
  console.log('Creating plot with dimensions:', dimensions, 'method:', method);
  console.log('First few points:', points.slice(0, 3));
  
  // Separate query and document points
  const queryPoints = points.filter(p => p.is_query);
  const docPoints = points.filter(p => !p.is_query);
  const retrievedPoints = docPoints.filter(p => p.score > 0);
  const otherPoints = docPoints.filter(p => p.score === 0);

  console.log('Point counts - query:', queryPoints.length, 'retrieved:', retrievedPoints.length, 'other:', otherPoints.length);

  const traces: any[] = [];

  // Query point trace
  if (queryPoints.length > 0) {
    const queryTrace: any = {
      x: queryPoints.map(p => p.x),
      y: queryPoints.map(p => p.y),
      mode: 'markers',
      type: dimensions === 3 ? 'scatter3d' : 'scatter',
      name: '🎯 Your Query',
      marker: {
        size: 15,
        color: '#ff6b6b',
        symbol: 'star',
        line: { color: '#ffffff', width: 2 }
      },
      text: queryPoints.map(p => `Query: ${p.text}`),
      hovertemplate: '<b>%{text}</b><extra></extra>',
      customdata: queryPoints
    };
    
    // Only add z coordinate for 3D plots
    if (dimensions === 3) {
      queryTrace.z = queryPoints.map(p => p.z);
    }
    
    traces.push(queryTrace);
  }

  // Retrieved chunks trace
  if (retrievedPoints.length > 0) {
    const retrievedTrace: any = {
      x: retrievedPoints.map(p => p.x),
      y: retrievedPoints.map(p => p.y),
      mode: 'markers',
      type: dimensions === 3 ? 'scatter3d' : 'scatter',
      name: '✅ Retrieved Chunks',
      marker: {
        size: retrievedPoints.map(p => 8 + p.score * 8),
        color: retrievedPoints.map(p => p.score),
        colorscale: 'Viridis',
        showscale: true,
        colorbar: { 
          title: 'Similarity Score', 
          x: 1.02,
          len: 0.6,
          y: 0.4,
          yanchor: 'middle'
        },
        line: { color: '#ffffff', width: 1 }
      },
      text: retrievedPoints.map(p => `${p.source} (chunk ${p.chunk_idx})<br>Score: ${p.score.toFixed(3)}<br>${p.text.substring(0, 100)}...`),
      hovertemplate: '<b>%{text}</b><extra></extra>',
      customdata: retrievedPoints
    };
    
    // Only add z coordinate for 3D plots
    if (dimensions === 3) {
      retrievedTrace.z = retrievedPoints.map(p => p.z);
    }
    
    traces.push(retrievedTrace);
  }

  // Other document points trace  
  if (otherPoints.length > 0) {
    const otherTrace: any = {
      x: otherPoints.map(p => p.x),
      y: otherPoints.map(p => p.y),
      mode: 'markers',
      type: dimensions === 3 ? 'scatter3d' : 'scatter',
      name: '📄 Other Documents',
      marker: {
        size: 5,
        color: '#666666',
        opacity: 0.6,
        line: { color: '#333333', width: 0.5 }
      },
      text: otherPoints.map(p => `${p.source} (chunk ${p.chunk_idx})<br>${p.text.substring(0, 100)}...`),
      hovertemplate: '<b>%{text}</b><extra></extra>',
      customdata: otherPoints
    };
    
    // Only add z coordinate for 3D plots
    if (dimensions === 3) {
      otherTrace.z = otherPoints.map(p => p.z);
    }
    
    traces.push(otherTrace);
  }

  // Plot layout - separate for 2D and 3D
  const baseLayout = {
    title: {
      text: `${method.toUpperCase()} Embedding Visualization (${dimensions}D)`,
      font: { color: '#ffffff', size: 16 }
    },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    legend: {
      font: { color: '#ffffff', size: 12 },
      bgcolor: 'rgba(0,0,0,0.7)',
      bordercolor: '#444444',
      borderwidth: 1,
      orientation: 'v',
      x: -0.15,
      y: 1,
      xanchor: 'right',
      yanchor: 'top',
      itemsizing: 'constant',
      itemwidth: 30,
      tracegroupgap: 8
    },
    margin: { t: 50, l: 50, r: 120, b: 50 }
  };

  const layout = dimensions === 3 ? {
    ...baseLayout,
    scene: {
      bgcolor: 'rgba(0,0,0,0)',
      xaxis: { 
        gridcolor: '#444444', 
        color: '#ffffff' 
      },
      yaxis: { 
        gridcolor: '#444444', 
        color: '#ffffff' 
      },
      zaxis: { 
        gridcolor: '#444444', 
        color: '#ffffff' 
      },
      camera: {
        eye: { x: 1.5, y: 1.5, z: 1.5 }
      }
    }
  } : {
    ...baseLayout,
    xaxis: {
      gridcolor: '#444444',
      color: '#ffffff'
    },
    yaxis: {
      gridcolor: '#444444',
      color: '#ffffff'
    }
  };

  const config = {
    displayModeBar: true,
    modeBarButtonsToRemove: ['pan2d', 'lasso2d'],
    displaylogo: false,
    responsive: true
  };

  // Create the plot
  console.log('About to create plot with traces:', traces.length, 'layout:', layout);
  console.log('Plot config:', config);
  
  window.Plotly.newPlot(elements.visualizationPlot, traces, layout, config)
    .then(() => {
      // Add click handler for points
      (elements.visualizationPlot as any).on('plotly_click', (data: any) => {
        if (data.points && data.points.length > 0) {
          const point = data.points[0].customdata as EmbeddingPoint;
          showPointDetails(point);
        }
      });
    })
    .catch(error => {
      console.error('Error creating plot:', error);
      elements.visualizationPlot.innerHTML = '<div style="color: #ff6b6b; text-align: center; padding: 20px;">Error creating visualization</div>';
    });
}

function showPointDetails(point: EmbeddingPoint): void {
  // Create a modal or update info panel with point details
  const infoHtml = `
    <div class="viz-stats">
      <div class="viz-stat">
        <div class="viz-stat-value">${point.is_query ? 'QUERY' : point.source}</div>
        <div class="viz-stat-label">${point.is_query ? 'Your Query' : 'Document'}</div>
      </div>
      <div class="viz-stat">
        <div class="viz-stat-value">${point.chunk_idx}</div>
        <div class="viz-stat-label">Chunk</div>
      </div>
      <div class="viz-stat">
        <div class="viz-stat-value">${point.score.toFixed(3)}</div>
        <div class="viz-stat-label">Score</div>
      </div>
    </div>
    <div style="margin-top: 12px; padding: 12px; background: var(--bg-secondary); border-radius: 6px; font-size: 14px; color: var(--text-secondary);">
      ${point.text}
    </div>
  `;
  
  elements.visualizationInfo.innerHTML = infoHtml;
}

async function generateVisualization(query: string): Promise<void> {
  try {
    // Show loading state
    elements.vizLoading.style.display = 'flex';
    elements.visualizationPlot.innerHTML = '';
    elements.visualizationInfo.innerHTML = '';
    
    // Get visualization parameters
    const method = elements.vizMethod.value;
    const dimensions = parseInt(elements.vizDimensions.value);
    const topK = parseInt(elements.topKSelect.value);
    
    console.log('Generating visualization:', { method, dimensions, topK });
    
    // Fetch visualization data
    const vizData = await fetchVisualization({
      query,
      method,
      dimensions,
      top_k: topK,
      max_points: 1000
    });
    
    console.log('Visualization data received:', vizData);
    
    if (!vizData.success) {
      throw new Error(vizData.error || 'Visualization failed');
    }
    
    // Validate data
    if (!vizData.points || vizData.points.length === 0) {
      throw new Error('No visualization points received');
    }
    
    // Check for required coordinates
    const samplePoint = vizData.points[0];
    if (dimensions === 3 && samplePoint && (samplePoint.z === null || samplePoint.z === undefined)) {
      throw new Error('3D visualization requested but z coordinates are missing');
    }
    
    // Hide loading, show visualization
    elements.vizLoading.style.display = 'none';
    elements.visualizationContainer.style.display = 'block';
    
    // Create the plot
    createEmbeddingPlot(vizData);
    
    // Update info
    const infoHtml = `
      <div class="viz-stats">
        <div class="viz-stat">
          <div class="viz-stat-value">${vizData.total_points}</div>
          <div class="viz-stat-label">Total Points</div>
        </div>
        <div class="viz-stat">
          <div class="viz-stat-value">${vizData.retrieved_count}</div>
          <div class="viz-stat-label">Retrieved</div>
        </div>
        <div class="viz-stat">
          <div class="viz-stat-value">${vizData.method.toUpperCase()}</div>
          <div class="viz-stat-label">Method</div>
        </div>
        <div class="viz-stat">
          <div class="viz-stat-value">${vizData.dimensions}D</div>
          <div class="viz-stat-label">Dimensions</div>
        </div>
      </div>
      <div style="color: var(--text-muted); font-size: 12px;">
        Click any point to see details • Query shown as ⭐ • Retrieved chunks highlighted
      </div>
    `;
    elements.visualizationInfo.innerHTML = infoHtml;
    
  } catch (error) {
    elements.vizLoading.style.display = 'none';
    elements.visualizationPlot.innerHTML = `
      <div style="color: var(--accent-error); text-align: center; padding: 20px;">
        <strong>Visualization Error:</strong><br>
        ${error instanceof Error ? error.message : String(error)}
      </div>
    `;
    console.error('Visualization error:', error);
  }
}

// ===============================
// App Entry Point
// ===============================
document.addEventListener('DOMContentLoaded', () => {
  initializeApp().catch((error) => {
    console.error('Failed to initialize app:', error);
    appState.isHealthy = false;
    elements.statusText.textContent = 'Initialization failed';
  });
});

// Export for potential external use
export type { 
  SearchResult, 
  QueryRequest, 
  QueryResponse, 
  HealthResponse,
  VisualizationRequest,
  VisualizationResponse,
  EmbeddingPoint
};