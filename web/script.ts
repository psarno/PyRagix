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
export type { SearchResult, QueryRequest, QueryResponse, HealthResponse };