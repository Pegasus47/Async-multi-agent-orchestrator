import React, { useState, useEffect, useRef } from 'react'
import { 
  Play, 
  Terminal, 
  History, 
  Layers, 
  Code2, 
  RotateCw, 
  ExternalLink, 
  Copy, 
  Check, 
  Key, 
  Bot, 
  Cpu, 
  FileText,
  Clock,
  Zap,
  DollarSign,
  ShieldCheck,
  ChevronRight,
  Sliders,
  Sparkles,
  AlertCircle
} from 'lucide-react'

export default function App() {
  // Navigation: only 'main' and 'history'
  const [activeTab, setActiveTab] = useState('main')
  const [theme, setTheme] = useState(() => localStorage.getItem('helm_theme') || 'dark')
  useEffect(() => {
    document.documentElement.dataset.theme = theme
    localStorage.setItem('helm_theme', theme)
  }, [theme])
  useEffect(() => {
    fetch('/api/runs?limit=1').then(r => r.json()).then(data => {
      if (data.runs?.length) {
        setCurrentRunId(data.runs[0].run_id)
        loadRunFiles(data.runs[0].run_id)
      }
    }).catch(console.error)
  }, [])

  // Task & Config State
  const [task, setTask] = useState('Build an interactive coffee shop landing page with dark theme and dynamic menu filtering')
  const [numAgents, setNumAgents] = useState(3)
  const [apiKey, setApiKey] = useState(localStorage.getItem('openrouter_api_key') || '')
  const [tempApiKey, setTempApiKey] = useState(apiKey)
  const [showKeyModal, setShowKeyModal] = useState(false)
  const [showModelConfig, setShowModelConfig] = useState(false)

  // Models State
  const [freeModels, setFreeModels] = useState([])
  const [helmModel, setHelmModel] = useState('google/gemma-4-31b-it:free')
  const [workerModels, setWorkerModels] = useState({
    agent_1: 'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free',
    agent_2: 'cohere/north-mini-code:free',
    agent_3: 'nvidia/nemotron-3-super-120b-a12b:free',
    agent_4: 'google/gemma-4-26b-a4b-it:free',
    agent_5: 'liquid/lfm-2.5-2.6b:free',
    agent_6: 'openrouter/free',
  })

  // Execution & Live Telemetry State
  const [isExecuting, setIsExecuting] = useState(false)
  const [telemetryState, setTelemetryState] = useState({
    status: 'idle',
    planSummary: '',
    workers: {},
    wallTime: null,
    speedup: null,
  })

  // Workspace Files & Preview State
  const [currentRunId, setCurrentRunId] = useState(null)
  const [files, setFiles] = useState({})
  const [selectedFile, setSelectedFile] = useState('')
  const [copied, setCopied] = useState(false)

  // History & Billing State
  const [billingData, setBillingData] = useState(null)
  const [historyLoading, setHistoryLoading] = useState(false)

  // WebSocket reference
  const wsRef = useRef(null)
  const iframeRef = useRef(null)

  // Fetch Free Models
  useEffect(() => {
    async function loadModels() {
      try {
        const url = '/api/models/free'
        const res = await fetch(url)
        const data = await res.json()
        if (data.models && data.models.length > 0) {
          setFreeModels(data.models)
          const ids = new Set(data.models.map(m => m.id))
          setHelmModel(prev => ids.has(prev) ? prev : data.models[0].id)
          setWorkerModels(prev => Object.fromEntries(Object.entries(prev).map(([id, model], i) =>
            [id, ids.has(model) ? model : data.models[i % data.models.length].id])))
        }
      } catch (err) {
        console.error('Failed to load free models:', err)
      }
    }
    loadModels()
  }, [apiKey])

  // WebSocket Connection
  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`
    const ws = new WebSocket(wsUrl)
    wsRef.current = ws

    ws.onmessage = (evt) => {
      try {
        const { event, data } = JSON.parse(evt.data)
        handleWsEvent(event, data)
      } catch (e) {
        console.error('WS parse error:', e)
      }
    }

    return () => {
      ws.close()
    }
  }, [])

  function handleWsEvent(event, data) {
    if (event === 'run_init') {
      setTelemetryState({
        status: 'planning',
        planSummary: '',
        workers: {},
        wallTime: null,
        speedup: null,
      })
    } else if (event === 'helm_planning_complete') {
      setTelemetryState(prev => ({
        ...prev,
        status: 'executing',
        planSummary: data.plan_summary,
      }))
    } else if (event === 'worker_start') {
      setTelemetryState(prev => ({
        ...prev,
        workers: {
          ...prev.workers,
          [data.agent_id]: { role: data.role, filename: data.filename, status: 'running' }
        }
      }))
    } else if (event === 'worker_complete') {
      setTelemetryState(prev => ({
        ...prev,
        workers: {
          ...prev.workers,
          [data.agent_id]: { ...prev.workers[data.agent_id], status: 'done', duration_s: data.duration_s }
        }
      }))
    } else if (event === 'run_finished') {
      setTelemetryState(prev => ({
        ...prev,
        status: 'finished',
        wallTime: data.wall_clock_s,
        speedup: data.speedup,
      }))
    }
  }

  // Execute Orchestration (100% Live)
  async function handleRun() {
    if (!task.trim()) return

    if (!apiKey.trim()) {
      setShowKeyModal(true)
      alert('OpenRouter API key is required. Please enter your API key to proceed.')
      return
    }

    setIsExecuting(true)
    setTelemetryState({ status: 'starting', planSummary: '', workers: {}, wallTime: null, speedup: null })

    const payload = {
      task: task.trim(),
      num_agents: parseInt(numAgents, 10),
      helm_model: helmModel,
      worker_models: workerModels,
      api_key: apiKey.trim(),
    }

    try {
      const res = await fetch('/api/orchestrate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })

      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Orchestration execution failed')
      }

      const report = await res.json()
      setCurrentRunId(report.run_id)
      await loadRunFiles(report.run_id)

    } catch (err) {
      alert(`Execution failed: ${err.message}`)
    } finally {
      setIsExecuting(false)
    }
  }

  // Load Run Files
  async function loadRunFiles(runId) {
    try {
      const res = await fetch(`/api/runs/${runId}/files`)
      const data = await res.json()
      const loadedFiles = data.files || {}
      setFiles(loadedFiles)

      const fnames = Object.keys(loadedFiles)
      if (fnames.length > 0) {
        const defaultFile = fnames.find(f => f.endsWith('.html')) || fnames[0]
        setSelectedFile(defaultFile)
      }
    } catch (e) {
      console.error('Error fetching run files:', e)
    }
  }

  // Load History & Billing Data
  async function loadBillingAndHistory() {
    setHistoryLoading(true)
    try {
      const res = await fetch('/api/billing')
      const data = await res.json()
      setBillingData(data)
    } catch (err) {
      console.error('Error loading billing history:', err)
    } finally {
      setHistoryLoading(false)
    }
  }

  useEffect(() => {
    if (activeTab === 'history') {
      loadBillingAndHistory()
    }
  }, [activeTab])

  // Presets
  const presets = [
    { label: 'Coffee Shop Landing', prompt: 'Build an interactive coffee shop landing page with dark theme, responsive grid, and menu filter' },
    { label: 'Tip Calculator', prompt: 'Build a responsive tip and bill split calculator with custom tip percentage and rounding' },
    { label: 'Kanban Board', prompt: 'Build a Kanban task board with todo, in-progress, and done columns with priority tags' },
    { label: 'Relational Mini-DB', prompt: 'Build an in-memory SQL database engine with lexer parser, storage, and executor' },
  ]

  const copyToClipboard = () => {
    if (!selectedFile || !files[selectedFile]) return
    navigator.clipboard.writeText(files[selectedFile])
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }

  const htmlFileExists = Object.keys(files).some(f => f.endsWith('.html'))

  return (
    <div className="min-h-screen bg-black text-zinc-100 flex flex-col font-sans">
      
      {/* Top Navbar */}
      <header className="border-b border-zinc-800 bg-zinc-950/80 backdrop-blur sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 h-14 flex items-center justify-between">
          
          {/* Logo */}
          <div className="flex items-center gap-3">
            <div className="w-7 h-7 bg-white text-black rounded flex items-center justify-center font-bold text-xs">
              H
            </div>
            <span className="font-semibold text-sm tracking-tight text-white">
              HELM ORCHESTRATOR
            </span>
          </div>

          {/* Navigation Tabs (Only MAIN and HISTORY) */}
          <div className="flex items-center gap-1 bg-zinc-900 border border-zinc-800 p-1 rounded-md">
            <button
              onClick={() => setActiveTab('main')}
              className={`px-4 py-1 text-xs font-medium rounded transition-colors ${
                activeTab === 'main'
                  ? 'bg-white text-black font-semibold'
                  : 'text-zinc-400 hover:text-white'
              }`}
            >
              MAIN
            </button>
            <button
              onClick={() => setActiveTab('history')}
              className={`px-4 py-1 text-xs font-medium rounded transition-colors ${
                activeTab === 'history'
                  ? 'bg-white text-black font-semibold'
                  : 'text-zinc-400 hover:text-white'
              }`}
            >
              HISTORY
            </button>
          </div>

          {/* Theme and API settings */}
          <div className="flex items-center gap-2">
            <button onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
              aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
              className="text-xs px-3 py-1 rounded border border-zinc-800 bg-zinc-900 text-zinc-300">
              {theme === 'dark' ? 'Light mode' : 'Dark mode'}
            </button>
            <button
              onClick={() => {
                setTempApiKey(apiKey)
                setShowKeyModal(true)
              }}
              className="text-xs px-2.5 py-1 rounded border border-zinc-800 bg-zinc-900 text-zinc-300 hover:bg-zinc-800 flex items-center gap-1.5 transition-colors font-mono"
            >
              <span className={`w-1.5 h-1.5 rounded-full ${apiKey ? 'bg-emerald-400' : 'bg-amber-400'}`}></span>
              <span>{apiKey ? 'KEY CONFIGURED' : 'SET OPENROUTER KEY'}</span>
              <Key className="w-3 h-3 text-zinc-400" />
            </button>
          </div>

        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 py-6">

        {/* ========================================================= */}
        {/* TAB 1: MAIN */}
        {/* ========================================================= */}
        {activeTab === 'main' && (
          <div className="space-y-6">

            {/* Task Input Section */}
            <div className="border border-zinc-800 bg-zinc-950 rounded-lg p-5 space-y-4">
              <div className="flex items-center justify-between">
                <label className="text-xs font-semibold text-zinc-200 uppercase tracking-wider">
                  Task Specification
                </label>

                {/* Agents Count Selector */}
                <div className="flex items-center gap-2 bg-zinc-900 border border-zinc-800 px-2.5 py-1 rounded">
                  <span className="text-xs text-zinc-400 font-mono">AGENTS:</span>
                  <div className="flex items-center gap-1">
                    {[2, 3, 4, 5, 6].map(n => (
                      <button
                        key={n}
                        onClick={() => setNumAgents(n)}
                        className={`w-5 h-5 text-xs font-mono rounded flex items-center justify-center transition-colors ${
                          numAgents === n 
                            ? 'bg-white text-black font-bold' 
                            : 'text-zinc-400 hover:text-white'
                        }`}
                      >
                        {n}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              {/* Textarea */}
              <textarea
                rows={2}
                value={task}
                onChange={(e) => setTask(e.target.value)}
                placeholder="Enter project task..."
                className="w-full bg-black border border-zinc-800 rounded p-3 text-sm text-zinc-100 placeholder-zinc-600 focus:outline-none focus:border-zinc-500 font-mono resize-none"
              />

              {/* Presets and Controls */}
              <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="text-[11px] text-zinc-500 mr-1">Presets:</span>
                  {presets.map((p, idx) => (
                    <button
                      key={idx}
                      onClick={() => setTask(p.prompt)}
                      className="text-[11px] px-2 py-0.5 rounded border border-zinc-800 bg-zinc-900/60 text-zinc-300 hover:border-zinc-600 transition-colors"
                    >
                      {p.label}
                    </button>
                  ))}
                </div>

                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setShowModelConfig(!showModelConfig)}
                    className="text-xs px-3 py-1.5 rounded border border-zinc-800 bg-zinc-900 text-zinc-300 hover:bg-zinc-800 flex items-center gap-1.5"
                  >
                    <Sliders className="w-3.5 h-3.5" />
                    <span>Models ({numAgents + 1})</span>
                  </button>

                  <button
                    onClick={handleRun}
                    disabled={isExecuting || !task.trim()}
                    className="px-5 py-1.5 rounded bg-white text-black font-semibold text-xs hover:bg-zinc-200 transition-colors flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {isExecuting ? (
                      <>
                        <RotateCw className="w-3.5 h-3.5 animate-spin" />
                        <span>Orchestrating Live...</span>
                      </>
                    ) : (
                      <>
                        <Play className="w-3.5 h-3.5 fill-current" />
                        <span>Run Build</span>
                      </>
                    )}
                  </button>
                </div>
              </div>

              {/* Collapsible Model Assignment Grid */}
              {showModelConfig && (
                <div role="dialog" aria-label="Model assignments" className="fixed right-0 top-14 bottom-0 z-50 w-full sm:w-96 bg-zinc-950 border border-zinc-800 p-6 overflow-y-auto flex flex-col gap-5 font-mono text-xs shadow-2xl">
                  <div className="flex justify-between items-center"><strong>OpenRouter free models</strong><button onClick={() => setShowModelConfig(false)} aria-label="Close model drawer">Close ×</button></div>
                  <p className="text-zinc-400">Choose a planning model and a model for each worker. HPC models are not connected.</p>
                  <div className="space-y-1">
                    <label className="text-[11px] text-zinc-400">Helm Orchestrator</label>
                    <select
                      value={helmModel}
                      onChange={(e) => setHelmModel(e.target.value)}
                      className="w-full bg-black border border-zinc-800 rounded p-1.5 text-xs text-zinc-200 focus:outline-none"
                    >
                      {freeModels.map(m => (
                        <option key={m.id} value={m.id}>{m.name}</option>
                      ))}
                    </select>
                  </div>

                  {Array.from({ length: numAgents }).map((_, i) => {
                    const aid = `agent_${i + 1}`
                    return (
                      <div key={aid} className="space-y-1">
                        <label className="text-[11px] text-zinc-400">Worker {i + 1}</label>
                        <select
                          value={workerModels[aid] || freeModels[i % freeModels.length]?.id || 'openrouter/free'}
                          onChange={(e) => setWorkerModels({ ...workerModels, [aid]: e.target.value })}
                          className="w-full bg-black border border-zinc-800 rounded p-1.5 text-xs text-zinc-200 focus:outline-none"
                        >
                          {freeModels.map(m => (
                            <option key={m.id} value={m.id}>{m.name}</option>
                          ))}
                        </select>
                      </div>
                    )
                  })}
                </div>
              )}
            </div>

            {/* Execution Flow Pipeline */}
            {(isExecuting || telemetryState.status !== 'idle') && (
              <div className="border border-zinc-800 bg-zinc-950 rounded-lg p-4 font-mono text-xs space-y-3">
                <div className="flex items-center justify-between text-zinc-400 border-b border-zinc-800 pb-2">
                  <span className="flex items-center gap-2">
                    <Zap className="w-3.5 h-3.5 text-white" />
                    <span>STATUS: {telemetryState.status.toUpperCase()}</span>
                  </span>
                  {telemetryState.wallTime && (
                    <span>
                      TIME: {telemetryState.wallTime}s | SPEEDUP: {telemetryState.speedup}x
                    </span>
                  )}
                </div>

                {telemetryState.planSummary && (
                  <p className="text-zinc-300 text-[11px]">
                    <span className="text-zinc-500">[Plan]:</span> {telemetryState.planSummary}
                  </p>
                )}

                {/* Agent status chips */}
                <div className="flex flex-wrap gap-2 pt-1">
                  {Array.from({ length: numAgents }).map((_, i) => {
                    const aid = `agent_${i + 1}`
                    const info = telemetryState.workers[aid]
                    return (
                      <div
                        key={aid}
                        className="flex items-center gap-2 px-2.5 py-1 rounded border border-zinc-800 bg-black text-[11px]"
                      >
                        <span className={`w-1.5 h-1.5 rounded-full ${
                          info?.status === 'done' ? 'bg-white' : (info?.status === 'running' ? 'bg-zinc-400 animate-ping' : 'bg-zinc-700')
                        }`} />
                        <span className="font-semibold text-zinc-200">Agent {i + 1}</span>
                        {info?.filename && <span className="text-zinc-400">{info.filename}</span>}
                        {info?.duration_s && <span className="text-zinc-500 font-mono">({info.duration_s}s)</span>}
                      </div>
                    )
                  })}
                </div>
              </div>
            )}

            {/* Code Inspector & Live Preview Split */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 h-[650px]">
              
              {/* Left Column: Code Viewer */}
              <div className="border border-zinc-800 bg-zinc-950 rounded-lg flex flex-col overflow-hidden">
                {/* File Tabs */}
                <div className="border-b border-zinc-800 bg-black px-3 py-2 flex items-center justify-between">
                  <div className="flex items-center gap-1 overflow-x-auto">
                    {Object.keys(files).length === 0 ? (
                      <span className="text-xs text-zinc-500 font-mono">No files generated yet</span>
                    ) : (
                      Object.keys(files).map((fname) => (
                        <button
                          key={fname}
                          onClick={() => setSelectedFile(fname)}
                          className={`px-2.5 py-1 rounded text-xs font-mono transition-colors ${
                            selectedFile === fname
                              ? 'bg-zinc-800 text-white font-medium border border-zinc-700'
                              : 'text-zinc-400 hover:text-white'
                          }`}
                        >
                          {fname}
                        </button>
                      ))
                    )}
                  </div>
                  {selectedFile && files[selectedFile] && (
                    <button
                      onClick={copyToClipboard}
                      className="p-1 rounded text-zinc-400 hover:text-white transition-colors"
                      title="Copy code"
                    >
                      {copied ? <Check className="w-3.5 h-3.5 text-white" /> : <Copy className="w-3.5 h-3.5" />}
                    </button>
                  )}
                </div>

                {/* Code Content */}
                <div className="flex-1 p-4 bg-black overflow-auto font-mono text-xs text-zinc-300">
                  {selectedFile && files[selectedFile] ? (
                    <pre className="whitespace-pre-wrap break-words leading-6 tab-size-2">{files[selectedFile]}</pre>
                  ) : (
                    <div className="h-full flex items-center justify-center text-zinc-600 font-mono text-xs">
                      Run an orchestration task to view source code output.
                    </div>
                  )}
                </div>
              </div>

              {/* Right Column: Live Sandboxed Iframe */}
              <div className="border border-zinc-800 bg-zinc-950 rounded-lg flex flex-col overflow-hidden">
                <div className="border-b border-zinc-800 bg-black px-3 py-2 flex items-center justify-between font-mono text-xs text-zinc-400">
                  <span className="flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-zinc-600"></span>
                    <span>PREVIEW: {currentRunId || 'standby'}</span>
                  </span>
                  {htmlFileExists && currentRunId && (
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => {
                          if (iframeRef.current) iframeRef.current.src = iframeRef.current.src
                        }}
                        className="hover:text-white"
                        title="Reload"
                      >
                        <RotateCw className="w-3.5 h-3.5" />
                      </button>
                      <a
                        href={`/api/preview/${currentRunId}/index.html`}
                        target="_blank"
                        rel="noreferrer"
                        className="hover:text-white"
                        title="Open in new window"
                      >
                        <ExternalLink className="w-3.5 h-3.5" />
                      </a>
                    </div>
                  )}
                </div>

                <div className="flex-1 bg-white relative">
                  {htmlFileExists && currentRunId ? (
                    <iframe
                      ref={iframeRef}
                      src={`/api/preview/${currentRunId}/index.html`}
                      className="w-full h-full border-0"
                      sandbox="allow-scripts"
                    />
                  ) : (
                    <div className="absolute inset-0 bg-black flex flex-col items-center justify-center text-center p-6 text-zinc-500 font-mono text-xs">
                      <Cpu className="w-8 h-8 mb-2 text-zinc-700" />
                      <span>Live interactive sandbox standby.</span>
                      <span className="text-[11px] text-zinc-600 mt-1">Generated web applications will render here.</span>
                    </div>
                  )}
                </div>
              </div>

            </div>

          </div>
        )}

        {/* ========================================================= */}
        {/* TAB 2: HISTORY & USAGE BILL */}
        {/* ========================================================= */}
        {activeTab === 'history' && (
          <div className="space-y-6 font-mono">
            
            {/* Header & Refresh */}
            <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
              <div>
                <h2 className="text-sm font-semibold text-white uppercase tracking-wider">
                  Usage & Billing Breakdown
                </h2>
                <p className="text-xs text-zinc-500">Historical runs, token consumption, and cost telemetry. Runs marked [SAMPLE] contain synthetic usage, not live measurements.</p>
              </div>
              <button
                onClick={loadBillingAndHistory}
                className="text-xs px-3 py-1.5 rounded border border-zinc-800 bg-zinc-900 text-zinc-300 hover:text-white flex items-center gap-1.5"
              >
                <RotateCw className="w-3.5 h-3.5" />
                <span>Refresh</span>
              </button>
            </div>

            {/* Billing Summary Cards */}
            {billingData?.summary && (
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <div className="border border-zinc-800 bg-zinc-950 p-4 rounded-lg">
                  <span className="text-[11px] text-zinc-500 uppercase">Total Bill (USD)</span>
                  <p className="text-xl font-bold text-white mt-1">
                    ${billingData.summary.total_cost_usd.toFixed(4)}
                  </p>
                  <span className="text-[10px] text-zinc-500">Free tier rate: $0.00</span>
                </div>

                <div className="border border-zinc-800 bg-zinc-950 p-4 rounded-lg">
                  <span className="text-[11px] text-zinc-500 uppercase">Total Tokens</span>
                  <p className="text-xl font-bold text-white mt-1">
                    {billingData.summary.total_tokens.toLocaleString()}
                  </p>
                  <span className="text-[10px] text-zinc-500">In: {billingData.summary.total_tokens_in} | Out: {billingData.summary.total_tokens_out}</span>
                </div>

                <div className="border border-zinc-800 bg-zinc-950 p-4 rounded-lg">
                  <span className="text-[11px] text-zinc-500 uppercase">Commercial Equivalent</span>
                  <p className="text-xl font-bold text-white mt-1">
                    ${billingData.summary.free_tier_savings_usd.toFixed(3)}
                  </p>
                  <span className="text-[10px] text-zinc-500">Est. API value saved</span>
                </div>

                <div className="border border-zinc-800 bg-zinc-950 p-4 rounded-lg">
                  <span className="text-[11px] text-zinc-500 uppercase">Total Completed Runs</span>
                  <p className="text-xl font-bold text-white mt-1">
                    {billingData.summary.total_runs}
                  </p>
                  <span className="text-[10px] text-zinc-500">Archived in SQLite</span>
                </div>
              </div>
            )}

            {/* Per-Model Usage Breakdown */}
            {billingData?.model_breakdown && billingData.model_breakdown.length > 0 && (
              <div className="border border-zinc-800 bg-zinc-950 rounded-lg overflow-hidden">
                <div className="bg-black px-4 py-2.5 border-b border-zinc-800 text-xs font-semibold text-zinc-300">
                  Model Usage Distribution
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead className="text-zinc-500 border-b border-zinc-800 bg-zinc-950">
                      <tr>
                        <th className="p-3 font-medium">Model</th>
                        <th className="p-3 font-medium text-center">Invocations</th>
                        <th className="p-3 font-medium text-right">Tokens In</th>
                        <th className="p-3 font-medium text-right">Tokens Out</th>
                        <th className="p-3 font-medium text-right">Total Tokens</th>
                        <th className="p-3 font-medium text-right">Billed ($)</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-zinc-850 text-zinc-300">
                      {billingData.model_breakdown.map((m, idx) => (
                        <tr key={idx} className="hover:bg-zinc-900/40">
                          <td className="p-3 text-white font-medium">{m.model}</td>
                          <td className="p-3 text-center text-zinc-400">{m.calls}</td>
                          <td className="p-3 text-right text-zinc-400">{m.tokens_in.toLocaleString()}</td>
                          <td className="p-3 text-right text-zinc-400">{m.tokens_out.toLocaleString()}</td>
                          <td className="p-3 text-right text-white font-semibold">{m.total_tokens.toLocaleString()}</td>
                          <td className="p-3 text-right text-zinc-300">${m.estimated_cost.toFixed(4)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Runs Ledger */}
            <div className="border border-zinc-800 bg-zinc-950 rounded-lg overflow-hidden">
              <div className="bg-black px-4 py-2.5 border-b border-zinc-800 text-xs font-semibold text-zinc-300">
                Detailed Run Ledger
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="text-zinc-500 border-b border-zinc-800 bg-zinc-950">
                    <tr>
                      <th className="p-3 font-medium">Run ID</th>
                      <th className="p-3 font-medium">Task</th>
                      <th className="p-3 font-medium text-center">Agents</th>
                      <th className="p-3 font-medium text-right">Wall Time</th>
                      <th className="p-3 font-medium text-right">Speedup</th>
                      <th className="p-3 font-medium text-right">Tokens</th>
                      <th className="p-3 font-medium text-right">Bill</th>
                      <th className="p-3 font-medium text-center">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-zinc-850 text-zinc-300">
                    {billingData?.runs?.length > 0 ? (
                      billingData.runs.map((r) => (
                        <tr key={r.run_id} className="hover:bg-zinc-900/40">
                          <td className="p-3 text-zinc-500">{r.run_id}</td>
                          <td className="p-3 text-white max-w-xs truncate" title={r.task}>{r.task}</td>
                          <td className="p-3 text-center text-zinc-400">{r.num_agents || 3}</td>
                          <td className="p-3 text-right text-zinc-300">{r.wall_clock_s}s</td>
                          <td className="p-3 text-right text-zinc-300 font-semibold">{r.concurrency_speedup}x</td>
                          <td className="p-3 text-right text-zinc-400">{(r.total_tokens_in + r.total_tokens_out).toLocaleString()}</td>
                          <td className="p-3 text-right text-zinc-400">${r.total_cost_est.toFixed(4)}</td>
                          <td className="p-3 text-center">
                            <button
                              onClick={() => {
                                setTask(r.task)
                                setNumAgents(r.num_agents)
                                setCurrentRunId(r.run_id)
                                loadRunFiles(r.run_id)
                                setActiveTab('main')
                              }}
                              className="px-2 py-1 rounded bg-zinc-900 border border-zinc-800 hover:bg-white hover:text-black text-zinc-300 text-[11px] transition-colors"
                            >
                              Inspect
                            </button>
                          </td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan={8} className="p-6 text-center text-zinc-500">
                          {historyLoading ? 'Loading runs...' : 'No historical runs found.'}
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>

          </div>
        )}

      </main>

      {/* API Key Modal */}
      {showKeyModal && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-zinc-950 border border-zinc-800 rounded-lg max-w-md w-full p-5 space-y-4 font-mono">
            <div className="flex items-center justify-between border-b border-zinc-800 pb-2">
              <span className="text-xs font-semibold text-white">OPENROUTER API KEY</span>
              {true && (
                <button onClick={() => setShowKeyModal(false)} className="text-zinc-500 hover:text-white">&times;</button>
              )}
            </div>
            
            <p className="text-[11px] text-zinc-400 font-sans">
              Enter your OpenRouter API key to execute tasks with free models. The key is saved in this browser and sent to your local server for OpenRouter requests.
            </p>

            <div className="space-y-1.5">
              <label className="text-[11px] text-zinc-400">API Key:</label>
              <input
                type="password"
                value={tempApiKey}
                onChange={(e) => setTempApiKey(e.target.value)}
                placeholder="sk-or-v1-..."
                className="w-full bg-black border border-zinc-800 rounded p-2 text-xs text-white focus:outline-none focus:border-zinc-500"
              />
            </div>

            <div className="flex justify-end gap-2 pt-2">
              <button
                onClick={() => {
                  const cleaned = tempApiKey.trim()
                  if (!cleaned) {
                    alert('Please enter a valid API key.')
                    return
                  }
                  setApiKey(cleaned)
                  localStorage.setItem('openrouter_api_key', cleaned)
                  setShowKeyModal(false)
                }}
                className="px-4 py-1.5 rounded bg-white text-black text-xs font-semibold hover:bg-zinc-200 transition-colors"
              >
                Save Key
              </button>
            </div>
          </div>
        </div>
      )}

    </div>
  )
}
