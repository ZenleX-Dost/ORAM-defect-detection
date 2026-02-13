/**
 * ORAM – Intelligent Inspection Platform
 * Frontend Application Logic
 *
 * Handles: camera connection, WebSocket streaming, image upload,
 * detection rendering, training management, and UI state.
 */

// 
// State
// 

// const API = window.location.origin; // Moved to config.js
let ws = null;
let isStreaming = false;
let isConnected = false;

// Stats
let stats = {
    framesAnalyzed: 0,
    totalDetections: 0,
    latencies: [],
    confidences: [],
};

// Detection history
let detectionHistory = [];

// Canvas context
const canvas = document.getElementById('videoCanvas');
const ctx = canvas.getContext('2d');

// 
// Initialization
// 

document.addEventListener('DOMContentLoaded', () => {
    checkServerStatus();
    setInterval(checkServerStatus, 10000);
});

async function checkServerStatus() {
    try {
        const res = await fetch(`${API}/api/status`);
        const data = await res.json();

        if (data.status === 'ok') {
            setStatus('online', 'Online');
            document.getElementById('modelStatus').textContent =
                data.models.pytorch_available ? 'PyTorch ' : 'Simulation';

            if (data.models.sam2_available) {
                document.getElementById('samBadge').style.display = 'flex';
            }

            if (data.camera.connected) {
                setConnected(true);
            }
        }
    } catch {
        setStatus('offline', 'Offline');
        document.getElementById('modelStatus').textContent = 'Unreachable';
    }
}

// 
// Camera Connection
// 

async function connectCamera() {
    const url = document.getElementById('cameraUrl').value.trim();
    const protocol = document.getElementById('cameraProtocol').value;

    if (!url) {
        showToast('Please enter a camera URL or IP', 'error');
        return;
    }

    setStatus('connecting', 'Connecting...');

    try {
        const res = await fetch(`${API}/api/camera/connect`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url, protocol }),
        });

        const data = await res.json();

        if (res.ok) {
            setConnected(true);
            showToast(`Connected: ${data.protocol} (${data.frame_size[0]}×${data.frame_size[1]})`, 'success');
        } else {
            setConnected(false);
            showToast(data.detail || 'Connection failed', 'error');
        }
    } catch (err) {
        setConnected(false);
        showToast(`Connection error: ${err.message}`, 'error');
    }
}

async function disconnectCamera() {
    try {
        await fetch(`${API}/api/camera/disconnect`, { method: 'POST' });
        setConnected(false);
        stopStream();
        showToast('Camera disconnected', 'info');
    } catch {
        showToast('Disconnect failed', 'error');
    }
}

function setConnected(connected) {
    isConnected = connected;
    document.getElementById('btnConnect').style.display = connected ? 'none' : '';
    document.getElementById('btnDisconnect').style.display = connected ? '' : 'none';
    document.getElementById('btnStartStream').disabled = !connected;

    if (connected) {
        setStatus('online', 'Connected');
    } else {
        setStatus('offline', 'Disconnected');
        document.getElementById('videoPlaceholder').style.display = '';
    }
}

// 
// WebSocket Streaming
// 

function startStream() {
    if (isStreaming) return;

    const wsUrl = `${API.replace('http', 'ws')}/api/stream`;
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        const settings = getSettings();
        ws.send(JSON.stringify({
            action: 'start',
            threshold: settings.threshold,
            use_sam2: settings.useSam2,
            preprocess: settings.preprocess,
        }));

        isStreaming = true;
        document.getElementById('btnStartStream').style.display = 'none';
        document.getElementById('btnStopStream').style.display = '';
        document.getElementById('liveBadge').style.display = '';
        document.getElementById('videoPlaceholder').style.display = 'none';

        showToast('Live inspection started', 'success');
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.annotated_image) {
            // Render annotated frame
            renderBase64Image(data.annotated_image);
            updateStats(data);
            renderDetections(data.detections || []);
        } else if (data.raw_image) {
            renderBase64Image(data.raw_image);
        }

        if (data.processing_time_ms) {
            document.getElementById('latencyBadge').textContent =
                `${Math.round(data.processing_time_ms)} ms`;
        }
    };

    ws.onerror = (err) => {
        showToast('Stream error', 'error');
        console.error('WebSocket error:', err);
    };

    ws.onclose = () => {
        isStreaming = false;
        document.getElementById('btnStartStream').style.display = '';
        document.getElementById('btnStopStream').style.display = 'none';
        document.getElementById('liveBadge').style.display = 'none';
    };
}

function stopStream() {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ action: 'stop' }));
    }
    if (ws) ws.close();
    ws = null;
    isStreaming = false;

    document.getElementById('btnStartStream').style.display = '';
    document.getElementById('btnStopStream').style.display = 'none';
    document.getElementById('liveBadge').style.display = 'none';

    showToast('Inspection stopped', 'info');
}

// 
// Image Upload / Drop
// 

async function uploadImage(event) {
    const file = event.target.files[0];
    if (!file) return;
    await analyzeFile(file);
}

function handleDrop(event) {
    event.preventDefault();
    event.target.classList.remove('dragover');
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        analyzeFile(file);
    }
}

async function analyzeFile(file) {
    const settings = getSettings();
    const formData = new FormData();
    formData.append('file', file);

    showToast('Analyzing image...', 'info');
    document.getElementById('videoPlaceholder').style.display = 'none';

    try {
        const url = new URL(`${API}/api/analyze`);
        url.searchParams.set('threshold', settings.threshold);
        url.searchParams.set('use_sam2', settings.useSam2);
        url.searchParams.set('preprocess', settings.preprocess);

        const res = await fetch(url, { method: 'POST', body: formData });
        const data = await res.json();

        if (data.annotated_image) {
            renderBase64Image(data.annotated_image);
        }

        updateStats(data);
        renderDetections(data.detections || []);

        if (data.has_anomalies) {
            showToast(` ${data.total_detections} anomalies detected!`, 'error');
        } else {
            showToast(' No anomalies detected', 'success');
        }
    } catch (err) {
        showToast(`Analysis failed: ${err.message}`, 'error');
    }
}

async function captureSnapshot() {
    if (!isConnected) {
        showToast('No camera connected', 'error');
        return;
    }

    try {
        const settings = getSettings();
        const url = new URL(`${API}/api/camera/snapshot`);
        url.searchParams.set('threshold', settings.threshold);
        url.searchParams.set('use_sam2', settings.useSam2);
        url.searchParams.set('preprocess', settings.preprocess);

        const res = await fetch(url);
        const data = await res.json();

        if (data.annotated_image) {
            renderBase64Image(data.annotated_image);
            document.getElementById('videoPlaceholder').style.display = 'none';
        }

        updateStats(data);
        renderDetections(data.detections || []);
        showToast('Snapshot captured', 'success');
    } catch (err) {
        showToast(`Snapshot failed: ${err.message}`, 'error');
    }
}

// 
// Rendering
// 

function renderBase64Image(base64) {
    const img = new Image();
    img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
    };
    img.src = `data:image/jpeg;base64,${base64}`;
}

const SEVERITY_COLORS = {
    critical: '#ef4444',
    high: '#f97316',
    medium: '#eab308',
    low: '#22c55e',
};

function renderDetections(detections) {
    const list = document.getElementById('detectionList');
    const empty = document.getElementById('emptyDetections');

    if (!detections || detections.length === 0) {
        list.innerHTML = '';
        list.appendChild(empty);
        empty.style.display = '';
        return;
    }

    empty.style.display = 'none';
    list.innerHTML = '';

    detections.forEach((det, i) => {
        const severity = det.severity || 'medium';
        const confidence = det.confidence || 0;
        const color = SEVERITY_COLORS[severity] || SEVERITY_COLORS.medium;

        const card = document.createElement('div');
        card.className = `detection-card severity-${severity}`;
        card.innerHTML = `
            <div class="det-header">
                <span class="det-type">${det.anomaly_type || 'Unknown'}</span>
                <span class="det-confidence" style="color:${color}">
                    ${(confidence * 100).toFixed(1)}%
                </span>
            </div>
            <div class="det-meta">
                <span>Severity: ${severity}</span>
                ${det.bbox ? `<span>Pos: (${det.bbox.x}, ${det.bbox.y})</span>` : ''}
            </div>
            <div class="confidence-bar">
                <div class="confidence-bar-fill" style="width:${confidence * 100}%; background:${color};"></div>
            </div>
        `;
        list.appendChild(card);

        // Add to history
        detectionHistory.push({
            ...det,
            timestamp: new Date().toLocaleTimeString(),
        });
    });

    // Update history tab
    updateHistory();
}

function updateHistory() {
    const histList = document.getElementById('historyList');
    const recent = detectionHistory.slice(-20).reverse();

    if (recent.length === 0) return;

    histList.innerHTML = '';
    recent.forEach(det => {
        const severity = det.severity || 'medium';
        const color = SEVERITY_COLORS[severity] || SEVERITY_COLORS.medium;
        const card = document.createElement('div');
        card.className = `detection-card severity-${severity}`;
        card.innerHTML = `
            <div class="det-header">
                <span class="det-type">${det.anomaly_type || 'Unknown'}</span>
                <span class="det-confidence" style="color:${color}">
                    ${((det.confidence || 0) * 100).toFixed(1)}%
                </span>
            </div>
            <div class="det-meta">
                <span>${det.timestamp}</span>
                <span>Severity: ${severity}</span>
            </div>
        `;
        histList.appendChild(card);
    });
}

function clearDetections() {
    detectionHistory = [];
    document.getElementById('detectionList').innerHTML =
        '<div class="empty-state" id="emptyDetections"><div class="icon"></div><p>No anomalies detected</p></div>';
    document.getElementById('historyList').innerHTML =
        '<div class="empty-state"><div class="icon"></div><p>Detection history will appear here</p></div>';
}

// 
// Stats
// 

function updateStats(data) {
    stats.framesAnalyzed++;
    stats.totalDetections += (data.total_detections || 0);

    if (data.processing_time_ms) {
        stats.latencies.push(data.processing_time_ms);
        // Keep last 50
        if (stats.latencies.length > 50) stats.latencies.shift();
    }

    if (data.confidence_scores) {
        const values = Object.values(data.confidence_scores);
        if (values.length > 0) {
            stats.confidences.push(Math.max(...values));
            if (stats.confidences.length > 50) stats.confidences.shift();
        }
    }

    document.getElementById('statFrames').textContent = stats.framesAnalyzed;
    document.getElementById('statDetections').textContent = stats.totalDetections;

    if (stats.latencies.length > 0) {
        const avgLat = stats.latencies.reduce((a, b) => a + b, 0) / stats.latencies.length;
        document.getElementById('statLatency').textContent = `${Math.round(avgLat)}ms`;
    }

    if (stats.confidences.length > 0) {
        const avgConf = stats.confidences.reduce((a, b) => a + b, 0) / stats.confidences.length;
        document.getElementById('statAccuracy').textContent = `${(avgConf * 100).toFixed(0)}%`;
    }

    // FPS estimate
    if (stats.latencies.length >= 2) {
        const avgMs = stats.latencies.reduce((a, b) => a + b, 0) / stats.latencies.length;
        const fps = Math.min(30, Math.round(1000 / avgMs));
        document.getElementById('fpsBadge').textContent = `${fps} FPS`;
    }
}

// 
// Training
// 

let trainingJobId = null;
let trainingInterval = null;

async function startTraining() {
    const agent = document.getElementById('trainAgent').value;
    const dataset = document.getElementById('trainDataset').value;
    const epochs = parseInt(document.getElementById('trainEpochs').value);

    try {
        const res = await fetch(`${API}/api/train`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                agent,
                dataset_key: dataset,
                epochs,
                model_name: 'efficientnet_b0',
                batch_size: 32,
                learning_rate: 0.001,
            }),
        });

        let data;
        const text = await res.text();
        try {
            data = JSON.parse(text);
        } catch (e) {
            throw new Error(`Server returned non-JSON: ${text.substring(0, 100)}...`);
        }

        if (data.job_id) {
            trainingJobId = data.job_id;
            document.getElementById('trainingProgress').style.display = '';
            document.getElementById('trainStatusText').textContent = 'Training started...';

            // Poll for progress
            trainingInterval = setInterval(pollTrainingProgress, 2000);
            showToast('Training job started', 'success');
        } else if (data.status === 'completed (simulated)') {
            showToast('Training completed (simulated)', 'success');
            displayTrainingResult(data.metrics);
        } else {
            showToast('Training start failed', 'error');
        }
    } catch (err) {
        showToast(`Training error: ${err.message}`, 'error');
        console.error("Full training error:", err);
    }
}

async function pollTrainingProgress() {
    if (!trainingJobId) return;

    try {
        const res = await fetch(`${API}/api/train/${trainingJobId}`);
        const data = await res.json();

        if (data.status === 'running') {
            const pct = data.total_epochs > 0
                ? Math.round((data.current_epoch / data.total_epochs) * 100) : 0;
            document.getElementById('trainProgressPct').textContent = `${pct}%`;
            document.getElementById('trainProgressBar').style.width = `${pct}%`;
            document.getElementById('trainStatusText').textContent =
                `${data.current_agent || 'Agent'} — Epoch ${data.current_epoch}/${data.total_epochs}`;
        } else if (data.status === 'completed') {
            clearInterval(trainingInterval);
            document.getElementById('trainProgressPct').textContent = '100%';
            document.getElementById('trainProgressBar').style.width = '100%';
            document.getElementById('trainStatusText').textContent = 'Completed ';
            showToast('Training completed successfully!', 'success');

            if (data.metrics) displayTrainingResult(data.metrics);
        } else if (data.status === 'failed') {
            clearInterval(trainingInterval);
            document.getElementById('trainStatusText').textContent = 'Failed ';
            showToast(`Training failed: ${data.error}`, 'error');
        }
    } catch {
        // Server might be busy
    }
}

function displayTrainingResult(metrics) {
    const container = document.getElementById('trainMetrics');
    container.innerHTML = '';

    if (Array.isArray(metrics)) {
        metrics.forEach(m => {
            container.innerHTML += `
                <div class="metric-item">
                    <div class="metric-label">${m.agent || 'Agent'}</div>
                    <div class="metric-value">${m.best_val_acc || m.status || '--'}%</div>
                </div>
            `;
        });
    } else if (typeof metrics === 'object') {
        Object.entries(metrics).forEach(([key, val]) => {
            const acc = val.best_val_acc || '--';
            container.innerHTML += `
                <div class="metric-item">
                    <div class="metric-label">${key}</div>
                    <div class="metric-value">${acc}%</div>
                </div>
            `;
        });
    }
}

// 
// UI Helpers
// 

function getSettings() {
    return {
        threshold: parseInt(document.getElementById('threshold').value) / 100,
        useSam2: document.getElementById('toggleSam2').checked,
        preprocess: document.getElementById('togglePreprocess').checked,
    };
}

function updateThreshold(val) {
    document.getElementById('thresholdValue').textContent = `${val}%`;
}

function setStatus(state, text) {
    const dot = document.getElementById('statusDot');
    dot.className = `status-dot ${state}`;
    document.getElementById('statusText').textContent = text;
}

function switchTab(name) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));

    event.target.classList.add('active');
    document.getElementById(`tab-${name}`).classList.add('active');
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
        if (toast.parentNode) toast.remove();
    }, 3500);
}
