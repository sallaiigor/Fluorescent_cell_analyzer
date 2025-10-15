document.addEventListener('DOMContentLoaded', () => {
    // HTML element selectors
    const imageUpload = document.getElementById('imageUpload');
    const uploadInfo = document.getElementById('uploadInfo');
    const paramsContainer = document.getElementById('paramsContainer');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const segChannelSelect = document.getElementById('segmentationChannelSelect');
    const minAreaInput = document.getElementById('minArea');
    const maxAreaInput = document.getElementById('maxArea');
    const minIntensityInput = document.getElementById('minIntensity');
    const maxIntensityInput = document.getElementById('maxIntensity');
    const progressContainer = document.getElementById('progressContainer');
    const statusText = document.getElementById('statusText');
    const progressBar = document.getElementById('progressBar');
    const elapsedTime = document.getElementById('elapsedTime');
    const estimatedTime = document.getElementById('estimatedTime');
    const resultsDiv = document.getElementById('results');
    const resultsGrid = document.getElementById('resultsGrid');
    const downloadCsvBtn = document.getElementById('downloadCsvBtn');
    const downloadPlotsBtn = document.getElementById('downloadPlotsBtn');
    const analysisModeSelect = document.getElementById('analysisMode');
    const quantifyOptions = document.getElementById('quantify-options');
    const compMatrixContainer = document.getElementById('compMatrixContainer');
    
    // Global state variables
    let files = [];
    let timerInterval = null;
    let bgCoords = { x1: 0, y1: 0, x2: 0, y2: 0 };
    let currentTool = 'rectangle';
    let isDrawing = false;
    let startPos = {};
    let endPos = {};
    let lassoPoints = [];

    // --- 1. UI Logic ---
    analysisModeSelect.addEventListener('change', () => {
        quantifyOptions.style.display = (analysisModeSelect.value === 'quantify') ? 'block' : 'none';
    });
    // Trigger change on load to set initial state
    analysisModeSelect.dispatchEvent(new Event('change'));


    imageUpload.addEventListener('change', (event) => {
        files = Array.from(event.target.files);
        uploadInfo.innerHTML = '';
        segChannelSelect.innerHTML = '';
        
        if (files.length === 0) {
            paramsContainer.classList.add('hidden');
            analyzeBtn.disabled = true;
            return;
        }

        const channels = {};
        files.forEach(file => {
            const pathParts = file.webkitRelativePath.split('/');
            if (pathParts.length > 1) {
                const channelName = pathParts[pathParts.length - 2];
                if (!channels[channelName]) channels[channelName] = 0;
                channels[channelName]++;
            }
        });

        const channelNames = Object.keys(channels);
        if (channelNames.length > 0) {
            let infoText = `<strong>Detected ${channelNames.length} channels:</strong><ul>`;
            channelNames.forEach(name => {
                infoText += `<li>${name} (${channels[name]} tiles)</li>`;
                const option = document.createElement('option');
                option.value = name;
                option.textContent = name;
                segChannelSelect.appendChild(option);
            });
            infoText += "</ul>";
            uploadInfo.innerHTML = infoText;
            paramsContainer.classList.remove('hidden');
            analyzeBtn.disabled = false;
            setupCompMatrix(channelNames);
        } else {
            uploadInfo.innerHTML = "<p style='color: red;'>Error: No channel subfolders detected. Please select the parent folder.</p>";
            analyzeBtn.disabled = true;
        }
    });

    // --- 2. Background and UI Helpers ---
    function setupBackgroundSelection(canvas, ctx, originalImage) {
        const scaleX = canvas.width / canvas.clientWidth;
        const scaleY = canvas.height / canvas.clientHeight;

        const redraw = () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(originalImage, 0, 0);
            if ((!isDrawing && !Object.keys(startPos).length) && lassoPoints.length === 0) return;
            ctx.strokeStyle = 'white'; ctx.lineWidth = 2 * scaleX; ctx.setLineDash([5 * scaleX, 5 * scaleX]); ctx.beginPath();
            if (currentTool === 'rectangle') {
                ctx.rect(startPos.x, startPos.y, endPos.x - startPos.x, endPos.y - startPos.y);
            } else if (currentTool === 'circle') {
                const radius = Math.sqrt(Math.pow(endPos.x - startPos.x, 2) + Math.pow(endPos.y - startPos.y, 2));
                ctx.arc(startPos.x, startPos.y, radius, 0, Math.PI * 2);
            } else if (currentTool === 'lasso' && lassoPoints.length > 1) {
                ctx.moveTo(lassoPoints[0].x, lassoPoints[0].y);
                lassoPoints.forEach(p => ctx.lineTo(p.x, p.y));
                if (!isDrawing) ctx.closePath();
            }
            ctx.stroke(); ctx.setLineDash([]);
        };
        canvas.addEventListener('mousedown', (e) => {
            isDrawing = true; startPos = { x: e.offsetX * scaleX, y: e.offsetY * scaleY };
            endPos = { ...startPos }; if (currentTool === 'lasso') lassoPoints = [startPos];
        });
        canvas.addEventListener('mousemove', (e) => {
            if (!isDrawing) return; endPos = { x: e.offsetX * scaleX, y: e.offsetY * scaleY };
            if (currentTool === 'lasso') lassoPoints.push(endPos);
            redraw();
        });
        canvas.addEventListener('mouseup', (e) => {
            isDrawing = false; endPos = { x: e.offsetX * scaleX, y: e.offsetY * scaleY }; redraw();
            if (currentTool === 'rectangle') {
                bgCoords = { x1: Math.min(startPos.x, endPos.x), y1: Math.min(startPos.y, endPos.y), x2: Math.max(startPos.x, endPos.x), y2: Math.max(startPos.y, endPos.y) };
            } else if (currentTool === 'circle') {
                const radius = Math.sqrt(Math.pow(endPos.x - startPos.x, 2) + Math.pow(endPos.y - startPos.y, 2));
                bgCoords = { x1: startPos.x - radius, y1: startPos.y - radius, x2: startPos.x + radius, y2: startPos.y + radius };
            } else if (currentTool === 'lasso' && lassoPoints.length > 1) {
                const xCoords = lassoPoints.map(p => p.x); const yCoords = lassoPoints.map(p => p.y);
                bgCoords = { x1: Math.min(...xCoords), y1: Math.min(...yCoords), x2: Math.max(...xCoords), y2: Math.max(...yCoords) };
            }
            startPos = {}; endPos = {};
        });
    }

    function setupCompMatrix(channelNames) {
        compMatrixContainer.innerHTML = '';
        compMatrixContainer.style.gridTemplateColumns = `auto repeat(${channelNames.length}, 1fr)`;
        
        compMatrixContainer.appendChild(document.createElement('div'));
        channelNames.forEach(name => {
            const label = document.createElement('div'); label.textContent = name; label.className = 'comp-label';
            compMatrixContainer.appendChild(label);
        });

        channelNames.forEach((rowName, r) => {
            const rowLabel = document.createElement('div'); rowLabel.textContent = rowName; rowLabel.className = 'comp-label';
            compMatrixContainer.appendChild(rowLabel);
            channelNames.forEach((colName, c) => {
                const input = document.createElement('input');
                input.type = 'number'; input.className = 'comp-input'; input.id = `comp_${r}_${c}`;
                input.value = (r === c) ? '1.0' : '0.0'; input.step = '0.01';
                compMatrixContainer.appendChild(input);
            });
        });
    }
    
    function startTimer() { let seconds = 0; elapsedTime.textContent = '0s'; timerInterval = setInterval(() => { seconds++; elapsedTime.textContent = `${seconds}s`; }, 1000); }
    function stopTimer() { clearInterval(timerInterval); }
    function updateProgress(progress, status) { statusText.textContent = status; progressBar.style.width = `${progress}%`; progressBar.textContent = `${progress}%`; }

    // --- 3. Status Polling and Result Display ---
    function checkStatus(taskId) {
        fetch(`/status/${taskId}`)
            .then(response => response.json())
            .then(data => {
                if (!data || Object.keys(data).length === 0) { setTimeout(() => checkStatus(taskId), 2000); return; }
                if (data.status === 'Error') {
                    stopTimer(); alert('Error during analysis: ' + data.error);
                    progressContainer.classList.add('hidden'); analyzeBtn.disabled = false; return;
                }
                updateProgress(data.progress, data.status);
                if (data.status === 'Complete') {
                    stopTimer();
                    resultsGrid.innerHTML = '';
                    
                    const timestamp = new Date().getTime();
                    const plotTitles = {
                        'reconstruction': 'Reconstructed Segmented Image', 'channel_comparison': 'Comparative Cell Count',
                        'cell_count': 'Cell Count (Initial vs. Filtered)', 'area_histogram': 'Area Distribution',
                        'histogram': 'Intensity Distribution', 'boxplot': 'Normalized Intensity'
                    };
                    
                    if (data.result && data.result.plot_names) {
                        data.result.plot_names.forEach(name => {
                            const card = document.createElement('div'); card.className = 'card';
                            const title = document.createElement('h3'); title.textContent = plotTitles[name] || name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                            const wrapper = document.createElement('div'); wrapper.className = 'chart-wrapper';
                            const img = document.createElement('img');
                            img.src = `/get_plot/${taskId}/${name}?v=${timestamp}`;
                            wrapper.appendChild(img);
                            card.appendChild(title); card.appendChild(wrapper);
                            resultsGrid.appendChild(card);
                        });
                    }

                    downloadCsvBtn.href = `/download_csv/${taskId}`;
                    downloadPlotsBtn.href = `/download_plots_pdf/${taskId}`;
                    resultsDiv.classList.remove('hidden');
                    progressContainer.classList.add('hidden');
                    analyzeBtn.disabled = false;
                } else {
                    setTimeout(() => checkStatus(taskId), 2000);
                }
            })
            .catch(error => { console.error("Status check failed:", error); setTimeout(() => checkStatus(taskId), 5000); });
    }

    // --- 4. Run Analysis ---
    analyzeBtn.addEventListener('click', async () => {
        resultsDiv.classList.add('hidden');
        progressContainer.classList.remove('hidden');
        analyzeBtn.disabled = true;
        updateProgress(0, 'Sending data to server...');
        estimatedTime.textContent = '~...';
        startTimer();

        const formData = new FormData();
        files.forEach(file => formData.append('images', file, file.webkitRelativePath));
        
        formData.append('analysisMode', analysisModeSelect.value);
        formData.append('segChannel', segChannelSelect.value);
        formData.append('minArea', minAreaInput.value);
        formData.append('maxArea', maxAreaInput.value);
        formData.append('minIntensity', minIntensityInput.value);
        formData.append('maxIntensity', maxIntensityInput.value);
        formData.append('bgX1', bgCoords.x1); formData.append('bgY1', bgCoords.y1);
        formData.append('bgX2', bgCoords.x2); formData.append('bgY2', bgCoords.y2);

        try {
            const response = await fetch('/analyze', { method: 'POST', body: formData });
            const data = await response.json();
            if (data.success && data.task_id) {
                checkStatus(data.task_id);
            } else {
                stopTimer(); alert('Failed to start analysis: ' + (data.error || 'Unknown error'));
                progressContainer.classList.add('hidden'); analyzeBtn.disabled = false;
            }
        } catch (error) {
            stopTimer(); alert('Server error: ' + error);
            progressContainer.classList.add('hidden'); analyzeBtn.disabled = false;
        }
    });
});
