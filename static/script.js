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
    
    // Global state variables
    let files = [];
    let timerInterval = null;

    // --- 1. Folder Upload Logic ---
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
                if (!channels[channelName]) {
                    channels[channelName] = 0;
                }
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
        } else {
            uploadInfo.innerHTML = "<p style='color: red;'>Error: No channel subfolders detected. Please select the parent folder containing the tiled channel folders.</p>";
            analyzeBtn.disabled = true;
        }
    });

    // --- 2. UI Helper Functions ---
    function startTimer() {
        let seconds = 0;
        elapsedTime.textContent = '0s';
        timerInterval = setInterval(() => {
            seconds++;
            elapsedTime.textContent = `${seconds}s`;
        }, 1000);
    }

    function stopTimer() {
        clearInterval(timerInterval);
    }

    function updateProgress(progress, status) {
        statusText.textContent = status;
        progressBar.style.width = `${progress}%`;
        progressBar.textContent = `${progress}%`;
    }

    // --- 3. Status Polling and Result Display ---
    function checkStatus(taskId) {
        fetch(`/status/${taskId}`)
            .then(response => response.json())
            .then(data => {
                if (!data || Object.keys(data).length === 0) {
                    setTimeout(() => checkStatus(taskId), 2000);
                    return;
                }
                if (data.status === 'Error') {
                    stopTimer();
                    alert('Error during analysis: ' + data.error);
                    progressContainer.classList.add('hidden');
                    analyzeBtn.disabled = false;
                    return;
                }
                updateProgress(data.progress, data.status);
                if (data.status === 'Complete') {
                    stopTimer();
                    resultsGrid.innerHTML = ''; // Clear previous results
                    
                    const plotNames = ['cell_count', 'area_histogram', 'histogram', 'boxplot'];
                    const plotTitles = {
                        'cell_count': 'Cell Count',
                        'area_histogram': 'Area Distribution',
                        'histogram': 'Intensity Distribution',
                        'boxplot': 'Normalized Intensity'
                    };

                    plotNames.forEach(name => {
                        const card = document.createElement('div');
                        card.className = 'card';
                        const title = document.createElement('h3');
                        title.textContent = plotTitles[name];
                        const wrapper = document.createElement('div');
                        wrapper.className = 'chart-wrapper';
                        const img = document.createElement('img');
                        const timestamp = new Date().getTime();
                        img.src = `/get_plot/${taskId}/${name}?v=${timestamp}`;
                        wrapper.appendChild(img);
                        card.appendChild(title);
                        card.appendChild(wrapper);
                        resultsGrid.appendChild(card);
                    });

                    downloadCsvBtn.href = `/download_csv/${taskId}`;
                    downloadPlotsBtn.href = `/download_plots_pdf/${taskId}`;
                    resultsDiv.classList.remove('hidden');
                    progressContainer.classList.add('hidden');
                    analyzeBtn.disabled = false;
                } else {
                    setTimeout(() => checkStatus(taskId), 2000);
                }
            })
            .catch(error => {
                console.error("Status check failed:", error);
                setTimeout(() => checkStatus(taskId), 5000);
            });
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
        files.forEach(file => {
            formData.append('images', file, file.webkitRelativePath);
        });
        formData.append('segChannel', segChannelSelect.value);
        formData.append('minArea', minAreaInput.value);
        formData.append('maxArea', maxAreaInput.value);
        formData.append('minIntensity', minIntensityInput.value);
        formData.append('maxIntensity', maxIntensityInput.value);
        
        // Background coords and comp matrix are not used in tiling workflow, but we send placeholders
        formData.append('bgX1', 0); formData.append('bgY1', 0);
        formData.append('bgX2', 0); formData.append('bgY2', 0);

        try {
            const response = await fetch('/analyze', { method: 'POST', body: formData });
            const data = await response.json();
            if (data.success && data.task_id) {
                checkStatus(data.task_id);
            } else {
                stopTimer();
                alert('Failed to start analysis: ' + (data.error || 'Unknown error'));
                progressContainer.classList.add('hidden');
                analyzeBtn.disabled = false;
            }
        } catch (error) {
            stopTimer();
            alert('Server error: ' + error);
            progressContainer.classList.add('hidden');
            analyzeBtn.disabled = false;
        }
    });
});
