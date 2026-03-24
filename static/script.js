document.addEventListener('DOMContentLoaded', () => {
    // --- Elements ---
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const webcam = document.getElementById('webcam');
    const canvas = document.getElementById('canvas');
    const startCamBtn = document.getElementById('start-cam');
    const captureBtn = document.getElementById('capture-btn');
    
    // Panels
    const idleState = document.getElementById('idle-state');
    const predictionState = document.getElementById('prediction-state');
    const processingOverlay = document.getElementById('processing-overlay');
    
    // Result details
    const resultIconMain = document.getElementById('result-icon-main');
    const classificationTag = document.getElementById('classification-tag');
    const predictionLabel = document.getElementById('prediction-label');
    const confidencePct = document.getElementById('confidence-pct');
    const confidenceFill = document.getElementById('confidence-fill');
    const feedbackMsg = document.getElementById('feedback-msg');
    const resetBtn = document.getElementById('reset-btn');
    const statusIndicator = document.getElementById('status-indicator');

    // Tabs
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    let stream = null;

    // --- Tab Logic ---
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const target = btn.dataset.tab;
            
            // Toggle active button
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Toggle content
            tabContents.forEach(content => {
                if (content.id === `${target}-content`) {
                    content.classList.remove('hidden');
                } else {
                    content.classList.add('hidden');
                }
            });

            // Stop webcam if switching to upload
            if (target === 'upload' && stream) {
                stopWebcam();
            }
        });
    });

    // --- File Upload Logic ---
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--primary-color)';
    });

    ['dragleave', 'dragend'].forEach(type => {
        dropZone.addEventListener(type, () => {
            dropZone.style.borderColor = 'var(--glass-border)';
        });
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--glass-border)';
        if (e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) {
            handleFile(fileInput.files[0]);
        }
    });

    // --- Refinement Logic ---
    const markRealBtn = document.getElementById('mark-real-btn');
    const markFakeBtn = document.getElementById('mark-fake-btn');
    const retrainBtn = document.getElementById('retrain-btn');
    const retrainStatus = document.getElementById('retrain-status');
    let lastImageData = null;

    markRealBtn.addEventListener('click', () => saveToDataset('real'));
    markFakeBtn.addEventListener('click', () => saveToDataset('fake'));

    async function saveToDataset(label) {
        if (!lastImageData) return;
        
        try {
            const response = await fetch('/save_to_dataset', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ label: label, image_data: lastImageData })
            });
            const data = await response.json();
            if (data.success) {
                retrainStatus.innerText = `Saved as ${label}! Now click Update for the AI to learn.`;
                retrainStatus.style.color = 'var(--primary-color)';
            }
        } catch (err) {
            console.error(err);
        }
    }

    retrainBtn.addEventListener('click', async () => {
        retrainStatus.innerText = "Retraining AI... Please wait (this takes ~1 min).";
        retrainBtn.disabled = true;
        try {
            const response = await fetch('/retrain', { method: 'POST' });
            const data = await response.json();
            if (data.success) {
                retrainStatus.innerText = "Success! AI has been updated with your data.";
                alert("The AI model has been successfully updated and reloaded!");
            } else {
                retrainStatus.innerText = "Error during retraining: " + data.error;
            }
        } catch (err) {
            retrainStatus.innerText = "Request failed.";
        } finally {
            retrainBtn.disabled = false;
        }
    });

    function handleFile(file) {
        if (file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onload = () => {
                lastImageData = reader.result;
                // Remove old preview if exists
                const existingPreview = dropZone.querySelector('.drop-zone-preview');
                if (existingPreview) existingPreview.remove();

                const preview = document.createElement('div');
                preview.classList.add('drop-zone-preview', 'fade-in');
                preview.style.backgroundImage = `url('${reader.result}')`;
                dropZone.appendChild(preview);
                
                // Process
                uploadFile(file);
            };
        }
    }

    async function uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);
        
        showLoading(true);
        try {
            const response = await fetch('/predict', { method: 'POST', body: formData });
            const data = await response.json();
            showResult(data);
        } catch (err) {
            handleError(err);
        } finally {
            showLoading(false);
        }
    }

    // --- Webcam Logic ---
    startCamBtn.addEventListener('click', async () => {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ 
                video: { width: 1280, height: 720, facingMode: "environment" } 
            });
            webcam.srcObject = stream;
            
            startCamBtn.classList.add('hidden');
            captureBtn.disabled = false;
        } catch (err) {
            console.error(err);
            alert("Camera access denied or unavailable.");
        }
    });

    captureBtn.addEventListener('click', async () => {
        if (!stream) return;

        canvas.width = webcam.videoWidth;
        canvas.height = webcam.videoHeight;
        const context = canvas.getContext('2d');
        
        // Mirror horizontally back for capture
        context.translate(webcam.videoWidth, 0);
        context.scale(-1, 1);
        context.drawImage(webcam, 0, 0, canvas.width, canvas.height);

        const imageData = canvas.toDataURL('image/jpeg', 0.8);
        lastImageData = imageData;
        const formData = new FormData();
        formData.append('image_data', imageData);

        showLoading(true);
        try {
            const response = await fetch('/predict', { method: 'POST', body: formData });
            const data = await response.json();
            showResult(data);
        } catch (err) {
            handleError(err);
        } finally {
            showLoading(false);
        }
    });

    function stopWebcam() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            webcam.srcObject = null;
            stream = null;
            startCamBtn.classList.remove('hidden');
            captureBtn.disabled = true;
        }
    }

    // --- UI Helpers ---
    function showLoading(isLoading) {
        if (isLoading) {
            processingOverlay.classList.remove('hidden');
        } else {
            processingOverlay.classList.add('hidden');
        }
    }

    function showResult(data) {
        idleState.classList.add('hidden');
        predictionState.classList.remove('hidden');
        predictionState.classList.add('fade-in');

        if (data.error) {
            handleError(data.error);
            return;
        }

        const isReal = data.is_real;
        const isInvalid = data.invalid;
        const confidence = (data.confidence * 100).toFixed(1);

        // Reset classes
        statusIndicator.className = 'status-indicator';
        classificationTag.className = 'classification-tag';
        predictionLabel.className = '';
        confidenceFill.className = 'meter-fill';

        if (isInvalid) {
            // Invalid Image Logic
            resultIconMain.className = 'fas fa-triangle-exclamation';
            statusIndicator.classList.add('invalid-stat');
            classificationTag.innerText = 'System Warning';
            classificationTag.classList.add('invalid-stat');
            predictionLabel.innerText = 'Invalid Input';
            predictionLabel.classList.add('invalid-stat');
            
            confidenceFill.style.width = `${confidence}%`;
            confidenceFill.classList.add('invalid-fill');
            confidencePct.innerText = `${confidence}% Match`;
            
            feedbackMsg.innerText = data.message || "The system could not identify this as a currency note. Please ensure the note is centered and clearly visible.";
        } else {
            // Success Logic (Real or Fake)
            resultIconMain.className = isReal ? 'fas fa-check-shield' : 'fas fa-shield-xmark';
            
            if (isReal) {
                statusIndicator.classList.add('real-stat');
                classificationTag.innerText = 'Verified Authentic';
                classificationTag.classList.add('real-stat');
                predictionLabel.innerText = 'Real Currency';
                predictionLabel.classList.add('real-stat');
                confidenceFill.classList.add('real-fill');
                feedbackMsg.innerText = "Security patterns match official neural signatures. This note is classified as GENUINE.";
            } else {
                statusIndicator.classList.add('fake-stat');
                classificationTag.innerText = 'Security Breach';
                classificationTag.classList.add('fake-stat');
                predictionLabel.innerText = 'Fake Currency';
                predictionLabel.classList.add('fake-stat');
                confidenceFill.classList.add('fake-fill');
                feedbackMsg.innerText = "Warning: Neural pattern mismatch detected. This note shows characteristics commonly found in COUNTERFEIT currency.";
            }

            confidenceFill.style.width = `${confidence}%`;
            confidencePct.innerText = `${confidence}%`;
        }
    }

    function handleError(err) {
        idleState.classList.add('hidden');
        predictionState.classList.remove('hidden');
        
        resultIconMain.className = 'fas fa-bug';
        predictionLabel.innerText = 'System Error';
        feedbackMsg.innerText = "An unexpected error occurred: " + (err.message || err);
        classificationTag.innerText = 'Error';
    }

    resetBtn.addEventListener('click', () => {
        predictionState.classList.add('hidden');
        idleState.classList.remove('hidden');
        idleState.classList.add('fade-in');
        
        // Reset Uploads
        fileInput.value = '';
        lastImageData = null;
        retrainStatus.innerText = '';
        const preview = dropZone.querySelector('.drop-zone-preview');
        if (preview) preview.remove();
        
        // If camera tab is active, keep it active
    });
});

