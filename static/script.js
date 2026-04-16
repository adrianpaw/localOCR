// Get DOM elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const progressContainer = document.getElementById('progressContainer');
const progressFill = document.getElementById('progressFill');
const errorMessage = document.getElementById('errorMessage');
const resultsSection = document.getElementById('resultsSection');
const extractedText = document.getElementById('extractedText');

// Handle click on upload area
uploadArea.addEventListener('click', () => fileInput.click());

// Handle file selection
fileInput.addEventListener('change', handleFileSelect);

// Handle drag and drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        fileInput.files = files;
        handleFileSelect();
    }
});

// Handle file selection
function handleFileSelect() {
    const file = fileInput.files[0];
    
    if (!file) return;
    
    // Validate file
    const allowedTypes = ['image/png', 'image/jpeg', 'image/bmp', 'image/gif', 'image/webp', 'application/pdf'];
    if (!allowedTypes.includes(file.type)) {
        showError('Invalid file type. Please upload an image or PDF.');
        return;
    }
    
    if (file.size > 50 * 1024 * 1024) {
        showError('File is too large. Maximum size is 50MB.');
        return;
    }
    
    // Process file
    extractTextFromImage(file);
}

// Extract text from image
async function extractTextFromImage(file) {
    try {
        // Show progress
        progressContainer.style.display = 'block';
        errorMessage.style.display = 'none';
        resultsSection.style.display = 'none';
        
        // Create form data
        const formData = new FormData();
        formData.append('file', file);
        
        // Send to server
        const response = await fetch('/api/extract', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Failed to extract text');
        }
        
        // Display results
        progressContainer.style.display = 'none';
        displayResults(data);
        
    } catch (error) {
        progressContainer.style.display = 'none';
        showError(error.message);
    }
}

// Display results
function displayResults(data) {
    document.getElementById('fileNameDisplay').textContent = `File: ${data.filename}`;
    extractedText.value = data.text;
    resultsSection.style.display = 'block';
    
    // Scroll to results
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }, 100);
}

// Show error message
function showError(message) {
    document.getElementById('errorText').textContent = message;
    errorMessage.style.display = 'flex';
    resultsSection.style.display = 'none';
}

// Close error message
function closeError() {
    errorMessage.style.display = 'none';
}

// Copy text to clipboard
async function copyText() {
    try {
        await navigator.clipboard.writeText(extractedText.value);
        showNotification('Copied to clipboard!');
    } catch (error) {
        showError('Failed to copy text');
    }
}

// Show notification
function showNotification(message) {
    const notification = document.createElement('div');
    notification.className = 'copy-notification';
    notification.textContent = message;
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.remove();
    }, 3000);
}

// Reset form
function resetForm() {
    fileInput.value = '';
    progressContainer.style.display = 'none';
    resultsSection.style.display = 'none';
    errorMessage.style.display = 'none';
    extractedText.value = '';
    
    // Scroll to top
    uploadArea.scrollIntoView({ behavior: 'smooth' });
}

// Load model info on page load
document.addEventListener('DOMContentLoaded', async () => {
    try {
        const response = await fetch('/api/model-info');
        const data = await response.json();
        
        if (response.ok && data.loaded) {
            document.getElementById('modelName').textContent = data.languages.join(', ');
            document.getElementById('deviceType').textContent = data.gpu ? 'GPU' : 'CPU';
            document.getElementById('modelInfo').style.display = 'block';
        } else {
            // Show error if model isn't loaded
            const errorBox = document.getElementById('modelInfo');
            errorBox.style.borderLeftColor = '#ef4444';
            errorBox.style.backgroundColor = '#fee';
            errorBox.innerHTML = `<p style="color: #ef4444;"><strong>Model Loading Error:</strong> ${data.error || 'Unknown error'}</p>`;
            errorBox.style.display = 'block';
        }
    } catch (error) {
        console.error('Failed to load model info:', error);
        const errorBox = document.getElementById('modelInfo');
        errorBox.style.borderLeftColor = '#ef4444';
        errorBox.style.backgroundColor = '#fee';
        errorBox.innerHTML = '<p style="color: #ef4444;"><strong>Error:</strong> Failed to connect to server</p>';
        errorBox.style.display = 'block';
    }
});
