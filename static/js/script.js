// Chatbot functionality
let chatbotOpen = false;
let cameraStream = null;
let currentCameraType = null;

function toggleChatbot() {
    const chatbotWindow = document.getElementById('chatbotWindow');
    chatbotOpen = !chatbotOpen;
    
    if (chatbotOpen) {
        chatbotWindow.style.display = 'flex';
    } else {
        chatbotWindow.style.display = 'none';
    }
}

function sendMessage() {
    const input = document.getElementById('chatInput');
    const message = input.value.trim();
    
    if (message === '') return;
    
    // Add user message to chat
    addMessage(message, 'user');
    input.value = '';
    
    // Send to server
    fetch('/api/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: message })
    })
    .then(response => response.json())
    .then(data => {
        addMessage(data.response, 'bot');
    })
    .catch(error => {
        addMessage('Sorry, I encountered an error. Please try again.', 'bot');
    });
}

function addMessage(text, sender) {
    const messagesContainer = document.getElementById('chatbotMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    messageDiv.textContent = text;
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Camera functionality
async function openCamera(cameraType) {
    currentCameraType = cameraType;
    const modal = document.getElementById('cameraModal');
    const video = document.getElementById('cameraVideo');
    const canvas = document.getElementById('cameraCanvas');
    const captureBtn = document.getElementById('captureBtn');
    const modalTitle = document.getElementById('modalTitle');
    
    modal.style.display = 'block';
    modalTitle.textContent = `Capture ${cameraType === 'cloth' ? 'Clothing' : 'Care Label'} Image`;
    captureBtn.textContent = `Capture ${cameraType === 'cloth' ? 'Clothing' : 'Care Label'}`;
    
    try {
        // Try to access browser camera directly
        cameraStream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: 'environment'
            } 
        });
        
        video.srcObject = cameraStream;
        video.style.display = 'block';
        canvas.style.display = 'none';
        
        // Show controls
        document.getElementById('captureBtn').style.display = 'inline-block';
        document.getElementById('retakeBtn').style.display = 'none';
        document.getElementById('useBtn').style.display = 'none';
        
    } catch (error) {
        console.error('Error accessing camera:', error);
        alert('Could not access camera. Please check permissions and ensure your camera is connected.');
        closeCamera();
    }
}

function closeCamera() {
    const modal = document.getElementById('cameraModal');
    const video = document.getElementById('cameraVideo');
    const canvas = document.getElementById('cameraCanvas');
    
    // Stop browser camera
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }
    
    video.srcObject = null;
    video.style.display = 'none';
    canvas.style.display = 'none';
    modal.style.display = 'none';
    currentCameraType = null;
}

function captureImage() {
    const video = document.getElementById('cameraVideo');
    const canvas = document.getElementById('cameraCanvas');
    const ctx = canvas.getContext('2d');
    
    if (!video.srcObject) {
        alert('Camera not available. Please try again.');
        return;
    }
    
    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Draw current video frame to canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Stop the video stream
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }
    
    // Hide video, show canvas
    video.style.display = 'none';
    canvas.style.display = 'block';
    
    // Update buttons
    document.getElementById('captureBtn').style.display = 'none';
    document.getElementById('retakeBtn').style.display = 'inline-block';
    document.getElementById('useBtn').style.display = 'inline-block';
}

function retakePhoto() {
    const video = document.getElementById('cameraVideo');
    const canvas = document.getElementById('cameraCanvas');
    
    // Hide canvas
    canvas.style.display = 'none';
    
    // Restart camera
    openCamera(currentCameraType);
}

function usePhoto() {
    const canvas = document.getElementById('cameraCanvas');
    const imageData = canvas.toDataURL('image/jpeg');
    
    // Store image in hidden field and show preview
    if (currentCameraType === 'cloth') {
        document.getElementById('cloth_image_data').value = imageData;
        document.getElementById('clothPreview').src = imageData;
        document.getElementById('clothPreview').style.display = 'block';
        showNotification('Clothing image captured successfully!', 'success');
    } else {
        document.getElementById('care_label_image_data').value = imageData;
        document.getElementById('careLabelPreview').src = imageData;
        document.getElementById('careLabelPreview').style.display = 'block';
        showNotification('Care label image captured successfully!', 'success');
    }
    
    closeCamera();
}

// Keyboard shortcut for capture (Q key)
document.addEventListener('keydown', function(e) {
    if (e.key === 'q' || e.key === 'Q') {
        const modal = document.getElementById('cameraModal');
        if (modal.style.display === 'block') {
            const captureBtn = document.getElementById('captureBtn');
            if (captureBtn.style.display !== 'none') {
                captureImage();
            }
        }
    }
});

// Mark clothing as worn
function markAsWorn(clothId) {
    fetch('/api/mark_worn', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ cloth_id: clothId })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showNotification('Item marked as worn!', 'success');
        }
    })
    .catch(error => {
        showNotification('Error marking item as worn', 'error');
    });
}

// Show notification
function showNotification(message, type) {
    const flashContainer = document.getElementById('flashMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `flash-message flash-${type}`;
    messageDiv.textContent = message;
    flashContainer.appendChild(messageDiv);
    
    setTimeout(() => {
        messageDiv.remove();
    }, 3000);
}

// Initialize sustainability score circle
function initSustainabilityScore(score) {
    const circle = document.querySelector('.score-circle');
    if (circle) {
        const percent = (score / 100) * 360;
        circle.style.setProperty('--score-percent', `${percent}deg`);
    }
}

// Weather selection for recommendations
function updateWeatherRecommendations() {
    const weatherSelect = document.getElementById('weatherSelect');
    if (weatherSelect) {
        const weather = weatherSelect.value;
        window.location.href = `/recommendations?weather=${weather}`;
    }
}

// Page animations
document.addEventListener('DOMContentLoaded', function() {
    // Add fade-in animation to cards
    const cards = document.querySelectorAll('.card');
    cards.forEach((card, index) => {
        card.style.animationDelay = `${index * 0.1}s`;
        card.classList.add('fade-in');
    });
    
    // Initialize sustainability score if present
    const scoreElement = document.querySelector('.score-text');
    if (scoreElement) {
        const score = parseInt(scoreElement.textContent);
        initSustainabilityScore(score);
    }
    
    // Close camera modal when clicking outside
    const modal = document.getElementById('cameraModal');
    if (modal) {
        modal.addEventListener('click', function(e) {
            if (e.target === modal) {
                closeCamera();
            }
        });
    }
});

// Auto-hide flash messages after 5 seconds
setTimeout(() => {
    const flashMessages = document.querySelectorAll('.flash-message');
    flashMessages.forEach(message => {
        message.style.transition = 'opacity 0.5s ease';
        message.style.opacity = '0';
        setTimeout(() => message.remove(), 500);
    });
}, 5000);