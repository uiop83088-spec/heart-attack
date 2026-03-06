// Medical Image Analyzer - Uses Python backend with real medical algorithms
// This replaces the browser-based TensorFlow.js approach with server-side analysis

async function analyzeMedicalChestXray(imageFile) {
    try {
        console.log('🔍 Starting medical chest X-ray analysis...');
        console.log('📤 Sending image to medical analysis API...');
        
        // Convert image to base64
        const base64Image = await fileToBase64(imageFile);
        
        // Send to Python backend for analysis
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image: base64Image
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            
            // Handle invalid medical image
            if (response.status === 400 && errorData.reason) {
                throw new Error(
                    `❌ ${errorData.error}\n\n` +
                    `Reason: ${errorData.reason}\n\n` +
                    `${errorData.suggestion || 'Please upload a valid chest X-ray image.'}`
                );
            }
            
            throw new Error(`Server error: ${response.status} - ${errorData.error || 'Unknown error'}`);
        }
        
        const result = await response.json();
        
        if (!result.success) {
            throw new Error(result.error || 'Analysis failed');
        }
        
        console.log('✅ Medical analysis complete!');
        console.log('📊 Validation:', result.validation);
        console.log('🏥 Analysis:', result.analysis);
        
        return result.analysis;
        
    } catch (error) {
        console.error('❌ Medical analysis error:', error);
        throw error;
    }
}

function fileToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

// Quick client-side validation before sending to server
async function validateImageBeforeUpload(imageFile) {
    return new Promise((resolve, reject) => {
        // Check file size
        if (imageFile.size > 10 * 1024 * 1024) {
            reject(new Error('Image too large. Please use an image smaller than 10MB.'));
            return;
        }
        
        // Check file type
        if (!imageFile.type.startsWith('image/')) {
            reject(new Error('Invalid file type. Please upload an image file.'));
            return;
        }
        
        // Load image to check dimensions
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                // Check if image is too small
                if (img.width < 100 || img.height < 100) {
                    reject(new Error('Image resolution too low. Please use a higher quality image.'));
                    return;
                }
                
                // Check aspect ratio (chest X-rays are typically portrait)
                const aspectRatio = img.height / img.width;
                if (aspectRatio < 0.7 || aspectRatio > 2.5) {
                    console.warn('⚠️ Warning: Image aspect ratio unusual for chest X-ray');
                }
                
                resolve(true);
            };
            img.onerror = () => reject(new Error('Failed to load image'));
            img.src = e.target.result;
        };
        reader.onerror = () => reject(new Error('Failed to read file'));
        reader.readAsDataURL(imageFile);
    });
}

// Export functions
window.analyzeMedicalChestXray = analyzeMedicalChestXray;
window.validateImageBeforeUpload = validateImageBeforeUpload;

console.log('✅ Medical Analyzer loaded - Using Python backend with medical imaging algorithms');
