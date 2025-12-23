/**
 * Utility Functions
 * Common helper functions used across the application
 */

/**
 * Format time to readable string
 * @param {Date} date - Date object to format
 * @returns {string} Formatted time string (e.g., "10:30 AM")
 */
export function formatTime(date) {
    if (!date) return '-';
    
    const d = new Date(date);
    const hours = d.getHours();
    const minutes = d.getMinutes();
    const ampm = hours >= 12 ? 'PM' : 'AM';
    const displayHours = hours % 12 || 12;
    const displayMinutes = minutes.toString().padStart(2, '0');
    
    return `${displayHours}:${displayMinutes} ${ampm}`;
}

/**
 * Format date to readable string
 * @param {string|Date} date - Date string or Date object
 * @returns {string} Formatted date string (e.g., "Jan 15, 2024")
 */
export function formatDate(date) {
    if (!date) return '-';
    
    const d = new Date(date);
    return d.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    });
}

/**
 * Format date and time together
 * @param {string|Date} date - Date string or Date object
 * @returns {string} Formatted datetime string
 */
export function formatDateTime(date) {
    if (!date) return '-';
    
    const d = new Date(date);
    return `${formatDate(d)} ${formatTime(d)}`;
}

/**
 * Show toast notification
 * @param {string} message - Message to display
 * @param {string} type - Type of toast: 'success', 'error', 'warning', 'info'
 * @param {number} duration - Duration in milliseconds (default: 3000)
 */
export function showToast(message, type = 'info', duration = 3000) {
    const container = document.getElementById('toastContainer');
    if (!container) {
        console.warn('Toast container not found');
        return;
    }

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    
    const icon = {
        success: '✓',
        error: '✗',
        warning: '⚠',
        info: 'ℹ'
    }[type] || 'ℹ';

    toast.innerHTML = `
        <span style="font-size: 20px;">${icon}</span>
        <span>${message}</span>
    `;

    container.appendChild(toast);

    // Trigger animation
    setTimeout(() => {
        toast.classList.add('show');
    }, 10);

    // Remove after duration
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 300);
    }, duration);
}

/**
 * Debounce function to limit function calls
 * @param {Function} func - Function to debounce
 * @param {number} wait - Wait time in milliseconds
 * @returns {Function} Debounced function
 */
export function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Validate email format
 * @param {string} email - Email to validate
 * @returns {boolean} True if valid email
 */
export function isValidEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
}

/**
 * Validate phone number format
 * @param {string} phone - Phone number to validate
 * @returns {boolean} True if valid phone
 */
export function isValidPhone(phone) {
    const phoneRegex = /^[\d\s\-\+\(\)]+$/;
    return phoneRegex.test(phone) && phone.replace(/\D/g, '').length >= 10;
}

/**
 * Format file size to human readable
 * @param {number} bytes - File size in bytes
 * @returns {string} Formatted size (e.g., "1.5 MB")
 */
export function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

/**
 * Get time ago string (e.g., "2 hours ago")
 * @param {Date|string} date - Date to compare
 * @returns {string} Time ago string
 */
export function getTimeAgo(date) {
    if (!date) return '-';
    
    const now = new Date();
    const past = new Date(date);
    const diffInSeconds = Math.floor((now - past) / 1000);
    
    if (diffInSeconds < 60) return 'Just now';
    if (diffInSeconds < 3600) return `${Math.floor(diffInSeconds / 60)} minutes ago`;
    if (diffInSeconds < 86400) return `${Math.floor(diffInSeconds / 3600)} hours ago`;
    if (diffInSeconds < 604800) return `${Math.floor(diffInSeconds / 86400)} days ago`;
    
    return formatDate(past);
}

/**
 * Copy text to clipboard
 * @param {string} text - Text to copy
 * @returns {Promise<boolean>} True if successful
 */
export async function copyToClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
        return true;
    } catch (err) {
        console.error('Failed to copy:', err);
        return false;
    }
}

/**
 * Download file
 * @param {string} content - File content
 * @param {string} filename - Filename
 * @param {string} mimeType - MIME type
 */
export function downloadFile(content, filename, mimeType = 'text/plain') {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

/**
 * Check if user is logged in
 * @returns {boolean} True if logged in
 */
export function isLoggedIn() {
    // TODO: Implement actual session check
    const token = localStorage.getItem('authToken');
    return !!token;
}

/**
 * Get stored auth token
 * @returns {string|null} Auth token or null
 */
export function getAuthToken() {
    return localStorage.getItem('authToken');
}

/**
 * Set auth token
 * @param {string} token - Auth token
 */
export function setAuthToken(token) {
    localStorage.setItem('authToken', token);
}

/**
 * Remove auth token (logout)
 */
export function removeAuthToken() {
    localStorage.removeItem('authToken');
}

/**
 * Redirect to login if not authenticated
 */
export function requireAuth() {
    if (!isLoggedIn()) {
        window.location.href = 'login.html';
    }
}

