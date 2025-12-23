/**
 * API Service
 * Handles all API calls to the backend
 */

import { API_CONFIG, STORAGE_KEYS } from '../config/config.js';
import { showToast } from './utils.js';

const BASE_URL = API_CONFIG.BASE_URL;

/**
 * Make API request
 * @param {string} endpoint - API endpoint
 * @param {object} options - Fetch options
 * @returns {Promise} Response data
 */
async function apiRequest(endpoint, options = {}) {
    const url = `${BASE_URL}${endpoint}`;
    const token = localStorage.getItem(STORAGE_KEYS.AUTH_TOKEN);
    
    const defaultHeaders = {
        'Content-Type': 'application/json',
    };
    
    if (token) {
        defaultHeaders['Authorization'] = `Bearer ${token}`;
    }
    
    const config = {
        ...options,
        headers: {
            ...defaultHeaders,
            ...options.headers,
        },
    };
    
    try {
        const response = await fetch(url, config);
        
        if (!response.ok) {
            const error = await response.json().catch(() => ({ message: 'Request failed' }));
            throw new Error(error.message || `HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API request failed:', error);
        throw error;
    }
}

// ==================== Authentication ====================

/**
 * Login user
 * @param {string} username - Username
 * @param {string} password - Password
 * @returns {Promise<object>} User data and token
 */
export async function login(username, password) {
    const data = await apiRequest(API_CONFIG.ENDPOINTS.LOGIN, {
        method: 'POST',
        body: JSON.stringify({ username, password }),
    });
    
    if (data.token) {
        localStorage.setItem(STORAGE_KEYS.AUTH_TOKEN, data.token);
        if (data.user) {
            localStorage.setItem(STORAGE_KEYS.USER_DATA, JSON.stringify(data.user));
        }
    }
    
    return data;
}

/**
 * Logout user
 * @returns {Promise<void>}
 */
export async function logout() {
    try {
        await apiRequest(API_CONFIG.ENDPOINTS.LOGOUT, {
            method: 'POST',
        });
    } catch (error) {
        console.error('Logout error:', error);
    } finally {
        localStorage.removeItem(STORAGE_KEYS.AUTH_TOKEN);
        localStorage.removeItem(STORAGE_KEYS.USER_DATA);
    }
}

// ==================== Employees ====================

/**
 * Get all employees
 * @returns {Promise<Array>} List of employees
 */
export async function getEmployees() {
    return await apiRequest(API_CONFIG.ENDPOINTS.EMPLOYEES);
}

/**
 * Get employee by ID
 * @param {string} employeeId - Employee ID
 * @returns {Promise<object>} Employee data
 */
export async function getEmployeeById(employeeId) {
    return await apiRequest(API_CONFIG.ENDPOINTS.EMPLOYEE_BY_ID(employeeId));
}

/**
 * Register new employee
 * @param {object} employeeData - Employee data
 * @param {Array<File>} photos - Employee photos
 * @returns {Promise<object>} Created employee data
 */
export async function registerEmployee(employeeData, photos) {
    const formData = new FormData();
    
    // Add employee data
    Object.keys(employeeData).forEach(key => {
        formData.append(key, employeeData[key]);
    });
    
    // Add photos
    photos.forEach((photo, index) => {
        formData.append(`photo_${index}`, photo);
    });
    
    return await apiRequest(API_CONFIG.ENDPOINTS.REGISTER_EMPLOYEE, {
        method: 'POST',
        headers: {}, // Let browser set Content-Type for FormData
        body: formData,
    });
}

// ==================== Attendance ====================

/**
 * Get today's attendance records
 * @returns {Promise<Array>} Attendance records
 */
export async function getTodayAttendance() {
    try {
        const response = await apiRequest(API_CONFIG.ENDPOINTS.ATTENDANCE_TODAY);
        return response.attendance || [];
    } catch (error) {
        console.error('Error fetching today attendance:', error);
        return [];
    }
}

/**
 * Get attendance records by date
 * @param {string} date - Date string (YYYY-MM-DD)
 * @returns {Promise<Array>} Attendance records
 */
export async function getAttendanceByDate(date) {
    return await apiRequest(API_CONFIG.ENDPOINTS.ATTENDANCE_BY_DATE(date));
}

/**
 * Get attendance records by employee
 * @param {string} employeeId - Employee ID
 * @returns {Promise<Array>} Attendance records
 */
export async function getAttendanceByEmployee(employeeId) {
    return await apiRequest(API_CONFIG.ENDPOINTS.ATTENDANCE_BY_EMPLOYEE(employeeId));
}

/**
 * Get all attendance records with filters
 * @param {object} filters - Filter options
 * @returns {Promise<Array>} Attendance records
 */
export async function getAttendanceRecords(filters = {}) {
    const queryParams = new URLSearchParams(filters).toString();
    const endpoint = queryParams 
        ? `${API_CONFIG.ENDPOINTS.ATTENDANCE}?${queryParams}`
        : API_CONFIG.ENDPOINTS.ATTENDANCE;
    
    return await apiRequest(endpoint);
}

// ==================== Statistics ====================

/**
 * Get statistics
 * @param {object} options - Statistics options
 * @returns {Promise<object>} Statistics data
 */
export async function getStatistics(options = {}) {
    try {
        if (options.date) {
            return await apiRequest(`${API_CONFIG.ENDPOINTS.STATISTICS_TODAY}?date=${options.date}`);
        }
        const response = await apiRequest(API_CONFIG.ENDPOINTS.STATISTICS);
        return response.statistics || {
            total_employees: 0,
            present_today: 0,
            late_today: 0,
            absent_today: 0,
        };
    } catch (error) {
        console.error('Error fetching statistics:', error);
        return {
            total_employees: 0,
            present_today: 0,
            late_today: 0,
            absent_today: 0,
        };
    }
}

// ==================== Face Recognition ====================

/**
 * Recognize face from image
 * @param {Blob|File} image - Image blob or file
 * @returns {Promise<object>} Recognition result
 */
export async function recognizeFace(image) {
    const formData = new FormData();
    formData.append('image', image);
    
    const response = await apiRequest(API_CONFIG.ENDPOINTS.RECOGNIZE, {
        method: 'POST',
        headers: {}, // Let browser set Content-Type for FormData
        body: formData,
    });
    
    if (response.recognized) {
        return {
            employee_id: response.employee_id,
            name: response.name,
            department: response.department,
            confidence: response.confidence,
            timestamp: response.timestamp
        };
    }
    
    return null;
}

/**
 * Detect faces in image
 * @param {Blob|File} image - Image blob or file
 * @returns {Promise<Array>} Detected faces
 */
export async function detectFaces(image) {
    const formData = new FormData();
    formData.append('image', image);
    
    return await apiRequest(API_CONFIG.ENDPOINTS.DETECT_FACES, {
        method: 'POST',
        headers: {},
        body: formData,
    });
}

// ==================== Reports ====================

/**
 * Export attendance as CSV
 * @param {object} filters - Filter options
 * @returns {Promise<Blob>} CSV file blob
 */
export async function exportAttendanceCSV(filters = {}) {
    const queryParams = new URLSearchParams(filters).toString();
    const endpoint = queryParams 
        ? `${API_CONFIG.ENDPOINTS.EXPORT_CSV}?${queryParams}`
        : API_CONFIG.ENDPOINTS.EXPORT_CSV;
    
    const response = await fetch(`${BASE_URL}${endpoint}`, {
        method: 'GET',
        headers: {
            'Authorization': `Bearer ${localStorage.getItem(STORAGE_KEYS.AUTH_TOKEN)}`,
        },
    });
    
    if (!response.ok) {
        throw new Error('Export failed');
    }
    
    return await response.blob();
}

// ==================== Settings ====================

/**
 * Get system settings
 * @returns {Promise<object>} Settings data
 */
export async function getSettings() {
    try {
        return await apiRequest(API_CONFIG.ENDPOINTS.SETTINGS);
    } catch (error) {
        console.error('Error fetching settings:', error);
        return null;
    }
}

/**
 * Update system settings
 * @param {object} settings - Settings to update
 * @returns {Promise<object>} Updated settings
 */
export async function updateSettings(settings) {
    return await apiRequest(API_CONFIG.ENDPOINTS.UPDATE_SETTINGS, {
        method: 'POST',
        body: JSON.stringify(settings),
    });
}

// Export default API object
export default {
    login,
    logout,
    getEmployees,
    getEmployeeById,
    registerEmployee,
    getTodayAttendance,
    getAttendanceByDate,
    getAttendanceByEmployee,
    getAttendanceRecords,
    getStatistics,
    recognizeFace,
    detectFaces,
    exportAttendanceCSV,
    getSettings,
    updateSettings,
};

