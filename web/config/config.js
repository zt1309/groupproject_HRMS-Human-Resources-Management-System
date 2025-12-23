/**
 * Application Configuration
 * API endpoints, thresholds, and system settings
 */

// API Configuration
export const API_CONFIG = {
    // Base URL for API calls
    BASE_URL: 'http://localhost:5000', // Finova API Server
    
    // API Endpoints
    ENDPOINTS: {
        // Authentication
        LOGIN: '/api/auth/login',
        LOGOUT: '/api/auth/logout',
        REGISTER: '/api/auth/register',
        
        // Employees
        EMPLOYEES: '/api/employees',
        EMPLOYEE_BY_ID: (id) => `/api/employees/${id}`,
        REGISTER_EMPLOYEE: '/api/employees/register',
        
        // Attendance
        ATTENDANCE: '/api/attendance',
        ATTENDANCE_TODAY: '/api/attendance/today',
        ATTENDANCE_BY_DATE: (date) => `/api/attendance/date/${date}`,
        ATTENDANCE_BY_EMPLOYEE: (empId) => `/api/attendance/employee/${empId}`,
        
        // Statistics
        STATISTICS: '/api/statistics',
        STATISTICS_TODAY: '/api/statistics/today',
        STATISTICS_RANGE: '/api/statistics/range',
        
        // Face Recognition
        RECOGNIZE: '/api/recognize',
        VERIFY: '/api/verify',
        DETECT_FACES: '/api/detect',
        
        // Reports
        REPORTS: '/api/reports',
        EXPORT_CSV: '/api/reports/export/csv',
        EXPORT_EXCEL: '/api/reports/export/excel',
        
        // Settings
        SETTINGS: '/api/settings',
        UPDATE_SETTINGS: '/api/settings/update',
    }
};

// Face Recognition Configuration
export const RECOGNITION_CONFIG = {
    // Confidence threshold for face recognition (0.0 - 1.0)
    CONFIDENCE_THRESHOLD: 0.7,
    
    // Recognition intervals (frames)
    RECOGNITION_INTERVAL: 5, // Process every 5th frame
    
    // Cooldown periods (milliseconds)
    EMPLOYEE_COOLDOWN: 30000, // 30 seconds between same employee
    GLOBAL_COOLDOWN: 5000,   // 5 seconds between any recognition
    
    // Face detection settings
    MIN_FACE_SIZE: 50,        // Minimum face size in pixels
    MAX_FACE_SIZE: 500,       // Maximum face size in pixels
    
    // Anti-spoofing
    ENABLE_ANTI_SPOOFING: true,
    ANTI_SPOOFING_THRESHOLD: 0.5,
};

// Camera Configuration
export const CAMERA_CONFIG = {
    // Default camera constraints
    DEFAULT_CONSTRAINTS: {
        video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            facingMode: 'user'
        }
    },
    
    // Supported resolutions
    RESOLUTIONS: [
        { width: 640, height: 480, label: '640x480' },
        { width: 1280, height: 720, label: '1280x720' },
        { width: 1920, height: 1080, label: '1920x1080' }
    ],
    
    // FPS calculation
    FPS_INTERVAL: 1000, // Calculate FPS every second
};

// Attendance Configuration
export const ATTENDANCE_CONFIG = {
    // Work hours
    WORK_START_HOUR: 8,
    WORK_START_MINUTE: 0,
    WORK_END_HOUR: 17,
    WORK_END_MINUTE: 0,
    
    // Late threshold (minutes after work start)
    LATE_THRESHOLD: 15, // 15 minutes
    
    // Timezone
    TIMEZONE: 'Asia/Ho_Chi_Minh',
    
    // Auto check-out (hours after check-in)
    AUTO_CHECKOUT_HOURS: 8,
};

// UI Configuration
export const UI_CONFIG = {
    // Toast notification duration (milliseconds)
    TOAST_DURATION: 3000,
    
    // Table pagination
    RECORDS_PER_PAGE: 20,
    
    // Refresh intervals (milliseconds)
    AUTO_REFRESH_INTERVAL: 30000, // 30 seconds
    
    // Animation durations
    FADE_DURATION: 300,
};

// Storage Keys
export const STORAGE_KEYS = {
    AUTH_TOKEN: 'authToken',
    USER_DATA: 'userData',
    REMEMBER_ME: 'rememberMe',
    CAMERA_SETTINGS: 'cameraSettings',
    SYSTEM_SETTINGS: 'systemSettings',
};

// Export all config
export default {
    API_CONFIG,
    RECOGNITION_CONFIG,
    CAMERA_CONFIG,
    ATTENDANCE_CONFIG,
    UI_CONFIG,
    STORAGE_KEYS,
};

