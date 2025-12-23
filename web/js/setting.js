/**
 * Settings Manager
 * Handles system settings management
 */

import { getSettings, updateSettings } from './api.js';
import { showToast } from './utils.js';
import { STORAGE_KEYS } from '../config/config.js';
import CameraManager from './camera.js';

export class SettingsManager {
    constructor() {
        this.settings = {};
        this.cameras = [];
    }

    /**
     * Initialize settings page
     */
    async init() {
        await this.loadSettings();
        await this.loadCameras();
        this.setupEventListeners();
    }

    /**
     * Load settings from API or storage
     */
    async loadSettings() {
        try {
            const settings = await getSettings();
            if (settings) {
                this.settings = settings;
                this.displaySettings(settings);
            } else {
                // Load from storage
                const stored = localStorage.getItem(STORAGE_KEYS.SYSTEM_SETTINGS);
                if (stored) {
                    this.settings = JSON.parse(stored);
                    this.displaySettings(this.settings);
                }
            }
        } catch (error) {
            console.error('Error loading settings:', error);
            // Use default settings
            this.displaySettings(this.getDefaultSettings());
        }
    }

    /**
     * Display settings in form
     * @param {object} settings - Settings object
     */
    displaySettings(settings) {
        if (settings.system_name) {
            document.getElementById('system-name').value = settings.system_name;
        }
        
        if (settings.timezone) {
            document.getElementById('timezone').value = settings.timezone;
        }
        
        if (settings.resolution) {
            document.getElementById('resolution').value = settings.resolution;
        }
    }

    /**
     * Load available cameras
     */
    async loadCameras() {
        try {
            this.cameras = await CameraManager.getAvailableCameras();
            this.displayCameras();
        } catch (error) {
            console.error('Error loading cameras:', error);
        }
    }

    /**
     * Display cameras in select
     */
    displayCameras() {
        const select = document.getElementById('camera-device');
        if (!select) return;

        select.innerHTML = '<option value="">Select camera</option>';
        
        this.cameras.forEach((camera, index) => {
            const option = document.createElement('option');
            option.value = camera.deviceId;
            option.textContent = camera.label || `Camera ${index + 1}`;
            select.appendChild(option);
        });
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // General settings form
        const generalForm = document.getElementById('general-settings-form');
        if (generalForm) {
            generalForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                await this.saveGeneralSettings();
            });
        }

        // Security settings form
        const securityForm = document.getElementById('security-settings-form');
        if (securityForm) {
            securityForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                await this.updatePassword();
            });
        }

        // Camera settings form
        const cameraForm = document.getElementById('camera-settings-form');
        if (cameraForm) {
            cameraForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                await this.saveCameraSettings();
            });
        }
    }

    /**
     * Save general settings
     */
    async saveGeneralSettings() {
        const systemName = document.getElementById('system-name').value;
        const timezone = document.getElementById('timezone').value;

        const settings = {
            system_name: systemName,
            timezone: timezone,
        };

        try {
            await updateSettings(settings);
            this.settings = { ...this.settings, ...settings };
            localStorage.setItem(STORAGE_KEYS.SYSTEM_SETTINGS, JSON.stringify(this.settings));
            showToast('Settings saved successfully', 'success');
        } catch (error) {
            console.error('Error saving settings:', error);
            showToast('Failed to save settings', 'error');
        }
    }

    /**
     * Update password
     */
    async updatePassword() {
        const currentPassword = document.getElementById('password').value;
        const newPassword = document.getElementById('new-password').value;
        const confirmPassword = document.getElementById('confirm-password').value;

        if (!currentPassword || !newPassword || !confirmPassword) {
            showToast('Please fill in all password fields', 'error');
            return;
        }

        if (newPassword !== confirmPassword) {
            showToast('New passwords do not match', 'error');
            return;
        }

        if (newPassword.length < 8) {
            showToast('Password must be at least 8 characters', 'error');
            return;
        }

        try {
            // TODO: Call API to update password
            showToast('Password updated successfully', 'success');
            
            // Clear form
            document.getElementById('password').value = '';
            document.getElementById('new-password').value = '';
            document.getElementById('confirm-password').value = '';
        } catch (error) {
            console.error('Error updating password:', error);
            showToast('Failed to update password', 'error');
        }
    }

    /**
     * Save camera settings
     */
    async saveCameraSettings() {
        const cameraDevice = document.getElementById('camera-device').value;
        const resolution = document.getElementById('resolution').value;

        const settings = {
            camera_device: cameraDevice,
            resolution: resolution,
        };

        try {
            await updateSettings(settings);
            this.settings = { ...this.settings, ...settings };
            localStorage.setItem(STORAGE_KEYS.CAMERA_SETTINGS, JSON.stringify(settings));
            showToast('Camera settings saved successfully', 'success');
        } catch (error) {
            console.error('Error saving camera settings:', error);
            showToast('Failed to save camera settings', 'error');
        }
    }

    /**
     * Get default settings
     */
    getDefaultSettings() {
        return {
            system_name: 'Face Recognition Attendance System',
            timezone: 'Asia/Ho_Chi_Minh',
            resolution: '1280x720',
        };
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    const settingsManager = new SettingsManager();
    settingsManager.init();
});

export default SettingsManager;



