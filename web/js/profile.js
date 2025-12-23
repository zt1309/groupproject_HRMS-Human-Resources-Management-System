/**
 * Profile Manager
 * Handles user profile management
 */

import { getEmployeeById, getAttendanceByEmployee, updateSettings } from './api.js';
import { showToast, formatTime, formatDate } from './utils.js';
import { STORAGE_KEYS } from '../config/config.js';

export class ProfileManager {
    constructor() {
        this.userData = null;
        this.employeeId = null;
    }

    /**
     * Initialize profile page
     */
    async init() {
        // Load user data from storage or API
        const storedUser = localStorage.getItem(STORAGE_KEYS.USER_DATA);
        if (storedUser) {
            this.userData = JSON.parse(storedUser);
            this.employeeId = this.userData.employee_id || this.userData.id;
        }

        await this.loadProfile();
        await this.loadAttendanceHistory();
        this.setupEventListeners();
    }

    /**
     * Load profile data
     */
    async loadProfile() {
        try {
            if (this.employeeId) {
                const data = await getEmployeeById(this.employeeId);
                this.userData = data;
                this.displayProfile(data);
            } else {
                // Use mock data for demonstration
                this.displayProfile(this.getMockProfile());
            }
        } catch (error) {
            console.error('Error loading profile:', error);
            // Use mock data on error
            this.displayProfile(this.getMockProfile());
        }
    }

    /**
     * Display profile data
     * @param {object} data - Profile data
     */
    displayProfile(data) {
        // Update header
        const fullName = `${data.first_name || ''} ${data.last_name || ''}`.trim() || data.name || 'User';
        document.getElementById('profile-name').textContent = fullName;
        document.getElementById('profile-role').textContent = data.role || data.department || 'Employee';
        document.getElementById('profile-email').textContent = data.email || 'user@example.com';

        // Update photo
        const photoElement = document.getElementById('profile-photo');
        if (data.photo_url) {
            photoElement.src = data.photo_url;
        }

        // Update form fields
        document.getElementById('first-name').value = data.first_name || '';
        document.getElementById('last-name').value = data.last_name || '';
        document.getElementById('email').value = data.email || '';
        document.getElementById('phone').value = data.phone || '';
        document.getElementById('department').value = data.department || '';
        
        if (data.join_date) {
            const joinDate = new Date(data.join_date);
            document.getElementById('join-date').value = joinDate.toISOString().split('T')[0];
        }
    }

    /**
     * Load attendance history
     */
    async loadAttendanceHistory() {
        try {
            let records = [];
            
            if (this.employeeId) {
                records = await getAttendanceByEmployee(this.employeeId);
            } else {
                // Mock data
                records = this.getMockAttendanceHistory();
            }

            this.displayAttendanceHistory(records);
        } catch (error) {
            console.error('Error loading attendance history:', error);
            this.displayAttendanceHistory([]);
        }
    }

    /**
     * Display attendance history
     * @param {Array} records - Attendance records
     */
    displayAttendanceHistory(records) {
        const container = document.getElementById('attendance-history');
        if (!container) return;

        if (records.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <span class="empty-icon">📋</span>
                    <p>No attendance history available</p>
                </div>
            `;
            return;
        }

        container.innerHTML = records.slice(0, 10).map(record => {
            const date = formatDate(record.date || record.timestamp);
            const checkIn = record.check_in ? formatTime(new Date(record.check_in)) : '-';
            const checkOut = record.check_out ? formatTime(new Date(record.check_out)) : '-';
            const status = record.status || 'on-time';
            const statusClass = status === 'on-time' ? 'badge-success' : status === 'late' ? 'badge-warning' : 'badge-danger';

            return `
                <div class="attendance-item">
                    <div class="attendance-info">
                        <h3>${date}</h3>
                        <p>Check In: ${checkIn} | Check Out: ${checkOut}</p>
                    </div>
                    <span class="badge ${statusClass}">${status}</span>
                </div>
            `;
        }).join('');
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Profile form submission
        const profileForm = document.getElementById('profile-form');
        if (profileForm) {
            profileForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                await this.updateProfile();
            });
        }

        // Change photo button
        const changePhotoBtn = document.getElementById('btn-change-photo');
        const photoUpload = document.getElementById('photo-upload');
        if (changePhotoBtn && photoUpload) {
            changePhotoBtn.addEventListener('click', () => {
                photoUpload.click();
            });

            photoUpload.addEventListener('change', (e) => {
                const file = e.target.files[0];
                if (file) {
                    this.updateProfilePhoto(file);
                }
            });
        }

        // Change password button
        const changePasswordBtn = document.getElementById('btn-change-password');
        if (changePasswordBtn) {
            changePasswordBtn.addEventListener('click', () => {
                window.location.href = 'settings.html#security';
            });
        }

        // Logout button
        const logoutBtn = document.getElementById('btn-logout');
        if (logoutBtn) {
            logoutBtn.addEventListener('click', () => {
                this.logout();
            });
        }
    }

    /**
     * Update profile
     */
    async updateProfile() {
        const formData = {
            first_name: document.getElementById('first-name').value,
            last_name: document.getElementById('last-name').value,
            email: document.getElementById('email').value,
            phone: document.getElementById('phone').value,
        };

        try {
            // TODO: Call API to update profile
            showToast('Profile updated successfully', 'success');
            await this.loadProfile();
        } catch (error) {
            console.error('Error updating profile:', error);
            showToast('Failed to update profile', 'error');
        }
    }

    /**
     * Update profile photo
     * @param {File} file - Photo file
     */
    async updateProfilePhoto(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            document.getElementById('profile-photo').src = e.target.result;
        };
        reader.readAsDataURL(file);

        try {
            // TODO: Upload photo to server
            showToast('Photo updated successfully', 'success');
        } catch (error) {
            console.error('Error updating photo:', error);
            showToast('Failed to update photo', 'error');
        }
    }

    /**
     * Logout user
     */
    logout() {
        if (confirm('Are you sure you want to logout?')) {
            localStorage.removeItem(STORAGE_KEYS.AUTH_TOKEN);
            localStorage.removeItem(STORAGE_KEYS.USER_DATA);
            window.location.href = 'login.html';
        }
    }

    /**
     * Get mock profile data (for demonstration)
     */
    getMockProfile() {
        return {
            first_name: 'John',
            last_name: 'Doe',
            name: 'John Doe',
            email: 'john.doe@example.com',
            phone: '+1234567890',
            department: 'IT',
            role: 'Software Engineer',
            join_date: '2023-01-15',
            photo_url: 'assets/placeholder-avatar.png',
        };
    }

    /**
     * Get mock attendance history (for demonstration)
     */
    getMockAttendanceHistory() {
        const records = [];
        const today = new Date();
        
        for (let i = 0; i < 10; i++) {
            const date = new Date(today);
            date.setDate(date.getDate() - i);
            
            records.push({
                date: date.toISOString().split('T')[0],
                check_in: new Date(date.setHours(8 + Math.floor(Math.random() * 2), Math.floor(Math.random() * 60))),
                check_out: new Date(date.setHours(17 + Math.floor(Math.random() * 2), Math.floor(Math.random() * 60))),
                status: Math.random() > 0.7 ? 'late' : 'on-time',
            });
        }
        
        return records;
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    const profileManager = new ProfileManager();
    profileManager.init();
});

export default ProfileManager;



