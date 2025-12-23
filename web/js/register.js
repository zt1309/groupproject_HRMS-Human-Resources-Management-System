/**
 * Registration Manager
 * Handles employee registration with face capture
 */

import { registerEmployee } from './api.js';
import { showToast, isValidEmail, isValidPhone } from './utils.js';
import CameraManager from './camera.js';

export class RegistrationManager {
    constructor() {
        this.currentStep = 1;
        this.totalSteps = 3;
        this.capturedPhotos = [];
        this.employeeData = {};
        this.camera = null;
    }

    /**
     * Initialize registration process
     * @param {HTMLElement} videoElement - Video element
     * @param {HTMLElement} canvasElement - Canvas element
     */
    init(videoElement, canvasElement) {
        this.camera = new CameraManager(videoElement, canvasElement);
        this.updateStepIndicator();
    }

    /**
     * Go to next step
     */
    nextStep() {
        if (this.currentStep < this.totalSteps) {
            this.currentStep++;
            this.updateStepIndicator();
            this.showCurrentStep();
        }
    }

    /**
     * Go to previous step
     */
    previousStep() {
        if (this.currentStep > 1) {
            this.currentStep--;
            this.updateStepIndicator();
            this.showCurrentStep();
        }
    }

    /**
     * Go to specific step
     * @param {number} step - Step number
     */
    goToStep(step) {
        if (step >= 1 && step <= this.totalSteps) {
            this.currentStep = step;
            this.updateStepIndicator();
            this.showCurrentStep();
        }
    }

    /**
     * Update step indicator
     */
    updateStepIndicator() {
        const steps = document.querySelectorAll('.step');
        steps.forEach((step, index) => {
            if (index + 1 === this.currentStep) {
                step.classList.add('active');
            } else {
                step.classList.remove('active');
            }
        });
    }

    /**
     * Show current step content
     */
    showCurrentStep() {
        const steps = document.querySelectorAll('.registration-step');
        steps.forEach((step, index) => {
            if (index + 1 === this.currentStep) {
                step.classList.add('active');
            } else {
                step.classList.remove('active');
            }
        });
    }

    /**
     * Validate step 1 (Personal Information)
     * @returns {boolean} True if valid
     */
    validateStep1() {
        const firstName = document.getElementById('firstName')?.value.trim();
        const lastName = document.getElementById('lastName')?.value.trim();
        const employeeId = document.getElementById('employeeId')?.value.trim();
        const email = document.getElementById('email')?.value.trim();
        const phone = document.getElementById('phone')?.value.trim();
        const department = document.getElementById('department')?.value;

        if (!firstName || !lastName || !employeeId || !email || !phone || !department) {
            showToast('Please fill in all required fields', 'error');
            return false;
        }

        if (!isValidEmail(email)) {
            showToast('Please enter a valid email address', 'error');
            return false;
        }

        if (!isValidPhone(phone)) {
            showToast('Please enter a valid phone number', 'error');
            return false;
        }

        this.employeeData = {
            first_name: firstName,
            last_name: lastName,
            employee_id: employeeId,
            email: email,
            phone: phone,
            department: department,
        };

        return true;
    }

    /**
     * Capture photo
     * @returns {Promise<Blob>} Captured photo
     */
    async capturePhoto() {
        if (!this.camera || !this.camera.isStreaming) {
            showToast('Camera is not ready', 'error');
            return null;
        }

        try {
            const photo = await this.camera.captureFrame();
            return photo;
        } catch (error) {
            console.error('Error capturing photo:', error);
            showToast('Failed to capture photo', 'error');
            return null;
        }
    }

    /**
     * Capture multiple photos with countdown
     * @param {number} count - Number of photos to capture
     * @returns {Promise<Array<Blob>>} Array of captured photos
     */
    async capturePhotos(count = 5) {
        const photos = [];
        const countdownElement = document.querySelector('.capture-countdown');

        for (let i = 0; i < count; i++) {
            // Show countdown
            if (countdownElement) {
                countdownElement.textContent = count - i;
                countdownElement.style.display = 'block';
            }

            // Wait before capture
            await new Promise(resolve => setTimeout(resolve, 1000));

            // Capture photo
            const photo = await this.capturePhoto();
            if (photo) {
                photos.push(photo);
                this.capturedPhotos.push(photo);
                this.updatePhotoPreview(photos.length - 1, photo);
            }

            // Hide countdown
            if (countdownElement) {
                countdownElement.style.display = 'none';
            }

            // Wait between captures
            if (i < count - 1) {
                await new Promise(resolve => setTimeout(resolve, 500));
            }
        }

        return photos;
    }

    /**
     * Update photo preview
     * @param {number} index - Photo index
     * @param {Blob} photo - Photo blob
     */
    updatePhotoPreview(index, photo) {
        const photosGrid = document.querySelector('.photos-grid');
        if (!photosGrid) return;

        const photoItem = photosGrid.children[index];
        if (photoItem) {
            const img = document.createElement('img');
            img.src = URL.createObjectURL(photo);
            photoItem.innerHTML = '';
            photoItem.appendChild(img);
            photoItem.classList.remove('photo-placeholder');
            photoItem.classList.add('photo-item');
        }
    }

    /**
     * Submit registration
     * @returns {Promise<boolean>} Success status
     */
    async submitRegistration() {
        if (this.capturedPhotos.length < 5) {
            showToast('Please capture at least 5 photos', 'error');
            return false;
        }

        try {
            showToast('Submitting registration...', 'info');

            const result = await registerEmployee(this.employeeData, this.capturedPhotos);

            if (result) {
                showToast('Registration successful!', 'success');
                return true;
            } else {
                showToast('Registration failed', 'error');
                return false;
            }
        } catch (error) {
            console.error('Registration error:', error);
            showToast('Registration failed: ' + error.message, 'error');
            return false;
        }
    }

    /**
     * Reset registration form
     */
    reset() {
        this.currentStep = 1;
        this.capturedPhotos = [];
        this.employeeData = {};
        this.updateStepIndicator();
        this.showCurrentStep();

        // Reset form
        const form = document.getElementById('registrationForm');
        if (form) {
            form.reset();
        }

        // Reset photo previews
        const photosGrid = document.querySelector('.photos-grid');
        if (photosGrid) {
            photosGrid.querySelectorAll('.photo-item').forEach(item => {
                item.classList.remove('photo-item');
                item.classList.add('photo-placeholder');
                item.innerHTML = `${Array.from(photosGrid.children).indexOf(item) + 1}`;
            });
        }
    }
}

export default RegistrationManager;



