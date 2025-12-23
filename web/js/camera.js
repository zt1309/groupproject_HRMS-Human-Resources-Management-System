/**
 * Camera Manager
 * Handles webcam access, video streaming, and frame capture
 */

import { CAMERA_CONFIG, RECOGNITION_CONFIG } from '../config/config.js';

export default class CameraManager {
    constructor(videoElement, canvasElement) {
        this.video = videoElement;
        this.canvas = canvasElement;
        this.ctx = canvasElement ? canvasElement.getContext('2d') : null;
        this.stream = null;
        this.isStreaming = false;
        this.isRecognizing = false;
        this.recognitionCallback = null;
        this.animationFrame = null;
        this.fps = 0;
        this.fpsCounter = 0;
        this.lastFpsTime = Date.now();
        
        // Recognition throttling
        this.frameCount = 0;
        this.lastRecognitionTime = 0;
    }

    /**
     * Start camera stream
     * @param {object} constraints - Video constraints
     * @returns {Promise<boolean>} Success status
     */
    async start(constraints = null) {
        try {
            // Check if browser supports getUserMedia
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                console.error('getUserMedia is not supported in this browser');
                alert('Trình duyệt của bạn không hỗ trợ truy cập camera. Vui lòng dùng Chrome, Edge hoặc Firefox mới nhất.');
                return false;
            }

            const videoConstraints = constraints || CAMERA_CONFIG.DEFAULT_CONSTRAINTS.video;
            
            console.log('Requesting camera access...');
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: videoConstraints,
                audio: false
            });
            
            console.log('Camera access granted');
            this.video.srcObject = this.stream;
            this.isStreaming = true;
            
            await new Promise((resolve, reject) => {
                const timeout = setTimeout(() => {
                    reject(new Error('Camera timeout - không thể tải video'));
                }, 10000); // 10 seconds timeout
                
                this.video.onloadedmetadata = () => {
                    clearTimeout(timeout);
                    console.log('Video metadata loaded');
                    this.video.play().then(() => {
                        console.log('Video playing');
                        this.startFpsCounter();
                        resolve();
                    }).catch(reject);
                };
                
                this.video.onerror = (e) => {
                    clearTimeout(timeout);
                    console.error('Video error:', e);
                    reject(new Error('Lỗi phát video'));
                };
            });
            
            console.log('Camera started successfully');
            return true;
        } catch (error) {
            console.error('Error accessing camera:', error);
            
            // Provide user-friendly error messages
            let errorMessage = 'Không thể truy cập camera. ';
            if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
                errorMessage += 'Vui lòng cho phép truy cập camera trong cài đặt browser.';
            } else if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
                errorMessage += 'Không tìm thấy camera. Vui lòng kiểm tra camera đã kết nối chưa.';
            } else if (error.name === 'NotReadableError' || error.name === 'TrackStartError') {
                errorMessage += 'Camera đang được sử dụng bởi ứng dụng khác.';
            } else {
                errorMessage += error.message || 'Lỗi không xác định.';
            }
            
            alert(errorMessage);
            return false;
        }
    }

    /**
     * Stop camera stream
     */
    stop() {
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
            this.animationFrame = null;
        }
        
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        
        if (this.video) {
            this.video.srcObject = null;
        }
        
        this.isStreaming = false;
        this.isRecognizing = false;
        this.recognitionCallback = null;
    }

    /**
     * Start face recognition
     * @param {Function} callback - Recognition callback
     */
    startRecognition(callback) {
        if (!this.isStreaming) {
            console.warn('Camera is not streaming');
            return;
        }
        
        this.isRecognizing = true;
        this.recognitionCallback = callback;
        this.frameCount = 0;
        this.lastRecognitionTime = 0;
        
        this.processFrame();
    }

    /**
     * Stop face recognition
     */
    stopRecognition() {
        this.isRecognizing = false;
        this.recognitionCallback = null;
        
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
            this.animationFrame = null;
        }
    }

    /**
     * Process video frame
     */
    processFrame() {
        if (!this.isRecognizing || !this.isStreaming) {
            return;
        }
        
        // Draw video frame to canvas
        if (this.canvas && this.ctx && this.video) {
            this.canvas.width = this.video.videoWidth;
            this.canvas.height = this.video.videoHeight;
            this.ctx.drawImage(this.video, 0, 0);
        }
        
        // Update FPS counter
        this.fpsCounter++;
        
        // Process recognition at intervals
        this.frameCount++;
        const now = Date.now();
        
        if (this.frameCount % RECOGNITION_CONFIG.RECOGNITION_INTERVAL === 0) {
            if (this.recognitionCallback && this.canvas) {
                // Check cooldown
                const timeSinceLastRecognition = now - this.lastRecognitionTime;
                if (timeSinceLastRecognition >= RECOGNITION_CONFIG.GLOBAL_COOLDOWN) {
                    // Capture frame for recognition
                    this.canvas.toBlob((blob) => {
                        if (blob && this.recognitionCallback) {
                            this.recognitionCallback(blob);
                            this.lastRecognitionTime = now;
                        }
                    }, 'image/jpeg', 0.9);
                }
            }
        }
        
        // Continue processing
        this.animationFrame = requestAnimationFrame(() => this.processFrame());
    }

    /**
     * Start FPS counter
     */
    startFpsCounter() {
        const updateFps = () => {
            const now = Date.now();
            const elapsed = now - this.lastFpsTime;
            
            if (elapsed >= CAMERA_CONFIG.FPS_INTERVAL) {
                this.fps = Math.round((this.fpsCounter * 1000) / elapsed);
                this.fpsCounter = 0;
                this.lastFpsTime = now;
                
                // Dispatch FPS event
                if (this.video) {
                    const event = new CustomEvent('fpsupdate', { detail: { fps: this.fps } });
                    this.video.dispatchEvent(event);
                }
            }
            
            if (this.isStreaming) {
                requestAnimationFrame(updateFps);
            }
        };
        
        this.lastFpsTime = Date.now();
        updateFps();
    }

    /**
     * Capture current frame as image
     * @returns {Promise<Blob>} Image blob
     */
    captureFrame() {
        return new Promise((resolve, reject) => {
            if (!this.canvas || !this.isStreaming) {
                reject(new Error('Camera is not ready'));
                return;
            }
            
            this.canvas.toBlob((blob) => {
                if (blob) {
                    resolve(blob);
                } else {
                    reject(new Error('Failed to capture frame'));
                }
            }, 'image/jpeg', 0.9);
        });
    }

    /**
     * Get current video dimensions
     * @returns {object} Width and height
     */
    getDimensions() {
        if (!this.video) {
            return { width: 0, height: 0 };
        }
        
        return {
            width: this.video.videoWidth,
            height: this.video.videoHeight
        };
    }

    /**
     * Get available cameras
     * @returns {Promise<Array>} List of available cameras
     */
    static async getAvailableCameras() {
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            return devices.filter(device => device.kind === 'videoinput');
        } catch (error) {
            console.error('Error enumerating cameras:', error);
            return [];
        }
    }
}

