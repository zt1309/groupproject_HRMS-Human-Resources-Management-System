/**
 * Attendance Manager
 * Handles attendance logging and management
 */

import { ATTENDANCE_CONFIG } from '../config/config.js';
import { formatTime, formatDate } from './utils.js';

export class AttendanceManager {
    constructor() {
        this.records = [];
        this.lastCheckIn = null;
    }

    /**
     * Check if employee should be marked as late
     * @param {Date} checkInTime - Check-in time
     * @returns {boolean} True if late
     */
    isLate(checkInTime) {
        const workStart = new Date(checkInTime);
        workStart.setHours(ATTENDANCE_CONFIG.WORK_START_HOUR, ATTENDANCE_CONFIG.WORK_START_MINUTE, 0, 0);
        
        const lateThreshold = new Date(workStart);
        lateThreshold.setMinutes(lateThreshold.getMinutes() + ATTENDANCE_CONFIG.LATE_THRESHOLD);
        
        return checkInTime > lateThreshold;
    }

    /**
     * Get attendance status
     * @param {Date} checkInTime - Check-in time
     * @returns {string} Status: 'on-time', 'late', or 'absent'
     */
    getStatus(checkInTime) {
        if (!checkInTime) {
            return 'absent';
        }
        
        return this.isLate(checkInTime) ? 'late' : 'on-time';
    }

    /**
     * Format attendance record
     * @param {object} record - Raw attendance record
     * @returns {object} Formatted record
     */
    formatRecord(record) {
        const checkInTime = record.check_in ? new Date(record.check_in) : null;
        const checkOutTime = record.check_out ? new Date(record.check_out) : null;
        
        return {
            ...record,
            date: formatDate(record.date || checkInTime),
            check_in: checkInTime ? formatTime(checkInTime) : '-',
            check_out: checkOutTime ? formatTime(checkOutTime) : '-',
            status: this.getStatus(checkInTime),
            timestamp: checkInTime || new Date(record.date),
        };
    }

    /**
     * Group records by date
     * @param {Array} records - Attendance records
     * @returns {object} Records grouped by date
     */
    groupByDate(records) {
        const grouped = {};
        
        records.forEach(record => {
            const date = record.date || new Date(record.timestamp).toISOString().split('T')[0];
            if (!grouped[date]) {
                grouped[date] = [];
            }
            grouped[date].push(record);
        });
        
        return grouped;
    }

    /**
     * Calculate statistics from records
     * @param {Array} records - Attendance records
     * @returns {object} Statistics
     */
    calculateStatistics(records) {
        const stats = {
            total: records.length,
            present: 0,
            late: 0,
            absent: 0,
        };
        
        records.forEach(record => {
            const status = record.status || this.getStatus(record.check_in ? new Date(record.check_in) : null);
            
            if (status === 'on-time') {
                stats.present++;
            } else if (status === 'late') {
                stats.late++;
            } else {
                stats.absent++;
            }
        });
        
        return stats;
    }
}

export default AttendanceManager;

