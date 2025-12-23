"""
Test script to verify anti-spoofing configuration
"""
import cv2
import numpy as np
from src.antispoof import check_liveness
from config import ENABLE_ANTISPOOF, ANTISPOOF_THRESHOLD

print("=" * 60)
print("ANTI-SPOOFING CONFIGURATION TEST")
print("=" * 60)
print(f"ENABLE_ANTISPOOF: {ENABLE_ANTISPOOF}")
print(f"ANTISPOOF_THRESHOLD: {ANTISPOOF_THRESHOLD}")
print("=" * 60)

# Test with a dummy face image
print("\n[TEST] Creating test face image...")
test_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)

print("[TEST] Running liveness check...")
result = check_liveness(test_face)

print(f"\n[RESULT] Liveness check returned: {result}")
print(f"[RESULT] Type: {'REAL' if result else 'FAKE'}")

print("\n" + "=" * 60)
print("TEST COMPLETED")
print("=" * 60)
print("\nNOTE: To adjust anti-spoofing behavior:")
print("1. Edit config.py")
print("2. Set ENABLE_ANTISPOOF = False to disable")
print("3. Adjust ANTISPOOF_THRESHOLD (0.0-1.0)")
print("   - Higher = stricter (more rejections)")
print("   - Lower = lenient (more acceptances)")
print("=" * 60)
