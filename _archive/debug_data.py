import numpy as np
import glob
import os
import logging
from bvh import Bvh
# We will temporarily comment out normalization to isolate the issue
# from normalization import normalize_skeleton 

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_ROOT = "Data/" 

def process_bvh_debug(filepath):
    print(f"\n--- DEBUGGING FILE: {os.path.basename(filepath)} ---")
    try:
        with open(filepath) as f:
            mocap = Bvh(f.read())
            print(f"Successfully parsed BVH. Total Frames: {mocap.nframes}")
            print(f"Joint names found: {mocap.get_joints()[:5]}...") # Print first 5 joints
    except Exception as e:
        logging.error(f"Failed to read/parse {filepath}: {e}")
        return

    # Try processing just the FIRST frame to see where it crashes
    i = 0 
    print(f"Attempting to process Frame {i}...")

    # TEST 1: Check Auto-Labeling Logic
    try:
        # Check if 'Chest' and 'Hips' exist
        chest_channels = mocap.joint_channels('Chest')
        hips_channels = mocap.joint_channels('Hips')
        print(f"Chest Channels: {chest_channels}")
        print(f"Hips Channels: {hips_channels}")

        # Try to read rotation
        # Note: Bvh library is case-sensitive. It might be 'Zrotation' or 'ZRotation'
        spine_bend = mocap.frame_joint_channel(i, 'Chest', 'Zrotation')
        print(f"Spine Bend Value: {spine_bend}")
        
    except Exception as e:
        print(f"❌ CRASHED at Auto-Labeling: {e}")
        # Stop here if this fails, as we need labels
        return 

    # TEST 2: Check Skeleton Extraction
    try:
        raw_pose = np.array(mocap.frame_pose(i))
        print(f"Raw Pose Data Length: {len(raw_pose)}")
        
        if len(raw_pose) < 51:
            print(f"❌ ERROR: Raw pose is too short! Expected >51, got {len(raw_pose)}")
        else:
            print("✅ Raw pose length is sufficient.")
            
    except Exception as e:
        print(f"❌ CRASHED at Skeleton Extraction: {e}")

if __name__ == "__main__":
    # Find just ONE file to test
    search_pattern = os.path.join(DATA_ROOT, "**", "*.bvh")
    files = glob.glob(search_pattern, recursive=True)
    
    if len(files) == 0:
        print("No files found.")
    else:
        # Run debug on the first file only
        process_bvh_debug(files[0])