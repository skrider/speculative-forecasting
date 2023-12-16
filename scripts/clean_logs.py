import os
import sys
from tensorboard.backend.event_processing import event_accumulator

def delete_small_logs(directory_path):
    # Iterate over all directories and files recursively
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            # Check if the file is a TensorBoard log file
            if file.endswith(".tfevents"):
                file_path = os.path.join(root, file)
                
                # Create an EventAccumulator and load the log file
                ea = event_accumulator.EventAccumulator(file_path)
                ea.Reload()
                
                __import__('pdb').set_trace()
                # Get all scalar keys
                scalar_keys = ea.Tags()['scalars']

                # Iterate over all scalar keys
                for key in scalar_keys:
                    steps = ea.Scalars(key)
                    
                    # If the log file has fewer than 10000 steps, delete it
                    if len(steps) < 10000:
                        os.remove(file_path)
                        break  # No need to check other keys if file is already deleted

delete_small_logs(sys.argv[1])
