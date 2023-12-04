import subprocess
import sys

def run_evaluation_script(clean_data_filename, poisoned_data_filename, model_filename, threshold):
    command = [
        "python",      # assuming your script is a Python script
        "eval.py",     # replace with the actual path to your script
        clean_data_filename,
        poisoned_data_filename,
        model_filename,
        str(threshold)  # pass the threshold as a command-line argument
    ]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error while running the script: {e}")
        print(e.stderr)
        sys.exit(1)

if __name__ == "__main__":
    # Set a default threshold
    default_threshold = 10

    # Check if a threshold is provided as a command-line argument
    if len(sys.argv) > 1:
        threshold = int(sys.argv[1])
    else:
        threshold = default_threshold

    print("For Pruning Threshold: " + str(threshold) + '%')

    clean_data_filename = "data/cl/test.h5"
    poisoned_data_filename = "data/bd/bd_test.h5"
    model_filename = "models/Repaired_net_" + str(threshold) + "_percent_threshold.h5"

    run_evaluation_script(clean_data_filename, poisoned_data_filename, model_filename, threshold)
