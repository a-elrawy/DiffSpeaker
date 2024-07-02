import os
import json
import shutil
# Define the function to compute the duration from the filename
def compute_duration(start_time, end_time):
    return int(end_time) - int(start_time)

# Define the main function to process the files and create the JSON structure
def process_files(file_list):
    data = {}
    
    for file in file_list:
        parts = file.split('#')
        subject_id = parts[0]
        start_time = parts[2]
        end_time = parts[3].split('.')[0]
        file_path = file
        
        duration = compute_duration(start_time, end_time)
        
        if subject_id not in data:
            data[subject_id] = {
                'files': [],
                'duration': 0
            }
        
        data[subject_id]['files'].append(file_path)
        data[subject_id]['duration'] += duration
    
    # Convert the data to the desired JSON format
    json_data = []
    for subject_id, info in data.items():
        json_data.append({
            'subject_id': subject_id,
            'files': info['files'],
            'duration': info['duration']
        })
    
    return json_data


# Function to get top N subjects by duration
def get_top_n_subjects(json_data, n=10):
    sorted_data = sorted(json_data, key=lambda x: x['duration'], reverse=True)
    return sorted_data[:n]


def move_top_n(top_subjects, base_dir='voxceleb'):
    """
    Create directories and move files to the respective directories.
    New structure:
    VoxCeleb
    ├── audio
    │   ├── file1.wav
    │   ├── file2.wav
    │   └── ...
    ├── flame
    │   ├── file1.npz
    │   ├── file2.npz
    │   └── ...

    
    """
    os.makedirs(base_dir, exist_ok=True)
    audio_path = os.path.join(base_dir, 'audio')
    os.makedirs(audio_path, exist_ok=True)
    flame_path = os.path.join(base_dir, 'flame')
    os.makedirs(flame_path, exist_ok=True)



    for subject in top_subjects:
        
        for file_path in subject['files']:
            # Assuming file_path includes the full path. If not, modify accordingly.
            audio_file = os.path.join('VoxCeleb', file_path)
            flame_file = file_path.replace('.wav', '.npz')
            flame_file = os.path.join('VoxCeleb', flame_file)
            try:
                shutil.move(audio_file, audio_path)
                shutil.move(flame_file, flame_path)
            except:
                print(f"Error moving {audio_file} and {flame_file}")
                continue

            
    print("Files moved successfully.")
        
# Example file list
file_list = os.listdir('VoxCeleb') 
file_list = [f for f in file_list if f.endswith('.wav')]


# Process the files
json_data = process_files(file_list)

# Get the top 10 subjects by duration
top_subjects = get_top_n_subjects(json_data, n=10)

print([t['subject_id'] for t in top_subjects])
print([t['duration'] for t in top_subjects])
print([len(t['files']) for t in top_subjects])
# Create directories and copy files
# create_directories_and_copy_files(top_subjects)

# Move files to the respective directories
move_top_n(top_subjects)

print("Directories created and files copied successfully.")



# ['id10715', 'id11181', 'id11211', 'id10231', 'id10756', 'id10931', 'id10104', 'id10786', 'id10720', 'id11182']
# [75787, 54425, 42142, 35317, 32065, 30732, 25024, 21621, 21354, 20607]
# [308, 221, 161, 175, 118, 116, 126, 119, 109, 90]