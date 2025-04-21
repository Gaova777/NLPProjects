from glob import glob
import pandas as pd

def load_subtitles_dataset(dataset_path):
    subtitles_path = glob(dataset_path + "/*.ass")
    
    scripts = []
    episode_num = []

    for path in subtitles_path:
        try:
            # Read Lines with UTF-8 encoding
            with open(path, 'r', encoding="utf-8") as file:
                lines = file.readlines()
                lines = lines[27:]  # Skip metadata/header lines
                lines = [",".join(line.split(',')[9:]) for line in lines]

            lines = [line.replace('\\N', '') for line in lines]
            script = " ".join(lines)

            episode = int(path.split('-')[-1].split('.')[0].strip())

            scripts.append(script)
            episode_num.append(episode)
        
        except UnicodeDecodeError:
            print(f"Error decoding file: {path}. Trying UTF-8-SIG encoding...")
            with open(path, 'r', encoding="utf-8-sig") as file:
                lines = file.readlines()
                lines = lines[27:]
                lines = [",".join(line.split(',')[9:]) for line in lines]

            lines = [line.replace('\\N', '') for line in lines]
            script = " ".join(lines)

            episode = int(path.split('-')[-1].split('.')[0].strip())

            scripts.append(script)
            episode_num.append(episode)
    
    df = pd.DataFrame({"episode": episode_num, "script": scripts})
    return df
