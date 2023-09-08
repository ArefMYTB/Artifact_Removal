import os
import shutil

# Set the root directory where your image folders are located
root_directory = 'Data/Dataset/DHI'

# Iterate through each folder in the root directory
for folder_name in os.listdir(root_directory):
    folder_path = os.path.join(root_directory, folder_name)

    # Get a list of all image files in the current folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]

    if not image_files:
        print(f"No image files found in '{folder_name}'")
        continue

    # Create a new folder for each image in the folder
    for i, image_file in enumerate(image_files):
        name = os.path.splitext(os.path.basename(image_file))
        new_folder_path = os.path.join(root_directory, f"{folder_name}_to_{name[0]}")
        os.makedirs(new_folder_path, exist_ok=True)

        flawless_name = "flawless" + os.path.splitext(image_file)[1]
        shutil.copy2(os.path.join(folder_path, image_file), os.path.join(new_folder_path, flawless_name))

        for j, other_image in enumerate(image_files):
            if i != j:
                reference_name = f"reference_{j + 1}{os.path.splitext(other_image)[1]}"
                shutil.copy2(os.path.join(folder_path, other_image), os.path.join(new_folder_path, reference_name))

    shutil.rmtree(folder_path)

print("Processing complete.")
