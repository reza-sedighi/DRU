import os
import shutil

# Cities split into train, val, and test
train_cities = [
    "aachen", "bochum", "bremen", "cologne", "darmstadt", "dusseldorf", 
    "erfurt", "hamburg", "hanover", "jena", "krefeld", "monchengladbach", 
    "strasbourg", "stuttgart", "tubingen", "ulm", "weimar", "zurich"
]

val_cities = ["frankfurt", "lindau", "munster"]

# test_cities = ["berlin", "bielefeld", "bonn", "leverkusen", "mainz", "munich"]

# Paths to the original and new directories
base_path = "/home/reza/Desktop/Dissertation/New/ConfMix/leftImg8bit_trainvaltest_foggy/leftImg8bit_foggy"
new_base_path = "/home/reza/Desktop/Dissertation/New/ConfMix/image_beta0.02"

# Function to handle the file moving process
def move_images(set_type, cities):
    original_set_path = os.path.join(base_path, set_type)
    new_set_path = os.path.join(new_base_path, set_type)
    
    # Create the new directory for train, val, test
    os.makedirs(new_set_path, exist_ok=True)
    
    # Loop through each city folder within the current set_type
    for city in cities:
        original_city_path = os.path.join(original_set_path, city)
        new_city_path = os.path.join(new_set_path, city)
        
        # Create the corresponding city directory in the new location
        os.makedirs(new_city_path, exist_ok=True)
        
        # Check if the city directory exists in the original path
        if os.path.exists(original_city_path):
            # Loop through each image in the city folder
            for image_name in os.listdir(original_city_path):
                # Move only images containing "_beta_0.02" in their name
                if "_beta_0.02" in image_name:
                    original_image_path = os.path.join(original_city_path, image_name)
                    new_image_path = os.path.join(new_city_path, image_name)
                    
                    # Move the image to the new directory
                    shutil.move(original_image_path, new_image_path)
                    print(f"Moved: {original_image_path} -> {new_image_path}")
        else:
            print(f"City folder not found: {original_city_path}")

# Process each set (train, val, test) with the correct cities
move_images("train", train_cities)
move_images("val", val_cities)
# move_images("test", test_cities)
