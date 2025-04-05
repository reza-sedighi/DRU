import json
import os

def filter_car_annotations(input_json_path, output_json_path):
    # Read the original JSON file
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    # Filter categories to keep only 'car' category
    car_categories = [cat for cat in data['categories'] if cat['name'] == 'car']
    car_category_ids = [cat['id'] for cat in car_categories]
    
    # Filter annotations to keep only car annotations
    car_annotations = [ann for ann in data['annotations'] 
                      if ann['category_id'] in car_category_ids]
    
    # Get unique image IDs that contain cars
    car_image_ids = set(ann['image_id'] for ann in car_annotations)
    
    # Filter images to keep only those containing cars
    car_images = [img for img in data['images'] 
                 if img['id'] in car_image_ids]
    
    # Create new data structure
    car_only_data = {
        'images': car_images,
        'annotations': car_annotations,
        'categories': car_categories
    }
    
    # Save to new JSON file
    with open(output_json_path, 'w') as f:
        json.dump(car_only_data, f, indent=2)
    
    print(f"Processed {len(car_annotations)} car annotations in {len(car_images)} images")
    print(f"Saved to {output_json_path}")

def main():
    # Process both train and val datasets
    filter_car_annotations(
        'cityscapes_train_cocostyle.json',
        'cityscapes_train_caronly_cocostyle.json'
    )
    filter_car_annotations(
        'cityscapes_val_cocostyle.json',
        'cityscapes_val_caronly_cocostyle.json'
    )

if __name__ == '__main__':
    main()