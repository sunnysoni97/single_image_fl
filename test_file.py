import os

if __name__ == "__main__":
    print("Trying to create a directory")
    root_dir = "./"
    new_dir = "test_dir/test_subdir"
    final_path = os.path.join(root_dir, new_dir)
    os.makedirs(new_dir, exist_ok=True)
    print(final_path)
    print("Dir created successfully")
    