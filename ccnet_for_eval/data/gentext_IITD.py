import os

left_dir = "./data/script2_after/Left"
right_dir = "./data/script2_after/Right"

# Train = Left Hand
with open("train_script2.txt", "w") as f:
    files = sorted(os.listdir(left_dir))

    for filename in files:
        if not filename.endswith(".bmp"):
            continue

        person_id = int(filename.split("_")[0])

        image_path = os.path.join(left_dir, filename).replace("\\", "/")

        f.write(f"{image_path} {person_id-1}\n")

# Test = Right Hand
with open("test_script2.txt", "w") as f:
    files = sorted(os.listdir(right_dir))

    for filename in files:
        if not filename.endswith(".bmp"):
            continue

        person_id = int(filename.split("_")[0])

        image_path = os.path.join(right_dir, filename).replace("\\", "/")

        f.write(f"{image_path} {person_id-1}\n")

print("Done!")