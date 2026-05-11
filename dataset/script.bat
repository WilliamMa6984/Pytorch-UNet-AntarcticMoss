dir train /b | grep -v ".csv" | grep -v "mask" > "train_img.txt"
dir train /b | grep -v ".csv" | grep "mask" > "train_mask.txt"

dir valid /b | grep -v ".csv" | grep -v "mask" > "valid_img.txt"
dir valid /b | grep -v ".csv" | grep "mask" > "valid_mask.txt"