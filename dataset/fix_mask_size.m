fileList = dir(fullfile("to_fix", '**', '*.png'));

for filename={fileList.name}
    mask = imread("to_fix/"+filename);
    mask = imresize(im2gray(mask), [572 572]);
    imwrite(mask, "to_fix_out/"+filename)
end