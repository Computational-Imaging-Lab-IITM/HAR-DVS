function norm_img = normalize_image(image)
    norm_img = image - min(image(:));
    norm_img = norm_img./max(norm_img(:));
end