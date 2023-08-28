We introduce a bran new dataset called DHI which is short for Distorted Human Image.

This dataset contains 3 major distortion type "Deformation", "Texture" and "None:

Deformation

│

├── 1 ├── Distorted -- Mask -- Original

├── 2

├── ...

└── n

Mask & Original images may be more than one image.(e.g. Mask_1 -- Mask_2)

Original_1 consistently represents the non-defective version of the distorted image across all dataset.

Same thing for Texture.

For "None" we simply copy the non-defective image as distorted.
