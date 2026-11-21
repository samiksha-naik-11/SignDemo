
import cv2
from pathlib import Path

# Resolve paths relative to this script so the program works from any
# current working directory.
folder = Path(__file__).resolve().parent

image_path = folder / "darkhand_files" / "sn.png"

print("Image path:", image_path)
print("File exists:", image_path.exists())

image = cv2.imread(str(image_path))
print("Image loaded:", image is not None)

if image is not None:
    print("Image dimensions:", image.shape)
def enhance_brightness(frame):

    # Convert image from BGR to LAB
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    # Split LAB channels
    l, a, b = cv2.split(lab)

    # Apply CLAHE to brightness channel
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced_l = clahe.apply(l)

    # Merge channels
    enhanced_lab = cv2.merge((enhanced_l, a, b))

    # Convert back to BGR
    enhanced = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR
    )

    return enhanced


# Test with a dark image
# Image is already loaded above

if image is None:
    raise FileNotFoundError(f"Unable to load image: {image_path}")
else:
    result = enhance_brightness(image)

    output_path = folder / "enhanced_hand.jpg"
    if not cv2.imwrite(str(output_path), result):
        raise RuntimeError(f"Unable to save enhanced image: {output_path}")
    print("Enhanced image saved to:", output_path)

    cv2.imshow("Original Image", image)
    cv2.imshow("Enhanced Image", result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
