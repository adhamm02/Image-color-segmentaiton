import cv2
import numpy as np
from sklearn.cluster import KMeans
import os

def cluster_image(image_path, k):
    """
    Perform KMeans clustering on an image to reduce colors and segment it into k clusters.

    Parameters:
        image_path (str): The path to the image file.
        k (int): The number of color clusters (must be between 1 and 256).

    Returns:
        List[np.ndarray]: A list of images, each representing one color cluster.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the image could not be loaded or if k is out of valid range.
    """

    # Check that the file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError("Image path does not exist.")

    # Check that k is a valid number of clusters
    if k <= 0 or k > 256:
        raise ValueError("Invalid number of clusters (k). Must be 1–256.")

    # Load image using OpenCV (BGR format by default)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Failed to load image.")

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Flatten image from (H, W, 3) to (H*W, 3) for clustering
    pixels = image.reshape((-1, 3))

    # Apply KMeans clustering to pixel colors
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pixels)

    # Get cluster labels and cluster centers
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # Prepare an image for each cluster
    clustered_images = []
    for cluster_num in range(k):
        # Create a mask for pixels that belong to this cluster
        mask = (labels == cluster_num).reshape(image.shape[:2])

        # Create a blank image and paint the cluster color where the mask is true
        cluster_img = np.zeros_like(image)
        cluster_img[mask] = centers[cluster_num].astype(np.uint8)

        # Add the cluster image to the list
        clustered_images.append(cluster_img)

    return clustered_images


# -----------------------------
# Test Cases for Validation
# -----------------------------

def test_valid_clustering():
    """
    Test case to verify that clustering works on a valid image with a valid k value.
    """
    test_image = "test_data/test.jpg"  # Change this path if needed
    try:
        clusters = cluster_image(test_image, 3)
        assert len(clusters) == 3
        print("✅ test_valid_clustering passed")
    except Exception as e:
        print(f"❌ test_valid_clustering failed: {e}")

def test_invalid_path():
    """
    Test case to verify that a non-existent image path raises FileNotFoundError.
    """
    try:
        cluster_image("non_existent.jpg", 3)
        print("❌ test_invalid_path failed (did not raise error)")
    except FileNotFoundError:
        print("✅ test_invalid_path passed")

def test_invalid_k():
    """
    Test case to verify that invalid k values raise ValueError.
    """
    try:
        cluster_image("test_data/test.jpg", -1)  # Invalid k value
        print("❌ test_invalid_k failed (did not raise error)")
    except ValueError:
        print("✅ test_invalid_k passed")

# Main function to run all tests
if __name__ == "__main__":
    test_valid_clustering()
    test_invalid_path()
    test_invalid_k()
