using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement; // For scene navigation
using System.IO; // For file handling
using System; // For DateTime

public class CapImg : MonoBehaviour
{
    WebCamTexture webcam; // To get the camera feed
    public RawImage cameraDisplay; // UI RawImage to display the camera feed
    public Button captureButton; // Button to capture and save the image

    void Start()
    {
        // Initialize the webcam
        webcam = new WebCamTexture();
        cameraDisplay.texture = webcam;
        cameraDisplay.material.mainTexture = webcam;
        webcam.Play();

        // Attach the CapturePhotoAndNavigate method to the button's onClick event
        if (captureButton != null)
        {
            captureButton.onClick.AddListener(CapturePhotoAndNavigate);
        }
    }

    void CapturePhotoAndNavigate()
    {
        // Perform photo capture functionality
        CapturePhoto();

        // Navigate to the specific scene (Scene index 4)
        SceneManager.LoadScene(4);
    }

    void CapturePhoto()
    {
        // Check if the webcam is ready
        if (webcam.isPlaying)
        {
            // Create a Texture2D with the same size as the webcam feed
            Texture2D photo = new Texture2D(webcam.width, webcam.height);

            // Copy pixels from the webcam feed to the Texture2D
            photo.SetPixels(webcam.GetPixels());
            photo.Apply();

            // Encode the Texture2D to a PNG (or use EncodeToJPG for JPG format)
            byte[] photoBytes = photo.EncodeToPNG();

            // Construct the save path within the Assets/ImageCaptures folder
            string saveFolder = Path.Combine(Application.dataPath, "ImageCaptures");

            // Ensure the target folder exists
            if (!Directory.Exists(saveFolder))
            {
                Directory.CreateDirectory(saveFolder);
            }

            // Generate a unique file name using a timestamp
            string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss"); // e.g., 20241119_095112
            string fileName = $"CapturedPhoto_{timestamp}.png";

            // Full path to save the file
            string fullPath = Path.Combine(saveFolder, fileName);
            File.WriteAllBytes(fullPath, photoBytes);

            Debug.Log($"Photo saved to: {fullPath}");

            // Stop the webcam
            webcam.Stop();
            Debug.Log("Webcam has been stopped.");
        }
        else
        {
            Debug.LogError("Webcam is not active.");
        }
    }

    void OnDestroy()
    {
        // Stop the webcam when the script is destroyed
        if (webcam != null)
        {
            webcam.Stop();
        }
    }
}
