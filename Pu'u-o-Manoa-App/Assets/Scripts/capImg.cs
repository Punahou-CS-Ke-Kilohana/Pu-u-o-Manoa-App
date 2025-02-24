using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement; // switching scenes
using System.IO; // for files
using System; // for timestamps
public class CapImg : MonoBehaviour
{
    WebCamTexture webcam;
    // variables for the image display and take picture button
    public RawImage cameraDisplay;
    public Button captureButton;
    // Reference to the background panel that needs to be black
    public Image backgroundPanel;

    void Start()
    {
        // initialize the webcam
        webcam = new WebCamTexture();
        cameraDisplay.texture = webcam;
        cameraDisplay.material.mainTexture = webcam;
        webcam.Play();
        // Attach "CapturePhotoAndNavigate" to the onClick event of the camera button
        if (captureButton != null)
        {
            captureButton.onClick.AddListener(CapturePhotoAndNavigate);
        }

        // If a background panel is assigned, make sure it's not visible during camera operation
        if (backgroundPanel != null)
        {
            backgroundPanel.enabled = false;
        }
    }

    void CapturePhotoAndNavigate()
    {
        // call CapturePhoto method and take photo
        CapturePhoto();

        // Set background to black before navigating to next scene
        SetBackgroundToBlack();

        // Wait a small amount before scene transition to ensure background appears
        StartCoroutine(NavigateAfterDelay(0.1f));
    }

    IEnumerator NavigateAfterDelay(float delay)
    {
        yield return new WaitForSeconds(delay);
        // Bring user to the more species information scene (scene 4 in scene manager)
        SceneManager.LoadScene(4);
    }

    void SetBackgroundToBlack()
    {
        // If you've assigned a background panel in the inspector
        if (backgroundPanel != null)
        {
            backgroundPanel.color = Color.black;
            backgroundPanel.enabled = true;
        }
        // If no panel is assigned, create one
        else
        {
            // Create a new GameObject with a Canvas Group component
            GameObject blackBackground = new GameObject("BlackBackground");
            Canvas canvas = FindObjectOfType<Canvas>();
            if (canvas != null)
            {
                blackBackground.transform.SetParent(canvas.transform, false);

                // Add an Image component to fill the screen with black
                Image bgImage = blackBackground.AddComponent<Image>();
                bgImage.color = Color.black;

                // Make it fill the entire screen and be behind everything else
                RectTransform rectTransform = blackBackground.GetComponent<RectTransform>();
                rectTransform.anchorMin = Vector2.zero;
                rectTransform.anchorMax = Vector2.one;
                rectTransform.sizeDelta = Vector2.zero;
                rectTransform.SetAsFirstSibling(); // Put it at the back

                // Position the black background behind the UI elements
                blackBackground.transform.SetSiblingIndex(0);
            }
            else
            {
                Debug.LogError("No Canvas found in the scene. Cannot create black background.");
            }
        }

        // Make sure the camera display is hidden
        if (cameraDisplay != null)
        {
            cameraDisplay.enabled = false;
        }
    }

    void CapturePhoto()
    {
        // Check if the webcam is on
        if (webcam.isPlaying)
        {
            // Create a 2D texture that matches the size of the webcam
            Texture2D photo = new Texture2D(webcam.width, webcam.height);
            // Set texture pixels to match the webcam
            photo.SetPixels(webcam.GetPixels());
            photo.Apply();
            // Encode image as a PNG
            byte[] photoBytes = photo.EncodeToPNG();
            // Save in ImageCaptures folder
            string saveFolder = Path.Combine(Application.dataPath, "ImageCaptures");
            // If the directory doesn't exist, create it
            if (!Directory.Exists(saveFolder))
            {
                Directory.CreateDirectory(saveFolder);
            }
            // Generate a unique filename with a timestamp
            string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss"); // e.g., 20241119_095112
            string fileName = $"CapturedPhoto_{timestamp}.png";
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