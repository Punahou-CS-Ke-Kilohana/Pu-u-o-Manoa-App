using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement; // switching scenes
using System.IO; // for files
using System; // for timestamps
using CompactExifLib;

public class CapImg : MonoBehaviour
{

    WebCamTexture webcam; 

    // variables for the image display and take picture button
    public RawImage cameraDisplay; 
    public Button captureButton; 
    public ImagePlacer imagePlacer;

    void Start()
    {
        // initialize the webcam
        webcam = new WebCamTexture();
        cameraDisplay.texture = webcam;
        cameraDisplay.material.mainTexture = webcam;
        webcam.Play();

        imagePlacer = GameObject.Find("ImagePlacerManager").GetComponent<ImagePlacer>();

        // Attach "CapturePhotoAndNavigate" to the onClick event of the camera button
        if (captureButton != null)
        {
            captureButton.onClick.AddListener(CapturePhotoAndNavigate);
        }
    }

    void CapturePhotoAndNavigate()
    {
        // call CapturePhoto method and take photo
        CapturePhoto();

        // Bring user to the more species information scene (scene 4 in scene manager)
        // you can edit this in build settings
        GetComponent<RunPython>().RunPythonScript();

        SceneManager.LoadScene(4);
    }

    void CapturePhoto()
    {
        // Check if the webcam is on
        if (webcam.isPlaying)
        {
            // create a 2d texture that matches the size of the webcam
            Texture2D photo = new Texture2D(webcam.width, webcam.height);

            // set texture pixels to match the webcam
            photo.SetPixels(webcam.GetPixels());
            photo.Apply();

            // Encode image as a PNG
            byte[] photoBytes = photo.EncodeToJPG();

            // Save in ImageCaptures flder
            string saveFolder = Path.Combine(Application.dataPath, "ImageCaptures");

            // If the director doesn't exist, create it
            if (!Directory.Exists(saveFolder))
            {
                Directory.CreateDirectory(saveFolder);
            }

            // Generate a unique filename with a timestamp
            string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss"); // e.g., 20241119_095112
            string fileName = $"CapturedPhoto_{timestamp}.jpg";

            // 
            string fullPath = Path.Combine(saveFolder, fileName);
            File.WriteAllBytes(fullPath, photoBytes);

            Debug.Log($"Photo saved to: {fullPath}");

            // Stop the webcam
            webcam.Stop();
            Debug.Log("Webcam has been stopped.");

            float lat = 21.1816f;
            float lon = 157.4930f;
            if (!Input.location.isEnabledByUser)
            {
                Debug.Log("Location not enabled on device or app does not have permission to access location");
            }
            else
            {
                Input.location.Start(10f, 10f);
                int maxWait = 10;
                while (Input.location.status == LocationServiceStatus.Initializing && maxWait > 0)
                {
                    new WaitForSeconds(1);
                    maxWait--;
                }
                if (maxWait < 1)
                {
                    Debug.Log("Timed out");
                }

                if (Input.location.status == LocationServiceStatus.Failed)
                {
                    Debug.LogError("Unable to determine device location");
                }
                else
                {
                    Debug.Log("Location: " + Input.location.lastData.latitude + " " + Input.location.lastData.longitude + " " + Input.location.lastData.altitude + " " + Input.location.lastData.horizontalAccuracy + " " + Input.location.lastData.timestamp);
                    lat = Input.location.lastData.latitude;
                    lon = Input.location.lastData.longitude;
                }
                Input.location.Stop();
            }

            ImagePlacer.Instance.Place(lat, lon, "e");
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

            // unused exifdata code :(
            // ExifData PictureExif;
            // GeoCoordinate lat;
            // GeoCoordinate lon;
            // DateTime DateTaken;
            // try
            // {
            //     PictureExif = new ExifData(fullPath);
            //     //PictureExif = new ExifData(Path.Combine(Path.Combine(Application.dataPath, "ImageCaptures"), "test.jpg"));
            //     PictureExif.GetDateTaken(out DateTaken);
            //     PictureExif.GetGpsLatitude(out lat);
            //     PictureExif.GetGpsLongitude(out lon);
            //     Debug.Log(DateTaken);
            //     Debug.Log(lat.Degree);
            //     Debug.Log(lon.Degree);

            // }
            // catch
            // {
            //     print("error");
            //     // Error occurred while reading image file
            // }