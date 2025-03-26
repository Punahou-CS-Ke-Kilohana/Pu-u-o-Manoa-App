// uses functions from the plant load to put data onto screen

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using UnityEngine.UI;
using System.IO;

public class MoreSpeciesDisplay : MonoBehaviour
{
    // Assuming PlantLoader loads and manages PlantData
    private PlantLoader plantLoader = new PlantLoader();
    public TMP_Text speciesNameText;
    public TMP_Text hawaiianNameText;
    public TMP_Text speciesStatusText;
    public TMP_Text BioText;
    public RawImage speciesImage;

    // Start is called before the first frame update
    void Start()
    {
        if (PlayerPrefs.HasKey("PlantName"))
        {
            string plantName = PlayerPrefs.GetString("PlantName");
            speciesNameText.text = plantName; // Update the text in the UI
        }
        else
        {
            speciesNameText.text = "Unknown Plant"; // Fallback text
        }

        // Ensure speciesName.text has a valid value
        if (!string.IsNullOrEmpty(speciesNameText.text))
        {
            // Get Hawaiian names for the species name
            List<string> hawaiianNames = plantLoader.GetHawaiianNames(speciesNameText.text);

            string speciesStatus = plantLoader.GetConservationStatus(speciesNameText.text);
            string bio = plantLoader.GetPlantBio(speciesNameText.text);

            hawaiianNameText.text = string.Join(", ", hawaiianNames);
            speciesStatusText.text = speciesStatus;
            BioText.text = bio;

            string projectRoot = Directory.GetParent(Directory.GetParent(Application.dataPath).FullName).FullName;
            string folderPath = Path.Combine(projectRoot, "ml_backend/images/local/" + speciesNameText.text);


            if (Directory.Exists(folderPath))
            {
                // Get all image files from the folder
                string[] imageFiles = Directory.GetFiles(folderPath, "*.jpg");

                if (imageFiles.Length > 0)
                {
                    string firstImagePath = imageFiles[0];
                    StartCoroutine(LoadImage(firstImagePath));
                }
                else
                {
                    Debug.LogWarning("No images found in " + folderPath);
                }
            }
            else
            {
                Debug.LogError("Folder not found: " + folderPath);
            }

            System.Collections.IEnumerator LoadImage(string path)
            {
                byte[] imageData = File.ReadAllBytes(path);
                Texture2D texture = new Texture2D(2, 2);

                if (texture.LoadImage(imageData))
                {
                    speciesImage.texture = texture;
                }
                else
                {
                    Debug.LogError("Failed to load image: " + path);
                }
                yield return null;
            }
        }
    }
}