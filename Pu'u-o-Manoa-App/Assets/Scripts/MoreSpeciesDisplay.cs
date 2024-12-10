// uses functions from the plant load to put data onto screen

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class MoreSpeciesDisplay : MonoBehaviour
{
    // Assuming PlantLoader loads and manages PlantData
    private PlantLoader plantLoader = new PlantLoader();
    public TMP_Text speciesNameText;
    public TMP_Text hawaiianNameText;
    public TMP_Text speciesStatusText;
    public TMP_Text BioText; 

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
        }
    }
}