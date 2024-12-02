using UnityEngine;
using TMPro;

public class PlantNameDisplay : MonoBehaviour
{
    public TMP_Text plantNameText; // Reference to the TMP_Text component

    void Start()
    {
        // Retrieve the plant name from PlayerPrefs
        if (PlayerPrefs.HasKey("PlantName"))
        {
            string plantName = PlayerPrefs.GetString("PlantName");
            plantNameText.text = plantName; // Update the text in the UI
        }
        else
        {
            plantNameText.text = "Unknown Plant"; // Fallback text
        }
    }
}
