using UnityEngine;
using System.Collections.Generic;
using Newtonsoft.Json;


// loads the plant names from the Json file
public class PlantLoader : MonoBehaviour
{
    // creates a list of strings by calling the GetPlantNames() method
    public List<string> GetPlantNames()
    {
        // declares a new list
        List <string> plantNames = new List<string>();

        // load a TextAsset from the json file in "Resources" directory
        TextAsset jsonFile = Resources.Load<TextAsset>("plant_data");
        if (jsonFile != null)
        {
            // Deserialize JSON data into PlantData using Newtonsoft.Json
            PlantData plantData = JsonConvert.DeserializeObject<PlantData>(jsonFile.text);

            // Print the name of every plant with underscores replaced by spaces
            foreach (var plant in plantData.Plants)
            {
                string plantName = plant.Key.Replace("_", " ");
                //Debug.Log("Plant Name: " + plantName);
                plantNames.Add(plantName);
            }
        }
        else
        {
            Debug.LogError("JSON file not found!");
        }
        return plantNames;
    }
}
