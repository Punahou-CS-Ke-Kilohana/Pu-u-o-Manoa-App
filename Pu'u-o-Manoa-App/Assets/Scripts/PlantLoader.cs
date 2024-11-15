using UnityEngine;
using System.Collections.Generic;
using Newtonsoft.Json;

public class PlantLoader : MonoBehaviour
{
    public List<string> GetPlantNames()
    {
        List <string> plantNames = new List<string>();
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
