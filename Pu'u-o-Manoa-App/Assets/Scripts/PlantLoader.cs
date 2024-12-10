// functions to load from the json file to return the data

using UnityEngine;
using System.Collections.Generic;
using Newtonsoft.Json;
using System.Linq;


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

    public List<string> GetHawaiianNames(string inputPlantName)
    {
        // Load the JSON file
        TextAsset jsonFile = Resources.Load<TextAsset>("plant_data");
        if (jsonFile == null)
        {
            Debug.LogError("Plant data JSON file not found!");
            return new List<string> { "Plant data file not found." };
        }

        // Deserialize the JSON file into the PlantData object
        PlantData plantData = JsonConvert.DeserializeObject<PlantData>(jsonFile.text);
        string newInputPlantName = inputPlantName.Replace(" ", "_");

        // Check if the plant exists in the dictionary
        if (plantData.Plants.TryGetValue(newInputPlantName, out PlantInfo plantInfo))
        {
            var hawaiianNames = plantInfo.HawaiianNameswithDiacritics;
            if (hawaiianNames != null && hawaiianNames.Count > 0)
            {
                return hawaiianNames;
            }
            else
            {
                return new List<string> { "No Hawaiian names found for the given input." };
            }
        }
        else
        {
            return new List<string> { "Plant name not found in the data." };
        }
    }

    public string GetConservationStatus(string inputPlantName)
    {
        // Load the JSON file
        TextAsset jsonFile = Resources.Load<TextAsset>("plant_data");
        if (jsonFile == null)
        {
            Debug.LogError("Plant data JSON file not found!");
            return "Plant data file not found.";
        }

        // Deserialize the JSON file into the PlantData object
        PlantData plantData = JsonConvert.DeserializeObject<PlantData>(jsonFile.text);
        string newInputPlantName = inputPlantName.Replace(" ", "_");

        // Check if the plant exists in the dictionary
        if (plantData.Plants.TryGetValue(newInputPlantName, out PlantInfo plantInfo))
        {
            var speciesStatus = plantInfo.EndangeredSpeciesStatus;
            if (speciesStatus != null)
            {
                return speciesStatus;
            }
            else
            {
                return "Conservation Status: N/A";
            }
        }
        else
        {
            return "Plant name not found in the data.";
        }
    }

    public string GetPlantBio(string inputPlantName)
    {
        // Load the JSON file
        TextAsset jsonFile = Resources.Load<TextAsset>("plant_data");
        if (jsonFile == null)
        {
            Debug.LogError("Plant data JSON file not found!");
            return "Plant data file not found.";
        }

        // Deserialize the JSON file into the PlantData object
        PlantData plantData = JsonConvert.DeserializeObject<PlantData>(jsonFile.text);
        string newInputPlantName = inputPlantName.Replace(" ", "_");

        // Check if the plant exists in the dictionary
        if (plantData.Plants.TryGetValue(newInputPlantName, out PlantInfo plantInfo))
        {
            var bio = plantInfo.bio;
            if (bio != null)
            {
                return bio;
            }
            else
            {
                return "No Bio for this plant";
            }
        }
        else
        {
            return "Plant name not found in the data.";
        }
    }
}
