using System.Diagnostics;
using UnityEngine;
using System.IO;
using System;
using UnityEngine.SceneManagement;
using Newtonsoft.Json;
using System.Collections.Generic;
using Debug = UnityEngine.Debug;

public class RunPython : MonoBehaviour
{
    public string pythonScriptRelativePath = "ml_backend/main.py";

    public void RunPythonScript()
    {
        string projectRoot = Directory.GetParent(Directory.GetParent(Application.dataPath).FullName).FullName;
        string pythonScriptPath = Path.Combine(projectRoot, pythonScriptRelativePath);
        string pythonPath = Path.Combine(projectRoot, "ml_backend");
        ProcessStartInfo start = new ProcessStartInfo
        {
            FileName = "python3",
            Arguments = $"\"{pythonScriptPath}\" --s interpret --d cpu --n test_model --e 10",
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true
        };

        start.EnvironmentVariables["PYTHONPATH"] = pythonPath;

        try
        {
            using (Process process = Process.Start(start))
            {
                using (System.IO.StreamReader reader = process.StandardOutput)
                {
                    string result = reader.ReadToEnd();
                    string[] lines = result.Split(new[] { '\r', '\n' }, StringSplitOptions.None);
                    string lastLine = lines[lines.Length - 2].Trim();
                    lastLine = lastLine.ToLower();
                    lastLine = lastLine.Replace("_", " ");
                    lastLine = char.ToUpper(lastLine[0]) + lastLine.Substring(1);
                    UnityEngine.Debug.Log("Python Output: " + lastLine);

                    PlantLoader plantLoader = new PlantLoader();

                    List<string> plantNames = plantLoader.GetPlantNames();

                    foreach (var plant in plantNames)
                    {
                        PlayerPrefs.SetString("PlantName", plant);
                        List<string> commonNames = plantLoader.GetCommonNames("Abutilon eremitopetalum");

                        foreach (var commonName in commonNames)
                        {
                            Debug.Log(commonName);
                            if (commonName == lastLine)
                            {
                                // Save the plant name in PlayerPrefs
                                PlayerPrefs.SetString("PlantName", plant);
                            }
                        }
                    }
                    PlayerPrefs.Save(); // Optional, ensures data is written immediately

                    // Load the next scene
                    SceneManager.LoadScene(4);
                }

                using (System.IO.StreamReader errorReader = process.StandardError)
                {
                    string error = errorReader.ReadToEnd();
                    if (!string.IsNullOrEmpty(error))
                    {
                        UnityEngine.Debug.LogError("Python Error: " + error);
                    }
                }
            }
        }
        catch (System.Exception ex)
        {
            UnityEngine.Debug.LogError("Failed to run Python script: " + ex.Message);
        }
    }
}
