using System.Diagnostics;
using UnityEngine;

public class RunPython : MonoBehaviour
{
    public string pythonScriptPath = "/Users/dyee25/Documents/GitHub/Pu-u-o-Manoa-App/Pu'u-o-Manoa-App/Assets/PythonScripts/test.py";
    public string inputFilePath = "/Users/dyee25/Documents/GitHub/Pu-u-o-Manoa-App/Pu'u-o-Manoa-App/Assets/PythonScripts/text.txt";

    void Start()
    {
        RunPythonScript();
    }

    public void RunPythonScript()
    {
        ProcessStartInfo start = new ProcessStartInfo
        {
            FileName = "python3", // Adjust if necessary
            Arguments = $"\"{pythonScriptPath}\" \"{inputFilePath}\"",
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true
        };

        try
        {
            using (Process process = Process.Start(start))
            {
                using (System.IO.StreamReader reader = process.StandardOutput)
                {
                    string result = reader.ReadToEnd();
                    UnityEngine.Debug.Log("Python Output: " + result);
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
