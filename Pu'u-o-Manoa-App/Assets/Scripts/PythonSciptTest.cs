using System.Diagnostics;
using UnityEngine;
using System.IO;

public class RunPython : MonoBehaviour
{
    public string pythonScriptRelativePath = "ml_backend/main.py";
    public void Start()
    {
        RunPythonScript();
    }
   
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
