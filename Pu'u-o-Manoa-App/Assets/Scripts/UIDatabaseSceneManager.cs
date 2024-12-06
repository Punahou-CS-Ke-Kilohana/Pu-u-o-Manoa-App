// passes data from UI database scene into more species page

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using TMPro;

public class UIDatabaseSceneManager : MonoBehaviour
{
    public TMP_Text plantName;

    public void StartMoreSpeciesScene()
    {
        // Save the plant name in PlayerPrefs
        PlayerPrefs.SetString("PlantName", plantName.text);
        PlayerPrefs.Save(); // Optional, ensures data is written immediately

        // Load the next scene
        SceneManager.LoadScene(4);
    }

}

