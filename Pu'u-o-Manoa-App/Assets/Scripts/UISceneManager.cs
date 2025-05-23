// each scene has its own scene manager which we may want to change later

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class UISceneManager : MonoBehaviour
{
    public static UISceneManager Instance;

    public void StartAIScene()
    {
        SceneManager.LoadScene(2);
    }

    public void StartMapScene()
    {
        SceneManager.LoadScene(1);
    }

    public void StartCameraScene()
    {
        SceneManager.LoadScene(3);
    }

    public void StartMainScene()
    {
        SceneManager.LoadScene(0);
    }

    public void StartMoreSpeciesScene()
    {
        SceneManager.LoadScene(4);
    }
}

