using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ImagePlacer : MonoBehaviour
{

    public static ImagePlacer Instance { get; private set; }
    public float startLat = 21.1816f;
    public float startLon = 157.4930f;
    public GameObject imagePrefab;

    private void Awake()
    {
        if (Instance == null)
        {
            Instance = this;
            DontDestroyOnLoad(gameObject);
        }
        else
        {
            Destroy(gameObject);
        }
    }

    void Start()
    {

    }

    void Update()
    {
        
    }

    public void Place(float Latitude, float Longitude, string image_loc)
    {
        float xPos = Longitude - startLon;
        float zPos = Latitude - startLat;

        Vector3 spawnPos = new Vector3(xPos, 100f, zPos);
        GameObject image = Instantiate(imagePrefab, spawnPos, Quaternion.identity);
        print("aaa");

    }

}
