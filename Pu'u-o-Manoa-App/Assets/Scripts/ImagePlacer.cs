using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ImagePlacer : MonoBehaviour
{
    public float startLat = 21.1816f;
    public float startLon = 157.4930f;
    public GameObject imagePrefab; 
    

    // Start is called before the first frame update
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public void Place(float Latitude, float Longitude, string image_loc)
    {
        float xPos = Longitude - startLon;
        float zPos = Latitude - startLat;

        // create image with yPos and xPos using image
        Vector3 spawnPos = new Vector3(xPos, 100f, zPos);
        GameObject image = Instantiate(imagePrefab, spawnPos, Quaternion.identity);

    }

}
