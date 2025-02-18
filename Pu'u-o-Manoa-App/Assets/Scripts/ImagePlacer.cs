using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ImagePlacer : MonoBehaviour
{

    public static ImagePlacer Instance { get; private set; }
    public float startLat = 21.1816f;
    public float startLon = 157.4930f;
    public GameObject imagePrefab;
    public List<ImageObject> images;

    private void Awake()
    {
        if (Instance == null)
        {
            Instance = this;
            images = new List<ImageObject>();
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
        images.Add(new ImageObject(imagePrefab, spawnPos, Quaternion.identity));
        //Instantiate(imagePrefab, spawnPos, Quaternion.identity)

    }

}

public class ImageObject
{
    public GameObject Prefab { get; private set; }
    public Vector3 SpawnPos { get; private set; }
    public Quaternion Rotation { get; private set; }

    public ImageObject(GameObject prefab, Vector3 spawnPos, Quaternion rotation)
    {
        Prefab = prefab;
        SpawnPos = spawnPos;
        Rotation = rotation;
    }

}
