using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class NewBehaviourScript : MonoBehaviour
{
    public double startLat = 21.1816;
    public double startLon = 157.4930;

    // Start is called before the first frame update
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {
        
    }

    void Place(float Latitude, float Longitude, string image)
    {
        double yPos = Longitude - startLon;
        double xPos = Latitude - startLat;

        // create image with yPos and xPos using image
    }

}
