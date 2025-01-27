using UnityEngine;

public class FirstPersonCamera : MonoBehaviour
{
    public float sensitivity = 100f;
    public Transform playerBody;
    public POVJoystick povJoystick; // Reference to the POVJoystick script

    private float xRotation = 0f;

    private void Start()
    {
        // Assign the povJoystick dynamically
        povJoystick = FindObjectOfType<POVJoystick>();
        if (povJoystick == null)
        {
            Debug.LogError("POVJoystick not found in the scene!");
        }
    }

    void Update()
    {
        // Get joystick input
        float joystickX = povJoystick.Direction.x/10; // Horizontal input
        float joystickY = povJoystick.Direction.y/10; // Vertical input

        // Rotate player body horizontally
        float mouseX = joystickX * sensitivity * Time.deltaTime;
        playerBody.Rotate(Vector3.up * mouseX);

        // Rotate camera vertically
        float mouseY = joystickY * sensitivity * Time.deltaTime;
        xRotation -= mouseY;
        xRotation = Mathf.Clamp(xRotation, -90f, 90f);

        transform.localRotation = Quaternion.Euler(xRotation, 0f, 0f);
    }
}
