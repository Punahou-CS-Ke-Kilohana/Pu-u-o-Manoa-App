using UnityEngine;

public class FirstPersonCamera : MonoBehaviour
{
    public float sensitivity = 1000f;
    public Transform playerBody;
    public POVJoystick povJoystick; // Reference to the POVJoystick script

    private float xRotation = 0f;

    void Update()
    {
        // Get joystick input
        float joystickX = povJoystick.Direction.x; // Horizontal input
        float joystickY = povJoystick.Direction.y; // Vertical input

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
