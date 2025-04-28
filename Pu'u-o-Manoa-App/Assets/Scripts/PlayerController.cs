using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/*

Purpose: This is the PlayerController script. It handles keypresses, movement, physics, and camera. It also limits the player's speed.
Date: 11/25/24
Project: Pu'u-o-Manoa App
By: Isaac Verbrugge

*/

public class PlayerController : MonoBehaviour
{

    Rigidbody rb;
    public float jumpForce = 3.5f;
    public float speedMultiplier = 0.1f;
    public float maxSpeed = 2.5f;
    bool canJump;
    public Transform cameraTransform;
    private GameObject player;
    public Camera mainCamera;

    // joystick
    public Joystick joystick;


    // Called before start
    private void Awake()
    {
        rb = GetComponent<Rigidbody>();
    }

    // Start is called before the first frame update
    void Start()
    {
        player = GameObject.FindGameObjectWithTag("Player");
        joystick = FindObjectOfType<Joystick>();
    }

    // Update is called once per frame
    void Update()
    {
        Vector3 cameraForward = cameraTransform.forward;
        Vector3 cameraRight = cameraTransform.right;
        cameraForward.y = 0f;
        cameraRight.y = 0f;
        cameraForward.Normalize();
        cameraRight.Normalize();
        Vector2 joystickDirection = joystick.Direction;
        //Debug.Log($"Horizontal: {joystickDirection.x}, Vertical: {joystickDirection.y}");
        CheckPressedKeys(cameraForward, cameraRight);
        CheckJoystick(joystickDirection, cameraForward, cameraRight);

        LimitSpeed();

    }

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.tag == "Ground")
        {
            canJump = true;
        }
    }

    private void OnCollisionExit(Collision collision)
    {
        if (collision.gameObject.tag == "Ground")
        {
            canJump = false;
        }
    }

    private void Move(Vector3 force)
    {
        rb.AddForce(force, ForceMode.Impulse);
    }

    private void CheckPressedKeys(Vector3 cameraForward, Vector3 cameraRight)
    {
        if (Input.GetKeyDown(KeyCode.Space) && canJump)
        {
            rb.AddForce(Vector3.up * jumpForce, ForceMode.Impulse);
        }

        Vector3 moveDirection = Vector3.zero;

        if (Input.GetKey(KeyCode.W))
        {
            this.Move(moveDirection += cameraForward * speedMultiplier);
        }
        if (Input.GetKey(KeyCode.S))
        {
            this.Move(moveDirection -= cameraForward * speedMultiplier);
        }
        if (Input.GetKey(KeyCode.A))
        {
            this.Move(moveDirection -= cameraRight * speedMultiplier);
        }
        if (Input.GetKey(KeyCode.D))
        {
            this.Move(moveDirection += cameraRight * speedMultiplier);
        }

        if (Input.GetKey(KeyCode.V))
        {
            Destroy(player);
            mainCamera.enabled = true;
        }
    }

    private void CheckJoystick(Vector2 joyDir, Vector3 cameraForward, Vector3 cameraRight)
    {

        Vector3 moveDirection = Vector3.zero;

        if (joyDir.y > 0)
        {
            moveDirection += cameraForward * speedMultiplier;
        }
        if (joyDir.y < 0)
        {
            moveDirection -= cameraForward * speedMultiplier;
        }

        if (joyDir.x != 0)
        {
            moveDirection += cameraRight * joyDir.x * speedMultiplier;
        }

        this.Move(moveDirection);

    }

    private void LimitSpeed()
    {
        Vector3 velocity = rb.velocity;
        Vector3 horizontalVelocity = new Vector3(velocity.x, 0f, velocity.z);

        if (horizontalVelocity.magnitude > maxSpeed)
        {
            horizontalVelocity = horizontalVelocity.normalized * maxSpeed;
            rb.velocity = new Vector3(horizontalVelocity.x, velocity.y, horizontalVelocity.z);
        }
    }

}