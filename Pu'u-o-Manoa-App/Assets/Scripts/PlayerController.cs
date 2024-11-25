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
        CheckJoystick(joystickDirection);

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

    private void CheckJoystick(Vector2 Direction)
    {
        // Convert 2D joystick input to 3D movement direction
        Vector3 moveDirection = new Vector3(Direction.x, 0, Direction.y).normalized;

        // Apply movement if there's input
        if (moveDirection != Vector3.zero)
        {
            this.Move(moveDirection * speedMultiplier);
        }



        // Access the joystick input data

        //Vector3 moveDirection = Vector3.zero;
        //Vector3 newDirection = new Vector3(Direction.x, Direction.y, 0);

        //if (newDirection.y > 0)
        //{
        //    this.Move(moveDirection += newDirection * speedMultiplier);
        //}
        //if (newDirection.y < 0)
        //{
        //    this.Move(moveDirection -= newDirection * speedMultiplier);
        //}
        //if (newDirection.x < 0)
        //{
        //    this.Move(moveDirection -= newDirection * speedMultiplier);
        //}
        //if (newDirection.x > 0)
        //{
        //    this.Move(moveDirection += newDirection * speedMultiplier);
        //}
    }

    //    //Debug.Log($"Joystick Horizontal: {horizontal}, Vertical: {vertical}");

    //    // Create a movement direction based on the joystick input and camera orientation
    //    //Vector3 moveDirection = (cameraForward * vertical + cameraRight * horizontal).normalized;

    //    //// Use the move direction to move the player
    //    //this.Move(moveDirection += cameraForward * speedMultiplier);
    //

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