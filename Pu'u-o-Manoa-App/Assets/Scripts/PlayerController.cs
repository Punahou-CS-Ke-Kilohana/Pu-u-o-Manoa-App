using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PlayerController : MonoBehaviour
{

    Rigidbody rb;
    public float jumpForce = 3.5f;
    public float speed = 0.1f;
    bool canJump;
    public Transform cameraTransform;

    // Called before start
    private void Awake()
    {
        rb = GetComponent<Rigidbody>();
    }

    // Start is called before the first frame update
    void Start()
    {
        
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

        CheckPressedKeys(cameraForward, cameraRight);

        

    }

    private void OnCollisionEnter(Collision collision)
    {
        if(collision.gameObject.tag == "Ground")
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
            this.Move(moveDirection += cameraForward * speed);
        }
        if (Input.GetKey(KeyCode.S))
        {
            this.Move(moveDirection -= cameraForward * speed);
        }
        if (Input.GetKey(KeyCode.A))
        {
            this.Move(moveDirection -= cameraRight * speed);
        }
        if (Input.GetKey(KeyCode.D))
        {
            this.Move(moveDirection += cameraRight * speed);
        }
    }

}
