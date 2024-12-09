using UnityEngine;
using UnityEngine.EventSystems;

/*

Purpose: This is the Joystick script. It allows the user to move around using a joystick, particularly useful for mobile use where the WASD keys are not available.
Date: 12/06/24
Project: Pu'u-o-Manoa App
By: Mark Chen

*/

public class Joystick : MonoBehaviour, IDragHandler, IPointerUpHandler, IPointerDownHandler
{
    private RectTransform joystickBackground;
    private RectTransform joystickHandle;

    private Vector2 inputVector;  // Stores the joystick's input direction

    public Vector2 Direction => inputVector;

    private void Start()
    {
        joystickBackground = GetComponent<RectTransform>();
        joystickHandle = transform.GetChild(0).GetComponent<RectTransform>();  // First child is the handle
    }

    //private void Update()
    //{
    //    Debug.Log($"Horizontal: {inputVector.x}, Vertical: {inputVector.y}");
    //}


    public void OnDrag(PointerEventData eventData)
    {
        // Convert touch/mouse position to joystick background space
        Vector2 pos;
        RectTransformUtility.ScreenPointToLocalPointInRectangle(joystickBackground, eventData.position, eventData.pressEventCamera, out pos);

        // Calculate joystick's movement relative to its size
        pos.x = (pos.x / joystickBackground.sizeDelta.x);
        pos.y = (pos.y / joystickBackground.sizeDelta.y);

        // Clamp joystick movement inside the joystick background
        inputVector = new Vector2(pos.x * 2, pos.y * 2);
        inputVector = (inputVector.magnitude > 1.0f) ? inputVector.normalized : inputVector;

        // Move the joystick handle within the background bounds
        joystickHandle.anchoredPosition = new Vector2(inputVector.x * (joystickBackground.sizeDelta.x / 2), inputVector.y * (joystickBackground.sizeDelta.y / 2));
    }

    public void OnPointerDown(PointerEventData eventData)
    {
        OnDrag(eventData);  // Start dragging when pointer goes down
    }

    public void OnPointerUp(PointerEventData eventData)
    {
        inputVector = Vector2.zero;  // Reset input when pointer is lifted
        joystickHandle.anchoredPosition = Vector2.zero;  // Return the joystick to center
    }
}
