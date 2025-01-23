using UnityEngine;
using UnityEngine.EventSystems;

/*

Purpose: This is the POVJoystick script. It allows the user to control the camera's rotation using a joystick, particularly useful for mobile use where mouse input is not available.
Date: 12/06/24
Project: Pu'u-o-Manoa App
By: Mark Chen

*/

public class POVJoystick : MonoBehaviour, IDragHandler, IPointerUpHandler, IPointerDownHandler
{
    private RectTransform joystickBackground;
    private RectTransform joystickHandle;

    private Vector2 inputVector;  // Stores the joystick's input direction

    public Vector2 Direction => inputVector;

    // On Start, initialize the RectTransform components:
    // - `joystickBackground` is the RectTransform of the parent object (the visible joystick base).
    // - `joystickHandle` is the RectTransform of the first child (the movable handle the user interacts with).
    private void Start()
    {
        joystickBackground = GetComponent<RectTransform>();
        joystickHandle = transform.GetChild(0).GetComponent<RectTransform>();  // First child is the handle
    }

    // OnDrag processes the user's drag input on the joystick
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

    // OnPointerDown is called when the user touches or clicks on the joystick:
    // - Immediately triggers OnDrag to calculate the initial joystick input and position.
    public void OnPointerDown(PointerEventData eventData)
    {
        OnDrag(eventData);  // Start dragging when pointer goes down
    }

    // OnPointerUp is called when the user releases the joystick:
    // - Resets the inputVector to (0, 0) to stop movement.
    // - Returns the joystick handle to its default centered position.
    public void OnPointerUp(PointerEventData eventData)
    {
        inputVector = Vector2.zero;  // Reset input when pointer is lifted
        joystickHandle.anchoredPosition = Vector2.zero;  // Return the joystick to center
    }
}