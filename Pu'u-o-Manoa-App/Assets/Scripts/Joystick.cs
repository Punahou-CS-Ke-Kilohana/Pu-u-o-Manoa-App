using UnityEngine;
using UnityEngine.EventSystems;

public class Joystick : MonoBehaviour, IDragHandler, IPointerUpHandler, IPointerDownHandler
{
    private RectTransform joystickBackground;
    private RectTransform joystickHandle;

    private Vector2 inputVector;  // Stores the joystick's input direction

    private void Start()
    {
        joystickBackground = GetComponent<RectTransform>();
        joystickHandle = transform.GetChild(0).GetComponent<RectTransform>();  // First child is the handle
    }

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

    public float Horizontal()
    {
        return inputVector.x;
    }

    public float Vertical()
    {
        return inputVector.y;
    }
}
