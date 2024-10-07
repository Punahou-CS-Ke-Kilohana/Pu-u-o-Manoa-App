using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

// these are the class thingys you need for drop and drag
public class DraggableUI : MonoBehaviour, IBeginDragHandler, IDragHandler, IEndDragHandler
{
    public GameObject placementMarkerPrefab; // 3D object to place on the terrain
    public Camera mainCamera;
    private RectTransform rectTransform;
    private Canvas canvas;


    // just getting the position and the canvas button pos
    private void Start()
    {
        rectTransform = GetComponent<RectTransform>();
        canvas = GetComponentInParent<Canvas>();
    }

    // begin draggin ui object
    public void OnBeginDrag(PointerEventData eventData)
    {
        // nothing needs to happen, unless we want like a ghost transparent preview of where guy goes instead
    }

    // drag the UI object
    public void OnDrag(PointerEventData eventData)
    {
        rectTransform.anchoredPosition += eventData.delta / canvas.scaleFactor;
    }

    // when drag ends (basically dropping)
    public void OnEndDrag(PointerEventData eventData)
    {
        // chatgpted (raycasts r confusing)
        Vector3 worldPosition = Input.mousePosition;
        Ray ray = mainCamera.ScreenPointToRay(worldPosition);
        RaycastHit hit;

        if (Physics.Raycast(ray, out hit))
        {
            // if the ray hits something (I chatgpted this part, don't really understand raycasts)
            Vector3 hitPosition = new Vector3(hit.point.x, hit.point.y + 1, hit.point.z);

            // spawn 3d marker prefab
            Instantiate(placementMarkerPrefab, hitPosition, Quaternion.identity);

            // hide the marker after placing
            gameObject.SetActive(false);

            // coords to console
            Debug.Log($"Placed at: X: {hitPosition.x}, Y: {hitPosition.y}, Z: {hitPosition.z}");

            Destroy(mainCamera);
        }
    }
}