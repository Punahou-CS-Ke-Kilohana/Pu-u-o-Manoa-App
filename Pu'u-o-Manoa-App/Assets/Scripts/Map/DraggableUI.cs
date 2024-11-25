using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

/*

Purpose: This is the draggable UI script. It's a UI used to drag and drop the player prefab on the map. Once "dropped," you enter the player's POV.
Date: 11/25/24
Project: Pu'u-o-Manoa App
By: Rodney Fujiyama & Isaac Verbrugge

*/

// these are the class thingys you need for drop and drag
public class DraggableUI : MonoBehaviour, IBeginDragHandler, IDragHandler, IEndDragHandler
{
    private RectTransform rectTransform;
    private Canvas canvas;
    private Button[] buttons;
    public Camera mainCamera;
    public GameObject playerPrefab; 
    private GameObject player;


    // just getting the position and the canvas button pos
    private void Start()
    {
        rectTransform = GetComponent<RectTransform>();
        canvas = GetComponentInParent<Canvas>();
        //buttons = canvas.GetComponentsInChildren<Button>();

        //foreach (Button btn in buttons)
        //{
        //    btn.onClick.AddListener(() => OnButtonClick(btn));
        //}
    }

    private void OnButtonClick(Button clickedButton)
    {
        Debug.Log("Button " + clickedButton.name + " clicked!");
        if (clickedButton.name.Equals("Exit"))
        {
            ExitMap();
        }


    }

    void Update()
    {
        CheckPressedKeys();
    }

    private void CheckPressedKeys()
    {
        if (Input.GetKeyDown(KeyCode.V))
        {
            Debug.Log("AAAA");
            ExitMap();
        }
    }

    private void ExitMap()
    {
        if (player != null)
        {
            player.SetActive(false);
        }
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
            player = Instantiate(playerPrefab, hitPosition, Quaternion.identity);
            PlayerController playerController = player.GetComponent<PlayerController>();
            playerController.mainCamera = this.mainCamera;

            // hide the marker after placing
            gameObject.SetActive(false);
            mainCamera.enabled = false;

            // coords to console
            Debug.Log($"Placed at: X: {hitPosition.x}, Y: {hitPosition.y}, Z: {hitPosition.z}");

            //Destroy(mainCamera);
        }
    }
}