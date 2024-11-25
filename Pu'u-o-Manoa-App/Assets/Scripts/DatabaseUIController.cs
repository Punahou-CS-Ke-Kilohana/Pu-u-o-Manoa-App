using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class DatabaseUIController : MonoBehaviour
{
    List<string> cells = new List<string>() {};
    public GameObject speciesCellPrefab;  // Prefab for the species cell (plantCell)
    public RectTransform contentPanel;    // Reference to the Scroll View's content panel (content)

    public Button AnimalButton;
    public Button PlantButton;

    public float spacing = 100f;           // Space between each prefab
    public int itemsPerRow = 3;           // Number of prefabs per row
    public ScrollRect scrollRect;

    PlantLoader plantLoader = new PlantLoader(); // Create an instance of PlantLoader

    void Start()
    {
        PopulatePlantsCells();
    }

    void PopulateScrollView(List<string> cellList)
    {
        // Clear existing cells from the content panel
        foreach (Transform child in contentPanel)
        {
            GameObject.Destroy(child.gameObject);
        }

        // Get the width and height of each prefab (speciesCell)
        RectTransform cellRect = speciesCellPrefab.GetComponent<RectTransform>();
        float cellWidth = cellRect.rect.width;
        float cellHeight = cellRect.rect.height;

        // Set a fixed width for the content panel
        float fixedContentWidth = 0; // Fixed width

        // Calculate total number of rows needed
        int totalRows = Mathf.CeilToInt((float)cellList.Count / itemsPerRow);
        
        // Calculate total height needed
        float totalHeight = Mathf.Max((totalRows * 79), 175);

        // Set the content panel size with fixed width and dynamic height
        contentPanel.sizeDelta = new Vector2(fixedContentWidth, totalHeight);

        // Enable or disable scrolling based on height
        scrollRect.vertical = totalHeight > 175;

        // Loop through the plant list and create each prefab in the correct position
        for (int i = 0; i < cellList.Count; i++)
        {
            // Instantiate the prefab
            GameObject newCell = Instantiate(speciesCellPrefab, contentPanel);

            // Calculate the row and column for this cell
            int row = i / itemsPerRow;
            int column = i % itemsPerRow;

            // Calculate the x and y position for this cell
            float xPos = (column + 1) * 85 + column * cellWidth; // Adjust spacing as needed
            float yPos = -(row + 1) * 45 - row * cellHeight / 3; // Negative to move down

            // Set the position of the new cell
            RectTransform cellRectTransform = newCell.GetComponent<RectTransform>();
            cellRectTransform.anchoredPosition = new Vector2(xPos, yPos);

            // Set the plant name in the cell
            TMPro.TextMeshProUGUI cellText = newCell.GetComponentInChildren<TMPro.TextMeshProUGUI>();
            if (cellText != null)
            {
                cellText.text = cellList[i];
            }
        }

        // After creating all cells, reset scroll position to top
        scrollRect.normalizedPosition = new Vector2(0, 1);
    }

    public void PopulateAnimalCells()
    {
        List<string> cells = new List<string>() { "NENE" };  // Add more animals if needed
        PopulateScrollView(cells);
    }

    public void PopulatePlantsCells()
    {
        List<string> plantNames = plantLoader.GetPlantNames();
        PopulateScrollView(plantNames);
    }
}
