using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class DatabaseUIController : MonoBehaviour
{
    List<string> plants = new List<string>() { "Ohia", "Ti", "Koa", "Hau", "Ohia", "Ti", "Koa", "Hau", "Ti", "Koa", "Hau","Ti", "Koa", "Hau" };  // Add more plants if needed
    public GameObject speciesCellPrefab;  // Prefab for the species cell (plantCell)
    public RectTransform contentPanel;    // Reference to the Scroll View's content panel (content)
    public float spacing = 100f;           // Space between each prefab
    public int itemsPerRow = 3;           // Number of prefabs per row
    public ScrollRect scrollRect;
    
    public RectTransform content;  // Reference to the ScrollRect component

    void Start()
    {
        PopulateScrollView();
        // Set scroll position to top after populating
        scrollRect.verticalNormalizedPosition = 1f;
    }

    void PopulateScrollView()
    {
        // Get the width and height of each prefab (speciesCell)
        RectTransform cellRect = speciesCellPrefab.GetComponent<RectTransform>();
        float cellWidth = cellRect.rect.width;
        float cellHeight = cellRect.rect.height;

        // Calculate the available width in the content panel
        float contentWidth = contentPanel.rect.width;

        // Calculate horizontal spacing based on screen width
        float horizontalSpacing = (contentWidth - (cellWidth * itemsPerRow)) / (itemsPerRow + 1);

        // Set vertical spacing to a smaller value, e.g., 5f
        float verticalSpacing = 5f;

        // Calculate total number of rows needed
        int totalRows = Mathf.CeilToInt((float)plants.Count / itemsPerRow);

        // Calculate the total height of all rows
        float totalContentHeight = totalRows * cellHeight + (totalRows + 1) * verticalSpacing;

        // Set minimum height for 4 cells (2 rows)
        float minHeight = 210f;

        // Adjust the content panel's height
        float adjustedHeight = Mathf.Max(minHeight, totalContentHeight);
        contentPanel.sizeDelta = new Vector2(contentWidth, adjustedHeight);

        // Loop through the plant list and create each prefab in the correct position
        for (int i = 0; i < plants.Count; i++)
        {
            // Instantiate the prefab
            GameObject newCell = Instantiate(speciesCellPrefab, contentPanel);

            // Calculate the row and column for this cell
            int row = i / itemsPerRow;
            int column = i % itemsPerRow;

            // Calculate the x and y position for this cell
            float xPos = (column + 1) * horizontalSpacing + column * cellWidth;
            float yPos = -(row + 1) * verticalSpacing - row * cellHeight - 30;  // Negative to move down

            // Set the position of the new cell
            RectTransform cellRectTransform = newCell.GetComponent<RectTransform>();
            cellRectTransform.anchoredPosition = new Vector2(xPos, yPos);

            // Set the plant name in the cell
            TMPro.TextMeshProUGUI cellText = newCell.GetComponentInChildren<TMPro.TextMeshProUGUI>();
            if (cellText != null)
            {
                cellText.text = plants[i];
            }
        }

        // Get the height of the scroll rect's viewport or use the RectTransform if viewport is not assigned
        float viewportHeight;
        if (scrollRect.viewport != null)
        {
            viewportHeight = scrollRect.viewport.rect.height;
        }
        else
        {
            RectTransform scrollRectTransform = scrollRect.GetComponent<RectTransform>();
            viewportHeight = scrollRectTransform.rect.height;
        }

        // Enable vertical scrolling if content exceeds viewport height
        scrollRect.vertical = adjustedHeight > viewportHeight;
    }
}
