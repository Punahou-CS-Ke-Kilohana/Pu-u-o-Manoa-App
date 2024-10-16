using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class DatabaseUIController : MonoBehaviour
{
    List<string> plants = new List<string>() { "Ohia", "Ti", "Koa", "Hau" };  // Add more plants if needed
    public GameObject speciesCellPrefab;  // Prefab for the species cell (plantCell)
    public RectTransform contentPanel;    // Reference to the Scroll View's content panel (content)
    public float spacing = 10f;           // Space between each prefab
    public int itemsPerRow = 3;           // Number of prefabs per row

    void Start()
    {
        PopulateScrollView();
    }

    void PopulateScrollView()
    {
        // Get the width and height of each prefab (speciesCell)
        RectTransform cellRect = speciesCellPrefab.GetComponent<RectTransform>();
        float cellWidth = cellRect.rect.width;
        float cellHeight = cellRect.rect.height;

        // Calculate the available width in the content panel
        float contentWidth = contentPanel.rect.width;

        // Calculate total number of rows needed
        int totalRows = Mathf.CeilToInt((float)plants.Count / itemsPerRow);

        // Adjust the content panel's height to fit all rows
        float totalHeight = totalRows * (cellHeight + spacing) - spacing;
        contentPanel.sizeDelta = new Vector2(contentWidth, totalHeight);

        // Loop through the plant list and create each prefab in the correct position
        for (int i = 0; i < plants.Count; i++)
        {
            // Instantiate the prefab
            GameObject newCell = Instantiate(speciesCellPrefab, contentPanel);

            // Calculate the row and column for this cell
            int row = i / itemsPerRow;
            int column = i % itemsPerRow;

            // Calculate the x and y position for this cell
            float xPos = column * (cellWidth + spacing) + 480;
            float yPos = -row * (cellHeight + spacing) - 180;  // Negative to move down

            // Set the position of the new cell
            RectTransform cellRectTransform = newCell.GetComponent<RectTransform>();
            cellRectTransform.anchoredPosition = new Vector2(xPos, yPos);
        }
    }
}
