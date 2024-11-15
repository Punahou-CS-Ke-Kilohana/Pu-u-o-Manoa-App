using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(MeshFilter), typeof(MeshRenderer))]
public class TriangleMesh : MonoBehaviour
{
    private void Start()
    {
        Mesh mesh = new Mesh();
        GetComponent<MeshFilter>().mesh = mesh;

        // Define vertices for the triangle
        Vector3[] vertices = new Vector3[]
        {
            new Vector3(0, 0, 0),  // Bottom-left vertex
            new Vector3(1, 0, 0),  // Bottom-right vertex
            new Vector3(0.5f, 1, 0)  // Top vertex
        };

        // Define the triangle by connecting vertices in clockwise order
        int[] triangles = new int[]
        {
            0, 1, 2  // One triangle, using vertices 0, 1, and 2
        };

        // Optional: Define UVs if you want to texture it
        Vector2[] uvs = new Vector2[]
        {
            new Vector2(0, 0),
            new Vector2(1, 0),
            new Vector2(0.5f, 1)
        };

        // Assign data to mesh
        mesh.vertices = vertices;
        mesh.triangles = triangles;
        mesh.uv = uvs;

        // Optional: Calculate the normals to improve lighting
        mesh.RecalculateNormals();
    }
}
