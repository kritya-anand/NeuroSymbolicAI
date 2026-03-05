using UnityEngine;

public class GraphEdge : MonoBehaviour
{
    public Transform nodeA;
    public Transform nodeB;
    LineRenderer lr;

    void Start()
    {
        lr = GetComponent<LineRenderer>();
    }

    void Update()
    {
        lr.SetPosition(0, nodeA.position);
        lr.SetPosition(1, nodeB.position);
    }
}