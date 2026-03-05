using UnityEngine;

public class CrossingDetector : MonoBehaviour
{
    public GraphEdge[] edges;

    void Update()
    {
        int crossings = 0;

        for(int i=0;i<edges.Length;i++)
        {
            for(int j=i+1;j<edges.Length;j++)
            {
                if(EdgesCross(edges[i], edges[j]))
                    crossings++;
            }
        }

        Debug.Log("Crossings: " + crossings);
    }

    bool EdgesCross(GraphEdge e1, GraphEdge e2)
    {
        Vector2 a1 = new Vector2(e1.nodeA.position.x, e1.nodeA.position.z);
        Vector2 a2 = new Vector2(e1.nodeB.position.x, e1.nodeB.position.z);
        Vector2 b1 = new Vector2(e2.nodeA.position.x, e2.nodeA.position.z);
        Vector2 b2 = new Vector2(e2.nodeB.position.x, e2.nodeB.position.z);

        return LineIntersect(a1,a2,b1,b2);
    }

    bool LineIntersect(Vector2 p1, Vector2 p2, Vector2 p3, Vector2 p4)
    {
        float d = (p4.y-p3.y)*(p2.x-p1.x)-(p4.x-p3.x)*(p2.y-p1.y);
        if (d == 0) return false;

        float u = ((p4.x-p3.x)*(p1.y-p3.y)-(p4.y-p3.y)*(p1.x-p3.x))/d;
        float v = ((p2.x-p1.x)*(p1.y-p3.y)-(p2.y-p1.y)*(p1.x-p3.x))/d;

        return (u>0 && u<1 && v>0 && v<1);
    }
}