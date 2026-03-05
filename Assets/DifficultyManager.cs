using UnityEngine;

public class DifficultyManager : MonoBehaviour
{
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Alpha0))
            DeEscalate();

        if (Input.GetKeyDown(KeyCode.Alpha1))
            Maintain();

        if (Input.GetKeyDown(KeyCode.Alpha2))
            Escalate();
    }

    public void DeEscalate()
    {
        Debug.Log("Simplifying graph");
    }

    public void Maintain()
    {
        Debug.Log("No change");
    }

    public void Escalate()
    {
        Debug.Log("Making harder");
    }
}