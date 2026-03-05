using UnityEngine;
using TMPro;

public class Timer : MonoBehaviour
{
    public float timeLeft = 120;
    public TextMeshProUGUI text;

    void Update()
    {
        timeLeft -= Time.deltaTime;
        text.text = timeLeft.ToString("F1");
    }
}