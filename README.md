# ✈️ Flight Booking AI Agent

An intelligent Python agent that automates flight booking on **MakeMyTrip**. Give it a plain‐English command and it will:

1. **Parse** your request using the Groq LLM (Llama 3.3 70B).
2. **Ask** for any missing details (name, email, phone, etc.).
3. **Open Chrome**, navigate to MakeMyTrip, and **fill the search form** automatically.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Make sure .env contains your Groq API key
#    GROQ_API_KEY=gsk_...

# 3. Run the agent
python flight_agent.py
```

You'll be prompted:

```
Enter your booking command:
> Book a flight from Mumbai to Delhi on 20 April for 2 passengers
```

The agent extracts the details, opens Chrome, and fills MakeMyTrip's search form.

---

## Project Structure

| File               | Purpose                                      |
|--------------------|----------------------------------------------|
| `flight_agent.py`  | Main script — LLM extraction + Selenium automation |
| `config.py`        | Configuration & constants                    |
| `.env`             | Groq API key (not committed to version control) |
| `requirements.txt` | Python dependencies                          |

---

## Requirements

- **Python 3.9+**
- **Google Chrome** installed on your system
- A valid **Groq API key** ([console.groq.com](https://console.groq.com))

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `GROQ_API_KEY not found` | Add your key to `.env` |
| ChromeDriver version mismatch | `webdriver-manager` handles this automatically — make sure Chrome is up to date |
| Popup not closing | MakeMyTrip sometimes changes popup selectors — the agent will continue regardless |
| Date not selected | If the calendar DOM changes, select the date manually; search will still work |
