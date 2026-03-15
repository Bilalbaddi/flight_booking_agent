"""
Flight Booking AI Agent
=======================
Accepts a natural-language command, extracts flight details using a Groq LLM,
opens a browser, navigates to Google Flights, and fills the search form.

Usage:
    python flight_agent.py
"""

import sys
import time
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

import config
from logic.flight_extractor import extract_flight_results, wait_for_results_container


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1.  Pydantic model for structured flight data
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FlightInfo(BaseModel):
    """Structured flight-booking information extracted from the user command."""
    source_city: str = Field(default="", description="Departure city")
    destination_city: str = Field(default="", description="Arrival city")
    travel_date: str = Field(
        default="",
        description="Departure date in DD/MM/YYYY format",
    )
    return_date: str = Field(
        default="",
        description="Return date in DD/MM/YYYY format (empty for one-way)",
    )
    num_passengers: int = Field(
        default=config.DEFAULT_PASSENGERS,
        description="Number of passengers",
    )
    passenger_name: str = Field(default="", description="Full name of the primary passenger")
    email: str = Field(default="", description="Contact email address")
    phone: str = Field(default="", description="Contact phone number")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2.  LLM-powered extraction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SYSTEM_PROMPT = """\
You are an intelligent flight-booking assistant.
Extract structured flight information from the user's natural-language request.

Rules:
- source_city and destination_city should be proper city names (capitalised).
- travel_date is the departure date and MUST be in DD/MM/YYYY format.
- return_date is the return date and MUST be in DD/MM/YYYY format for round trips.
    If one-way, return_date can be empty.
    The current year is {current_year}.
  If the user only says a month & day, assume the current year or the next year
  if that date has already passed.
- num_passengers defaults to 1 if not mentioned.
- passenger_name, email, and phone may be empty if not provided.
- Return ONLY the JSON object, nothing else.
"""

USER_PROMPT = "User request: {user_command}"


def build_llm() -> ChatGroq:
    """Return a ChatGroq LLM instance."""
    return ChatGroq(
        api_key=config.GROQ_API_KEY,
        model=config.GROQ_MODEL,
        temperature=0,
    )


def extract_flight_info(user_command: str) -> FlightInfo:
    """Use Groq LLM to parse a natural-language command into FlightInfo."""
    llm = build_llm()
    structured_llm = llm.with_structured_output(FlightInfo)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT),
    ])

    chain = prompt | structured_llm
    result: FlightInfo = chain.invoke({
        "user_command": user_command,
        "current_year": datetime.now().year,
    })
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3.  Interactive gap-filling for missing fields
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def fill_missing_fields(info: FlightInfo) -> FlightInfo:
    """Prompt the user in the terminal for any fields the LLM left empty."""
    prompts = {
        "source_city":      "Enter departure city: ",
        "destination_city":  "Enter destination city: ",
        "travel_date":       "Enter departure date (DD/MM/YYYY): ",
        "return_date":       "Enter return date (DD/MM/YYYY): ",
        "passenger_name":    "Enter passenger name: ",
        "email":             "Enter email address: ",
        "phone":             "Enter phone number: ",
    }
    data = info.model_dump()
    for field, prompt_text in prompts.items():
        if not data.get(field):
            data[field] = input(prompt_text).strip()
    if data.get("num_passengers", 0) < 1:
        try:
            data["num_passengers"] = int(input("Enter number of passengers: ").strip())
        except ValueError:
            data["num_passengers"] = 1
    return FlightInfo(**data)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4.  Selenium helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def launch_browser() -> webdriver.Chrome:
    """Create and return a Chrome WebDriver instance."""
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.implicitly_wait(config.BROWSER_IMPLICIT_WAIT)
    return driver


def js_click(driver, element):
    """Click via JavaScript — bypasses overlay interceptions."""
    driver.execute_script("arguments[0].click();", element)


def safe_click(driver, element):
    """Try normal click, fall back to JS click."""
    try:
        element.click()
    except Exception:
        js_click(driver, element)


def _is_transient_driver_connection_error(exc: Exception) -> bool:
    """Return True when Selenium transport was briefly reset (common on Windows)."""
    msg = str(exc).lower()
    needles = [
        "winerror 10054",
        "forcibly closed",
        "connection reset",
        "remote end closed connection",
        "max retries exceeded",
    ]
    return any(n in msg for n in needles)


def _call_driver_with_retry(func, retries: int = 2, delay: float = 0.35):
    """Execute a webdriver call with short retry for transient socket resets."""
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return func()
        except Exception as exc:
            last_exc = exc
            if attempt < retries and _is_transient_driver_connection_error(exc):
                time.sleep(delay)
                continue
            raise
    if last_exc:
        raise last_exc


def browser_keys(driver, *keys):
    """
    Send keystrokes to the BROWSER (not to a specific element).
    This uses ActionChains which bypasses ElementNotInteractableException
    because it types into whatever the browser currently has focused.
    """
    actions = ActionChains(driver)
    for key in keys:
        actions.send_keys(key)
    actions.perform()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5.  Google Flights automation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MONTH_NAMES = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def _parse_date(date_str: str) -> datetime:
    """Parse DD/MM/YYYY into a datetime object."""
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d %m %Y"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {date_str}")


def _fill_city(driver, aria_label_variants: list, city_name: str) -> None:
    """
    Click a city field by trying multiple aria-label variants (with/without
    trailing space), clear it, type the city, and pick the first suggestion.

    Uses ActionChains (browser_keys) for ALL keyboard input to avoid
    ElementNotInteractableException on dynamically-created overlay inputs.
    """
    wait = WebDriverWait(driver, 15)

    # Step A — Click the city field to activate it
    clicked = False
    for label in aria_label_variants:
        try:
            field = driver.find_element(
                By.CSS_SELECTOR, f"input[aria-label='{label}']"
            )
            if field.is_displayed():
                safe_click(driver, field)
                clicked = True
                print(f"   ✓ Clicked field with aria-label='{label}'")
                break
        except Exception:
            continue

    if not clicked:
        # Fallback: find any visible combobox input
        try:
            all_inputs = driver.find_elements(By.CSS_SELECTOR, "input[role='combobox']")
            for inp in all_inputs:
                if inp.is_displayed():
                    safe_click(driver, inp)
                    clicked = True
                    print("   ✓ Clicked fallback combobox input")
                    break
        except Exception:
            pass

    if not clicked:
        print(f"   ⚠ Could not find city field for: {city_name}")
        return

    time.sleep(1.5)

    # Step B — Clear existing text and type city name
    # Using ActionChains (browser_keys) sends keystrokes to whatever the
    # browser currently has focused — immune to ElementNotInteractableException
    browser_keys(driver, Keys.CONTROL + "a")      # select all
    time.sleep(0.2)
    browser_keys(driver, Keys.BACKSPACE)           # delete
    time.sleep(0.3)
    browser_keys(driver, city_name)                # type city
    print(f"   ✓ Typed '{city_name}'")
    time.sleep(2.5)  # wait for suggestions to load

    # Step C — Pick the first suggestion: li[role='option']
    try:
        first_option = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "li[role='option']"))
        )
        safe_click(driver, first_option)
        print("   ✓ Selected first suggestion")
        time.sleep(1)
    except Exception:
        # Fallback: press Down + Enter via ActionChains
        print("   ⚠ Suggestion click failed, trying keyboard selection")
        browser_keys(driver, Keys.ARROW_DOWN)
        time.sleep(0.3)
        browser_keys(driver, Keys.ENTER)
        time.sleep(1)


def _fill_from_city(driver, city_name: str) -> None:
    """Fill the 'Where from?' field."""
    # Google Flights uses trailing space in aria-label inconsistently
    _fill_city(driver, ["Where from? ", "Where from?"], city_name)


def _fill_to_city(driver, city_name: str) -> None:
    """Fill the 'Where to?' field — must click explicitly (no auto-focus)."""
    _fill_city(driver, ["Where to? ", "Where to?"], city_name)


def _format_date_for_input(date_str: str) -> str:
    """Convert DD/MM/YYYY into a human-friendly format accepted by date inputs."""
    dt = _parse_date(date_str)
    return dt.strftime("%d %b %Y")  # e.g. 20 Apr 2026


def _fill_date_field_via_text(driver, field_name: str, date_str: str) -> bool:
    """Click a date field, type date text, and press Enter."""
    if not date_str:
        return True

    date_text = _format_date_for_input(date_str)
    field = None

    selectors = [
        f"input[aria-label='{field_name}']",
        f"input[aria-label^='{field_name}']",
        f"[role='combobox'][aria-label^='{field_name}']",
        f"[role='button'][aria-label^='{field_name}']",
        f"[aria-label^='{field_name}']",
    ]

    for sel in selectors:
        try:
            elems = _call_driver_with_retry(lambda: driver.find_elements(By.CSS_SELECTOR, sel))
            visible = [e for e in elems if e.is_displayed()]
            if visible:
                field = visible[0]
                break
        except Exception:
            continue

    if field is None:
        try:
            elems = _call_driver_with_retry(
                lambda: driver.find_elements(
                    By.XPATH,
                    f"//input[contains(@aria-label, '{field_name}')] | "
                    f"//*[@role='button' and contains(@aria-label, '{field_name}')] | "
                    f"//*[contains(normalize-space(.), '{field_name}') and (@role='button' or self::div)]"
                )
            )
            visible = [e for e in elems if e.is_displayed()]
            if visible:
                field = visible[0]
        except Exception:
            pass

    if field is None:
        print(f"   ⚠ Could not find {field_name} field.")
        return False

    try:
        safe_click(driver, field)
        time.sleep(0.25)

        # Clear only when the focused element is an actual text-entry control
        # for this field. This avoids accidentally wiping the other date field.
        can_clear = False
        try:
            can_clear = bool(_call_driver_with_retry(
                lambda: driver.execute_script(
                    """
                    const expected = (arguments[0] || '').toLowerCase();
                    const ae = document.activeElement;
                    if (!ae) return false;

                    const tag = (ae.tagName || '').toLowerCase();
                    const isTextEntry = tag === 'input' || tag === 'textarea' || ae.isContentEditable;
                    const aria = (ae.getAttribute('aria-label') || '').toLowerCase();

                    return isTextEntry && (!aria || aria.includes(expected));
                    """,
                    field_name,
                )
            ))
        except Exception:
            can_clear = False

        if can_clear:
            browser_keys(driver, Keys.CONTROL + "a")
            time.sleep(0.15)
            browser_keys(driver, Keys.BACKSPACE)
            time.sleep(0.15)

        browser_keys(driver, date_text)
        time.sleep(0.15)
        browser_keys(driver, Keys.ENTER)
        print(f"   ✓ Set {field_name}: {date_text}")
        time.sleep(0.35)
        return True
    except Exception:
        try:
            _call_driver_with_retry(
                lambda: driver.execute_script(
                    """
                    const target = document.activeElement;
                    if (!target) return false;
                    target.focus();
                    if ('value' in target) {
                        target.value = arguments[0];
                    }
                    target.dispatchEvent(new Event('input', { bubbles: true }));
                    target.dispatchEvent(new Event('change', { bubbles: true }));
                    return true;
                    """,
                    date_text,
                )
            )
            browser_keys(driver, Keys.ENTER)
            print(f"   ✓ Set {field_name}: {date_text} (JS fallback)")
            time.sleep(0.35)
            return True
        except Exception:
            print(f"   ⚠ Could not set {field_name} date.")
            return False


def _select_date(driver, departure_date: str, return_date: str) -> None:
    """Fill Departure and Return directly via text input and press Done."""
    print(f"   Setting Departure date from input: {departure_date}")
    ok_dep = _fill_date_field_via_text(driver, "Departure", departure_date)

    print(f"   Setting Return date from input: {return_date}")
    ok_ret = _fill_date_field_via_text(driver, "Return", return_date)

    # Stabilization for Google Flights: after setting Return, Departure can
    # occasionally get cleared by focus/overlay behavior. Re-apply once.
    if departure_date:
        print("   Re-checking Departure date after Return entry …")
        ok_dep = _fill_date_field_via_text(driver, "Departure", departure_date) and ok_dep

    clicked_done = _click_done(driver)
    if clicked_done:
        print("   ✓ Date picker closed with Done")
    else:
        print("   ⚠ Done button not clicked (might already be closed)")
    if not (ok_dep and ok_ret):
        print("   ⚠ One or more date fields could not be auto-filled. Please verify dates manually.")


def _click_done(driver, timeout: int = 8) -> bool:
    """Click visible 'Done' control in date picker. Returns True if clicked/closed."""
    wait = WebDriverWait(driver, timeout)
    time.sleep(0.25)

    done_xpath = (
        "//button[.//span[translate(normalize-space(.),'DONE','done')='done'] "
        "or translate(normalize-space(.),'DONE','done')='done']"
        " | "
        "//*[@role='button' and translate(normalize-space(.),'DONE','done')='done']"
    )

    # Attempt 1: Selenium element click
    try:
        done_buttons = driver.find_elements(By.XPATH, done_xpath)
        visible = [b for b in done_buttons if b.is_displayed() and b.get_attribute("aria-disabled") != "true"]
        if visible:
            safe_click(driver, visible[-1])
            print("   ✓ Clicked Done")
            try:
                wait.until(lambda d: len([e for e in d.find_elements(By.XPATH, done_xpath) if e.is_displayed()]) == 0)
            except Exception:
                pass
            return True
    except Exception:
        pass

    # Attempt 2: JS fallback click by visible text
    try:
        clicked = _call_driver_with_retry(
            lambda: driver.execute_script("""
                function isVisible(el) {
                    return !!(el.offsetParent || (el.getClientRects && el.getClientRects().length));
                }
                const nodes = Array.from(document.querySelectorAll('button, [role="button"]'));
                const candidates = nodes.filter(el => {
                    const txt = (el.innerText || el.textContent || '').trim().toLowerCase();
                    const dis = el.getAttribute('aria-disabled') === 'true' || el.disabled;
                    return isVisible(el) && !dis && txt === 'done';
                });
                if (!candidates.length) return false;
                candidates[candidates.length - 1].click();
                return true;
            """)
        )
        if clicked:
            print("   ✓ Clicked Done (JS fallback)")
            try:
                wait.until(lambda d: len([e for e in d.find_elements(By.XPATH, done_xpath) if e.is_displayed()]) == 0)
            except Exception:
                pass
            return True
    except Exception:
        pass

    return False


def _wait_until_main_search_is_interactive(driver, timeout: int = 12) -> None:
    """Wait until date picker is closed and Search button is clickable."""
    wait = WebDriverWait(driver, timeout)

    # If date picker is still open, try closing it first.
    _click_done(driver, timeout=3)

    # 1) Wait for date-picker Done button(s) to disappear.
    try:
        wait.until(lambda d: len([
            b for b in d.find_elements(
                By.XPATH,
                "//button[.//span[translate(normalize-space(.),'DONE','done')='done'] or translate(normalize-space(.),'DONE','done')='done'] | //*[@role='button' and translate(normalize-space(.),'DONE','done')='done']"
            ) if b.is_displayed()
        ]) == 0)
    except Exception:
        # If Done is not found at all, continue.
        pass

    # 2) Ensure primary Search button is present and clickable.
    search_locators = [
        (By.CSS_SELECTOR, "button[aria-label='Search']"),
        (By.XPATH, "//button[.//span[normalize-space()='Search'] or normalize-space()='Search']"),
    ]
    for by, value in search_locators:
        try:
            wait.until(EC.element_to_be_clickable((by, value)))
            return
        except Exception:
            continue


def _click_search_and_wait(driver, timeout: int = 20) -> bool:
    """Click primary Search button and wait for search/results to start loading."""
    _wait_until_main_search_is_interactive(driver, timeout=12)
    wait = WebDriverWait(driver, timeout)
    before_url = driver.current_url

    search_btn = None
    selectors = [
        (By.CSS_SELECTOR, "button[aria-label='Search']"),
        (By.CSS_SELECTOR, "button[aria-label*='Search']"),
        (By.XPATH, "//button[.//span[normalize-space()='Search'] or normalize-space()='Search']"),
        (By.XPATH, "//*[@role='button' and .//span[normalize-space()='Search']]"),
    ]
    for by, value in selectors:
        try:
            candidates = driver.find_elements(by, value)
            visible = [e for e in candidates if e.is_displayed()]
            if visible:
                search_btn = visible[0]
                break
        except Exception:
            continue

    if search_btn is None:
        # JS fallback for visible primary Search button
        try:
            clicked = _call_driver_with_retry(
                lambda: driver.execute_script("""
                    function isVisible(el) {
                        return !!(el.offsetParent || (el.getClientRects && el.getClientRects().length));
                    }
                    const nodes = Array.from(document.querySelectorAll('button, [role="button"]'));
                    const candidates = nodes.filter(el => {
                        const txt = (el.innerText || el.textContent || '').trim().toLowerCase();
                        const aria = (el.getAttribute('aria-label') || '').toLowerCase();
                        const dis = el.getAttribute('aria-disabled') === 'true' || el.disabled;
                        return isVisible(el) && !dis && (txt === 'search' || aria === 'search' || aria.includes('search'));
                    });
                    if (!candidates.length) return false;
                    candidates[0].click();
                    return true;
                """)
            )
            if not clicked:
                return False
        except Exception:
            return False
    else:
        safe_click(driver, search_btn)

    # Validation: URL change or common search-results markers appear.
    try:
        wait.until(
            lambda d: (
                d.current_url != before_url
                or len(d.find_elements(By.XPATH, "//*[contains(., 'Best flights') or contains(., 'Top flights') or contains(., 'Sort by') or contains(., 'Price graph') or contains(., 'Departing flights') or contains(., 'Returning flights')]") ) > 0
            )
        )
    except Exception:
        # Non-fatal; click may still have worked in SPA flows without obvious markers.
        pass

    return True


def _get_visible_flight_cards(driver):
    """Return visible flight-result cards using structural XPath locators."""
    card_xpaths = [
        "//ul[@role='listbox']/li",
        "//*[@role='listitem']",
        "//div[contains(normalize-space(.), '₹') and (contains(normalize-space(.), 'AM') or contains(normalize-space(.), 'PM'))]",
    ]

    cards = []
    seen = set()
    for xp in card_xpaths:
        try:
            for el in driver.find_elements(By.XPATH, xp):
                if not el.is_displayed():
                    continue
                if el.id in seen:
                    continue
                seen.add(el.id)
                cards.append(el)
        except Exception:
            continue
    return cards


def _retry_click_card(driver, element, retries: int = 3, delay: float = 0.5) -> bool:
    """Retry click for dynamic result cards."""
    for _ in range(retries):
        try:
            driver.execute_script(
                "arguments[0].scrollIntoView({behavior:'instant', block:'center', inline:'nearest'});",
                element,
            )
            safe_click(driver, element)
            return True
        except Exception:
            time.sleep(delay)
    return False


def _select_cheapest_from_sorted_list(driver, timeout: int = 30) -> bool:
    """
    Use Google Flights built-in "Cheapest" tab and pick first visible card.
    """
    wait_for_results_container(driver, timeout=timeout)
    wait = WebDriverWait(driver, timeout)

    before_cards = _get_visible_flight_cards(driver)
    before_first_text = before_cards[0].text.strip() if before_cards else ""

    cheapest_locators = [
        (By.XPATH, "//button[contains(normalize-space(.), 'Cheapest') or .//span[contains(normalize-space(.), 'Cheapest')]]"),
        (By.XPATH, "//*[@role='tab' and contains(normalize-space(.), 'Cheapest')]"),
    ]

    cheapest_tab = None
    for by, value in cheapest_locators:
        try:
            elems = driver.find_elements(by, value)
            visible = [e for e in elems if e.is_displayed() and e.get_attribute("aria-disabled") != "true"]
            if visible:
                cheapest_tab = visible[0]
                break
        except Exception:
            continue

    if cheapest_tab is None:
        return False

    print("Clicking Cheapest filter to sort flights by lowest price.")
    if not _retry_click_card(driver, cheapest_tab, retries=3, delay=0.6):
        return False

    # Wait for reorder: either tab becomes selected or first card text changes.
    try:
        wait.until(
            lambda d: (
                (cheapest_tab.get_attribute("aria-selected") or "").lower() == "true"
                or (cheapest_tab.get_attribute("aria-pressed") or "").lower() == "true"
                or (
                    len(_get_visible_flight_cards(d)) > 0
                    and _get_visible_flight_cards(d)[0].text.strip() != before_first_text
                )
            )
        )
    except Exception:
        # Continue best-effort even if explicit reorder signal is unavailable.
        pass

    wait.until(lambda d: len(_get_visible_flight_cards(d)) > 0)
    cards = _get_visible_flight_cards(driver)
    if not cards:
        return False

    first_card = cards[0]
    try:
        driver.execute_script(
            "arguments[0].scrollIntoView({behavior:'smooth', block:'center', inline:'nearest'});",
            first_card,
        )
        driver.execute_script("arguments[0].style.border = '3px solid red';", first_card)
    except Exception:
        pass

    if not _retry_click_card(driver, first_card, retries=3, delay=0.6):
        return False

    print("Cheapest flight selected from sorted list.")
    return True


def _safe_find_elements(driver, by, value):
    """find_elements with transient-driver retry; returns [] on persistent failure."""
    try:
        return _call_driver_with_retry(lambda: driver.find_elements(by, value)) or []
    except Exception:
        return []


def _wait_for_xpath_presence(driver, xpath: str, timeout: int = 20) -> bool:
    """Wait until at least one element exists for the given XPath."""
    wait = WebDriverWait(driver, timeout)
    try:
        wait.until(lambda d: len(_safe_find_elements(d, By.XPATH, xpath)) > 0)
        return True
    except Exception:
        return False


def _click_first_visible(driver, xpaths: list, timeout: int = 20, retries: int = 3):
    """Find and click the first visible element from a list of XPath locators."""
    last_exc = None

    for _ in range(retries):
        for xp in xpaths:
            try:
                _wait_for_xpath_presence(driver, xp, timeout=timeout)
                elems = _safe_find_elements(driver, By.XPATH, xp)
                visible = [e for e in elems if e.is_displayed() and e.get_attribute("aria-disabled") != "true"]
                if not visible:
                    continue
                el = visible[0]
                driver.execute_script(
                    "arguments[0].scrollIntoView({behavior:'smooth', block:'center', inline:'nearest'});",
                    el,
                )
                safe_click(driver, el)
                return el
            except Exception as exc:
                last_exc = exc
                continue
        time.sleep(0.5)

    if last_exc:
        raise last_exc
    raise RuntimeError("No clickable element found for provided locators")


def _expand_first_card_and_select_flight(driver, section_hint=None, timeout: int = 25) -> bool:
    """Expand first flight card and click its 'Select flight' button."""
    if section_hint:
        _wait_for_xpath_presence(
            driver,
            f"//*[contains(normalize-space(.), '{section_hint}') and not(self::script)]",
            timeout=timeout,
        )

    wait_for_results_container(driver, timeout=timeout)
    cards = _get_visible_flight_cards(driver)
    if not cards:
        return False

    first_card = cards[0]
    try:
        driver.execute_script(
            "arguments[0].scrollIntoView({behavior:'smooth', block:'center', inline:'nearest'});",
            first_card,
        )
    except Exception:
        pass

    # Try expanding with a dropdown/details button inside first card.
    expanded = False
    expand_locators = [
        ".//button[contains(@aria-label,'details') or contains(@aria-label,'Details') or contains(@aria-label,'expand') or contains(@aria-label,'Expand')]",
        ".//*[@role='button' and (contains(@aria-label,'details') or contains(@aria-label,'expand'))]",
    ]
    for loc in expand_locators:
        try:
            btns = _call_driver_with_retry(lambda: first_card.find_elements(By.XPATH, loc))
            btns = [b for b in btns if b.is_displayed() and b.get_attribute("aria-disabled") != "true"]
            if btns:
                _retry_click_card(driver, btns[0], retries=3, delay=0.5)
                expanded = True
                break
        except Exception:
            continue

    # If no explicit expand button was found, clicking the card itself often expands.
    if not expanded:
        _retry_click_card(driver, first_card, retries=2, delay=0.4)

    # Click first visible "Select flight".
    try:
        _click_first_visible(
            driver,
            [
                "//button[.//span[contains(normalize-space(.), 'Select flight')] or contains(normalize-space(.), 'Select flight')]",
                "//*[@role='button' and contains(normalize-space(.), 'Select flight')]",
            ],
            timeout=timeout,
            retries=4,
        )
        return True
    except Exception:
        return False


def _select_first_booking_provider(driver, timeout: int = 25) -> str:
    """Select first booking provider and click Continue. Returns provider name."""
    wait = WebDriverWait(driver, timeout)

    # STEP 1 — Wait for booking options section visibility.
    booking_heading_xpath = "//*[contains(normalize-space(.), 'Booking options')]"
    book_with_xpath = "//*[contains(normalize-space(.), 'Book with')]"

    try:
        wait.until(
            lambda d: any(e.is_displayed() for e in _safe_find_elements(d, By.XPATH, booking_heading_xpath))
            or any(e.is_displayed() for e in _safe_find_elements(d, By.XPATH, book_with_xpath))
        )
    except Exception:
        pass

    # STEP 2 — Locate booking option cards and collect their Continue buttons.
    provider_cards = _safe_find_elements(
        driver,
        By.XPATH,
        "//*[contains(normalize-space(.), 'Book with') and not(self::script)]"
    )
    provider_cards = [c for c in provider_cards if c.is_displayed()]

    continue_buttons = []
    provider_name = ""

    for idx, card in enumerate(provider_cards):
        try:
            btns = _call_driver_with_retry(
                lambda: card.find_elements(
                    By.XPATH,
                    ".//button[contains(normalize-space(.), 'Continue')] | .//*[@role='button' and contains(normalize-space(.), 'Continue')]"
                )
            )
            btns = [b for b in btns if b.is_displayed() and b.get_attribute("aria-disabled") != "true"]
            if btns:
                continue_buttons.append(btns[0])
                if idx == 0:
                    txt = card.text.strip()
                    provider_name = txt.replace("Book with", "").strip() if "Book with" in txt else txt
        except Exception:
            continue

    # Fallback: if cards did not yield buttons, collect visible Continue globally.
    if not continue_buttons:
        for xp in [
            "//button[contains(normalize-space(.), 'Continue')]",
            "//*[@role='button' and contains(normalize-space(.), 'Continue')]",
        ]:
            btns = _safe_find_elements(driver, By.XPATH, xp)
            btns = [b for b in btns if b.is_displayed() and b.get_attribute("aria-disabled") != "true"]
            if btns:
                continue_buttons.extend(btns)
                break

    if not continue_buttons:
        raise RuntimeError("No Continue button found in booking options section")

    # STEP 3/4 — Select first option and click with retry + safe click logic.
    first_continue = continue_buttons[0]
    print("Selecting first booking option.")

    clicked = False
    for _ in range(3):
        try:
            driver.execute_script(
                "arguments[0].scrollIntoView({behavior:'smooth', block:'center', inline:'nearest'});",
                first_continue,
            )
            safe_click(driver, first_continue)
            clicked = True
            break
        except Exception:
            time.sleep(0.4)

    if not clicked:
        raise RuntimeError("Could not click Continue on first booking provider")

    print("Clicked Continue.")

    # STEP 5 — Wait for contact info page.
    wait.until(
        lambda d: any(e.is_displayed() for e in _safe_find_elements(
            d,
            By.XPATH,
            "//*[contains(normalize-space(.), 'Enter Contact Information') or contains(normalize-space(.), 'Contact Information')]"
        ))
    )
    print("Contact information page loaded.")

    return provider_name or "first available provider"


def _is_booking_page_visible(driver) -> bool:
    """Return True when Selected flights/Booking options page is visible."""
    checks = [
        "//*[contains(normalize-space(.), 'Selected flights')]",
        "//*[contains(normalize-space(.), 'Booking options')]",
    ]
    for xp in checks:
        elems = _safe_find_elements(driver, By.XPATH, xp)
        if any(e.is_displayed() for e in elems):
            return True
    return False


def _wait_for_booking_page(driver, timeout: int = 25) -> bool:
    """Wait until booking page markers are visible."""
    wait = WebDriverWait(driver, timeout)
    try:
        wait.until(lambda d: _is_booking_page_visible(d))
        return True
    except Exception:
        return False


def _fill_contact_information(driver, phone: str, email: str, timeout: int = 30) -> bool:
    """Fill Mobile Number and Email ID fields on booking page."""
    def fill_field(label_variants, value):
        if not value:
            return True

        xps = []
        for label in label_variants:
            xps.extend([
                f"//input[contains(@aria-label, '{label}') or contains(@placeholder, '{label}') or contains(@name, '{label}')]",
                f"//label[contains(normalize-space(.), '{label}')]/following::input[1]",
                f"//*[contains(normalize-space(.), '{label}')]/following::input[1]",
            ])

        field = None
        for xp in xps:
            try:
                _wait_for_xpath_presence(driver, xp, timeout=timeout)
                elems = _safe_find_elements(driver, By.XPATH, xp)
                elems = [e for e in elems if e.is_displayed()]
                if elems:
                    field = elems[0]
                    break
            except Exception:
                continue

        if field is None:
            return False

        try:
            driver.execute_script(
                "arguments[0].scrollIntoView({behavior:'smooth', block:'center', inline:'nearest'});",
                field,
            )
            safe_click(driver, field)
            browser_keys(driver, Keys.CONTROL + "a")
            browser_keys(driver, Keys.BACKSPACE)
            browser_keys(driver, value)
            return True
        except Exception:
            return False

    ok_phone = fill_field(["Mobile Number", "Phone", "Mobile"], phone)
    ok_email = fill_field(["Email ID", "Email"], email)
    return ok_phone and ok_email


def _complete_booking_steps(driver, info: FlightInfo) -> None:
    """Complete post-selection booking steps up to payment screen."""
    current_stage = "departure_selection"

    # Stage 1 → Departure flight selection
    if current_stage == "departure_selection":
        if _is_booking_page_visible(driver):
            current_stage = "booking"
        else:
            dep_ok = _expand_first_card_and_select_flight(driver, section_hint="Top departing flights", timeout=30)
            if dep_ok:
                print("Departure flight selected.")
                current_stage = "return_selection"
            else:
                print("   ⚠ Could not select departure flight automatically.")
                return

    # Stage 2 → Return flight selection
    if current_stage == "return_selection":
        # If the booking page is already visible, do not run return dropdown logic.
        if _is_booking_page_visible(driver):
            print("Booking page detected.")
            current_stage = "booking"
        else:
            ret_ok = _expand_first_card_and_select_flight(driver, section_hint="Top returning flights", timeout=35)
            if ret_ok:
                print("Return flight selected.")
                if _wait_for_booking_page(driver, timeout=35):
                    print("Booking page detected.")
                    current_stage = "booking"
                else:
                    print("   ⚠ Return selected but booking page was not detected.")
                    return
            else:
                print("   ⚠ Could not select return flight automatically.")
                return

    # Stage 3 → Booking options page
    if current_stage == "booking":
        try:
            provider = _select_first_booking_provider(driver, timeout=35)
            print(f"Selected booking provider: {provider}.")
            current_stage = "contact_information"
        except Exception:
            print("   ⚠ Could not select booking provider automatically.")
            return

    # Stage 4 → Contact information page
    if current_stage == "contact_information":
        if _fill_contact_information(driver, info.phone, info.email, timeout=35):
            print("Contact information filled.")
            current_stage = "proceed_to_payment"
        else:
            print("   ⚠ Could not fully fill contact information.")
            return

    # Stage 5 → Proceed to payment
    if current_stage == "proceed_to_payment":
        try:
            _click_first_visible(
                driver,
                [
                    "//button[contains(normalize-space(.), 'Proceed To Pay')]",
                    "//*[@role='button' and contains(normalize-space(.), 'Proceed To Pay')]",
                ],
                timeout=30,
                retries=4,
            )
            print("Proceeding to payment page.")
        except Exception:
            print("   ⚠ Could not click Proceed To Pay automatically.")
            return

    print("Automation completed successfully. Awaiting manual payment.")


def _set_passengers(driver, num_passengers: int) -> None:
    """Adjust the passenger count."""
    if num_passengers <= 1:
        return
    try:
        pax_btns = driver.find_elements(
            By.CSS_SELECTOR, "button[aria-label*='passenger']"
        )
        visible = [b for b in pax_btns if b.is_displayed()]
        if not visible:
            return
        safe_click(driver, visible[0])
        time.sleep(1)

        for _ in range(num_passengers - 1):
            add_btns = driver.find_elements(
                By.CSS_SELECTOR,
                "button[aria-label*='Add adult'], "
                "button[aria-label='Increase number of adults']"
            )
            for btn in add_btns:
                if btn.is_displayed():
                    safe_click(driver, btn)
                    time.sleep(0.3)
                    break

        _click_done(driver)
    except Exception:
        print(f"   ⚠ Could not set {num_passengers} passengers. Adjust manually.")


def automate_search(info: FlightInfo) -> webdriver.Chrome:
    """Open Google Flights, fill the search form, and click Search."""
    driver = launch_browser()
    flights = []

    print("\n🌐 Opening Google Flights …")
    driver.get(config.GOOGLE_FLIGHTS_URL)
    time.sleep(5)  # let page fully render

    # Accept cookies/consent if present
    try:
        for text in ["Accept all", "I agree", "Accept"]:
            btns = driver.find_elements(
                By.XPATH, f"//button[contains(., '{text}')]"
            )
            for btn in btns:
                if btn.is_displayed():
                    safe_click(driver, btn)
                    time.sleep(1)
                    break
    except Exception:
        pass

    # ── From city ──────────────────────────────────────────────────────────
    print(f"\n📍 Setting departure: {info.source_city}")
    _fill_from_city(driver, info.source_city)

    # ── To city (must click explicitly — no auto-focus after From) ─────────
    print(f"\n📍 Setting destination: {info.destination_city}")
    _fill_to_city(driver, info.destination_city)

    # ── Travel date ────────────────────────────────────────────────────────
    print(f"\n📅 Setting dates: departure={info.travel_date}, return={info.return_date}")
    _select_date(driver, info.travel_date, info.return_date)

    # ── Passengers ─────────────────────────────────────────────────────────
    if info.num_passengers > 1:
        print(f"\n👥 Setting passengers: {info.num_passengers}")
        _set_passengers(driver, info.num_passengers)

    # ── Click Search ───────────────────────────────────────────────────────
    print("\n🔍 Clicking Search …")
    time.sleep(0.6)
    try:
        clicked = _click_search_and_wait(driver, timeout=20)
        if clicked:
            print("   ✓ Search clicked and submission triggered.")

            # ── Extract results from Google Flights search page ─────────────
            print("\n📊 Extracting flight results …")
            try:
                flights = extract_flight_results(driver, max_results=15, timeout=30)
                print(f"\nExtracted {len(flights)} flights")

                selected = _select_cheapest_from_sorted_list(driver, timeout=30)
                if not selected:
                    print("   ⚠ Could not auto-select flight from Cheapest sorted list.")
                else:
                    _complete_booking_steps(driver, info)

                for i, f in enumerate(flights[:5], start=1):
                    print(
                        f"      {i}. {f.get('airline', '')} | "
                        f"{f.get('departure', '')} → {f.get('arrival', '')} | "
                        f"{f.get('duration', '')} | {f.get('stops', '')} | {f.get('price', '')}"
                    )
            except Exception as exc:
                err_text = str(exc).strip() or repr(exc)
                print(f"   ⚠ Could not extract flight results ({type(exc).__name__}): {err_text}")
        else:
            print("   ⚠ Could not find Search button. Click it manually.")
    except Exception:
        print("   ⚠ Could not click Search. Click it manually.")

    # Store extracted list on driver object for optional downstream usage.
    driver.extracted_flights = flights
    driver.sorted_flights = []
    driver.cheapest_flight = None

    print("\n✅ Done! The browser will stay open for you.\n")
    return driver


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6.  Main entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main() -> None:
    print("=" * 60)
    print("  ✈️  Flight Booking AI Agent")
    print("=" * 60)
    print()

    if not config.GROQ_API_KEY:
        print("❌ GROQ_API_KEY not found in .env. Please add it and retry.")
        sys.exit(1)

    user_command = input(
        "Enter your booking command "
        "(e.g. 'Book a flight from Mumbai to Delhi on 20 April'):\n> "
    ).strip()
    if not user_command:
        print("❌ No command entered. Exiting.")
        sys.exit(1)

    print("\n🤖 Analysing your request with AI …")
    flight_info = extract_flight_info(user_command)
    flight_info = fill_missing_fields(flight_info)

    print("\n" + "─" * 40)
    print("  Extracted Flight Details")
    print("─" * 40)
    print(f"  From          : {flight_info.source_city}")
    print(f"  To            : {flight_info.destination_city}")
    print(f"  Departure Date: {flight_info.travel_date}")
    print(f"  Return Date   : {flight_info.return_date}")
    print(f"  Passengers    : {flight_info.num_passengers}")
    print(f"  Passenger Name: {flight_info.passenger_name}")
    print(f"  Email         : {flight_info.email}")
    print(f"  Phone         : {flight_info.phone}")
    print("─" * 40)

    driver = automate_search(flight_info)

    input("\nPress Enter to close the browser …")
    driver.quit()


if __name__ == "__main__":
    main()
