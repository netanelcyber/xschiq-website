const DAYS = ["ראשון", "שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת"];
const MONTHS = ["ינואר", "פברואר", "מרץ", "אפריל", "מאי", "יוני", "יולי", "אוגוסט", "ספטמבר", "אוקטובר", "נובמבר", "דצמבר"];
const STORAGE_KEY = "adl-secure-session-v1";
const TARGETS = { "מקלחת": 13, "סידור חדר": 4, "כביסה": 7 };

const TASK_META = {
  "מקלחת": {
    steps: ["להכין מגבת ובגדים", "להתקלח", "להתנגב ולהתלבש"],
  },
  "סידור חדר": {
    steps: ["לסדר מיטה", "לפנות משטח מרכזי", "להחזיר חפצים למקום"],
  },
  "כביסה": {
    steps: ["לאסוף בגדים לסל", "להפעיל מכונה", "לייבש או לתלות"],
  },
};

const WEEKLY_TEMPLATES = [
  { id: "sun-a", dayIndex: 0, task: "מקלחת", slot: "בוקר", note: "מקלחת קצרה לפתיחת היום." },
  { id: "sun-b", dayIndex: 0, task: "מקלחת", slot: "ערב", note: "מקלחת ערב עם סיום רגוע." },
  { id: "sun-c", dayIndex: 0, task: "סידור חדר", slot: "גמיש", note: "שלושה צעדים קבועים בלבד." },
  { id: "mon-a", dayIndex: 1, task: "מקלחת", slot: "בוקר", note: "שומרים על רצף קבוע." },
  { id: "mon-b", dayIndex: 1, task: "מקלחת", slot: "ערב", note: "מקלחת קצרה לפני שינה." },
  { id: "tue-a", dayIndex: 2, task: "מקלחת", slot: "בוקר", note: "התחלה ברורה של היום." },
  { id: "tue-b", dayIndex: 2, task: "מקלחת", slot: "ערב", note: "שטיפה, ייבוש, סיום." },
  { id: "tue-c", dayIndex: 2, task: "סידור חדר", slot: "אחה\"צ", note: "סדר בסיסי לפני הערב." },
  { id: "wed-a", dayIndex: 3, task: "מקלחת", slot: "בוקר", note: "מקלחת בוקר רגילה." },
  { id: "wed-b", dayIndex: 3, task: "מקלחת", slot: "ערב", note: "מקלחת קצרה לסגירת היום." },
  { id: "thu-a", dayIndex: 4, task: "מקלחת", slot: "בוקר", note: "רצף קבוע ורגוע." },
  { id: "thu-b", dayIndex: 4, task: "מקלחת", slot: "ערב", note: "מקלחת ערב קבועה." },
  { id: "thu-c", dayIndex: 4, task: "סידור חדר", slot: "גמיש", note: "סדר קצר באזור שינה ועבודה." },
  { id: "fri-a", dayIndex: 5, task: "מקלחת", slot: "בוקר", note: "מקלחת בוקר לקראת סוף השבוע." },
  { id: "fri-b", dayIndex: 5, task: "מקלחת", slot: "ערב", note: "יש לבדוק אם צריך להקדים לפני שבת או חג." },
  { id: "sat-a", dayIndex: 6, task: "מקלחת", slot: "ערב", note: "בדרך כלל מתבצע אחרי צאת שבת או חג." },
  { id: "sat-b", dayIndex: 6, task: "סידור חדר", slot: "גמיש", note: "סגירת שבוע והכנה לשבוע הבא." },
];

const nodes = {
  gregorianDate: document.getElementById("gregorianDate"),
  hebrewDate: document.getElementById("hebrewDate"),
  shabbatWindow: document.getElementById("shabbatWindow"),
  holidayList: document.getElementById("holidayList"),
  calendarAlert: document.getElementById("calendarAlert"),
  summaryShower: document.getElementById("summaryShower"),
  summaryRoom: document.getElementById("summaryRoom"),
  summaryLaundry: document.getElementById("summaryLaundry"),
  summaryShowerBar: document.getElementById("summaryShowerBar"),
  summaryRoomBar: document.getElementById("summaryRoomBar"),
  summaryLaundryBar: document.getElementById("summaryLaundryBar"),
  focusTitle: document.getElementById("focusTitle"),
  focusSubtitle: document.getElementById("focusSubtitle"),
  focusList: document.getElementById("focusList"),
  focusEmpty: document.getElementById("focusEmpty"),
  weekRange: document.getElementById("weekRange"),
  monthLabel: document.getElementById("monthLabel"),
  weeklyGrid: document.getElementById("weeklyGrid"),
  laundryGrid: document.getElementById("laundryGrid"),
  printButton: document.getElementById("printButton"),
  resetWeekButton: document.getElementById("resetWeekButton"),
  resetMonthButton: document.getElementById("resetMonthButton"),
};

function todayMidday() {
  const now = new Date();
  return new Date(now.getFullYear(), now.getMonth(), now.getDate(), 12, 0, 0, 0);
}

function isoDate(date) {
  return date.toISOString().slice(0, 10);
}

function formatDate(date) {
  return `${DAYS[date.getDay()]} ${String(date.getDate()).padStart(2, "0")}/${String(date.getMonth() + 1).padStart(2, "0")}/${date.getFullYear()}`;
}

function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function weekStart(date) {
  const copy = new Date(date);
  copy.setDate(copy.getDate() - copy.getDay());
  return copy;
}

function loadState() {
  try {
    return JSON.parse(sessionStorage.getItem(STORAGE_KEY)) || {};
  } catch {
    return {};
  }
}

function saveState(state) {
  sessionStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}

function getEntry(state, key) {
  return state[key] || {
    selfSignature: "",
    caregiverSignature: "",
    completedByUser: false,
    approvedByCaregiver: false,
    signedAt: "",
  };
}

function buildLaundryDates(date) {
  const year = date.getFullYear();
  const month = date.getMonth();
  const daysInMonth = new Date(year, month + 1, 0).getDate();
  const items = [];
  const used = new Set();

  for (let index = 0; index < TARGETS["כביסה"]; index += 1) {
    let day = Math.floor((index * daysInMonth) / TARGETS["כביסה"]) + 1;
    while (used.has(day) && day < daysInMonth) day += 1;
    used.add(day);
    items.push(new Date(year, month, day, 12, 0, 0, 0));
  }

  return items;
}

function contextualizeTask(task, dayEvents) {
  const clone = { ...task, note: task.note, slot: task.slot };
  const candles = dayEvents?.candles;
  const havdalah = dayEvents?.havdalah;

  if (candles && clone.slot !== "בוקר") {
    const title = candles.title?.replace("הדלקת נרות ", "") || "שבת/חג";
    clone.slot = "לפני כניסת שבת/חג";
    clone.note = `${clone.note} אם מבצעים היום, לסיים עד ${candles.time} לפני ${title}.`;
  }

  if (havdalah && clone.id === "sat-a") {
    clone.slot = "אחרי צאת שבת/חג";
    clone.note = `${clone.note} אפשר לבצע החל מ-${havdalah.time}.`;
  }

  if (dayEvents?.holidays?.length) {
    clone.note = `${clone.note} היום חל: ${dayEvents.holidays.join(" / ")}.`;
  }

  return clone;
}

function renderTask(task, saved) {
  const signed = Boolean(saved.signedAt);
  return `
    <article class="task-card">
      <div class="task-top">
        <span class="task-badge">${task.task}</span>
        <span class="task-time">${task.slot}</span>
      </div>
      <h4>${task.task}</h4>
      <p class="task-note">${task.note}</p>
      <ul class="task-steps">
        ${TASK_META[task.task].steps.map((step) => `<li>${step}</li>`).join("")}
      </ul>
      <div class="field-grid">
        <div class="field">
          <label for="self-${task.storageKey}">חתימה שלי</label>
          <input id="self-${task.storageKey}" data-key="${task.storageKey}" data-field="selfSignature" type="text" maxlength="40" autocomplete="off" spellcheck="false" placeholder="ראשי תיבות" value="${escapeHtml(saved.selfSignature)}">
        </div>
        <div class="field">
          <label for="care-${task.storageKey}">חתימת מטפל/ת</label>
          <input id="care-${task.storageKey}" data-key="${task.storageKey}" data-field="caregiverSignature" type="text" maxlength="40" autocomplete="off" spellcheck="false" placeholder="ראשי תיבות" value="${escapeHtml(saved.caregiverSignature)}">
        </div>
      </div>
      <div class="checks">
        <label><input data-key="${task.storageKey}" data-field="completedByUser" type="checkbox" ${saved.completedByUser ? "checked" : ""}>ביצעתי את המשימה</label>
        <label><input data-key="${task.storageKey}" data-field="approvedByCaregiver" type="checkbox" ${saved.approvedByCaregiver ? "checked" : ""}>מטפל/ת אישר/ה</label>
      </div>
      <div class="task-actions">
        <button type="button" data-action="save-task" data-key="${task.storageKey}">שמירת חתימה כפולה</button>
      </div>
      <div class="task-status ${signed ? "signed" : ""}" id="status-${task.storageKey}">
        ${signed ? `נחתם: ${saved.signedAt}` : "ממתין לביצוע, אישור וחתימה כפולה"}
      </div>
    </article>
  `;
}

function renderWeek(referenceDate, calendarData, state) {
  const start = weekStart(referenceDate);
  const days = DAYS.map((name, offset) => {
    const date = new Date(start);
    date.setDate(start.getDate() + offset);
    const dayKey = isoDate(date);
    const tasks = WEEKLY_TEMPLATES
      .filter((task) => task.dayIndex === offset)
      .map((task) => contextualizeTask({
        ...task,
        storageKey: `weekly:${isoDate(start)}:${task.id}`,
      }, calendarData.eventsByDate?.[dayKey]));

    return { name, date, tasks };
  });

  nodes.weekRange.textContent = `${formatDate(days[0].date)} - ${formatDate(days[6].date)}`;
  nodes.weeklyGrid.innerHTML = days.map((day) => {
    const signedCount = day.tasks.filter((task) => getEntry(state, task.storageKey).signedAt).length;
    return `
      <section class="panel day-card">
        <div class="card-head">
          <div>
            <h3>${day.name}</h3>
            <div class="muted">${formatDate(day.date)}</div>
          </div>
          <div class="day-status">${signedCount}/${day.tasks.length} חתומים</div>
        </div>
        <div class="task-stack">
          ${day.tasks.map((task) => renderTask(task, getEntry(state, task.storageKey))).join("")}
        </div>
      </section>
    `;
  }).join("");

  return days;
}

function renderLaundry(referenceDate, calendarData, state) {
  const items = buildLaundryDates(referenceDate).map((date, index) => {
    const task = contextualizeTask({
      id: `laundry-${index + 1}`,
      task: "כביסה",
      slot: "גמיש",
      note: "סבב כביסה אחד מלא.",
      storageKey: `monthly:${referenceDate.getFullYear()}-${referenceDate.getMonth() + 1}:laundry-${index + 1}`,
    }, calendarData.eventsByDate?.[isoDate(date)]);
    return { date, task };
  });

  nodes.monthLabel.textContent = `${MONTHS[referenceDate.getMonth()]} ${referenceDate.getFullYear()}`;
  nodes.laundryGrid.innerHTML = items.map((item, index) => `
    <section class="panel laundry-card">
      <div class="card-head">
        <div>
          <h3>כביסה ${index + 1}</h3>
          <div class="muted">${formatDate(item.date)}</div>
        </div>
      </div>
      ${renderTask(item.task, getEntry(state, item.task.storageKey))}
    </section>
  `).join("");

  return items;
}

function updateSummary(weekDays, laundryItems, state) {
  const showerCount = weekDays.flatMap((day) => day.tasks).filter((task) => task.task === "מקלחת" && getEntry(state, task.storageKey).signedAt).length;
  const roomCount = weekDays.flatMap((day) => day.tasks).filter((task) => task.task === "סידור חדר" && getEntry(state, task.storageKey).signedAt).length;
  const laundryCount = laundryItems.filter((item) => getEntry(state, item.task.storageKey).signedAt).length;

  nodes.summaryShower.textContent = `${showerCount}/${TARGETS["מקלחת"]}`;
  nodes.summaryRoom.textContent = `${roomCount}/${TARGETS["סידור חדר"]}`;
  nodes.summaryLaundry.textContent = `${laundryCount}/${TARGETS["כביסה"]}`;
  nodes.summaryShowerBar.style.width = `${(showerCount / TARGETS["מקלחת"]) * 100}%`;
  nodes.summaryRoomBar.style.width = `${(roomCount / TARGETS["סידור חדר"]) * 100}%`;
  nodes.summaryLaundryBar.style.width = `${(laundryCount / TARGETS["כביסה"]) * 100}%`;
}

function updateFocus(referenceDate, weekDays, laundryItems, state) {
  const todayKey = isoDate(referenceDate);
  const day = weekDays[referenceDate.getDay()];
  const tasks = [
    ...day.tasks.map((task) => ({ label: `${task.task} - ${task.slot}`, done: Boolean(getEntry(state, task.storageKey).signedAt) })),
    ...laundryItems.filter((item) => isoDate(item.date) === todayKey).map((item) => ({ label: `כביסה - ${item.task.slot}`, done: Boolean(getEntry(state, item.task.storageKey).signedAt) })),
  ];

  nodes.focusTitle.textContent = `היום: ${DAYS[referenceDate.getDay()]}`;
  nodes.focusSubtitle.textContent = `תאריך גרגוריאני: ${formatDate(referenceDate)}`;

  if (!tasks.length) {
    nodes.focusList.innerHTML = "";
    nodes.focusEmpty.classList.remove("hidden");
    return;
  }

  nodes.focusEmpty.classList.add("hidden");
  nodes.focusList.innerHTML = tasks.map((task) => `<li>${task.label}${task.done ? " | כבר נחתם" : " | ממתין לחתימה"}</li>`).join("");
}

function updateCalendar(calendarData) {
  nodes.gregorianDate.textContent = formatDate(todayMidday());
  nodes.hebrewDate.textContent = calendarData.hebrewDate || "לא זמין";
  nodes.shabbatWindow.textContent = calendarData.windowText || "אין כרגע נתוני שבת/חג.";
  nodes.holidayList.textContent = calendarData.highlights?.length ? `אירועים בולטים: ${calendarData.highlights.join(" | ")}` : "אין חג מרכזי בטווח הקרוב.";

  if (calendarData.alertText) {
    nodes.calendarAlert.textContent = calendarData.alertText;
    nodes.calendarAlert.classList.remove("hidden");
  } else {
    nodes.calendarAlert.classList.add("hidden");
  }
}

function formValues(key) {
  const query = (field) => document.querySelector(`[data-key="${key}"][data-field="${field}"]`);
  return {
    selfSignature: query("selfSignature")?.value.trim() || "",
    caregiverSignature: query("caregiverSignature")?.value.trim() || "",
    completedByUser: Boolean(query("completedByUser")?.checked),
    approvedByCaregiver: Boolean(query("approvedByCaregiver")?.checked),
  };
}

async function fetchCalendar(referenceDate) {
  const response = await fetch(`/api/calendar?date=${isoDate(referenceDate)}`, {
    method: "GET",
    credentials: "same-origin",
    headers: { "Accept": "application/json" },
  });

  if (!response.ok) throw new Error("calendar-api-failed");
  return response.json();
}

async function renderApp() {
  const referenceDate = todayMidday();
  const state = loadState();
  let calendarData = { eventsByDate: {}, highlights: [], windowText: "", alertText: "" };

  try {
    calendarData = await fetchCalendar(referenceDate);
  } catch {
    calendarData.alertText = "לא ניתן היה לטעון כרגע את נתוני הלוח העברי וזמני שבת/חג. הממשק ממשיך לפעול במצב מקומי בלבד.";
  }

  updateCalendar(calendarData);
  const weekDays = renderWeek(referenceDate, calendarData, state);
  const laundryItems = renderLaundry(referenceDate, calendarData, state);
  updateSummary(weekDays, laundryItems, state);
  updateFocus(referenceDate, weekDays, laundryItems, state);
}

function resetByPrefix(prefix) {
  const state = loadState();
  const next = Object.fromEntries(Object.entries(state).filter(([key]) => !key.startsWith(prefix)));
  saveState(next);
}

document.addEventListener("click", async (event) => {
  const button = event.target.closest("[data-action='save-task']");
  if (!button) return;

  const key = button.dataset.key;
  const values = formValues(key);
  const statusNode = document.getElementById(`status-${key}`);

  if (!values.selfSignature || !values.caregiverSignature) {
    if (statusNode) statusNode.textContent = "יש למלא שתי חתימות בראשי תיבות.";
    return;
  }

  if (!values.completedByUser || !values.approvedByCaregiver) {
    if (statusNode) statusNode.textContent = "יש לסמן גם ביצוע וגם אישור מטפל/ת.";
    return;
  }

  const state = loadState();
  state[key] = { ...values, signedAt: new Date().toLocaleString("he-IL") };
  saveState(state);
  await renderApp();
});

nodes.printButton.addEventListener("click", () => window.print());
nodes.resetWeekButton.addEventListener("click", async () => {
  resetByPrefix(`weekly:${isoDate(weekStart(todayMidday()))}:`);
  await renderApp();
});
nodes.resetMonthButton.addEventListener("click", async () => {
  const today = todayMidday();
  resetByPrefix(`monthly:${today.getFullYear()}-${today.getMonth() + 1}:`);
  await renderApp();
});

renderApp();
