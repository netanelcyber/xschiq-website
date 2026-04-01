const STORAGE_KEY = "adl-planner-state-v2";
const SIGNATURES_KEY = "adl-planner-signatures-v2";

const ui = {
  gregorianDate: document.getElementById("gregorianDate"),
  hebrewDate: document.getElementById("hebrewDate"),
  shabbatWindow: document.getElementById("shabbatWindow"),
  holidayList: document.getElementById("holidayList"),
  calendarAlert: document.getElementById("calendarAlert"),
  summaryShower: document.getElementById("summaryShower"),
  summaryShowerBar: document.getElementById("summaryShowerBar"),
  summaryRoom: document.getElementById("summaryRoom"),
  summaryRoomBar: document.getElementById("summaryRoomBar"),
  summaryLaundry: document.getElementById("summaryLaundry"),
  summaryLaundryBar: document.getElementById("summaryLaundryBar"),
  focusTitle: document.getElementById("focusTitle"),
  focusSubtitle: document.getElementById("focusSubtitle"),
  focusList: document.getElementById("focusList"),
  focusEmpty: document.getElementById("focusEmpty"),
  breakdownTitle: document.getElementById("breakdownTitle"),
  breakdownBadge: document.getElementById("breakdownBadge"),
  breakdownSubtitle: document.getElementById("breakdownSubtitle"),
  subtaskProgressText: document.getElementById("subtaskProgressText"),
  subtaskProgressBar: document.getElementById("subtaskProgressBar"),
  subtaskList: document.getElementById("subtaskList"),
  signatureHint: document.getElementById("signatureHint"),
  selfSignButton: document.getElementById("selfSignButton"),
  caregiverSignButton: document.getElementById("caregiverSignButton"),
  completeSubtasksButton: document.getElementById("completeSubtasksButton"),
  clearSubtasksButton: document.getElementById("clearSubtasksButton"),
  weeklyGrid: document.getElementById("weeklyGrid"),
  weekRange: document.getElementById("weekRange"),
  laundryGrid: document.getElementById("laundryGrid"),
  monthLabel: document.getElementById("monthLabel"),
  printButton: document.getElementById("printButton"),
  resetWeekButton: document.getElementById("resetWeekButton"),
  resetMonthButton: document.getElementById("resetMonthButton")
};

const DAY_LABELS = ["יום א'", "יום ב'", "יום ג'", "יום ד'", "יום ה'", "יום ו'", "שבת"];
const SHOWER_PLAN = [
  [
    { title: "מקלחת בוקר", window: "07:00-09:00" },
    { title: "מקלחת ערב", window: "19:30-21:00" }
  ],
  [
    { title: "מקלחת בוקר", window: "07:00-09:00" },
    { title: "מקלחת ערב", window: "19:30-21:00" }
  ],
  [
    { title: "מקלחת בוקר", window: "07:00-09:00" },
    { title: "מקלחת ערב", window: "19:30-21:00" }
  ],
  [
    { title: "מקלחת בוקר", window: "07:00-09:00" },
    { title: "מקלחת ערב", window: "19:30-21:00" }
  ],
  [
    { title: "מקלחת בוקר", window: "07:00-09:00" },
    { title: "מקלחת ערב", window: "19:30-21:00" }
  ],
  [
    { title: "מקלחת בוקר", window: "07:30-09:30" },
    { title: "מקלחת ערב", window: "18:30-20:00" }
  ],
  [
    { title: "מקלחת יומית", window: "18:30-20:00" }
  ]
];

const ROOM_DAYS = new Set([0, 2, 4, 6]);
const LAUNDRY_DAYS = [2, 6, 10, 14, 18, 22, 26];

const SUBTASKS = {
  shower: [
    "להכין מגבת, בגדים נקיים וכלי רחצה מראש.",
    "להדליק מים בטמפרטורה נוחה ולהיכנס בלי דחייה נוספת.",
    "לסבן ולשטוף את הגוף לפי הסדר הקבוע.",
    "לחפוף או לשטוף את השיער לפי הצורך.",
    "להתנגב, ללבוש בגדים נקיים ולהעביר כביסה לסל."
  ],
  room: [
    "לאסוף אשפה, כלים ובגדים שלא במקום.",
    "לסדר את המיטה או פינת המנוחה.",
    "להחזיר חפצים למקום הקבוע שלהם.",
    "לנגב או לאוורר אזור אחד מרכזי בחדר."
  ],
  laundry: [
    "למיין כביסה לפי צבעים וסוגי בד.",
    "להפעיל מכונה עם התוכנית הנכונה.",
    "להעביר לייבוש או לתלייה בסיום.",
    "לקפל את הפריטים היבשים.",
    "להחזיר לארון או למקום אחסון מסודר."
  ]
};

const planner = {
  now: new Date(),
  weekStart: null,
  weekTasks: [],
  laundryTasks: [],
  allTasks: [],
  selectedTaskId: null,
  state: loadStore(localStorage, STORAGE_KEY),
  signatures: loadStore(sessionStorage, SIGNATURES_KEY)
};

initialize();

function initialize() {
  planner.weekStart = startOfWeek(planner.now);
  planner.weekTasks = buildWeekTasks(planner.weekStart);
  planner.laundryTasks = buildLaundryTasks(planner.now);
  planner.allTasks = [...planner.weekTasks, ...planner.laundryTasks];
  planner.allTasks.forEach(ensureTaskState);
  planner.selectedTaskId = pickInitialTaskId();
  bindEvents();
  renderAll();
  void hydrateCalendarNotes();
}

function bindEvents() {
  ui.printButton.addEventListener("click", () => window.print());
  ui.resetWeekButton.addEventListener("click", resetWeekTasks);
  ui.resetMonthButton.addEventListener("click", resetMonthTasks);

  ui.focusList.addEventListener("click", handleTaskSelection);
  ui.weeklyGrid.addEventListener("click", handleTaskSelection);
  ui.laundryGrid.addEventListener("click", handleTaskSelection);

  ui.subtaskList.addEventListener("change", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLInputElement)) {
      return;
    }

    const index = Number.parseInt(target.dataset.subtaskIndex || "", 10);
    if (Number.isNaN(index) || !planner.selectedTaskId) {
      return;
    }

    const state = getTaskState(planner.selectedTaskId);
    state.subtasks[index] = target.checked;
    saveStore(localStorage, STORAGE_KEY, planner.state);
    renderAll();
  });

  ui.selfSignButton.addEventListener("click", () => toggleSignature("self"));
  ui.caregiverSignButton.addEventListener("click", () => toggleSignature("caregiver"));
  ui.completeSubtasksButton.addEventListener("click", completeAllSubtasks);
  ui.clearSubtasksButton.addEventListener("click", clearSelectedTaskProgress);
}

function handleTaskSelection(event) {
  const trigger = event.target.closest("[data-task-id]");
  if (!trigger) {
    return;
  }

  planner.selectedTaskId = trigger.dataset.taskId;
  renderAll();
}

function buildWeekTasks(weekStart) {
  const tasks = [];

  SHOWER_PLAN.forEach((slots, offset) => {
    const date = addDays(weekStart, offset);
    const dateKey = formatDateKey(date);

    slots.forEach((slot, slotIndex) => {
      tasks.push({
        id: `week-${formatDateKey(weekStart)}-${dateKey}-shower-${slotIndex}`,
        category: "shower",
        group: "week",
        date,
        dateKey,
        dayLabel: DAY_LABELS[offset],
        title: slot.title,
        window: slot.window,
        description: "מקלחת עם רצף צעדים קבוע ואישור ביצוע כפול.",
        subtasks: [...SUBTASKS.shower]
      });
    });

    if (ROOM_DAYS.has(offset)) {
      tasks.push({
        id: `week-${formatDateKey(weekStart)}-${dateKey}-room`,
        category: "room",
        group: "week",
        date,
        dateKey,
        dayLabel: DAY_LABELS[offset],
        title: "סידור חדר",
        window: "17:00-19:00",
        description: "סידור קצר וממוקד של סביבת החדר.",
        subtasks: [...SUBTASKS.room]
      });
    }
  });

  return tasks;
}

function buildLaundryTasks(now) {
  const year = now.getFullYear();
  const month = now.getMonth();
  const daysInMonth = new Date(year, month + 1, 0).getDate();

  return LAUNDRY_DAYS.filter((day) => day <= daysInMonth).map((day, index) => {
    const date = new Date(year, month, day);
    return {
      id: `laundry-${year}-${month + 1}-${index + 1}`,
      category: "laundry",
      group: "month",
      date,
      dateKey: formatDateKey(date),
      dayLabel: `סבב ${index + 1}`,
      title: `כביסה ${index + 1}/7`,
      window: "במהלך היום",
      description: "מחזור כביסה חודשי עם פירוק לתתי-מטלות ואישור ביצוע.",
      subtasks: [...SUBTASKS.laundry]
    };
  });
}

function pickInitialTaskId() {
  const todayKey = formatDateKey(planner.now);
  const todayTask = planner.allTasks.find((task) => task.dateKey === todayKey);
  return todayTask ? todayTask.id : planner.allTasks[0]?.id || null;
}

function ensureTaskState(task) {
  if (!planner.state[task.id]) {
    planner.state[task.id] = {
      subtasks: task.subtasks.map(() => false)
    };
  }

  if (!planner.signatures[task.id]) {
    planner.signatures[task.id] = { self: false, caregiver: false };
  }
}

function getTaskState(taskId) {
  return planner.state[taskId];
}

function getSignatureState(taskId) {
  return planner.signatures[taskId];
}

function getTaskStatus(task) {
  const state = getTaskState(task.id);
  const signatures = getSignatureState(task.id);
  const completedSubtasks = state.subtasks.filter(Boolean).length;
  const totalSubtasks = task.subtasks.length;
  const fullySigned = signatures.self && signatures.caregiver;
  const completed = completedSubtasks === totalSubtasks && fullySigned;

  return {
    completedSubtasks,
    totalSubtasks,
    completed,
    signatures,
    progress: totalSubtasks ? Math.round((completedSubtasks / totalSubtasks) * 100) : 0
  };
}

function toggleSignature(type) {
  if (!planner.selectedTaskId) {
    return;
  }

  const signatures = getSignatureState(planner.selectedTaskId);
  signatures[type] = !signatures[type];
  saveStore(sessionStorage, SIGNATURES_KEY, planner.signatures);
  renderAll();
}

function completeAllSubtasks() {
  if (!planner.selectedTaskId) {
    return;
  }

  const task = getTaskById(planner.selectedTaskId);
  if (!task) {
    return;
  }

  planner.state[planner.selectedTaskId].subtasks = task.subtasks.map(() => true);
  saveStore(localStorage, STORAGE_KEY, planner.state);
  renderAll();
}

function clearSelectedTaskProgress() {
  if (!planner.selectedTaskId) {
    return;
  }

  const task = getTaskById(planner.selectedTaskId);
  if (!task) {
    return;
  }

  planner.state[planner.selectedTaskId].subtasks = task.subtasks.map(() => false);
  planner.signatures[planner.selectedTaskId] = { self: false, caregiver: false };
  saveStore(localStorage, STORAGE_KEY, planner.state);
  saveStore(sessionStorage, SIGNATURES_KEY, planner.signatures);
  renderAll();
}

function resetWeekTasks() {
  planner.weekTasks.forEach((task) => {
    planner.state[task.id] = { subtasks: task.subtasks.map(() => false) };
    planner.signatures[task.id] = { self: false, caregiver: false };
  });
  saveStore(localStorage, STORAGE_KEY, planner.state);
  saveStore(sessionStorage, SIGNATURES_KEY, planner.signatures);
  renderAll();
}

function resetMonthTasks() {
  planner.laundryTasks.forEach((task) => {
    planner.state[task.id] = { subtasks: task.subtasks.map(() => false) };
    planner.signatures[task.id] = { self: false, caregiver: false };
  });
  saveStore(localStorage, STORAGE_KEY, planner.state);
  saveStore(sessionStorage, SIGNATURES_KEY, planner.signatures);
  renderAll();
}

function renderAll() {
  renderDates();
  renderSummaries();
  renderFocus();
  renderWeeklyGrid();
  renderLaundryGrid();
  renderBreakdown();
}

function renderDates() {
  ui.gregorianDate.textContent = new Intl.DateTimeFormat("he-IL", {
    weekday: "long",
    day: "numeric",
    month: "long",
    year: "numeric"
  }).format(planner.now);

  ui.hebrewDate.textContent = new Intl.DateTimeFormat("he-IL-u-ca-hebrew", {
    weekday: "long",
    day: "numeric",
    month: "long",
    year: "numeric"
  }).format(planner.now);

  const weekEnd = addDays(planner.weekStart, 6);
  ui.weekRange.textContent = `${formatShortDate(planner.weekStart)} - ${formatShortDate(weekEnd)}`;
  ui.monthLabel.textContent = new Intl.DateTimeFormat("he-IL", {
    month: "long",
    year: "numeric"
  }).format(planner.now);
}

function renderSummaries() {
  const categories = [
    { name: "shower", label: ui.summaryShower, bar: ui.summaryShowerBar, tasks: planner.weekTasks.filter((task) => task.category === "shower") },
    { name: "room", label: ui.summaryRoom, bar: ui.summaryRoomBar, tasks: planner.weekTasks.filter((task) => task.category === "room") },
    { name: "laundry", label: ui.summaryLaundry, bar: ui.summaryLaundryBar, tasks: planner.laundryTasks }
  ];

  categories.forEach(({ label, bar, tasks }) => {
    const done = tasks.filter((task) => getTaskStatus(task).completed).length;
    const total = tasks.length;
    label.textContent = `${done}/${total}`;
    bar.style.width = `${total ? (done / total) * 100 : 0}%`;
  });
}

function renderFocus() {
  const todayKey = formatDateKey(planner.now);
  const todayTasks = planner.allTasks.filter((task) => task.dateKey === todayKey);

  ui.focusList.innerHTML = "";
  ui.focusEmpty.classList.toggle("hidden", todayTasks.length > 0);

  if (todayTasks.length === 0) {
    ui.focusTitle.textContent = "היום פנוי ממשימות קבועות";
    ui.focusSubtitle.textContent = "אפשר להתכונן למטלות של מחר או להשלים סבב חודשי אם צריך.";
    return;
  }

  ui.focusTitle.textContent = `היום יש ${todayTasks.length} מטלות פעילות`;
  ui.focusSubtitle.textContent = "כל מטלה נפתחת לפירוק ברור לתת-מטלות כדי להוריד עומס ולהפוך ביצוע לתהליך קצר וברור.";

  todayTasks.forEach((task) => {
    const status = getTaskStatus(task);
    const item = document.createElement("li");
    item.innerHTML = `
      <button type="button" class="focus-task-button" data-task-id="${task.id}">
        <strong>${task.title}</strong>
        <span class="task-meta">${task.window} · ${status.completedSubtasks}/${status.totalSubtasks} תתי-מטלות</span>
      </button>
    `;
    ui.focusList.appendChild(item);
  });
}

function renderWeeklyGrid() {
  ui.weeklyGrid.innerHTML = "";

  for (let offset = 0; offset < 7; offset += 1) {
    const date = addDays(planner.weekStart, offset);
    const dateKey = formatDateKey(date);
    const tasks = planner.weekTasks.filter((task) => task.dateKey === dateKey);
    const dayCard = document.createElement("article");
    dayCard.className = `panel day-card${dateKey === formatDateKey(planner.now) ? " today" : ""}`;

    const taskMarkup = tasks.map((task) => renderTaskMarkup(task)).join("");

    dayCard.innerHTML = `
      <div class="task-topline">
        <div>
          <div class="day-label">${DAY_LABELS[offset]}</div>
          <div class="task-meta">${formatShortDate(date)}</div>
        </div>
        ${dateKey === formatDateKey(planner.now) ? '<span class="mini-badge done">היום</span>' : ""}
      </div>
      <div class="task-stack">${taskMarkup}</div>
    `;

    ui.weeklyGrid.appendChild(dayCard);
  }
}

function renderLaundryGrid() {
  ui.laundryGrid.innerHTML = "";

  planner.laundryTasks.forEach((task) => {
    const card = document.createElement("article");
    const current = task.dateKey === formatDateKey(planner.now) ? " is-current" : "";
    card.className = `panel laundry-card${current}`;
    card.innerHTML = renderTaskMarkup(task, {
      heading: `<div class="task-topline"><div><div class="day-label">${task.title}</div><div class="task-meta">${formatShortDate(task.date)}</div></div></div>`
    });
    ui.laundryGrid.appendChild(card);
  });
}

function renderTaskMarkup(task, options = {}) {
  const status = getTaskStatus(task);
  const stateClass = status.completed ? "done" : status.completedSubtasks > 0 ? "pending" : "";

  return `
    ${options.heading || ""}
    <div class="task-shell">
      <div class="task-topline">
        <div>
          <div class="task-title">${task.title}</div>
          <span class="task-meta">${task.window} · ${task.description}</span>
        </div>
        <span class="status-pill ${stateClass || "pending"}">${status.completed ? "הושלם" : `${status.completedSubtasks}/${status.totalSubtasks}`}</span>
      </div>
      <div class="status-row" style="margin-top: 10px;">
        <span class="sign-pill ${status.signatures.self ? "active" : ""}">אני</span>
        <span class="sign-pill ${status.signatures.caregiver ? "active" : ""}">מטפל/ת</span>
      </div>
      <button type="button" class="task-open ${planner.selectedTaskId === task.id ? "active" : ""}" data-task-id="${task.id}">
        פתח/י פירוק מטלה
      </button>
    </div>
  `;
}

function renderBreakdown() {
  const task = getTaskById(planner.selectedTaskId);

  if (!task) {
    ui.breakdownTitle.textContent = "אין מטלה פעילה";
    ui.breakdownBadge.textContent = "0/0";
    ui.breakdownSubtitle.textContent = "בחרו מטלה כדי לראות פירוק ברור לתת-מטלות.";
    ui.subtaskList.innerHTML = "";
    return;
  }

  const status = getTaskStatus(task);
  ui.breakdownTitle.textContent = task.title;
  ui.breakdownSubtitle.textContent = `${task.dayLabel} · ${formatShortDate(task.date)} · ${task.window} · ${task.description}`;
  ui.breakdownBadge.textContent = `${status.completedSubtasks}/${status.totalSubtasks}`;
  ui.breakdownBadge.className = `mini-badge ${status.completed ? "done" : status.completedSubtasks > 0 ? "warning" : ""}`.trim();
  ui.subtaskProgressText.textContent = `${status.completedSubtasks}/${status.totalSubtasks}`;
  ui.subtaskProgressBar.style.width = `${status.progress}%`;

  const subtasks = getTaskState(task.id).subtasks;
  ui.subtaskList.innerHTML = task.subtasks.map((subtask, index) => `
    <li class="subtask-item ${subtasks[index] ? "done" : ""}">
      <label>
        <input type="checkbox" data-subtask-index="${index}" ${subtasks[index] ? "checked" : ""}>
        <span>${subtask}</span>
      </label>
    </li>
  `).join("");

  const signatures = getSignatureState(task.id);
  ui.selfSignButton.classList.toggle("active", signatures.self);
  ui.caregiverSignButton.classList.toggle("active", signatures.caregiver);
  ui.signatureHint.textContent = status.completed
    ? "המטלה הושלמה במלואה ונחתמה על ידי שני הצדדים."
    : "ביצוע מלא של מטלה נשמר רק לאחר סימון תתי-המשימות והחתימות הדרושות.";
}

async function hydrateCalendarNotes() {
  try {
    const response = await fetch(`/api/calendar?year=${planner.now.getFullYear()}&month=${planner.now.getMonth() + 1}`);
    const calendarData = await response.json();

    if (!response.ok) {
      throw new Error(calendarData.error || "Calendar fetch failed");
    }

    const lighting = calendarData.shabbat?.lighting;
    const havdalah = calendarData.shabbat?.havdalah;

    if (lighting && havdalah) {
      ui.shabbatWindow.textContent = `${lighting.title} · ${formatTime(new Date(lighting.date))} | ${havdalah.title} · ${formatTime(new Date(havdalah.date))}`;

      const hoursUntilLighting = (new Date(lighting.date).getTime() - planner.now.getTime()) / 36e5;
      if (hoursUntilLighting > 0 && hoursUntilLighting < 30) {
        ui.calendarAlert.textContent = "תזכורת: שבת או חג נכנסים בקרוב. כדאי להקדים משימות רועשות כמו כביסה וסידור.";
        ui.calendarAlert.classList.remove("hidden");
      }
    }

    const holidayItems = (calendarData.holidays || []).slice(0, 4).map((item) => item.title);

    ui.holidayList.textContent = holidayItems.length
      ? `אירועים קרובים: ${holidayItems.join(" · ")}`
      : "לא זוהו אירועים מיוחדים קרובים החודש.";
  } catch (error) {
    ui.shabbatWindow.textContent = "לא ניתן היה לטעון כרגע זמני שבת וחג. אפשר לנסות שוב בהמשך.";
    ui.holidayList.textContent = "לא ניתן היה לטעון כרגע אירועי חודש.";
  }
}

function getTaskById(taskId) {
  return planner.allTasks.find((task) => task.id === taskId) || null;
}

function loadStore(store, key) {
  try {
    const value = store.getItem(key);
    return value ? JSON.parse(value) : {};
  } catch (error) {
    return {};
  }
}

function saveStore(store, key, value) {
  try {
    store.setItem(key, JSON.stringify(value));
  } catch (error) {
    // Ignore storage failures in private browsing modes.
  }
}

function startOfWeek(date) {
  const start = new Date(date);
  start.setHours(0, 0, 0, 0);
  start.setDate(start.getDate() - start.getDay());
  return start;
}

function addDays(date, days) {
  const next = new Date(date);
  next.setDate(next.getDate() + days);
  next.setHours(0, 0, 0, 0);
  return next;
}

function formatDateKey(date) {
  const year = date.getFullYear();
  const month = `${date.getMonth() + 1}`.padStart(2, "0");
  const day = `${date.getDate()}`.padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function formatShortDate(date) {
  return new Intl.DateTimeFormat("he-IL", {
    day: "numeric",
    month: "short"
  }).format(date);
}

function formatTime(date) {
  return new Intl.DateTimeFormat("he-IL", {
    hour: "2-digit",
    minute: "2-digit"
  }).format(date);
}
