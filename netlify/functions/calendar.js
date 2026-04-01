const TEL_AVIV_GEONAMEID = "293397";

function isIsoDate(value) {
  return /^\d{4}-\d{2}-\d{2}$/.test(value || "");
}

function asDate(value) {
  return new Date(`${value}T12:00:00+03:00`);
}

function isoDate(date) {
  return date.toISOString().slice(0, 10);
}

function fmtTime(value) {
  return new Intl.DateTimeFormat("he-IL", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
    timeZone: "Asia/Jerusalem",
  }).format(new Date(value));
}

function pairWindows(items, now) {
  const timed = items
    .filter((item) => item.category === "candles" || item.category === "havdalah")
    .map((item) => ({
      kind: item.category === "candles" ? "start" : "end",
      title: item.title,
      date: new Date(item.date),
      time: fmtTime(item.date),
    }))
    .sort((a, b) => a.date - b.date);

  let active = null;
  let nextStart = null;
  let nextEnd = null;

  for (let index = 0; index < timed.length; index += 1) {
    const current = timed[index];
    if (current.kind === "start") {
      const end = timed.slice(index + 1).find((item) => item.kind === "end");
      if (end && current.date <= now && now < end.date) {
        active = { start: current, end };
        break;
      }
      if (!nextStart && current.date > now) nextStart = current;
    }
    if (current.kind === "end" && !nextEnd && current.date > now) {
      nextEnd = current;
    }
  }

  return { active, nextStart, nextEnd };
}

exports.handler = async function handler(event) {
  if (event.httpMethod !== "GET") {
    return { statusCode: 405, body: JSON.stringify({ error: "Method not allowed" }) };
  }

  const dateParam = event.queryStringParameters?.date;
  if (!isIsoDate(dateParam)) {
    return { statusCode: 400, body: JSON.stringify({ error: "Invalid date" }) };
  }

  const referenceDate = asDate(dateParam);
  const monthStart = new Date(referenceDate.getFullYear(), referenceDate.getMonth(), 1, 12, 0, 0, 0);
  const monthEnd = new Date(referenceDate.getFullYear(), referenceDate.getMonth() + 1, 0, 12, 0, 0, 0);
  const rangeEnd = new Date(monthEnd);
  rangeEnd.setDate(rangeEnd.getDate() + 7);

  const converterUrl = `https://www.hebcal.com/converter?cfg=json&date=${dateParam}&g2h=1`;
  const calendarUrl = `https://www.hebcal.com/hebcal?cfg=json&start=${isoDate(monthStart)}&end=${isoDate(rangeEnd)}&maj=on&min=on&mod=on&nx=on&mf=off&ss=off&c=on&M=on&i=on&lg=h&geo=geoname&geonameid=${TEL_AVIV_GEONAMEID}`;

  try {
    const [converterResp, calendarResp] = await Promise.all([fetch(converterUrl), fetch(calendarUrl)]);
    if (!converterResp.ok || !calendarResp.ok) {
      throw new Error("Upstream calendar provider error");
    }

    const converter = await converterResp.json();
    const calendar = await calendarResp.json();
    const items = Array.isArray(calendar.items) ? calendar.items : [];
    const eventsByDate = {};
    const highlightSet = new Set();

    for (const item of items) {
      const dayKey = String(item.date).slice(0, 10);
      eventsByDate[dayKey] = eventsByDate[dayKey] || { holidays: [] };

      if (item.category === "candles") {
        eventsByDate[dayKey].candles = { title: item.title, time: fmtTime(item.date) };
      } else if (item.category === "havdalah") {
        eventsByDate[dayKey].havdalah = { title: item.title, time: fmtTime(item.date) };
      } else if (item.category === "holiday") {
        eventsByDate[dayKey].holidays.push(item.title);
        if (highlightSet.size < 6) highlightSet.add(item.title);
      }
    }

    const now = new Date();
    const windows = pairWindows(items, now);
    let windowText = "אין כרגע חלון שבת/חג פעיל.";
    let alertText = "";

    if (windows.active) {
      windowText = `כעת שבת/חג. התחיל ב-${windows.active.start.time} ומסתיים ב-${windows.active.end.time}.`;
      alertText = "כעת חל זמן שבת/חג בתל אביב. מומלץ לעבוד עם עותק מודפס ולבצע חתימה דיגיטלית רק אחרי צאת שבת/חג.";
    } else if (windows.nextStart || windows.nextEnd) {
      const startText = windows.nextStart ? `כניסה קרובה: ${windows.nextStart.title} ב-${windows.nextStart.time}` : "";
      const endText = windows.nextEnd ? `צאת שבת/חג קרובה: ${windows.nextEnd.time}` : "";
      windowText = [startText, endText].filter(Boolean).join(" | ");
      if (windows.nextStart && String(windows.nextStart.date.toISOString()).slice(0, 10) === dateParam) {
        alertText = `היום יש כניסת שבת/חג ב-${windows.nextStart.time}. משימות ערב וגמישות יש להשלים לפני הזמן הזה.`;
      }
    }

    return {
      statusCode: 200,
      headers: {
        "Content-Type": "application/json; charset=utf-8",
        "Cache-Control": "public, max-age=900, stale-while-revalidate=60",
      },
      body: JSON.stringify({
        hebrewDate: converter.hebrew || `${converter.hd || ""} ${converter.hm || ""} ${converter.hy || ""}`.trim(),
        eventsByDate,
        highlights: Array.from(highlightSet),
        windowText,
        alertText,
      }),
    };
  } catch (error) {
    return {
      statusCode: 502,
      headers: { "Content-Type": "application/json; charset=utf-8" },
      body: JSON.stringify({ error: "Calendar fetch failed" }),
    };
  }
};
