function sanitizeYear(value) {
  const year = Number.parseInt(value || "", 10);
  const currentYear = new Date().getFullYear();
  return Number.isInteger(year) && year >= 2020 && year <= currentYear + 2 ? year : currentYear;
}

function sanitizeMonth(value) {
  const month = Number.parseInt(value || "", 10);
  const currentMonth = new Date().getMonth() + 1;
  return Number.isInteger(month) && month >= 1 && month <= 12 ? month : currentMonth;
}

exports.handler = async function handler(event) {
  const year = sanitizeYear(event.queryStringParameters?.year);
  const month = sanitizeMonth(event.queryStringParameters?.month);
  const city = "IL-Tel+Aviv";

  try {
    const [shabbatResponse, holidayResponse] = await Promise.all([
      fetch(`https://www.hebcal.com/shabbat?cfg=json&geo=city&city=${city}&M=on&b=18`),
      fetch(`https://www.hebcal.com/hebcal?cfg=json&maj=on&min=on&mod=on&nx=on&year=${year}&month=${month}&geo=city&city=${city}`)
    ]);

    if (!shabbatResponse.ok || !holidayResponse.ok) {
      throw new Error("Failed to fetch Hebcal data");
    }

    const shabbatData = await shabbatResponse.json();
    const holidayData = await holidayResponse.json();

    const lighting = shabbatData.items?.find((item) => item.category === "candles") || null;
    const havdalah = shabbatData.items?.find((item) => item.category === "havdalah") || null;
    const holidays = (holidayData.items || [])
      .filter((item) => item.category !== "candles" && item.category !== "havdalah")
      .slice(0, 6)
      .map((item) => ({
        title: item.title,
        date: item.date
      }));

    return {
      statusCode: 200,
      headers: {
        "Content-Type": "application/json; charset=utf-8",
        "Cache-Control": "no-store"
      },
      body: JSON.stringify({
        shabbat: {
          lighting,
          havdalah
        },
        holidays
      })
    };
  } catch (error) {
    return {
      statusCode: 502,
      headers: {
        "Content-Type": "application/json; charset=utf-8",
        "Cache-Control": "no-store"
      },
      body: JSON.stringify({
        error: "Unable to load calendar data right now."
      })
    };
  }
};
