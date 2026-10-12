import { NextResponse } from "next/server"

export const dynamic = "force-dynamic"

const API = process.env.CFB_FORECAST_API_URL

export async function GET(request: Request) {
  if (!API) {
    return NextResponse.json(
      {
        error: "Authoritative forecast API is not configured",
        detail: "Set CFB_FORECAST_API_URL. The web application no longer calculates a second forecast from scoreboard data.",
      },
      { status: 503 },
    )
  }

  const { searchParams } = new URL(request.url)
  const season = searchParams.get("season") ?? "2026"
  const week = searchParams.get("week")
  const upstream = new URL("/v1/forecast", API)
  upstream.searchParams.set("season", season)
  if (week) upstream.searchParams.set("week", week)

  try {
    const response = await fetch(upstream, { cache: "no-store" })
    const body = await response.text()
    return new NextResponse(body, {
      status: response.status,
      headers: { "content-type": response.headers.get("content-type") ?? "application/json" },
    })
  } catch (error) {
    return NextResponse.json(
      {
        error: "Authoritative forecast API unavailable",
        detail: error instanceof Error ? error.message : "Unknown API error",
      },
      { status: 503 },
    )
  }
}
