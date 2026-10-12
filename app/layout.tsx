import type { Metadata } from "next"
import "./globals.css"

export const metadata: Metadata = {
  title: "CFB-PIM FBS Terminal | College Football Forecasting",
  description: "Research-grade college football forecasts, market diagnostics, and historical validation.",
}

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en" className="dark"><body>{children}</body></html>
}
